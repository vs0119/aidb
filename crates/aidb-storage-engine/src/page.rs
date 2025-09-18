use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU32, Ordering};

use bytemuck::{Pod, Zeroable};
use tokio::fs::{File, OpenOptions};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::sync::RwLock;

use crate::{PageId, Result, StorageEngineError, VectorRow};

pub const PAGE_SIZE: usize = 8192; // 8KB pages like PostgreSQL
pub const MAX_ROWS_PER_PAGE: usize = 256;
pub const PAGE_HEADER_SIZE: usize = 64;
pub const SLOT_SIZE: usize = 4;

#[derive(Debug, Clone, Copy, Zeroable)]
#[repr(C, packed)]
pub struct PageHeader {
    pub page_id: PageId,
    pub page_type: u8,
    pub flags: u8,
    pub reserved: u16,
    pub lower_offset: u16, // End of slot array
    pub upper_offset: u16, // Start of free space from end
    pub row_count: u16,
    pub free_space: u16,
    pub checksum: u32,
    pub lsn: u64, // Log sequence number
    pub reserved2: [u8; 32],
}

impl Default for PageHeader {
    fn default() -> Self {
        Self {
            page_id: 0,
            page_type: 0,
            flags: 0,
            reserved: 0,
            lower_offset: PAGE_HEADER_SIZE as u16,
            upper_offset: PAGE_SIZE as u16,
            row_count: 0,
            free_space: (PAGE_SIZE - PAGE_HEADER_SIZE) as u16,
            checksum: 0,
            lsn: 0,
            reserved2: [0; 32],
        }
    }
}

#[derive(Debug, Clone, Copy, Pod, Zeroable)]
#[repr(C)]
pub struct SlotPointer {
    pub offset: u16,
    pub length: u16,
}

impl Default for SlotPointer {
    fn default() -> Self {
        Self {
            offset: 0,
            length: 0,
        }
    }
}

#[derive(Debug)]
pub struct Page {
    pub page_id: PageId,
    pub data: Vec<u8>,
    pub dirty: bool,
    pub pin_count: AtomicU32,
}

impl Page {
    pub fn new(page_id: PageId) -> Self {
        let mut data = vec![0u8; PAGE_SIZE];
        let mut header = PageHeader::default();
        header.page_id = page_id;

        unsafe {
            std::ptr::copy_nonoverlapping(
                &header as *const _ as *const u8,
                data.as_mut_ptr(),
                std::mem::size_of::<PageHeader>(),
            );
        }

        Self {
            page_id,
            data,
            dirty: false,
            pin_count: AtomicU32::new(0),
        }
    }

    pub fn from_bytes(page_id: PageId, bytes: Vec<u8>) -> Result<Self> {
        if bytes.len() != PAGE_SIZE {
            return Err(StorageEngineError::PageCorruption(format!(
                "Invalid page size: {} bytes",
                bytes.len()
            )));
        }

        let header = unsafe { *(bytes.as_ptr() as *const PageHeader) };

        let header_page_id = header.page_id;
        if header_page_id != page_id {
            return Err(StorageEngineError::PageCorruption(format!(
                "Page ID mismatch: expected {}, found {}",
                page_id, header_page_id
            )));
        }

        if !Self::verify_checksum(&bytes) {
            return Err(StorageEngineError::PageCorruption(
                "Invalid checksum".to_string(),
            ));
        }

        Ok(Self {
            page_id,
            data: bytes,
            dirty: false,
            pin_count: AtomicU32::new(0),
        })
    }

    pub fn header(&self) -> &PageHeader {
        unsafe { &*(self.data.as_ptr() as *const PageHeader) }
    }

    pub fn header_mut(&mut self) -> &mut PageHeader {
        self.dirty = true;
        unsafe { &mut *(self.data.as_mut_ptr() as *mut PageHeader) }
    }

    pub fn get_slot(&self, slot_id: u16) -> Option<SlotPointer> {
        let header = self.header();
        if slot_id >= header.row_count {
            return None;
        }

        let slot_offset = PAGE_HEADER_SIZE + (slot_id as usize * SLOT_SIZE);
        if slot_offset + SLOT_SIZE > self.data.len() {
            return None;
        }

        Some(unsafe { *(self.data.as_ptr().add(slot_offset) as *const SlotPointer) })
    }

    pub fn set_slot(&mut self, slot_id: u16, slot: SlotPointer) {
        self.dirty = true;
        let slot_offset = PAGE_HEADER_SIZE + (slot_id as usize * SLOT_SIZE);

        unsafe {
            std::ptr::copy_nonoverlapping(
                &slot as *const _ as *const u8,
                self.data.as_mut_ptr().add(slot_offset),
                SLOT_SIZE,
            );
        }
    }

    pub fn get_row_data(&self, slot_id: u16) -> Option<&[u8]> {
        let slot = self.get_slot(slot_id)?;
        if slot.offset == 0 || slot.length == 0 {
            return None;
        }

        let start = slot.offset as usize;
        let end = start + slot.length as usize;

        if end > self.data.len() {
            return None;
        }

        Some(&self.data[start..end])
    }

    pub fn insert_row(&mut self, row: &VectorRow) -> Result<u16> {
        let serialized = bincode::serialize(row).map_err(|e| {
            StorageEngineError::PageCorruption(format!("Serialization failed: {e}"))
        })?;

        let needed_space = serialized.len() + SLOT_SIZE;

        let (slot_id, data_offset) = {
            let header = self.header();

            if header.free_space < needed_space as u16 {
                return Err(StorageEngineError::OutOfSpace);
            }

            if header.row_count >= MAX_ROWS_PER_PAGE as u16 {
                return Err(StorageEngineError::OutOfSpace);
            }

            let slot_id = header.row_count;
            let data_offset = header.upper_offset - serialized.len() as u16;
            (slot_id, data_offset)
        };

        unsafe {
            std::ptr::copy_nonoverlapping(
                serialized.as_ptr(),
                self.data.as_mut_ptr().add(data_offset as usize),
                serialized.len(),
            );
        }

        let slot = SlotPointer {
            offset: data_offset,
            length: serialized.len() as u16,
        };
        self.set_slot(slot_id, slot);

        let header = self.header_mut();
        header.row_count += 1;
        header.upper_offset = data_offset;
        header.lower_offset += SLOT_SIZE as u16;
        header.free_space -= needed_space as u16;

        Ok(slot_id)
    }

    pub fn get_row(&self, slot_id: u16) -> Result<Option<VectorRow>> {
        if let Some(data) = self.get_row_data(slot_id) {
            let row: VectorRow = bincode::deserialize(data).map_err(|e| {
                StorageEngineError::PageCorruption(format!("Deserialization failed: {e}"))
            })?;
            Ok(Some(row))
        } else {
            Ok(None)
        }
    }

    pub fn update_row(&mut self, slot_id: u16, row: &VectorRow) -> Result<()> {
        let serialized = bincode::serialize(row).map_err(|e| {
            StorageEngineError::PageCorruption(format!("Serialization failed: {e}"))
        })?;

        if let Some(slot) = self.get_slot(slot_id) {
            if serialized.len() <= slot.length as usize {
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        serialized.as_ptr(),
                        self.data.as_mut_ptr().add(slot.offset as usize),
                        serialized.len(),
                    );
                }
                self.dirty = true;
                return Ok(());
            }
        }

        Err(StorageEngineError::OutOfSpace)
    }

    pub fn delete_row(&mut self, slot_id: u16) -> Result<bool> {
        if slot_id >= self.header().row_count {
            return Ok(false);
        }

        let slot = SlotPointer {
            offset: 0,
            length: 0,
        };
        self.set_slot(slot_id, slot);

        Ok(true)
    }

    pub fn compact(&mut self) -> Result<()> {
        let mut valid_rows = Vec::new();
        let header = self.header();

        for slot_id in 0..header.row_count {
            if let Some(row) = self.get_row(slot_id)? {
                if let Some(slot) = self.get_slot(slot_id) {
                    if slot.offset != 0 && slot.length != 0 {
                        valid_rows.push(row);
                    }
                }
            }
        }

        *self = Self::new(self.page_id);

        for row in valid_rows {
            self.insert_row(&row)?;
        }

        Ok(())
    }

    pub fn calculate_checksum(&mut self) {
        {
            let header = self.header_mut();
            header.checksum = 0;
        }

        let checksum = crc32fast::hash(&self.data);
        let header = self.header_mut();
        header.checksum = checksum;
    }

    pub fn verify_checksum(data: &[u8]) -> bool {
        let header = unsafe { &*(data.as_ptr() as *const PageHeader) };
        let stored_checksum = header.checksum;

        let mut temp_data = data.to_vec();
        let temp_header = unsafe { &mut *(temp_data.as_mut_ptr() as *mut PageHeader) };
        temp_header.checksum = 0;

        let calculated_checksum = crc32fast::hash(&temp_data);
        stored_checksum == calculated_checksum
    }

    pub fn free_space(&self) -> usize {
        self.header().free_space as usize
    }

    pub fn pin(&self) -> u32 {
        self.pin_count.fetch_add(1, Ordering::AcqRel)
    }

    pub fn unpin(&self) -> u32 {
        self.pin_count.fetch_sub(1, Ordering::AcqRel)
    }

    pub fn is_pinned(&self) -> bool {
        self.pin_count.load(Ordering::Acquire) > 0
    }
}

pub struct PageManager {
    data_dir: PathBuf,
    next_page_id: AtomicU32,
    file_handles: RwLock<HashMap<String, File>>,
}

impl PageManager {
    pub async fn new(data_dir: impl AsRef<Path>) -> Result<Self> {
        let data_dir = data_dir.as_ref().to_path_buf();

        tokio::fs::create_dir_all(&data_dir).await?;

        Ok(Self {
            data_dir,
            next_page_id: AtomicU32::new(1),
            file_handles: RwLock::new(HashMap::new()),
        })
    }

    pub async fn allocate_page(&self) -> Result<PageId> {
        Ok(self.next_page_id.fetch_add(1, Ordering::AcqRel))
    }

    pub async fn read_page(&self, page_id: PageId) -> Result<Page> {
        let file_path = self.page_file_path(page_id);

        if !file_path.exists() {
            return Ok(Page::new(page_id));
        }

        let mut file = OpenOptions::new().read(true).open(&file_path).await?;

        let mut buffer = vec![0u8; PAGE_SIZE];
        file.read_exact(&mut buffer).await?;

        Page::from_bytes(page_id, buffer)
    }

    pub async fn write_page(&self, page: &mut Page) -> Result<()> {
        if !page.dirty {
            return Ok(());
        }

        page.calculate_checksum();

        let file_path = self.page_file_path(page.page_id);

        if let Some(parent) = file_path.parent() {
            tokio::fs::create_dir_all(parent).await?;
        }

        let mut file = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(&file_path)
            .await?;

        file.write_all(&page.data).await?;
        file.sync_all().await?;

        page.dirty = false;
        Ok(())
    }

    fn page_file_path(&self, page_id: PageId) -> PathBuf {
        let shard = page_id / 1000;
        self.data_dir
            .join("pages")
            .join(format!("shard_{:04}", shard))
            .join(format!("page_{:08}.dat", page_id))
    }
}
