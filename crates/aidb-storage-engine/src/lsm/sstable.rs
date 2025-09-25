use std::cmp::Ordering;
use std::collections::BTreeMap;
use std::fs::{self, File, OpenOptions};
use std::io::{self, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use serde::{Deserialize, Serialize};

use super::bloom::BloomFilter;
use super::memtable::MemTableEntry;

const SSTABLE_MAGIC: u32 = 0xA1DB5A01;
const FOOTER_SIZE: i64 = (8 + 4 + 8 + 4 + 4 + 4) as i64;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SSTableEntry {
    pub key: Vec<u8>,
    pub value: Option<Vec<u8>>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub(crate) struct BlockMeta {
    offset: u64,
    length: u32,
    first_key: Vec<u8>,
    last_key: Vec<u8>,
    entry_count: u32,
}

#[derive(Clone)]
pub struct SSTable {
    id: u64,
    path: PathBuf,
    block_size: usize,
    blocks: Vec<BlockMeta>,
    bloom: BloomFilter,
}

impl SSTable {
    #[allow(clippy::too_many_arguments)]
    pub fn create(
        path: impl AsRef<Path>,
        id: u64,
        entries: &[MemTableEntry],
        block_size: usize,
        bloom_bits_per_key: usize,
        bloom_hashes: u32,
    ) -> io::Result<Self> {
        let path = path.as_ref().to_path_buf();
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        let mut file = OpenOptions::new()
            .create(true)
            .truncate(true)
            .write(true)
            .open(&path)?;

        let mut blocks = Vec::new();
        let mut offset = 0u64;
        let chunk_size = block_size.max(1);
        for chunk in entries.chunks(chunk_size) {
            if chunk.is_empty() {
                continue;
            }
            let sstable_entries: Vec<SSTableEntry> = chunk
                .iter()
                .map(|entry| SSTableEntry {
                    key: entry.key.clone(),
                    value: entry.value.clone(),
                })
                .collect();
            let encoded = bincode::serialize(&sstable_entries).map_err(|err| {
                io::Error::new(io::ErrorKind::Other, format!("serialize block: {err}"))
            })?;
            let len = encoded.len() as u32;
            file.write_all(&len.to_le_bytes())?;
            file.write_all(&encoded)?;
            blocks.push(BlockMeta {
                offset,
                length: len,
                first_key: chunk.first().map(|e| e.key.clone()).unwrap_or_default(),
                last_key: chunk.last().map(|e| e.key.clone()).unwrap_or_default(),
                entry_count: chunk.len() as u32,
            });
            offset += 4 + len as u64;
        }

        let index_bytes = bincode::serialize(&blocks).map_err(|err| {
            io::Error::new(io::ErrorKind::Other, format!("serialize index: {err}"))
        })?;
        let bloom = BloomFilter::from_keys(
            entries.iter().map(|entry| entry.key.as_slice()),
            bloom_bits_per_key,
            bloom_hashes,
        );
        let bloom_bytes = bincode::serialize(&bloom).map_err(|err| {
            io::Error::new(io::ErrorKind::Other, format!("serialize bloom: {err}"))
        })?;

        let index_offset = offset;
        file.write_all(&index_bytes)?;
        offset += index_bytes.len() as u64;
        let bloom_offset = offset;
        file.write_all(&bloom_bytes)?;

        file.write_all(&index_offset.to_le_bytes())?;
        file.write_all(&(index_bytes.len() as u32).to_le_bytes())?;
        file.write_all(&bloom_offset.to_le_bytes())?;
        file.write_all(&(bloom_bytes.len() as u32).to_le_bytes())?;
        file.write_all(&(block_size as u32).to_le_bytes())?;
        file.write_all(&SSTABLE_MAGIC.to_le_bytes())?;
        file.sync_data()?;

        Ok(Self {
            id,
            path,
            block_size,
            blocks,
            bloom,
        })
    }

    pub fn open(path: impl AsRef<Path>, id: u64) -> io::Result<Self> {
        let path = path.as_ref().to_path_buf();
        let mut file = File::open(&path)?;
        if file.metadata()?.len() < FOOTER_SIZE as u64 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "sstable file too small",
            ));
        }
        file.seek(SeekFrom::End(-FOOTER_SIZE))?;
        let mut footer_buf = [0u8; FOOTER_SIZE as usize];
        file.read_exact(&mut footer_buf)?;
        let index_offset = u64::from_le_bytes(footer_buf[0..8].try_into().unwrap());
        let index_len = u32::from_le_bytes(footer_buf[8..12].try_into().unwrap()) as usize;
        let bloom_offset = u64::from_le_bytes(footer_buf[12..20].try_into().unwrap());
        let bloom_len = u32::from_le_bytes(footer_buf[20..24].try_into().unwrap()) as usize;
        let block_size = u32::from_le_bytes(footer_buf[24..28].try_into().unwrap()) as usize;
        let magic = u32::from_le_bytes(footer_buf[28..32].try_into().unwrap());
        if magic != SSTABLE_MAGIC {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("invalid sstable magic: {magic:#x}"),
            ));
        }

        let mut index_bytes = vec![0u8; index_len];
        file.seek(SeekFrom::Start(index_offset))?;
        file.read_exact(&mut index_bytes)?;
        let blocks: Vec<BlockMeta> = bincode::deserialize(&index_bytes).map_err(|err| {
            io::Error::new(io::ErrorKind::Other, format!("deserialize index: {err}"))
        })?;

        let mut bloom_bytes = vec![0u8; bloom_len];
        file.seek(SeekFrom::Start(bloom_offset))?;
        file.read_exact(&mut bloom_bytes)?;
        let bloom: BloomFilter = bincode::deserialize(&bloom_bytes).map_err(|err| {
            io::Error::new(io::ErrorKind::Other, format!("deserialize bloom: {err}"))
        })?;

        Ok(Self {
            id,
            path,
            block_size,
            blocks,
            bloom,
        })
    }

    pub fn id(&self) -> u64 {
        self.id
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn block_size(&self) -> usize {
        self.block_size
    }

    pub fn bloom(&self) -> &BloomFilter {
        &self.bloom
    }

    pub fn get(&self, key: &[u8]) -> io::Result<Option<Option<Vec<u8>>>> {
        if !self.bloom.might_contain(key) {
            return Ok(None);
        }
        if let Some(block_meta) = self.locate_block(key) {
            let block = self.read_block(block_meta)?;
            for entry in block {
                match entry.key.as_slice().cmp(key) {
                    Ordering::Less => continue,
                    Ordering::Equal => return Ok(Some(entry.value)),
                    Ordering::Greater => break,
                }
            }
        }
        Ok(None)
    }

    pub fn load_all(&self) -> io::Result<Vec<SSTableEntry>> {
        let mut entries = Vec::new();
        for meta in &self.blocks {
            entries.extend(self.read_block(meta)?);
        }
        Ok(entries)
    }

    fn locate_block(&self, key: &[u8]) -> Option<&BlockMeta> {
        if self.blocks.is_empty() {
            return None;
        }
        let mut left = 0usize;
        let mut right = self.blocks.len();
        while left < right {
            let mid = (left + right) / 2;
            let meta = &self.blocks[mid];
            if key < meta.first_key.as_slice() {
                if mid == 0 {
                    return Some(meta);
                }
                right = mid;
            } else if key > meta.last_key.as_slice() {
                left = mid + 1;
            } else {
                return Some(meta);
            }
        }
        if left < self.blocks.len() {
            Some(&self.blocks[left])
        } else {
            self.blocks.last()
        }
    }

    fn read_block(&self, meta: &BlockMeta) -> io::Result<Vec<SSTableEntry>> {
        let mut file = File::open(&self.path)?;
        file.seek(SeekFrom::Start(meta.offset))?;
        let mut len_buf = [0u8; 4];
        file.read_exact(&mut len_buf)?;
        let len = u32::from_le_bytes(len_buf) as usize;
        if len != meta.length as usize {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "sstable block length mismatch",
            ));
        }
        let mut buf = vec![0u8; len];
        file.read_exact(&mut buf)?;
        bincode::deserialize(&buf).map_err(|err| {
            io::Error::new(io::ErrorKind::Other, format!("deserialize block: {err}"))
        })
    }
}

pub fn merge_tables(
    path: impl AsRef<Path>,
    id: u64,
    tables: &[Arc<SSTable>],
    block_size: usize,
    bloom_bits_per_key: usize,
    bloom_hashes: u32,
) -> io::Result<SSTable> {
    let mut merged = BTreeMap::<Vec<u8>, Option<Vec<u8>>>::new();
    for table in tables {
        let entries = table.load_all()?;
        for entry in entries {
            merged.insert(entry.key, entry.value);
        }
    }
    let merged_entries: Vec<MemTableEntry> = merged
        .into_iter()
        .map(|(key, value)| MemTableEntry { key, value })
        .collect();
    SSTable::create(
        path,
        id,
        &merged_entries,
        block_size,
        bloom_bits_per_key,
        bloom_hashes,
    )
}
