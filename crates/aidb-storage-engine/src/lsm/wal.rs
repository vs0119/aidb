use std::fs::{self, File, OpenOptions};
use std::io::{self, BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

use parking_lot::Mutex;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WalEntry {
    pub tombstone: bool,
    pub key: Vec<u8>,
    pub value: Option<Vec<u8>>,
}

impl WalEntry {
    pub fn insert(key: Vec<u8>, value: Vec<u8>) -> Self {
        Self {
            tombstone: false,
            key,
            value: Some(value),
        }
    }

    pub fn delete(key: Vec<u8>) -> Self {
        Self {
            tombstone: true,
            key,
            value: None,
        }
    }
}

pub struct WriteAheadLog {
    path: PathBuf,
    writer: Mutex<BufWriter<File>>,
}

impl WriteAheadLog {
    pub fn open(path: impl AsRef<Path>) -> io::Result<Self> {
        let path = path.as_ref().to_path_buf();
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .read(true)
            .open(&path)?;
        Ok(Self {
            writer: Mutex::new(BufWriter::new(file)),
            path,
        })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn append(&self, entry: &WalEntry) -> io::Result<()> {
        let mut writer = self.writer.lock();
        write_entry(&mut *writer, entry)?;
        writer.flush()?;
        writer.get_mut().sync_data()
    }

    pub fn sync(&self) -> io::Result<()> {
        let mut writer = self.writer.lock();
        writer.flush()?;
        writer.get_mut().sync_data()
    }

    pub fn replay(&self) -> io::Result<Vec<WalEntry>> {
        let mut file = OpenOptions::new().read(true).open(&self.path)?;
        file.seek(SeekFrom::Start(0))?;
        let mut reader = BufReader::new(file);
        let mut entries = Vec::new();
        while let Some(entry) = read_entry(&mut reader)? {
            entries.push(entry);
        }
        Ok(entries)
    }

    pub fn reset(&self) -> io::Result<()> {
        {
            let mut writer = self.writer.lock();
            writer.flush()?;
        }
        let file = OpenOptions::new()
            .write(true)
            .truncate(true)
            .open(&self.path)?;
        let mut writer = self.writer.lock();
        *writer = BufWriter::new(file);
        writer.flush()
    }
}

fn write_entry<W: Write>(writer: &mut W, entry: &WalEntry) -> io::Result<()> {
    writer.write_all(&[entry.tombstone as u8])?;
    let key_len = entry.key.len() as u32;
    writer.write_all(&key_len.to_le_bytes())?;
    writer.write_all(&entry.key)?;
    match &entry.value {
        Some(value) => {
            let value_len = value.len() as u32;
            writer.write_all(&value_len.to_le_bytes())?;
            writer.write_all(value)?;
        }
        None => {
            writer.write_all(&0u32.to_le_bytes())?;
        }
    }
    Ok(())
}

fn read_entry<R: Read>(reader: &mut R) -> io::Result<Option<WalEntry>> {
    let mut flag = [0u8; 1];
    match reader.read_exact(&mut flag) {
        Ok(()) => {}
        Err(err) if err.kind() == io::ErrorKind::UnexpectedEof => {
            return Ok(None);
        }
        Err(err) => return Err(err),
    }
    let mut len_buf = [0u8; 4];
    reader.read_exact(&mut len_buf)?;
    let key_len = u32::from_le_bytes(len_buf) as usize;
    let mut key = vec![0u8; key_len];
    reader.read_exact(&mut key)?;
    reader.read_exact(&mut len_buf)?;
    let value_len = u32::from_le_bytes(len_buf) as usize;
    let value = if value_len == 0 {
        None
    } else {
        let mut buf = vec![0u8; value_len];
        reader.read_exact(&mut buf)?;
        Some(buf)
    };
    Ok(Some(WalEntry {
        tombstone: flag[0] == 1,
        key,
        value,
    }))
}
