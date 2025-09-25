use std::collections::BTreeMap;
use std::io;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use parking_lot::RwLock;

use super::wal::{WalEntry, WriteAheadLog};

#[derive(Clone, Debug)]
pub struct MemTableEntry {
    pub key: Vec<u8>,
    pub value: Option<Vec<u8>>,
}

pub struct MemTable {
    id: u64,
    wal: Arc<WriteAheadLog>,
    data: RwLock<BTreeMap<Vec<u8>, MemTableEntry>>,
    approx_size: AtomicUsize,
    capacity_bytes: usize,
}

impl MemTable {
    pub fn new(id: u64, wal: Arc<WriteAheadLog>, capacity_bytes: usize) -> io::Result<Self> {
        wal.reset()?;
        Ok(Self {
            id,
            wal,
            data: RwLock::new(BTreeMap::new()),
            approx_size: AtomicUsize::new(0),
            capacity_bytes,
        })
    }

    pub fn recover(id: u64, wal: Arc<WriteAheadLog>, capacity_bytes: usize) -> io::Result<Self> {
        let mut data = BTreeMap::new();
        let mut approx_size = 0usize;
        for entry in wal.replay()? {
            match entry.tombstone {
                true => {
                    approx_size += entry.key.len();
                    data.insert(
                        entry.key.clone(),
                        MemTableEntry {
                            key: entry.key,
                            value: None,
                        },
                    );
                }
                false => {
                    if let Some(value) = entry.value {
                        approx_size += entry.key.len() + value.len();
                        data.insert(
                            entry.key.clone(),
                            MemTableEntry {
                                key: entry.key,
                                value: Some(value),
                            },
                        );
                    }
                }
            }
        }
        Ok(Self {
            id,
            wal,
            data: RwLock::new(data),
            approx_size: AtomicUsize::new(approx_size),
            capacity_bytes,
        })
    }

    pub fn id(&self) -> u64 {
        self.id
    }

    pub fn capacity(&self) -> usize {
        self.capacity_bytes
    }

    pub fn wal_path(&self) -> &std::path::Path {
        self.wal.path()
    }

    pub fn approximate_size(&self) -> usize {
        self.approx_size.load(Ordering::Relaxed)
    }

    pub fn is_full(&self) -> bool {
        self.approximate_size() >= self.capacity_bytes
    }

    pub fn len(&self) -> usize {
        self.data.read().len()
    }

    pub fn insert(&self, key: Vec<u8>, value: Vec<u8>) -> io::Result<()> {
        let entry = WalEntry::insert(key.clone(), value.clone());
        self.wal.append(&entry)?;
        let mut data = self.data.write();
        if let Some(previous) = data.get(key.as_slice()) {
            if let Some(old) = &previous.value {
                self.approx_size
                    .fetch_sub(key.len() + old.len(), Ordering::Relaxed);
            } else {
                self.approx_size.fetch_sub(key.len(), Ordering::Relaxed);
            }
        }
        self.approx_size
            .fetch_add(key.len() + value.len(), Ordering::Relaxed);
        data.insert(
            key.clone(),
            MemTableEntry {
                key,
                value: Some(value),
            },
        );
        Ok(())
    }

    pub fn delete(&self, key: Vec<u8>) -> io::Result<()> {
        let entry = WalEntry::delete(key.clone());
        self.wal.append(&entry)?;
        let mut data = self.data.write();
        if let Some(previous) = data.get(key.as_slice()) {
            if let Some(old) = &previous.value {
                self.approx_size
                    .fetch_sub(key.len() + old.len(), Ordering::Relaxed);
            } else {
                self.approx_size.fetch_sub(key.len(), Ordering::Relaxed);
            }
        }
        self.approx_size.fetch_add(key.len(), Ordering::Relaxed);
        data.insert(key.clone(), MemTableEntry { key, value: None });
        Ok(())
    }

    pub fn get(&self, key: &[u8]) -> Option<Option<Vec<u8>>> {
        let data = self.data.read();
        data.get(key).map(|entry| entry.value.clone())
    }

    pub fn snapshot(&self) -> Vec<MemTableEntry> {
        let data = self.data.read();
        data.values().cloned().collect()
    }
}
