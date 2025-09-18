use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io::{BufReader, Read, Write};
use std::path::{Path, PathBuf};

use aidb_core::{Id, JsonValue, MetadataFilter, Metric, SearchResult, Vector, VectorIndex};
use memmap2::Mmap;
use parking_lot::{Mutex, RwLock};
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use std::sync::Arc;

const WAL_LOG_INCREMENT_BYTES: u64 = 64 * 1024 * 1024; // 64 MiB increments for logging

#[cfg(test)]
const WAL_MMAP_THRESHOLD: u64 = 1 << 10; // 1 KiB in tests
#[cfg(not(test))]
const WAL_MMAP_THRESHOLD: u64 = 1 << 20; // 1 MiB in production

#[derive(thiserror::Error, Debug)]
pub enum StorageError {
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("codec error: {0}")]
    Codec(#[from] Box<bincode::ErrorKind>),
    #[error("dimension mismatch: expected {expected}, got {got}")]
    Dim { expected: usize, got: usize },
}

#[derive(Serialize, Deserialize, Debug)]
enum WalOp {
    Upsert {
        id: Id,
        vector: Vector,
        payload: Option<JsonValue>,
    },
    Remove {
        id: Id,
    },
}

pub(crate) struct Wal {
    path: PathBuf,
    f: RwLock<File>,
}

#[derive(Default, Clone)]
pub struct WalStats {
    pub size_bytes: u64,
    pub last_truncate: Option<std::time::SystemTime>,
    last_reported_bytes: u64,
    pub bytes_since_truncate: u64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use aidb_index_bf::BruteForceIndex;
    use std::path::PathBuf;
    use uuid::Uuid;

    fn temp_dir() -> PathBuf {
        std::env::temp_dir().join(format!("aidb-storage-{}", Uuid::new_v4()))
    }

    #[test]
    fn snapshot_resets_wal_and_preserves_state_across_recovery() -> Result<(), StorageError> {
        let base = temp_dir();
        std::fs::create_dir_all(&base)?;
        let wal_path = base.join("collection.wal");
        let snapshot_path = base.join("collection.snapshot");

        let index = BruteForceIndex::new(4);
        let collection = Collection::new("test", 4, Metric::Cosine, index, &wal_path)?;

        let id_a = Uuid::new_v4();
        collection.upsert(id_a, vec![0.1, 0.2, 0.3, 0.4], None)?;
        let wal_len_before = std::fs::metadata(&wal_path)?.len();
        assert!(wal_len_before > 0);

        collection.snapshot_to(&snapshot_path)?;
        let wal_len_after_snapshot = std::fs::metadata(&wal_path)?.len();
        assert_eq!(wal_len_after_snapshot, 0);
        let snapshot_stats = collection.wal_stats();
        assert_eq!(snapshot_stats.size_bytes, 0);
        assert!(snapshot_stats.last_truncate.is_some());

        let id_b = Uuid::new_v4();
        collection.upsert(id_b, vec![0.4, 0.3, 0.2, 0.1], None)?;
        let wal_len_post = std::fs::metadata(&wal_path)?.len();
        assert!(wal_len_post > 0);

        // Recover into a fresh collection using snapshot + WAL
        let fresh_index = BruteForceIndex::new(4);
        let restored = Collection::new("test", 4, Metric::Cosine, fresh_index, &wal_path)?;
        restored.load_snapshot(&snapshot_path)?;
        restored.recover()?;

        let res_a = restored.search(&[0.1, 0.2, 0.3, 0.4], 1, None);
        assert_eq!(res_a.len(), 1);
        assert_eq!(res_a[0].id, id_a);

        let res_b = restored.search(&[0.4, 0.3, 0.2, 0.1], 1, None);
        assert_eq!(res_b.len(), 1);
        assert_eq!(res_b[0].id, id_b);

        let stats = restored.wal_stats();
        assert!(stats.size_bytes > 0);
        assert!(stats.last_truncate.is_none());

        let _ = std::fs::remove_dir_all(&base);

        Ok(())
    }

    #[test]
    fn wal_load_all_stream_small_file() -> Result<(), StorageError> {
        let base = temp_dir();
        std::fs::create_dir_all(&base)?;
        let wal_path = base.join("wal-small.wal");
        let wal = super::Wal::open(&wal_path)?;

        wal.append(&super::WalOp::Upsert {
            id: Uuid::new_v4(),
            vector: vec![0.1, 0.2, 0.3],
            payload: None,
        })?;

        let ops = wal.load_all()?;
        assert_eq!(ops.len(), 1);

        let _ = std::fs::remove_dir_all(&base);
        Ok(())
    }

    #[test]
    fn wal_load_all_mmap_large_file() -> Result<(), StorageError> {
        let base = temp_dir();
        std::fs::create_dir_all(&base)?;
        let wal_path = base.join("wal-large.wal");
        let wal = super::Wal::open(&wal_path)?;

        let mut count = 0;
        while std::fs::metadata(&wal_path).map(|m| m.len()).unwrap_or(0) < super::WAL_MMAP_THRESHOLD
        {
            wal.append(&super::WalOp::Upsert {
                id: Uuid::new_v4(),
                vector: vec![count as f32; 32],
                payload: None,
            })?;
            count += 1;
        }

        let ops = wal.load_all()?;
        assert!(ops.len() >= count);

        let _ = std::fs::remove_dir_all(&base);
        Ok(())
    }
}

impl Wal {
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self, StorageError> {
        let f = OpenOptions::new()
            .create(true)
            .read(true)
            .append(true)
            .open(path.as_ref())?;
        Ok(Self {
            path: path.as_ref().to_path_buf(),
            f: RwLock::new(f),
        })
    }

    pub(crate) fn append(&self, op: &WalOp) -> Result<(), StorageError> {
        let mut f = self.f.write();
        let buf = bincode::serialize(op)?;
        let len = (buf.len() as u64).to_le_bytes();
        f.write_all(&len)?;
        f.write_all(&buf)?;
        f.flush()?;
        Ok(())
    }

    pub(crate) fn reset(&self) -> Result<(), StorageError> {
        let f = self.f.write();
        f.set_len(0)?;
        f.sync_all()?;
        Ok(())
    }

    pub(crate) fn load_all(&self) -> Result<Vec<WalOp>, StorageError> {
        let file = OpenOptions::new().read(true).open(&self.path)?;
        let len = file.metadata()?.len();
        if len == 0 {
            return Ok(Vec::new());
        }

        if len >= WAL_MMAP_THRESHOLD {
            if let Ok(map) = unsafe { Mmap::map(&file) } {
                return parse_wal_bytes(&map);
            }
        }

        load_all_stream(file, len as usize)
    }
}

fn load_all_stream(file: File, len: usize) -> Result<Vec<WalOp>, StorageError> {
    let mut reader = BufReader::new(file);
    let mut buf = Vec::with_capacity(len);
    reader.read_to_end(&mut buf)?;
    parse_wal_bytes(&buf)
}

fn parse_wal_bytes(bytes: &[u8]) -> Result<Vec<WalOp>, StorageError> {
    let mut ops = Vec::new();
    let mut offset = 0usize;
    while offset + 8 <= bytes.len() {
        let mut len_buf = [0u8; 8];
        len_buf.copy_from_slice(&bytes[offset..offset + 8]);
        let len = u64::from_le_bytes(len_buf) as usize;
        offset += 8;
        if offset + len > bytes.len() {
            break;
        }
        let payload = &bytes[offset..offset + len];
        offset += len;
        let op: WalOp = bincode::deserialize(payload)?;
        ops.push(op);
    }
    Ok(ops)
}

pub struct Collection<I: VectorIndex> {
    name: String,
    dim: usize,
    metric: Metric,
    index: RwLock<I>,
    wal: Wal,
    wal_lock: Mutex<()>,
    wal_stats: Mutex<WalStats>,
}

impl<I: VectorIndex> Collection<I> {
    pub fn new(
        name: impl Into<String>,
        dim: usize,
        metric: Metric,
        index: I,
        wal_path: impl AsRef<Path>,
    ) -> Result<Self, StorageError> {
        if index.dim() != dim {
            return Err(StorageError::Dim {
                expected: dim,
                got: index.dim(),
            });
        }
        let wal = Wal::open(wal_path)?;
        Ok(Self {
            name: name.into(),
            dim,
            metric,
            index: RwLock::new(index),
            wal,
            wal_lock: Mutex::new(()),
            wal_stats: Mutex::new(WalStats::default()),
        })
    }

    pub fn recover(&self) -> Result<(), StorageError> {
        let _guard = self.wal_lock.lock();
        let ops = self.wal.load_all()?;
        let mut idx = self.index.write();
        for op in ops {
            match op {
                WalOp::Upsert {
                    id,
                    vector,
                    payload,
                } => idx.add(id, vector, payload),
                WalOp::Remove { id } => {
                    idx.remove(&id);
                }
            }
        }
        drop(idx);
        self.refresh_wal_stats()?;
        Ok(())
    }

    pub fn upsert(
        &self,
        id: Id,
        vector: Vector,
        payload: Option<JsonValue>,
    ) -> Result<(), StorageError> {
        let _guard = self.wal_lock.lock();
        if vector.len() != self.dim {
            return Err(StorageError::Dim {
                expected: self.dim,
                got: vector.len(),
            });
        }
        self.wal.append(&WalOp::Upsert {
            id,
            vector: vector.clone(),
            payload: payload.clone(),
        })?;
        self.refresh_wal_stats()?;
        self.index.write().add(id, vector, payload);
        Ok(())
    }

    pub fn remove(&self, id: Id) -> Result<bool, StorageError> {
        let _guard = self.wal_lock.lock();
        self.wal.append(&WalOp::Remove { id })?;
        self.refresh_wal_stats()?;
        Ok(self.index.write().remove(&id))
    }

    pub fn search(
        &self,
        vector: &[f32],
        top_k: usize,
        filter: Option<&MetadataFilter>,
    ) -> Vec<SearchResult> {
        self.index.read().search(vector, top_k, self.metric, filter)
    }

    pub fn dim(&self) -> usize {
        self.dim
    }
    pub fn metric(&self) -> Metric {
        self.metric
    }

    pub fn len(&self) -> usize {
        self.index.read().len()
    }

    pub fn with_index_mut<T>(&self, f: impl FnOnce(&mut I) -> T) -> T {
        let mut guard = self.index.write();
        f(&mut *guard)
    }

    fn refresh_wal_stats(&self) -> Result<(), StorageError> {
        let size = std::fs::metadata(&self.wal.path)?.len();
        let mut stats = self.wal_stats.lock();
        let prev_size = stats.size_bytes;
        stats.size_bytes = size;
        if size >= prev_size {
            stats.bytes_since_truncate =
                stats.bytes_since_truncate.saturating_add(size - prev_size);
        }
        let prev_bucket = stats.last_reported_bytes / WAL_LOG_INCREMENT_BYTES;
        let new_bucket = size / WAL_LOG_INCREMENT_BYTES;
        let should_log = if size == 0 {
            false
        } else if stats.last_reported_bytes == 0 && size >= WAL_LOG_INCREMENT_BYTES {
            true
        } else {
            new_bucket > prev_bucket
        };
        if should_log {
            stats.last_reported_bytes = size;
            let since = stats
                .last_truncate
                .and_then(|ts| ts.elapsed().ok())
                .map(|d| d.as_secs())
                .unwrap_or_default();
            log::info!(
                target: "aidb::storage",
                "wal_growth collection={} size_bytes={} since_last_truncate_secs={}",
                self.name,
                size,
                since
            );
        }
        Ok(())
    }

    fn update_truncate_time(&self) {
        let mut stats = self.wal_stats.lock();
        stats.size_bytes = 0;
        stats.last_truncate = Some(std::time::SystemTime::now());
        stats.last_reported_bytes = 0;
        stats.bytes_since_truncate = 0;
        log::info!(
            target: "aidb::storage",
            "wal_truncated collection={}",
            self.name
        );
    }
}

pub struct Engine<I: VectorIndex> {
    pub collections: RwLock<HashMap<String, Arc<Collection<I>>>>,
}

impl<I: VectorIndex> Engine<I> {
    pub fn new() -> Self {
        Self {
            collections: RwLock::new(HashMap::new()),
        }
    }
    pub fn insert_collection(&self, c: Collection<I>) {
        self.collections.write().insert(c.name.clone(), Arc::new(c));
    }
}

impl<I> Collection<I>
where
    I: VectorIndex + Serialize + DeserializeOwned,
{
    pub fn snapshot_to(&self, path: impl AsRef<Path>) -> Result<(), StorageError> {
        let _wal_guard = self.wal_lock.lock();
        let idx = self.index.read();
        let bytes = bincode::serialize(&*idx)?;
        std::fs::write(path.as_ref(), bytes)?;
        drop(idx);
        self.wal.reset()?;
        self.update_truncate_time();
        Ok(())
    }

    pub fn load_snapshot(&self, path: impl AsRef<Path>) -> Result<(), StorageError> {
        let _wal_guard = self.wal_lock.lock();
        let bytes = std::fs::read(path)?;
        let idx: I = bincode::deserialize(&bytes)?;
        // quick sanity
        if idx.dim() != self.dim {
            return Err(StorageError::Dim {
                expected: self.dim,
                got: idx.dim(),
            });
        }
        *self.index.write() = idx;
        self.refresh_wal_stats()?;
        Ok(())
    }

    pub fn wal_stats(&self) -> WalStats {
        self.wal_stats.lock().clone()
    }
}
