use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io::{BufReader, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant, SystemTime};
use std::{sync::Arc, thread};

use aidb_core::{Id, JsonValue, MetadataFilter, Metric, SearchResult, Vector, VectorIndex};
use memmap2::Mmap;
use parking_lot::{Mutex, RwLock};
use serde::{de::DeserializeOwned, Deserialize, Serialize};

const WAL_LOG_INCREMENT_BYTES: u64 = 64 * 1024 * 1024; // 64 MiB increments for logging

#[cfg(test)]
const WAL_MMAP_THRESHOLD: u64 = 1 << 10; // 1 KiB in tests
#[cfg(not(test))]
const WAL_MMAP_THRESHOLD: u64 = 1 << 20; // 1 MiB in production

const WAL_GROUP_COMMIT_MAX_OPS: usize = 32;
const WAL_GROUP_COMMIT_MAX_BYTES: usize = 64 * 1024;
const WAL_GROUP_COMMIT_INTERVAL: Duration = Duration::from_millis(5);

#[derive(thiserror::Error, Debug)]
pub enum StorageError {
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("codec error: {0}")]
    Codec(#[from] Box<bincode::ErrorKind>),
    #[error("dimension mismatch: expected {expected}, got {got}")]
    Dim { expected: usize, got: usize },
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum WalOp {
    Upsert {
        id: Id,
        vector: Vector,
        payload: Option<JsonValue>,
    },
    Remove {
        id: Id,
    },
}

#[derive(Serialize, Deserialize, Debug, Clone)]
struct WalRecord {
    lsn: u64,
    timestamp: SystemTime,
    op: WalOp,
}

#[derive(Debug, Clone)]
pub struct WalEntry {
    pub lsn: u64,
    pub timestamp: SystemTime,
    pub op: WalOp,
}

impl From<WalRecord> for WalEntry {
    fn from(value: WalRecord) -> Self {
        Self {
            lsn: value.lsn,
            timestamp: value.timestamp,
            op: value.op,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub enum RecoveryTarget {
    Latest,
    Lsn(u64),
    Timestamp(SystemTime),
}

struct WalPending {
    buffer: Vec<u8>,
    op_count: usize,
    last_flush: Instant,
    first_enqueued_at: Option<Instant>,
}

impl WalPending {
    fn new() -> Self {
        Self {
            buffer: Vec::with_capacity(8 * WAL_GROUP_COMMIT_MAX_OPS),
            op_count: 0,
            last_flush: Instant::now(),
            first_enqueued_at: None,
        }
    }

    fn should_flush(&self) -> bool {
        self.buffer.len() >= WAL_GROUP_COMMIT_MAX_BYTES
            || self.op_count >= WAL_GROUP_COMMIT_MAX_OPS
            || self
                .first_enqueued_at
                .map(|ts| ts.elapsed() >= WAL_GROUP_COMMIT_INTERVAL)
                .unwrap_or(false)
    }

    fn clear(&mut self) {
        self.buffer.clear();
        self.op_count = 0;
        self.last_flush = Instant::now();
        self.first_enqueued_at = None;
    }
}

#[derive(Clone)]
pub(crate) struct Wal {
    inner: Arc<WalInner>,
}

struct WalInner {
    path: PathBuf,
    f: RwLock<File>,
    pending: Mutex<WalPending>,
    next_lsn: AtomicU64,
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
        assert!(collection.wal_stats().size_bytes > 0);

        collection.snapshot_to(&snapshot_path)?;
        let wal_len_after_snapshot = std::fs::metadata(&wal_path)?.len();
        assert_eq!(wal_len_after_snapshot, 0);
        let snapshot_stats = collection.wal_stats();
        assert_eq!(snapshot_stats.size_bytes, 0);
        assert!(snapshot_stats.last_truncate.is_some());

        let id_b = Uuid::new_v4();
        collection.upsert(id_b, vec![0.4, 0.3, 0.2, 0.1], None)?;
        assert!(collection.wal_stats().size_bytes > 0);

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
        let inner = Arc::new(WalInner {
            path: path.as_ref().to_path_buf(),
            f: RwLock::new(f),
            pending: Mutex::new(WalPending::new()),
            next_lsn: AtomicU64::new(1),
        });

        let wal = Self { inner };
        let last_lsn = wal.inner.scan_last_lsn()?;
        wal.inner.next_lsn.store(last_lsn + 1, Ordering::Release);

        Ok(wal)
    }

    pub(crate) fn append(&self, op: &WalOp) -> Result<(), StorageError> {
        let lsn = self.inner.next_lsn.fetch_add(1, Ordering::AcqRel);
        let record = WalRecord {
            lsn,
            timestamp: SystemTime::now(),
            op: op.clone(),
        };

        let buf = bincode::serialize(&record)?;
        let len = (buf.len() as u64).to_le_bytes();

        let mut pending = self.inner.pending.lock();
        let was_empty = pending.buffer.is_empty();
        if was_empty {
            pending.first_enqueued_at = Some(Instant::now());
        }
        pending.buffer.extend_from_slice(&len);
        pending.buffer.extend_from_slice(&buf);
        pending.op_count += 1;

        let should_flush_now = pending.should_flush();
        if should_flush_now {
            self.inner.flush_locked(&mut pending)?;
        }
        drop(pending);

        if was_empty && !should_flush_now {
            let inner = Arc::clone(&self.inner);
            thread::spawn(move || {
                thread::sleep(WAL_GROUP_COMMIT_INTERVAL);
                if let Err(err) = inner.flush_if_due() {
                    log::debug!(
                        target: "aidb::storage::wal",
                        "delayed_flush_failed error={:?}",
                        err
                    );
                }
            });
        }

        Ok(())
    }

    pub(crate) fn reset(&self) -> Result<(), StorageError> {
        self.inner.flush()?;
        let f = self.inner.f.write();
        f.set_len(0)?;
        f.sync_all()?;
        Ok(())
    }

    pub(crate) fn load_all(&self) -> Result<Vec<WalOp>, StorageError> {
        self.load_until(RecoveryTarget::Latest)
    }

    pub(crate) fn load_until(&self, target: RecoveryTarget) -> Result<Vec<WalOp>, StorageError> {
        let records = self.inner.load_records()?;
        let filtered: Vec<WalRecord> = match target {
            RecoveryTarget::Latest => records,
            RecoveryTarget::Lsn(lsn) => records.into_iter().filter(|rec| rec.lsn <= lsn).collect(),
            RecoveryTarget::Timestamp(ts) => records
                .into_iter()
                .filter(|rec| rec.timestamp <= ts)
                .collect(),
        };

        Ok(filtered.into_iter().map(|rec| rec.op).collect())
    }

    pub(crate) fn tail_from(&self, lsn: u64) -> Result<Vec<WalEntry>, StorageError> {
        let records = self.inner.load_records()?;
        Ok(records
            .into_iter()
            .filter(|rec| rec.lsn >= lsn)
            .map(WalEntry::from)
            .collect())
    }

    pub(crate) fn size_bytes(&self) -> Result<u64, StorageError> {
        self.inner.size_bytes()
    }
}

impl WalInner {
    fn flush_locked(&self, pending: &mut WalPending) -> Result<(), StorageError> {
        if pending.buffer.is_empty() {
            pending.last_flush = Instant::now();
            pending.first_enqueued_at = None;
            return Ok(());
        }

        let mut file = self.f.write();
        file.write_all(&pending.buffer)?;
        file.flush()?;
        pending.clear();
        Ok(())
    }

    fn flush(&self) -> Result<(), StorageError> {
        let mut pending = self.pending.lock();
        self.flush_locked(&mut pending)
    }

    fn flush_if_due(self: Arc<Self>) -> Result<(), StorageError> {
        let mut pending = self.pending.lock();
        let due = pending
            .first_enqueued_at
            .map(|ts| ts.elapsed() >= WAL_GROUP_COMMIT_INTERVAL)
            .unwrap_or(false);
        if due && !pending.buffer.is_empty() {
            self.flush_locked(&mut pending)?;
        }
        Ok(())
    }

    fn load_records(&self) -> Result<Vec<WalRecord>, StorageError> {
        self.flush()?;
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

    fn scan_last_lsn(&self) -> Result<u64, StorageError> {
        let records = self.load_records()?;
        Ok(records.last().map(|rec| rec.lsn).unwrap_or(0))
    }

    fn size_bytes(&self) -> Result<u64, StorageError> {
        let pending_len = {
            let pending = self.pending.lock();
            pending.buffer.len() as u64
        };
        let on_disk = std::fs::metadata(&self.path).map(|m| m.len()).unwrap_or(0);
        Ok(on_disk + pending_len)
    }
}

fn load_all_stream(file: File, len: usize) -> Result<Vec<WalRecord>, StorageError> {
    let mut reader = BufReader::new(file);
    let mut buf = Vec::with_capacity(len);
    reader.read_to_end(&mut buf)?;
    parse_wal_bytes(&buf)
}

fn parse_wal_bytes(bytes: &[u8]) -> Result<Vec<WalRecord>, StorageError> {
    let mut ops = Vec::new();
    let mut offset = 0usize;
    let mut fallback_lsn = 1u64;
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
        match bincode::deserialize::<WalRecord>(payload) {
            Ok(record) => {
                fallback_lsn = record.lsn.saturating_add(1);
                ops.push(record);
            }
            Err(_) => {
                let op: WalOp = bincode::deserialize(payload)?;
                ops.push(WalRecord {
                    lsn: fallback_lsn,
                    timestamp: SystemTime::UNIX_EPOCH,
                    op,
                });
                fallback_lsn = fallback_lsn.saturating_add(1);
            }
        }
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
        self.recover_to(RecoveryTarget::Latest)
    }

    pub fn recover_to(&self, target: RecoveryTarget) -> Result<(), StorageError> {
        let _guard = self.wal_lock.lock();
        let ops = self.wal.load_until(target)?;
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

    pub fn wal_entries_from(&self, lsn: u64) -> Result<Vec<WalEntry>, StorageError> {
        let _guard = self.wal_lock.lock();
        self.wal.tail_from(lsn)
    }

    pub fn recover_at_time(&self, timestamp: SystemTime) -> Result<(), StorageError> {
        self.recover_to(RecoveryTarget::Timestamp(timestamp))
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
        let size = self.wal.size_bytes()?;
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
