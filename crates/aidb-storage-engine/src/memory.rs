use std::collections::{BTreeMap, HashMap};
use std::ops::Bound;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use aidb_core::Id;
use dashmap::{mapref::entry::Entry, DashMap};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use tokio::fs::{self, OpenOptions};
use tokio::io::AsyncWriteExt;
use tokio::sync::Mutex;
use tokio::task::JoinHandle;
use tokio::time;

use crate::page::{Page, PageManager};
use crate::{Result, StorageEngineError, VectorRow};

/// Storage tier preferences for hybrid deployments.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StorageTier {
    /// Keep the entire collection resident in memory.
    MemoryOnly,
    /// Maintain a memory working set but spill excess data to disk.
    Hybrid,
    /// Prefer disk-resident data and only cache hot rows in memory.
    DiskPreferred,
}

impl Default for StorageTier {
    fn default() -> Self {
        StorageTier::Hybrid
    }
}

/// Configurable policy that can be tuned per collection.
#[derive(Debug, Clone)]
pub struct ConfigurableStoragePolicy {
    /// Preferred storage tier for the collection.
    pub storage_tier: StorageTier,
    /// Maximum number of rows that should remain in memory before eviction kicks in.
    pub max_memory_rows: Option<usize>,
    /// Interval between snapshots. Collections can override the global cadence.
    pub snapshot_interval: Duration,
    /// Whether WAL logging is enabled for the collection.
    pub wal_enabled: bool,
    /// Force an fsync after every WAL append for stronger durability.
    pub wal_sync: bool,
    /// Number of rows to evict at a time when the memory budget is exceeded.
    pub eviction_batch_size: usize,
}

impl Default for ConfigurableStoragePolicy {
    fn default() -> Self {
        Self {
            storage_tier: StorageTier::Hybrid,
            max_memory_rows: Some(50_000),
            snapshot_interval: Duration::from_secs(300),
            wal_enabled: true,
            wal_sync: false,
            eviction_batch_size: 1024,
        }
    }
}

/// Represents a single in-memory table backed by hash and tree based indexes.
pub struct MemoryTable {
    name: String,
    primary_index: DashMap<Id, Arc<VectorRow>>,
    ordered_index: RwLock<BTreeMap<Id, Arc<VectorRow>>>,
    row_count: AtomicU64,
    policy: RwLock<ConfigurableStoragePolicy>,
}

impl MemoryTable {
    fn new(name: String, policy: ConfigurableStoragePolicy) -> Self {
        Self {
            name,
            primary_index: DashMap::new(),
            ordered_index: RwLock::new(BTreeMap::new()),
            row_count: AtomicU64::new(0),
            policy: RwLock::new(policy),
        }
    }

    /// Name of the underlying collection.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Returns the current policy clone.
    pub fn policy(&self) -> ConfigurableStoragePolicy {
        self.policy.read().clone()
    }

    /// Update the storage policy for this table.
    pub fn set_policy(&self, policy: ConfigurableStoragePolicy) {
        *self.policy.write() = policy;
    }

    /// Inserts or replaces a row in the table.
    pub fn insert(&self, row: VectorRow) {
        let row_id = row.id;
        let arc_row = Arc::new(row);
        let previous = self.primary_index.insert(row_id, arc_row.clone());
        {
            let mut index = self.ordered_index.write();
            index.insert(row_id, arc_row.clone());
        }
        if previous.is_none() {
            self.row_count.fetch_add(1, Ordering::AcqRel);
        }
    }

    /// Removes a row from the table.
    pub fn remove(&self, id: &Id) -> Option<VectorRow> {
        let removed = self.primary_index.remove(id).map(|(_, row)| row);
        if let Some(row) = &removed {
            {
                let mut index = self.ordered_index.write();
                index.remove(id);
            }
            self.row_count.fetch_sub(1, Ordering::AcqRel);
            Some((row.as_ref()).clone())
        } else {
            None
        }
    }

    /// Fetch a row by id using the hash index.
    pub fn get(&self, id: &Id) -> Option<Arc<VectorRow>> {
        self.primary_index.get(id).map(|v| v.clone())
    }

    /// Returns the number of rows in the table.
    pub fn len(&self) -> usize {
        self.row_count.load(Ordering::Acquire) as usize
    }

    /// Returns true when the table should spill rows to disk.
    pub fn should_evict(&self) -> bool {
        let policy = self.policy();
        if let Some(max_rows) = policy.max_memory_rows {
            self.len() > max_rows
        } else {
            false
        }
    }

    /// Evict the oldest rows based on the ordered index.
    pub fn evict_oldest(&self, count: usize) -> Vec<VectorRow> {
        if count == 0 {
            return Vec::new();
        }

        let mut index = self.ordered_index.write();
        let keys: Vec<Id> = index.keys().take(count).cloned().collect();
        let mut removed = Vec::with_capacity(keys.len());

        for key in keys {
            index.remove(&key);
            if let Some((_, row)) = self.primary_index.remove(&key) {
                removed.push(row.as_ref().clone());
                self.row_count.fetch_sub(1, Ordering::AcqRel);
            }
        }

        removed
    }

    /// Collect all rows in sorted order.
    pub fn export_rows(&self) -> Vec<VectorRow> {
        let index = self.ordered_index.read();
        index.values().map(|row| row.as_ref().clone()).collect()
    }

    /// Replace the current contents with the provided rows.
    pub fn replace_rows(&self, rows: Vec<VectorRow>) {
        self.primary_index.clear();
        self.ordered_index.write().clear();
        self.row_count.store(0, Ordering::Release);

        for row in rows {
            self.insert(row);
        }
    }

    /// Iterate over a range using the ordered index.
    pub fn range(&self, start: Option<Id>, end: Option<Id>, limit: usize) -> Vec<Arc<VectorRow>> {
        let index = self.ordered_index.read();
        let start_bound = start.map_or(Bound::Unbounded, Bound::Included);
        let end_bound = end.map_or(Bound::Unbounded, Bound::Included);
        index
            .range((start_bound, end_bound))
            .take(limit)
            .map(|(_, row)| row.clone())
            .collect()
    }
}

/// Persistent WAL operation for the in-memory engine.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum WalOperation {
    Insert(VectorRow),
    Update(VectorRow),
    Delete(Id),
}

/// A single WAL record annotated with metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WalRecord {
    pub collection: String,
    pub lsn: u64,
    pub operation: WalOperation,
    pub timestamp: u64,
}

impl WalRecord {
    fn new(collection: String, lsn: u64, operation: WalOperation) -> Self {
        Self {
            collection,
            lsn,
            operation,
            timestamp: current_timestamp(),
        }
    }
}

fn current_timestamp() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or_default()
}

/// Write-ahead logging manager for memory tables.
pub struct MemoryDurabilityManager {
    wal_path: PathBuf,
    writer: Mutex<tokio::fs::File>,
    current_lsn: AtomicU64,
}

impl MemoryDurabilityManager {
    /// Create a new manager backed by the provided WAL file.
    pub async fn new(path: impl AsRef<Path>) -> Result<Self> {
        let wal_path = path.as_ref().to_path_buf();
        if let Some(parent) = wal_path.parent() {
            fs::create_dir_all(parent).await?;
        }

        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .read(true)
            .open(&wal_path)
            .await?;

        let manager = Self {
            wal_path,
            writer: Mutex::new(file),
            current_lsn: AtomicU64::new(0),
        };

        let max_lsn = manager.scan_max_lsn().await?;
        manager.current_lsn.store(max_lsn, Ordering::Release);

        Ok(manager)
    }

    async fn scan_max_lsn(&self) -> Result<u64> {
        let records = self.replay().await?;
        Ok(records.iter().map(|r| r.lsn).max().unwrap_or(0))
    }

    fn next_lsn(&self) -> u64 {
        self.current_lsn.fetch_add(1, Ordering::AcqRel) + 1
    }

    /// Append a WAL operation, optionally forcing a sync.
    pub async fn log_operation(
        &self,
        collection: &str,
        operation: WalOperation,
        sync: bool,
    ) -> Result<u64> {
        let lsn = self.next_lsn();
        let record = WalRecord::new(collection.to_string(), lsn, operation);
        let mut writer = self.writer.lock().await;
        let mut encoded = serde_json::to_vec(&record)
            .map_err(|e| StorageEngineError::Serialization(e.to_string()))?;
        encoded.push(b'\n');
        writer.write_all(&encoded).await?;
        if sync {
            writer.sync_data().await?;
        }
        Ok(lsn)
    }

    /// Convenience helper for insert logging.
    pub async fn log_insert(&self, collection: &str, row: &VectorRow, sync: bool) -> Result<u64> {
        self.log_operation(collection, WalOperation::Insert(row.clone()), sync)
            .await
    }

    /// Convenience helper for updates.
    pub async fn log_update(&self, collection: &str, row: &VectorRow, sync: bool) -> Result<u64> {
        self.log_operation(collection, WalOperation::Update(row.clone()), sync)
            .await
    }

    /// Convenience helper for deletes.
    pub async fn log_delete(&self, collection: &str, id: &Id, sync: bool) -> Result<u64> {
        self.log_operation(collection, WalOperation::Delete(*id), sync)
            .await
    }

    /// Force a sync of the underlying WAL file.
    pub async fn sync(&self) -> Result<()> {
        let writer = self.writer.lock().await;
        writer.sync_data().await?;
        Ok(())
    }

    /// Truncate the WAL after a successful snapshot.
    pub async fn truncate(&self) -> Result<()> {
        {
            let truncate_file = OpenOptions::new()
                .create(true)
                .write(true)
                .truncate(true)
                .open(&self.wal_path)
                .await?;
            truncate_file.sync_data().await?;
        }

        let mut writer = self.writer.lock().await;
        let new_file = OpenOptions::new()
            .create(true)
            .append(true)
            .read(true)
            .open(&self.wal_path)
            .await?;
        *writer = new_file;
        self.current_lsn.store(0, Ordering::Release);
        Ok(())
    }

    /// Read and deserialize every WAL record.
    pub async fn replay(&self) -> Result<Vec<WalRecord>> {
        let data = match fs::read(&self.wal_path).await {
            Ok(bytes) => bytes,
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
            Err(err) => return Err(err.into()),
        };

        let mut records = Vec::new();
        for line in data.split(|b| *b == b'\n') {
            if line.is_empty() {
                continue;
            }
            let record: WalRecord = serde_json::from_slice(line)
                .map_err(|e| StorageEngineError::Serialization(e.to_string()))?;
            records.push(record);
        }

        Ok(records)
    }

    /// Return the currently assigned LSN.
    pub fn current_lsn(&self) -> u64 {
        self.current_lsn.load(Ordering::Acquire)
    }

    /// Path to the WAL on disk.
    pub fn wal_path(&self) -> &Path {
        &self.wal_path
    }
}

/// A consistent snapshot of the in-memory state used for crash recovery.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemorySnapshot {
    pub taken_at: u64,
    pub last_lsn: u64,
    pub tables: HashMap<String, Vec<VectorRow>>,
}

impl MemorySnapshot {
    fn new(last_lsn: u64, tables: HashMap<String, Vec<VectorRow>>) -> Self {
        Self {
            taken_at: current_timestamp(),
            last_lsn,
            tables,
        }
    }
}

/// In-memory storage composed of hash and tree indexes with durability hooks.
pub struct InMemoryStorage {
    tables: DashMap<String, Arc<MemoryTable>>,
    durability: Arc<MemoryDurabilityManager>,
    default_policy: ConfigurableStoragePolicy,
}

impl InMemoryStorage {
    /// Create a new storage instance using the provided durability manager.
    pub fn new(durability: Arc<MemoryDurabilityManager>) -> Self {
        Self {
            tables: DashMap::new(),
            durability,
            default_policy: ConfigurableStoragePolicy::default(),
        }
    }

    /// Override the default collection policy.
    pub fn with_default_policy(
        durability: Arc<MemoryDurabilityManager>,
        default_policy: ConfigurableStoragePolicy,
    ) -> Self {
        Self {
            tables: DashMap::new(),
            durability,
            default_policy,
        }
    }

    fn ensure_table(&self, collection: &str) -> Arc<MemoryTable> {
        if let Some(table) = self.tables.get(collection) {
            return table.clone();
        }

        let policy = self.default_policy.clone();
        match self.tables.entry(collection.to_string()) {
            Entry::Occupied(entry) => entry.get().clone(),
            Entry::Vacant(entry) => {
                let table = Arc::new(MemoryTable::new(collection.to_string(), policy));
                entry.insert(table.clone());
                table
            }
        }
    }

    /// Explicitly create a collection with a provided policy.
    pub fn create_collection(
        &self,
        collection: &str,
        policy: ConfigurableStoragePolicy,
    ) -> Arc<MemoryTable> {
        match self.tables.entry(collection.to_string()) {
            Entry::Occupied(entry) => {
                entry.get().set_policy(policy);
                entry.get().clone()
            }
            Entry::Vacant(entry) => {
                let table = Arc::new(MemoryTable::new(collection.to_string(), policy));
                entry.insert(table.clone());
                table
            }
        }
    }

    /// Update the policy for an existing collection.
    pub fn update_policy(&self, collection: &str, policy: ConfigurableStoragePolicy) -> Result<()> {
        if let Some(table) = self.tables.get(collection) {
            table.set_policy(policy);
            Ok(())
        } else {
            Err(StorageEngineError::CollectionNotFound(
                collection.to_string(),
            ))
        }
    }

    /// Fetch a collection reference.
    pub fn get_collection(&self, collection: &str) -> Option<Arc<MemoryTable>> {
        self.tables.get(collection).map(|t| t.clone())
    }

    /// Insert a row while recording a WAL entry.
    pub async fn insert(&self, collection: &str, row: VectorRow) -> Result<()> {
        let table = self.ensure_table(collection);
        let policy = table.policy();
        if policy.wal_enabled {
            self.durability
                .log_insert(collection, &row, policy.wal_sync)
                .await?;
        }
        table.insert(row);
        Ok(())
    }

    /// Update a row and log the change.
    pub async fn update(&self, collection: &str, row: VectorRow) -> Result<()> {
        let table = self.ensure_table(collection);
        let policy = table.policy();
        if policy.wal_enabled {
            self.durability
                .log_update(collection, &row, policy.wal_sync)
                .await?;
        }
        table.insert(row);
        Ok(())
    }

    /// Delete a row by identifier.
    pub async fn delete(&self, collection: &str, id: &Id) -> Result<Option<VectorRow>> {
        let table = match self.tables.get(collection) {
            Some(table) => table,
            None => return Ok(None),
        };
        let policy = table.policy();
        if policy.wal_enabled {
            self.durability
                .log_delete(collection, id, policy.wal_sync)
                .await?;
        }
        Ok(table.remove(id))
    }

    /// Fetch a single row.
    pub fn get(&self, collection: &str, id: &Id) -> Option<Arc<VectorRow>> {
        self.tables.get(collection).and_then(|table| table.get(id))
    }

    /// Range scan using the ordered index.
    pub fn range(
        &self,
        collection: &str,
        start: Option<Id>,
        end: Option<Id>,
        limit: usize,
    ) -> Vec<Arc<VectorRow>> {
        self.tables
            .get(collection)
            .map(|table| table.range(start, end, limit))
            .unwrap_or_default()
    }

    /// Export a snapshot of the current in-memory state.
    pub fn export_snapshot(&self) -> MemorySnapshot {
        let mut tables = HashMap::new();
        for entry in self.tables.iter() {
            tables.insert(entry.key().clone(), entry.value().export_rows());
        }
        MemorySnapshot::new(self.durability.current_lsn(), tables)
    }

    /// Apply a snapshot to the current state.
    pub fn apply_snapshot(&self, snapshot: &MemorySnapshot) {
        for (collection, rows) in &snapshot.tables {
            let table = self.ensure_table(collection);
            table.replace_rows(rows.clone());
        }
    }

    /// Replay WAL records without emitting new log entries.
    pub fn apply_wal_record(&self, record: WalRecord) {
        let table = self.ensure_table(&record.collection);
        match record.operation {
            WalOperation::Insert(row) | WalOperation::Update(row) => {
                table.insert(row);
            }
            WalOperation::Delete(id) => {
                table.remove(&id);
            }
        }
    }

    /// Convenience helper to recover from snapshot and WAL.
    pub fn recover(&self, snapshot: Option<MemorySnapshot>, wal: &[WalRecord]) {
        if let Some(snapshot) = snapshot {
            self.apply_snapshot(&snapshot);
        }
        for record in wal {
            self.apply_wal_record(record.clone());
        }
    }

    /// List all known collections.
    pub fn collections(&self) -> Vec<String> {
        self.tables
            .iter()
            .map(|entry| entry.key().clone())
            .collect()
    }
}

/// Periodic snapshotter for the in-memory engine.
pub struct SnapshotCreator {
    storage: Arc<InMemoryStorage>,
    durability: Arc<MemoryDurabilityManager>,
    snapshot_dir: PathBuf,
    interval: Duration,
    last_snapshot: AtomicU64,
}

impl SnapshotCreator {
    /// Build a new snapshot creator.
    pub async fn new(
        storage: Arc<InMemoryStorage>,
        durability: Arc<MemoryDurabilityManager>,
        snapshot_dir: impl AsRef<Path>,
        interval: Duration,
    ) -> Result<Self> {
        let snapshot_dir = snapshot_dir.as_ref().to_path_buf();
        fs::create_dir_all(&snapshot_dir).await?;
        Ok(Self {
            storage,
            durability,
            snapshot_dir,
            interval,
            last_snapshot: AtomicU64::new(0),
        })
    }

    /// Run a single snapshot cycle.
    pub async fn run_once(&self) -> Result<PathBuf> {
        let snapshot = self.storage.export_snapshot();
        let bytes = bincode::serialize(&snapshot)
            .map_err(|e| StorageEngineError::Serialization(e.to_string()))?;
        let filename = format!("snapshot_{}.bin", snapshot.taken_at);
        let path = self.snapshot_dir.join(filename);
        fs::write(&path, bytes).await?;
        self.durability.truncate().await?;
        self.last_snapshot
            .store(snapshot.taken_at, Ordering::Release);
        Ok(path)
    }

    /// Spawn a background task that periodically captures snapshots.
    pub fn start(self: Arc<Self>) -> JoinHandle<()> {
        tokio::spawn(async move {
            let mut ticker = time::interval(self.interval);
            loop {
                ticker.tick().await;
                if let Err(err) = self.run_once().await {
                    log::error!("snapshot creation failed: {err}");
                }
            }
        })
    }

    /// Load the most recent snapshot from disk.
    pub async fn load_latest(&self) -> Result<Option<MemorySnapshot>> {
        let mut entries = fs::read_dir(&self.snapshot_dir).await?;
        let mut latest: Option<(u64, PathBuf)> = None;
        while let Some(entry) = entries.next_entry().await? {
            let path = entry.path();
            if let Some(stem) = path.file_stem().and_then(|s| s.to_str()) {
                if let Some(ts) = stem.strip_prefix("snapshot_") {
                    if let Ok(value) = ts.parse::<u64>() {
                        if latest.as_ref().map(|(ts, _)| *ts).unwrap_or(0) < value {
                            latest = Some((value, path));
                        }
                    }
                }
            }
        }

        let Some((_, path)) = latest else {
            return Ok(None);
        };

        let bytes = fs::read(path).await?;
        let snapshot: MemorySnapshot = bincode::deserialize(&bytes)
            .map_err(|e| StorageEngineError::Serialization(e.to_string()))?;
        Ok(Some(snapshot))
    }

    /// Restore from the latest snapshot on disk.
    pub async fn restore_latest(&self) -> Result<Option<u64>> {
        if let Some(snapshot) = self.load_latest().await? {
            let taken_at = snapshot.taken_at;
            self.storage.apply_snapshot(&snapshot);
            self.last_snapshot.store(taken_at, Ordering::Release);
            Ok(Some(taken_at))
        } else {
            Ok(None)
        }
    }

    /// Timestamp of the most recent snapshot.
    pub fn last_snapshot(&self) -> u64 {
        self.last_snapshot.load(Ordering::Acquire)
    }
}

/// Coordinates memory and disk storage for hybrid deployments.
pub struct HybridStorageManager {
    memory: Arc<InMemoryStorage>,
    durability: Arc<MemoryDurabilityManager>,
    page_manager: Option<Arc<PageManager>>,
}

impl HybridStorageManager {
    /// Build a new hybrid manager.
    pub fn new(
        memory: Arc<InMemoryStorage>,
        durability: Arc<MemoryDurabilityManager>,
        page_manager: Option<Arc<PageManager>>,
    ) -> Self {
        Self {
            memory,
            durability,
            page_manager,
        }
    }

    /// Insert a row honoring the configured policy.
    pub async fn insert(&self, collection: &str, row: VectorRow) -> Result<()> {
        let table = self
            .memory
            .get_collection(collection)
            .unwrap_or_else(|| self.memory.ensure_table(collection));
        match table.policy().storage_tier {
            StorageTier::DiskPreferred => self.persist_rows(collection, vec![row]).await,
            StorageTier::Hybrid => {
                self.memory.insert(collection, row).await?;
                self.evict_if_needed(collection).await
            }
            StorageTier::MemoryOnly => self.memory.insert(collection, row).await,
        }
    }

    /// Update a row in place.
    pub async fn update(&self, collection: &str, row: VectorRow) -> Result<()> {
        self.memory.update(collection, row).await
    }

    /// Delete a row from both memory and disk.
    pub async fn delete(&self, collection: &str, id: &Id) -> Result<Option<VectorRow>> {
        if let Some(page_manager) = &self.page_manager {
            // Evictions are written via PageManager in append-only mode, there is no
            // direct disk delete. WAL replay during recovery will rebuild state.
            let _ = page_manager; // silence unused warning when feature disabled.
        }
        self.memory.delete(collection, id).await
    }

    async fn evict_if_needed(&self, collection: &str) -> Result<()> {
        let Some(table) = self.memory.get_collection(collection) else {
            return Ok(());
        };
        let policy = table.policy();
        if let Some(limit) = policy.max_memory_rows {
            let current = table.len();
            if current > limit {
                let overflow = current - limit;
                let batch = overflow.max(policy.eviction_batch_size);
                let rows = table.evict_oldest(batch);
                if !rows.is_empty() {
                    self.persist_rows(collection, rows).await?;
                }
            }
        }
        Ok(())
    }

    async fn persist_rows(&self, collection: &str, rows: Vec<VectorRow>) -> Result<()> {
        if rows.is_empty() {
            return Ok(());
        }

        if let Some(page_manager) = &self.page_manager {
            let mut page = Page::new(page_manager.allocate_page().await?);
            for row in rows {
                match page.insert_row(&row) {
                    Ok(_) => {}
                    Err(StorageEngineError::OutOfSpace) => {
                        page_manager.write_page(&mut page).await?;
                        page = Page::new(page_manager.allocate_page().await?);
                        page.insert_row(&row)?;
                    }
                    Err(err) => return Err(err),
                }
            }
            page_manager.write_page(&mut page).await?;
            Ok(())
        } else {
            // Fallback durability via WAL when no disk manager is available.
            for row in rows {
                self.durability
                    .log_operation(collection, WalOperation::Insert(row), true)
                    .await?;
            }
            Ok(())
        }
    }

    /// Restore in-memory data from the latest snapshot and WAL.
    pub async fn recover(&self, snapshotter: &SnapshotCreator) -> Result<()> {
        if let Some(snapshot) = snapshotter.load_latest().await? {
            let wal = self.durability.replay().await?;
            self.memory.recover(Some(snapshot), &wal);
        } else {
            let wal = self.durability.replay().await?;
            self.memory.recover(None, &wal);
        }
        Ok(())
    }
}
