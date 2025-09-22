use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::Arc;

use aidb_core::{Id, JsonValue, Vector};
use serde::{Deserialize, Serialize};
use thiserror::Error;

pub mod buffer;
pub mod compression;
pub mod distributed;
pub mod lsm;
pub mod page;
pub mod transaction;
pub mod vacuum;

pub use buffer::*;
pub use compression::*;
pub use distributed::*;
pub use lsm::*;
pub use page::*;
pub use transaction::*;
pub use vacuum::*;

#[derive(Error, Debug)]
pub enum StorageEngineError {
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("page corruption detected: {0}")]
    PageCorruption(String),
    #[error("transaction conflict: {0}")]
    TransactionConflict(String),
    #[error("out of storage space")]
    OutOfSpace,
    #[error("invalid page reference: {0}")]
    InvalidPageRef(PageId),
    #[error("compression error: {0}")]
    CompressionError(String),
    #[error("distributed transaction error: {0}")]
    DistributedTransaction(String),
}

pub type Result<T> = std::result::Result<T, StorageEngineError>;
pub type PageId = u32;
pub type TransactionId = u64;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct RowId {
    pub page_id: PageId,
    pub slot_id: u16,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorRow {
    pub id: Id,
    pub vector: Vector,
    pub payload: Option<JsonValue>,
    pub created_xid: TransactionId,
    pub updated_xid: Option<TransactionId>,
    pub deleted_xid: Option<TransactionId>,
}

impl VectorRow {
    pub fn new(id: Id, vector: Vector, payload: Option<JsonValue>, xid: TransactionId) -> Self {
        Self {
            id,
            vector,
            payload,
            created_xid: xid,
            updated_xid: None,
            deleted_xid: None,
        }
    }

    pub fn is_visible(&self, snapshot: &TransactionSnapshot) -> bool {
        if let Some(deleted_xid) = self.deleted_xid {
            if snapshot.can_see(deleted_xid) {
                return false;
            }
        }

        if let Some(updated_xid) = self.updated_xid {
            return snapshot.can_see(updated_xid);
        }

        snapshot.can_see(self.created_xid)
    }
}

#[derive(Debug, Clone)]
pub struct TransactionSnapshot {
    pub xid: TransactionId,
    pub active_xids: Arc<Vec<TransactionId>>,
    pub xmin: TransactionId,
    pub xmax: TransactionId,
}

impl TransactionSnapshot {
    pub fn can_see(&self, xid: TransactionId) -> bool {
        if xid >= self.xmax {
            return false;
        }
        if xid < self.xmin {
            return true;
        }
        !self.active_xids.binary_search(&xid).is_ok()
    }
}

pub struct StorageEngine {
    pub buffer_pool: Arc<BufferPool>,
    pub transaction_manager: Arc<TransactionManager>,
    pub page_manager: Arc<PageManager>,
    pub compression_manager: Arc<CompressionManager>,
    pub vacuum_manager: Arc<VacuumManager>,
    pub stats: Arc<EngineStats>,
}

#[derive(Default)]
pub struct EngineStats {
    pub page_reads: AtomicU64,
    pub page_writes: AtomicU64,
    pub cache_hits: AtomicU64,
    pub cache_misses: AtomicU64,
    pub vacuum_runs: AtomicU64,
    pub compression_ratio: AtomicU32,
}

impl StorageEngine {
    pub async fn new(data_dir: impl AsRef<std::path::Path>) -> Result<Self> {
        Self::with_transaction_config(data_dir, transaction::TransactionManagerConfig::default())
            .await
    }

    pub async fn with_transaction_config(
        data_dir: impl AsRef<std::path::Path>,
        config: transaction::TransactionManagerConfig,
    ) -> Result<Self> {
        let buffer_pool = Arc::new(BufferPool::new(1024).await?);
        let transaction_manager = Arc::new(TransactionManager::with_config(config));
        let page_manager = Arc::new(PageManager::new(data_dir).await?);
        let compression_manager = Arc::new(CompressionManager::new());
        let vacuum_manager = Arc::new(VacuumManager::new());
        let stats = Arc::new(EngineStats::default());

        Ok(Self {
            buffer_pool,
            transaction_manager,
            page_manager,
            compression_manager,
            vacuum_manager,
            stats,
        })
    }

    pub async fn begin_transaction(&self) -> Result<Transaction> {
        self.transaction_manager.begin().await
    }

    pub async fn begin_transaction_with_options(
        &self,
        options: transaction::TransactionOptions,
    ) -> Result<Transaction> {
        self.transaction_manager.begin_with_options(options).await
    }

    pub async fn insert_vector(
        &self,
        txn: &Transaction,
        id: Id,
        vector: Vector,
        payload: Option<JsonValue>,
    ) -> Result<RowId> {
        let row = VectorRow::new(id, vector, payload, txn.id);

        let page_id = self.page_manager.allocate_page().await?;
        self.transaction_manager
            .acquire_lock(txn, page_id, transaction::LockType::Exclusive)
            .await?;
        self.transaction_manager
            .register_page_access(txn, page_id, transaction::AccessMode::Write);
        let page = self.buffer_pool.get_page(page_id).await?;

        let slot_id = page.write().await.insert_row(&row)?;
        let row_id = RowId { page_id, slot_id };

        self.stats.page_writes.fetch_add(1, Ordering::Relaxed);
        Ok(row_id)
    }

    pub async fn get_vector(&self, txn: &Transaction, row_id: RowId) -> Result<Option<VectorRow>> {
        self.transaction_manager
            .acquire_lock(txn, row_id.page_id, transaction::LockType::Shared)
            .await?;
        self.transaction_manager.register_page_access(
            txn,
            row_id.page_id,
            transaction::AccessMode::Read,
        );
        let page = self.buffer_pool.get_page(row_id.page_id).await?;
        let page_guard = page.read().await;

        if let Some(row) = page_guard.get_row(row_id.slot_id)? {
            if row.is_visible(&txn.snapshot()) {
                self.stats.cache_hits.fetch_add(1, Ordering::Relaxed);
                return Ok(Some(row));
            }
        }

        self.stats.cache_misses.fetch_add(1, Ordering::Relaxed);
        Ok(None)
    }

    pub async fn update_vector(
        &self,
        txn: &Transaction,
        row_id: RowId,
        vector: Vector,
        payload: Option<JsonValue>,
    ) -> Result<()> {
        self.transaction_manager
            .acquire_lock(txn, row_id.page_id, transaction::LockType::Exclusive)
            .await?;
        self.transaction_manager.register_page_access(
            txn,
            row_id.page_id,
            transaction::AccessMode::Write,
        );
        let page = self.buffer_pool.get_page(row_id.page_id).await?;
        let mut page_guard = page.write().await;

        if let Some(mut row) = page_guard.get_row(row_id.slot_id)? {
            if !row.is_visible(&txn.snapshot()) {
                return Err(StorageEngineError::TransactionConflict(
                    "Row not visible".to_string(),
                ));
            }

            row.vector = vector;
            row.payload = payload;
            row.updated_xid = Some(txn.id);

            page_guard.update_row(row_id.slot_id, &row)?;
            self.stats.page_writes.fetch_add(1, Ordering::Relaxed);
        }

        Ok(())
    }

    pub async fn delete_vector(&self, txn: &Transaction, row_id: RowId) -> Result<bool> {
        self.transaction_manager
            .acquire_lock(txn, row_id.page_id, transaction::LockType::Exclusive)
            .await?;
        self.transaction_manager.register_page_access(
            txn,
            row_id.page_id,
            transaction::AccessMode::Write,
        );
        let page = self.buffer_pool.get_page(row_id.page_id).await?;
        let mut page_guard = page.write().await;

        if let Some(mut row) = page_guard.get_row(row_id.slot_id)? {
            if !row.is_visible(&txn.snapshot()) {
                return Ok(false);
            }

            row.deleted_xid = Some(txn.id);
            page_guard.update_row(row_id.slot_id, &row)?;
            self.stats.page_writes.fetch_add(1, Ordering::Relaxed);

            return Ok(true);
        }

        Ok(false)
    }

    pub async fn commit_transaction(&self, txn: Transaction) -> Result<()> {
        self.transaction_manager.commit(txn).await
    }

    pub async fn rollback_transaction(&self, txn: Transaction) -> Result<()> {
        self.transaction_manager.rollback(txn).await
    }

    pub async fn vacuum(&self) -> Result<VacuumStats> {
        self.vacuum_manager
            .run(&self.buffer_pool, &self.page_manager)
            .await
    }

    pub fn get_stats(&self) -> EngineStats {
        EngineStats {
            page_reads: AtomicU64::new(self.stats.page_reads.load(Ordering::Relaxed)),
            page_writes: AtomicU64::new(self.stats.page_writes.load(Ordering::Relaxed)),
            cache_hits: AtomicU64::new(self.stats.cache_hits.load(Ordering::Relaxed)),
            cache_misses: AtomicU64::new(self.stats.cache_misses.load(Ordering::Relaxed)),
            vacuum_runs: AtomicU64::new(self.stats.vacuum_runs.load(Ordering::Relaxed)),
            compression_ratio: AtomicU32::new(self.stats.compression_ratio.load(Ordering::Relaxed)),
        }
    }
}
