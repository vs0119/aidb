use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use parking_lot::RwLock;
use tokio::time::{Duration, Instant};

use crate::{PageId, Result, StorageEngineError, TransactionId, TransactionSnapshot};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransactionState {
    Active,
    Preparing,
    Committed,
    Aborted,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IsolationLevel {
    ReadUncommitted,
    ReadCommitted,
    RepeatableRead,
    Serializable,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConcurrencyControl {
    Optimistic,
    Pessimistic,
}

#[derive(Debug, Clone, Copy)]
pub struct TransactionOptions {
    pub isolation_level: IsolationLevel,
    pub concurrency_control: ConcurrencyControl,
}

impl Default for TransactionOptions {
    fn default() -> Self {
        Self {
            isolation_level: IsolationLevel::ReadCommitted,
            concurrency_control: ConcurrencyControl::Pessimistic,
        }
    }
}

#[derive(Debug, Clone)]
pub struct TransactionManagerConfig {
    pub default_isolation_level: IsolationLevel,
    pub default_concurrency_control: ConcurrencyControl,
    pub lock_timeout: Duration,
}

impl Default for TransactionManagerConfig {
    fn default() -> Self {
        Self {
            default_isolation_level: IsolationLevel::ReadCommitted,
            default_concurrency_control: ConcurrencyControl::Pessimistic,
            lock_timeout: Duration::from_secs(30),
        }
    }
}

impl TransactionManagerConfig {
    fn default_options(&self) -> TransactionOptions {
        TransactionOptions {
            isolation_level: self.default_isolation_level,
            concurrency_control: self.default_concurrency_control,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AccessMode {
    Read,
    Write,
}

#[derive(Debug, Clone)]
pub struct Transaction {
    pub id: TransactionId,
    pub state: TransactionState,
    pub isolation_level: IsolationLevel,
    pub concurrency_control: ConcurrencyControl,
    pub snapshot: TransactionSnapshot,
    pub start_time: Instant,
    pub locks_held: Arc<RwLock<HashSet<PageId>>>,
    pub read_set: Arc<RwLock<HashSet<PageId>>>,
    pub write_set: Arc<RwLock<HashSet<PageId>>>,
}

impl Transaction {
    pub fn new(
        id: TransactionId,
        isolation_level: IsolationLevel,
        concurrency_control: ConcurrencyControl,
        snapshot: TransactionSnapshot,
    ) -> Self {
        Self {
            id,
            state: TransactionState::Active,
            isolation_level,
            concurrency_control,
            snapshot,
            start_time: Instant::now(),
            locks_held: Arc::new(RwLock::new(HashSet::new())),
            read_set: Arc::new(RwLock::new(HashSet::new())),
            write_set: Arc::new(RwLock::new(HashSet::new())),
        }
    }

    pub fn add_read_page(&self, page_id: PageId) {
        self.read_set.write().insert(page_id);
    }

    pub fn add_write_page(&self, page_id: PageId) {
        self.write_set.write().insert(page_id);
    }

    pub fn duration(&self) -> Duration {
        self.start_time.elapsed()
    }

    pub fn has_conflict_with(&self, other: &Transaction) -> bool {
        let my_read_set = self.read_set.read();
        let my_write_set = self.write_set.read();
        let other_read_set = other.read_set.read();
        let other_write_set = other.write_set.read();

        // Write-Write conflict
        if !my_write_set.is_disjoint(&other_write_set) {
            return true;
        }

        // Read-Write conflict
        if !my_read_set.is_disjoint(&other_write_set) || !my_write_set.is_disjoint(&other_read_set)
        {
            return true;
        }

        false
    }
}

#[derive(Debug)]
struct LockEntry {
    transaction_id: TransactionId,
    lock_type: LockType,
    granted: bool,
    requested_at: Instant,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LockType {
    Shared,
    Exclusive,
    Update,
}

pub struct LockManager {
    locks: RwLock<HashMap<PageId, Vec<LockEntry>>>,
    deadlock_detector: Arc<DeadlockDetector>,
}

impl LockManager {
    pub fn new() -> Self {
        Self {
            locks: RwLock::new(HashMap::new()),
            deadlock_detector: Arc::new(DeadlockDetector::new()),
        }
    }

    pub async fn acquire_lock(
        &self,
        transaction_id: TransactionId,
        page_id: PageId,
        lock_type: LockType,
        timeout: Duration,
    ) -> Result<()> {
        let deadline = Instant::now() + timeout;

        loop {
            enum LockAttempt {
                Granted,
                Waiting,
                TimedOut,
            }

            let attempt = {
                let mut locks = self.locks.write();
                let entries = locks.entry(page_id).or_insert_with(Vec::new);

                if self.can_grant_lock(entries, &lock_type) {
                    if let Some(entry) = entries
                        .iter_mut()
                        .find(|e| e.transaction_id == transaction_id)
                    {
                        entry.granted = true;
                        entry.lock_type = lock_type;
                        entry.requested_at = Instant::now();
                    } else {
                        entries.push(LockEntry {
                            transaction_id,
                            lock_type,
                            granted: true,
                            requested_at: Instant::now(),
                        });
                    }
                    self.deadlock_detector.clear_waiter(transaction_id);
                    LockAttempt::Granted
                } else if Instant::now() >= deadline {
                    entries.retain(|entry| entry.transaction_id != transaction_id);
                    if entries.is_empty() {
                        locks.remove(&page_id);
                    }
                    LockAttempt::TimedOut
                } else {
                    if !entries
                        .iter()
                        .any(|entry| entry.transaction_id == transaction_id)
                    {
                        entries.push(LockEntry {
                            transaction_id,
                            lock_type,
                            granted: false,
                            requested_at: Instant::now(),
                        });
                    }

                    let holders: Vec<_> = entries
                        .iter()
                        .filter(|e| e.granted)
                        .map(|e| e.transaction_id)
                        .collect();
                    self.deadlock_detector
                        .update_wait_edges(transaction_id, holders);
                    LockAttempt::Waiting
                }
            };

            match attempt {
                LockAttempt::Granted => return Ok(()),
                LockAttempt::TimedOut => {
                    self.deadlock_detector.remove_transaction(transaction_id);
                    return Err(StorageEngineError::TransactionConflict(
                        "Lock acquisition timeout".to_string(),
                    ));
                }
                LockAttempt::Waiting => {
                    if self.deadlock_detector.has_deadlock(transaction_id).await? {
                        self.remove_waiting_entry(page_id, transaction_id);
                        self.deadlock_detector.remove_transaction(transaction_id);
                        return Err(StorageEngineError::TransactionConflict(
                            "Deadlock detected".to_string(),
                        ));
                    }

                    tokio::time::sleep(Duration::from_millis(10)).await;
                }
            }
        }
    }

    fn can_grant_lock(&self, entries: &[LockEntry], lock_type: &LockType) -> bool {
        let granted_locks: Vec<_> = entries.iter().filter(|e| e.granted).collect();

        match lock_type {
            LockType::Shared => granted_locks
                .iter()
                .all(|e| e.lock_type != LockType::Exclusive),
            LockType::Exclusive => granted_locks.is_empty(),
            LockType::Update => granted_locks
                .iter()
                .all(|e| e.lock_type == LockType::Shared),
        }
    }

    pub fn release_locks(&self, transaction_id: TransactionId) {
        let mut locks = self.locks.write();

        locks.retain(|_, entries| {
            entries.retain(|e| e.transaction_id != transaction_id);
            !entries.is_empty()
        });

        self.grant_waiting_locks(&mut locks);
        self.deadlock_detector.remove_transaction(transaction_id);
    }

    fn grant_waiting_locks(&self, locks: &mut HashMap<PageId, Vec<LockEntry>>) {
        for entries in locks.values_mut() {
            let mut i = 0;
            while i < entries.len() {
                let lock_type = entries[i].lock_type;
                if !entries[i].granted && self.can_grant_lock(entries, &lock_type) {
                    entries[i].granted = true;
                    self.deadlock_detector
                        .clear_waiter(entries[i].transaction_id);
                }
                i += 1;
            }
        }
    }

    fn remove_waiting_entry(&self, page_id: PageId, transaction_id: TransactionId) {
        let mut locks = self.locks.write();
        if let Some(entries) = locks.get_mut(&page_id) {
            entries.retain(|entry| entry.transaction_id != transaction_id);
            if entries.is_empty() {
                locks.remove(&page_id);
            }
        }
    }
}

pub struct DeadlockDetector {
    wait_graph: RwLock<HashMap<TransactionId, HashSet<TransactionId>>>,
}

impl DeadlockDetector {
    pub fn new() -> Self {
        Self {
            wait_graph: RwLock::new(HashMap::new()),
        }
    }

    pub async fn has_deadlock(&self, transaction_id: TransactionId) -> Result<bool> {
        let graph = self.wait_graph.read();
        let mut visited = HashSet::new();
        let mut stack = HashSet::new();
        Ok(self.has_cycle_dfs(&graph, transaction_id, &mut visited, &mut stack))
    }

    fn has_cycle_dfs(
        &self,
        graph: &HashMap<TransactionId, HashSet<TransactionId>>,
        node: TransactionId,
        visited: &mut HashSet<TransactionId>,
        stack: &mut HashSet<TransactionId>,
    ) -> bool {
        if stack.contains(&node) {
            return true;
        }

        if !visited.insert(node) {
            return false;
        }

        stack.insert(node);

        if let Some(neighbors) = graph.get(&node) {
            for &neighbor in neighbors {
                if self.has_cycle_dfs(graph, neighbor, visited, stack) {
                    return true;
                }
            }
        }

        stack.remove(&node);
        false
    }

    pub fn update_wait_edges(&self, waiter: TransactionId, holders: Vec<TransactionId>) {
        let mut graph = self.wait_graph.write();
        if holders.is_empty() {
            graph.remove(&waiter);
            return;
        }

        let entry = graph.entry(waiter).or_insert_with(HashSet::new);
        entry.clear();
        for holder in holders {
            if holder != waiter {
                entry.insert(holder);
            }
        }
        if entry.is_empty() {
            graph.remove(&waiter);
        }
    }

    pub fn clear_waiter(&self, waiter: TransactionId) {
        self.wait_graph.write().remove(&waiter);
    }

    pub fn remove_transaction(&self, transaction_id: TransactionId) {
        let mut graph = self.wait_graph.write();
        graph.remove(&transaction_id);

        for edges in graph.values_mut() {
            edges.retain(|&id| id != transaction_id);
        }
    }
}

pub struct TransactionManager {
    next_xid: AtomicU64,
    active_transactions: RwLock<HashMap<TransactionId, Transaction>>,
    committed_transactions: RwLock<HashSet<TransactionId>>,
    aborted_transactions: RwLock<HashSet<TransactionId>>,
    lock_manager: Arc<LockManager>,
    global_snapshot: RwLock<TransactionSnapshot>,
    config: TransactionManagerConfig,
}

impl TransactionManager {
    pub fn new() -> Self {
        Self::with_config(TransactionManagerConfig::default())
    }

    pub fn with_config(config: TransactionManagerConfig) -> Self {
        let initial_snapshot = TransactionSnapshot {
            xid: 0,
            active_xids: Arc::new(Vec::new()),
            xmin: 0,
            xmax: 1,
        };

        Self {
            next_xid: AtomicU64::new(1),
            active_transactions: RwLock::new(HashMap::new()),
            committed_transactions: RwLock::new(HashSet::new()),
            aborted_transactions: RwLock::new(HashSet::new()),
            lock_manager: Arc::new(LockManager::new()),
            global_snapshot: RwLock::new(initial_snapshot),
            config,
        }
    }

    pub async fn begin(&self) -> Result<Transaction> {
        self.begin_with_options(self.config.default_options()).await
    }

    pub async fn begin_with_isolation(
        &self,
        isolation_level: IsolationLevel,
    ) -> Result<Transaction> {
        let mut options = self.config.default_options();
        options.isolation_level = isolation_level;
        self.begin_with_options(options).await
    }

    pub async fn begin_with_options(&self, options: TransactionOptions) -> Result<Transaction> {
        let xid = self.next_xid.fetch_add(1, Ordering::AcqRel);
        let snapshot = self.create_snapshot(xid, options.isolation_level);

        let transaction = Transaction::new(
            xid,
            options.isolation_level,
            options.concurrency_control,
            snapshot,
        );

        self.active_transactions
            .write()
            .insert(xid, transaction.clone());
        self.update_global_snapshot();

        Ok(transaction)
    }

    fn create_snapshot(
        &self,
        xid: TransactionId,
        isolation_level: IsolationLevel,
    ) -> TransactionSnapshot {
        let active_transactions = self.active_transactions.read();

        match isolation_level {
            IsolationLevel::ReadUncommitted => TransactionSnapshot {
                xid,
                active_xids: Arc::new(Vec::new()),
                xmin: 0,
                xmax: u64::MAX,
            },
            IsolationLevel::ReadCommitted => {
                let active_xids: Vec<_> = active_transactions.keys().copied().collect();
                TransactionSnapshot {
                    xid,
                    active_xids: Arc::new(active_xids),
                    xmin: active_transactions.keys().copied().min().unwrap_or(xid),
                    xmax: xid,
                }
            }
            IsolationLevel::RepeatableRead | IsolationLevel::Serializable => {
                let active_xids: Vec<_> = active_transactions.keys().copied().collect();
                TransactionSnapshot {
                    xid,
                    active_xids: Arc::new(active_xids),
                    xmin: active_transactions.keys().copied().min().unwrap_or(xid),
                    xmax: xid,
                }
            }
        }
    }

    pub async fn commit(&self, mut transaction: Transaction) -> Result<()> {
        if transaction.state != TransactionState::Active {
            return Err(StorageEngineError::TransactionConflict(
                "Transaction not active".to_string(),
            ));
        }

        if self.requires_conflict_detection(&transaction) && self.has_conflicts(&transaction)? {
            return self.rollback(transaction).await;
        }

        transaction.state = TransactionState::Committed;

        self.committed_transactions.write().insert(transaction.id);
        self.active_transactions.write().remove(&transaction.id);
        self.lock_manager.release_locks(transaction.id);
        transaction.locks_held.write().clear();
        transaction.read_set.write().clear();
        transaction.write_set.write().clear();

        self.update_global_snapshot();

        Ok(())
    }

    pub async fn rollback(&self, mut transaction: Transaction) -> Result<()> {
        transaction.state = TransactionState::Aborted;

        self.aborted_transactions.write().insert(transaction.id);
        self.active_transactions.write().remove(&transaction.id);
        self.lock_manager.release_locks(transaction.id);
        transaction.locks_held.write().clear();
        transaction.read_set.write().clear();
        transaction.write_set.write().clear();

        self.update_global_snapshot();

        Ok(())
    }

    fn requires_conflict_detection(&self, transaction: &Transaction) -> bool {
        matches!(
            transaction.concurrency_control,
            ConcurrencyControl::Optimistic
        ) || transaction.isolation_level == IsolationLevel::Serializable
    }

    fn has_conflicts(&self, transaction: &Transaction) -> Result<bool> {
        let active_transactions = self.active_transactions.read();

        for other_txn in active_transactions.values() {
            if other_txn.id != transaction.id && transaction.has_conflict_with(other_txn) {
                return Ok(true);
            }
        }

        Ok(false)
    }

    fn update_global_snapshot(&self) {
        let active_transactions = self.active_transactions.read();
        let active_xids: Vec<_> = active_transactions.keys().copied().collect();

        let mut global_snapshot = self.global_snapshot.write();
        global_snapshot.active_xids = Arc::new(active_xids);
        global_snapshot.xmin = active_transactions
            .keys()
            .copied()
            .min()
            .unwrap_or(global_snapshot.xmax);
        global_snapshot.xmax = self.next_xid.load(Ordering::Acquire);
    }

    pub async fn acquire_lock(
        &self,
        transaction: &Transaction,
        page_id: PageId,
        lock_type: LockType,
    ) -> Result<()> {
        if matches!(
            transaction.concurrency_control,
            ConcurrencyControl::Optimistic
        ) {
            return Ok(());
        }

        self.lock_manager
            .acquire_lock(transaction.id, page_id, lock_type, self.config.lock_timeout)
            .await?;
        transaction.locks_held.write().insert(page_id);
        Ok(())
    }

    pub fn register_page_access(
        &self,
        transaction: &Transaction,
        page_id: PageId,
        mode: AccessMode,
    ) {
        match mode {
            AccessMode::Read => {
                transaction.add_read_page(page_id);
            }
            AccessMode::Write => {
                transaction.add_read_page(page_id);
                transaction.add_write_page(page_id);
            }
        }
    }

    pub fn get_active_transaction_count(&self) -> usize {
        self.active_transactions.read().len()
    }

    pub fn get_transaction_stats(&self) -> TransactionStats {
        let active = self.active_transactions.read();
        let committed = self.committed_transactions.read();
        let aborted = self.aborted_transactions.read();

        TransactionStats {
            active_count: active.len(),
            committed_count: committed.len(),
            aborted_count: aborted.len(),
            oldest_active_transaction: active.values().map(|t| t.start_time).min(),
        }
    }

    pub async fn cleanup_old_transactions(&self, max_age: Duration) -> Result<usize> {
        let mut cleaned = 0;
        let _cutoff = Instant::now() - max_age;

        {
            let mut aborted = self.aborted_transactions.write();
            let initial_size = aborted.len();

            aborted.retain(|_| false);

            cleaned += initial_size - aborted.len();
        }

        {
            let mut committed = self.committed_transactions.write();
            let initial_size = committed.len();

            committed.retain(|_| false);

            cleaned += initial_size - committed.len();
        }

        Ok(cleaned)
    }
}

#[derive(Debug, Clone)]
pub struct TransactionStats {
    pub active_count: usize,
    pub committed_count: usize,
    pub aborted_count: usize,
    pub oldest_active_transaction: Option<Instant>,
}
