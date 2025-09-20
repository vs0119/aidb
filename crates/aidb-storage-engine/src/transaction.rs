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

#[derive(Debug, Clone)]
pub struct Transaction {
    pub id: TransactionId,
    pub state: TransactionState,
    pub isolation_level: IsolationLevel,
    pub snapshot: Arc<RwLock<TransactionSnapshot>>,
    pub start_time: Instant,
    pub locks_held: Arc<RwLock<HashSet<PageId>>>,
    pub read_set: Arc<RwLock<HashSet<PageId>>>,
    pub write_set: Arc<RwLock<HashSet<PageId>>>,
    savepoints: Arc<RwLock<Vec<Savepoint>>>,
    nested_stack: Arc<RwLock<Vec<String>>>,
}

impl Transaction {
    pub fn new(
        id: TransactionId,
        isolation_level: IsolationLevel,
        snapshot: TransactionSnapshot,
    ) -> Self {
        Self {
            id,
            state: TransactionState::Active,
            isolation_level,
            snapshot: Arc::new(RwLock::new(snapshot)),
            start_time: Instant::now(),
            locks_held: Arc::new(RwLock::new(HashSet::new())),
            read_set: Arc::new(RwLock::new(HashSet::new())),
            write_set: Arc::new(RwLock::new(HashSet::new())),
            savepoints: Arc::new(RwLock::new(Vec::new())),
            nested_stack: Arc::new(RwLock::new(Vec::new())),
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

    pub fn snapshot(&self) -> TransactionSnapshot {
        self.snapshot.read().clone()
    }

    pub fn create_savepoint(&self, name: impl Into<String>) -> Result<()> {
        let name = name.into();

        let mut savepoints = self.savepoints.write();
        if savepoints.iter().any(|sp| sp.name == name) {
            return Err(StorageEngineError::TransactionConflict(format!(
                "Savepoint '{}' already exists",
                name
            )));
        }

        let snapshot = self.snapshot();
        let read_set = self.read_set.read().clone();
        let write_set = self.write_set.read().clone();

        savepoints.push(Savepoint {
            name,
            snapshot,
            read_set,
            write_set,
            created_at: Instant::now(),
        });

        Ok(())
    }

    pub fn release_savepoint(&self, name: &str) -> Result<()> {
        let mut savepoints = self.savepoints.write();

        if let Some(pos) = savepoints.iter().rposition(|sp| sp.name == name) {
            savepoints.remove(pos);
            return Ok(());
        }

        Err(StorageEngineError::TransactionConflict(format!(
            "Savepoint '{}' not found",
            name
        )))
    }

    pub fn rollback_to_savepoint(&self, name: &str) -> Result<()> {
        let mut savepoints = self.savepoints.write();

        let Some(pos) = savepoints.iter().rposition(|sp| sp.name == name) else {
            return Err(StorageEngineError::TransactionConflict(format!(
                "Savepoint '{}' not found",
                name
            )));
        };

        let savepoint = savepoints[pos].clone();
        savepoints.truncate(pos + 1);
        drop(savepoints);

        *self.read_set.write() = savepoint.read_set;
        *self.write_set.write() = savepoint.write_set;
        *self.snapshot.write() = savepoint.snapshot;

        Ok(())
    }

    pub fn begin_nested(&self, name: impl Into<String>) -> Result<NestedTransaction> {
        let name = name.into();
        let mut nested_stack = self.nested_stack.write();
        let depth = nested_stack.len() + 1;
        let savepoint_name = format!("{}::{}::{}", self.id, depth, name);
        drop(nested_stack);

        self.create_savepoint(savepoint_name.clone())?;

        let mut nested_stack = self.nested_stack.write();
        nested_stack.push(savepoint_name.clone());

        Ok(NestedTransaction {
            transaction: self.clone(),
            name,
            savepoint: savepoint_name,
            depth,
        })
    }

    fn pop_nested(&self, savepoint: &str) {
        let mut nested_stack = self.nested_stack.write();
        if let Some(pos) = nested_stack.iter().rposition(|entry| entry == savepoint) {
            nested_stack.remove(pos);
        }
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

#[derive(Debug, Clone)]
struct Savepoint {
    name: String,
    snapshot: TransactionSnapshot,
    read_set: HashSet<PageId>,
    write_set: HashSet<PageId>,
    created_at: Instant,
}

pub struct NestedTransaction {
    transaction: Transaction,
    name: String,
    savepoint: String,
    depth: usize,
}

impl NestedTransaction {
    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn depth(&self) -> usize {
        self.depth
    }

    pub fn commit(self) -> Result<()> {
        self.transaction.release_savepoint(&self.savepoint)?;
        self.transaction.pop_nested(&self.savepoint);
        Ok(())
    }

    pub fn rollback(self) -> Result<()> {
        self.transaction.rollback_to_savepoint(&self.savepoint)?;
        self.transaction.release_savepoint(&self.savepoint)?;
        self.transaction.pop_nested(&self.savepoint);
        Ok(())
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
            {
                let mut locks = self.locks.write();
                let entries = locks.entry(page_id).or_insert_with(Vec::new);

                if self.can_grant_lock(entries, &lock_type) {
                    entries.push(LockEntry {
                        transaction_id,
                        lock_type,
                        granted: true,
                        requested_at: Instant::now(),
                    });
                    return Ok(());
                }

                if Instant::now() >= deadline {
                    return Err(StorageEngineError::TransactionConflict(
                        "Lock acquisition timeout".to_string(),
                    ));
                }

                entries.push(LockEntry {
                    transaction_id,
                    lock_type,
                    granted: false,
                    requested_at: Instant::now(),
                });
            }

            if self.deadlock_detector.has_deadlock(transaction_id).await? {
                return Err(StorageEngineError::TransactionConflict(
                    "Deadlock detected".to_string(),
                ));
            }

            tokio::time::sleep(Duration::from_millis(10)).await;
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
    }

    fn grant_waiting_locks(&self, locks: &mut HashMap<PageId, Vec<LockEntry>>) {
        for entries in locks.values_mut() {
            let mut i = 0;
            while i < entries.len() {
                let lock_type = entries[i].lock_type;
                if !entries[i].granted && self.can_grant_lock(entries, &lock_type) {
                    entries[i].granted = true;
                }
                i += 1;
            }
        }
    }
}

pub struct DeadlockDetector {
    wait_graph: RwLock<HashMap<TransactionId, Vec<TransactionId>>>,
}

impl DeadlockDetector {
    pub fn new() -> Self {
        Self {
            wait_graph: RwLock::new(HashMap::new()),
        }
    }

    pub async fn has_deadlock(&self, transaction_id: TransactionId) -> Result<bool> {
        let graph = self.wait_graph.read();
        Ok(self.has_cycle_dfs(&graph, transaction_id, &mut HashSet::new()))
    }

    fn has_cycle_dfs(
        &self,
        graph: &HashMap<TransactionId, Vec<TransactionId>>,
        node: TransactionId,
        visited: &mut HashSet<TransactionId>,
    ) -> bool {
        if visited.contains(&node) {
            return true;
        }

        visited.insert(node);

        if let Some(neighbors) = graph.get(&node) {
            for &neighbor in neighbors {
                if self.has_cycle_dfs(graph, neighbor, visited) {
                    return true;
                }
            }
        }

        visited.remove(&node);
        false
    }

    pub fn add_wait_edge(&self, waiter: TransactionId, holder: TransactionId) {
        let mut graph = self.wait_graph.write();
        graph.entry(waiter).or_insert_with(Vec::new).push(holder);
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
}

impl TransactionManager {
    pub fn new() -> Self {
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
        }
    }

    pub async fn begin(&self) -> Result<Transaction> {
        self.begin_with_isolation(IsolationLevel::ReadCommitted)
            .await
    }

    pub async fn begin_with_isolation(
        &self,
        isolation_level: IsolationLevel,
    ) -> Result<Transaction> {
        let xid = self.next_xid.fetch_add(1, Ordering::AcqRel);
        let snapshot = self.create_snapshot(xid, isolation_level);

        let transaction = Transaction::new(xid, isolation_level, snapshot);

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
                let mut active_xids: Vec<_> = active_transactions.keys().copied().collect();
                active_xids.sort_unstable();
                TransactionSnapshot {
                    xid,
                    active_xids: Arc::new(active_xids),
                    xmin: active_transactions.keys().copied().min().unwrap_or(xid),
                    xmax: xid,
                }
            }
            IsolationLevel::RepeatableRead | IsolationLevel::Serializable => {
                let mut active_xids: Vec<_> = active_transactions.keys().copied().collect();
                active_xids.sort_unstable();
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

        // Check for serialization conflicts
        if transaction.isolation_level == IsolationLevel::Serializable {
            if self.has_serialization_conflicts(&transaction)? {
                return self.rollback(transaction).await;
            }
        }

        transaction.state = TransactionState::Committed;

        self.committed_transactions.write().insert(transaction.id);
        self.active_transactions.write().remove(&transaction.id);
        self.lock_manager.release_locks(transaction.id);

        self.update_global_snapshot();

        Ok(())
    }

    pub async fn rollback(&self, mut transaction: Transaction) -> Result<()> {
        transaction.state = TransactionState::Aborted;

        self.aborted_transactions.write().insert(transaction.id);
        self.active_transactions.write().remove(&transaction.id);
        self.lock_manager.release_locks(transaction.id);

        self.update_global_snapshot();

        Ok(())
    }

    fn has_serialization_conflicts(&self, transaction: &Transaction) -> Result<bool> {
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
        let mut active_xids: Vec<_> = active_transactions.keys().copied().collect();
        active_xids.sort_unstable();

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
        self.lock_manager
            .acquire_lock(transaction.id, page_id, lock_type, Duration::from_secs(30))
            .await
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
