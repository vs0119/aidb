use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

use parking_lot::RwLock;
use tokio::task::JoinHandle;

use crate::{IsolationLevel, RowId, TransactionId, TransactionSnapshot, VectorRow};

#[derive(Debug, Clone)]
pub struct MVCCTupleVersion {
    pub data: VectorRow,
    pub begin_xid: TransactionId,
    pub end_xid: Option<TransactionId>,
}

impl MVCCTupleVersion {
    pub fn new(data: VectorRow, begin_xid: TransactionId) -> Self {
        Self {
            data,
            begin_xid,
            end_xid: None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct MVCCTuple {
    pub row_id: RowId,
    pub versions: VecDeque<MVCCTupleVersion>,
}

impl MVCCTuple {
    pub fn new(row_id: RowId) -> Self {
        Self {
            row_id,
            versions: VecDeque::new(),
        }
    }

    pub fn current_version_mut(&mut self) -> Option<&mut MVCCTupleVersion> {
        self.versions.front_mut()
    }

    pub fn push_version(&mut self, version: MVCCTupleVersion) {
        self.versions.push_front(version);
    }
}

#[derive(Default)]
pub struct VersionStore {
    tuples: RwLock<HashMap<RowId, MVCCTuple>>,
}

impl VersionStore {
    pub fn register_insert(&self, row_id: RowId, row: VectorRow) {
        let mut guard = self.tuples.write();
        let tuple = guard
            .entry(row_id)
            .or_insert_with(|| MVCCTuple::new(row_id));
        tuple.versions.clear();
        tuple.push_version(MVCCTupleVersion::new(row.clone(), row.created_xid));
    }

    pub fn register_update(&self, row_id: RowId, row: VectorRow, xid: TransactionId) {
        let mut guard = self.tuples.write();
        let tuple = guard
            .entry(row_id)
            .or_insert_with(|| MVCCTuple::new(row_id));

        if let Some(head) = tuple.current_version_mut() {
            if head.begin_xid == xid {
                head.data = row;
                head.end_xid = None;
                return;
            }

            if head.end_xid.is_none() {
                head.end_xid = Some(xid);
            }
        }

        tuple.push_version(MVCCTupleVersion::new(row, xid));
    }

    pub fn register_delete(&self, row_id: RowId, xid: TransactionId) {
        let mut guard = self.tuples.write();
        if let Some(tuple) = guard.get_mut(&row_id) {
            if let Some(head) = tuple.current_version_mut() {
                head.end_xid = Some(xid);
            }
        }
    }

    pub fn get_tuple(&self, row_id: &RowId) -> Option<MVCCTuple> {
        self.tuples.read().get(row_id).cloned()
    }

    pub fn cleanup_versions(&self, cutoff_xid: TransactionId) -> usize {
        let mut guard = self.tuples.write();
        let mut removed = 0;

        guard.retain(|_, tuple| {
            while tuple.versions.len() > 1 {
                let remove_oldest = match tuple.versions.back() {
                    Some(version) => match version.end_xid {
                        Some(end) => end < cutoff_xid,
                        None => false,
                    },
                    None => false,
                };

                if remove_oldest {
                    tuple.versions.pop_back();
                    removed += 1;
                } else {
                    break;
                }
            }

            !tuple.versions.is_empty()
        });

        removed
    }
}

pub struct VisibilityChecker;

impl VisibilityChecker {
    pub fn new() -> Self {
        Self
    }

    pub fn is_visible(&self, version: &MVCCTupleVersion, snapshot: &TransactionSnapshot) -> bool {
        if version.begin_xid == snapshot.xid {
            return version.end_xid != Some(snapshot.xid);
        }

        if !snapshot.can_see(version.begin_xid) {
            return false;
        }

        if let Some(end_xid) = version.end_xid {
            if end_xid == snapshot.xid {
                return false;
            }

            if snapshot.can_see(end_xid) {
                return false;
            }
        }

        true
    }

    pub fn select_visible_version(
        &self,
        tuple: &MVCCTuple,
        snapshot: &TransactionSnapshot,
    ) -> Option<VectorRow> {
        tuple
            .versions
            .iter()
            .find(|version| self.is_visible(version, snapshot))
            .map(|version| version.data.clone())
    }
}

#[derive(Debug)]
pub struct SnapshotManager {
    global_snapshot: RwLock<TransactionSnapshot>,
    active_snapshots: RwLock<HashMap<TransactionId, TransactionSnapshot>>,
}

impl SnapshotManager {
    pub fn new(initial_snapshot: TransactionSnapshot) -> Self {
        Self {
            global_snapshot: RwLock::new(initial_snapshot),
            active_snapshots: RwLock::new(HashMap::new()),
        }
    }

    pub fn create_snapshot<I>(
        &self,
        xid: TransactionId,
        isolation_level: IsolationLevel,
        active_xids: I,
        next_xid: TransactionId,
    ) -> TransactionSnapshot
    where
        I: IntoIterator<Item = TransactionId>,
    {
        let mut active: Vec<_> = active_xids
            .into_iter()
            .filter(|existing| existing != &xid)
            .collect();
        active.sort_unstable();

        match isolation_level {
            IsolationLevel::ReadUncommitted => TransactionSnapshot {
                xid,
                active_xids: Arc::new(Vec::new()),
                xmin: 0,
                xmax: u64::MAX,
            },
            IsolationLevel::ReadCommitted => TransactionSnapshot {
                xid,
                xmin: active.first().copied().unwrap_or(xid),
                xmax: next_xid,
                active_xids: Arc::new(active),
            },
            IsolationLevel::RepeatableRead | IsolationLevel::Serializable => {
                let global = self.global_snapshot.read();
                TransactionSnapshot {
                    xid,
                    xmin: global.xmin,
                    xmax: next_xid,
                    active_xids: Arc::new(active),
                }
            }
        }
    }

    pub fn register_snapshot(&self, snapshot: TransactionSnapshot) {
        self.active_snapshots.write().insert(snapshot.xid, snapshot);
    }

    pub fn release_snapshot(&self, xid: TransactionId) {
        self.active_snapshots.write().remove(&xid);
    }

    pub fn update_global_snapshot<I>(&self, active_xids: I, next_xid: TransactionId)
    where
        I: IntoIterator<Item = TransactionId>,
    {
        let mut active: Vec<_> = active_xids.into_iter().collect();
        active.sort_unstable();
        let xmin = active.first().copied().unwrap_or(next_xid);

        let snapshot = TransactionSnapshot {
            xid: next_xid,
            active_xids: Arc::new(active),
            xmin,
            xmax: next_xid,
        };

        *self.global_snapshot.write() = snapshot;
    }

    pub fn oldest_active_xid(&self) -> TransactionId {
        let snapshots = self.active_snapshots.read();
        snapshots
            .values()
            .map(|snapshot| snapshot.xmin)
            .min()
            .unwrap_or_else(|| self.global_snapshot.read().xmin)
    }

    pub fn global_snapshot(&self) -> TransactionSnapshot {
        self.global_snapshot.read().clone()
    }
}

pub struct MvccGarbageCollector {
    #[allow(dead_code)]
    handle: JoinHandle<()>,
    stop: Arc<AtomicBool>,
}

impl MvccGarbageCollector {
    pub fn start(
        store: Arc<VersionStore>,
        snapshot_manager: Arc<SnapshotManager>,
        interval: Duration,
    ) -> Self {
        let stop = Arc::new(AtomicBool::new(false));
        let stop_signal = stop.clone();

        let handle = tokio::spawn(async move {
            while !stop_signal.load(Ordering::Relaxed) {
                tokio::time::sleep(interval).await;
                let cutoff = snapshot_manager.oldest_active_xid();
                store.cleanup_versions(cutoff);
            }
        });

        Self { handle, stop }
    }

    pub fn stop(&self) {
        self.stop.store(true, Ordering::Relaxed);
    }
}
