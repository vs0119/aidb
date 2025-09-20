use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use async_trait::async_trait;
use parking_lot::RwLock;
use tokio::time::{timeout, Duration};

use crate::{Result, StorageEngineError, TransactionId};

#[async_trait]
pub trait TransactionParticipant: Send + Sync {
    async fn prepare(&self, xid: TransactionId) -> Result<()>;
    async fn commit(&self, xid: TransactionId) -> Result<()>;
    async fn abort(&self, xid: TransactionId) -> Result<()>;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DistributedTransactionState {
    Active,
    Preparing,
    Prepared,
    Committed,
    Aborted,
    Failed,
}

pub struct TwoPhaseCommitCoordinator {
    participants: RwLock<Vec<Arc<dyn TransactionParticipant>>>,
    states: RwLock<HashMap<TransactionId, DistributedTransactionState>>,
    prepare_timeout: Duration,
    next_xid: AtomicU64,
}

impl TwoPhaseCommitCoordinator {
    pub fn new(prepare_timeout: Duration) -> Self {
        Self {
            participants: RwLock::new(Vec::new()),
            states: RwLock::new(HashMap::new()),
            prepare_timeout,
            next_xid: AtomicU64::new(1),
        }
    }

    pub fn with_participants(
        prepare_timeout: Duration,
        participants: Vec<Arc<dyn TransactionParticipant>>,
    ) -> Self {
        Self {
            participants: RwLock::new(participants),
            states: RwLock::new(HashMap::new()),
            prepare_timeout,
            next_xid: AtomicU64::new(1),
        }
    }

    pub fn register_participant(&self, participant: Arc<dyn TransactionParticipant>) {
        self.participants.write().push(participant);
    }

    pub fn begin_transaction(&self) -> TransactionId {
        let xid = self.next_xid.fetch_add(1, Ordering::AcqRel);
        self.states
            .write()
            .insert(xid, DistributedTransactionState::Active);
        xid
    }

    pub fn get_state(&self, xid: TransactionId) -> Option<DistributedTransactionState> {
        self.states.read().get(&xid).copied()
    }

    pub async fn commit_transaction(&self, xid: TransactionId) -> Result<()> {
        self.transition_state(xid, DistributedTransactionState::Preparing);
        let participants = self.participants_snapshot();

        if let Err(err) = self.run_prepare_phase(xid, &participants).await {
            self.transition_state(xid, DistributedTransactionState::Failed);
            self.abort_internal(xid, &participants).await?;
            return Err(err);
        }

        self.transition_state(xid, DistributedTransactionState::Prepared);

        for participant in &participants {
            participant.commit(xid).await?;
        }

        self.transition_state(xid, DistributedTransactionState::Committed);
        self.states.write().remove(&xid);
        Ok(())
    }

    pub async fn abort_transaction(&self, xid: TransactionId) -> Result<()> {
        let participants = self.participants_snapshot();
        self.transition_state(xid, DistributedTransactionState::Aborted);
        self.abort_internal(xid, &participants).await?;
        self.states.write().remove(&xid);
        Ok(())
    }

    async fn run_prepare_phase(
        &self,
        xid: TransactionId,
        participants: &[Arc<dyn TransactionParticipant>],
    ) -> Result<()> {
        for participant in participants {
            timeout(self.prepare_timeout, participant.prepare(xid))
                .await
                .map_err(|_| {
                    StorageEngineError::DistributedTransaction(
                        "participant prepare timeout".to_string(),
                    )
                })??;
        }

        Ok(())
    }

    async fn abort_internal(
        &self,
        xid: TransactionId,
        participants: &[Arc<dyn TransactionParticipant>],
    ) -> Result<()> {
        for participant in participants {
            participant.abort(xid).await?;
        }
        Ok(())
    }

    fn participants_snapshot(&self) -> Vec<Arc<dyn TransactionParticipant>> {
        self.participants.read().iter().cloned().collect::<Vec<_>>()
    }

    fn transition_state(&self, xid: TransactionId, state: DistributedTransactionState) {
        self.states.write().insert(xid, state);
    }
}
