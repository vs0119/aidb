use std::sync::Arc;

use async_trait::async_trait;
use thiserror::Error;
use tokio::sync::mpsc;
use tokio::task::JoinHandle;

use super::memtable::MemTable;
use super::tree::LsmResult;

#[derive(Clone)]
pub enum CompactionTask {
    Flush(Arc<MemTable>),
    CompactLevel(usize),
}

#[async_trait]
pub trait CompactionExecutor: Send + Sync + 'static {
    async fn flush_memtable(&self, memtable: Arc<MemTable>) -> LsmResult<Vec<CompactionTask>>;
    async fn compact_level(&self, level: usize) -> LsmResult<Vec<CompactionTask>>;
}

#[derive(Error, Debug)]
pub enum SchedulerError {
    #[error("compaction scheduler stopped")]
    Stopped,
}

impl<T> From<mpsc::error::SendError<T>> for SchedulerError {
    fn from(_: mpsc::error::SendError<T>) -> Self {
        SchedulerError::Stopped
    }
}

pub struct CompactionScheduler {
    sender: mpsc::Sender<CompactionTask>,
    _worker: JoinHandle<()>,
}

impl CompactionScheduler {
    pub fn new(executor: Arc<dyn CompactionExecutor>) -> Self {
        let (tx, rx) = mpsc::channel(32);
        let worker_sender = tx.clone();
        let worker = tokio::spawn(async move {
            run_worker(executor, rx, worker_sender).await;
        });
        Self {
            sender: tx,
            _worker: worker,
        }
    }

    pub async fn schedule_flush(&self, memtable: Arc<MemTable>) -> Result<(), SchedulerError> {
        self.sender
            .send(CompactionTask::Flush(memtable))
            .await
            .map_err(Into::into)
    }

    pub async fn schedule_compaction(&self, level: usize) -> Result<(), SchedulerError> {
        self.sender
            .send(CompactionTask::CompactLevel(level))
            .await
            .map_err(Into::into)
    }
}

async fn run_worker(
    executor: Arc<dyn CompactionExecutor>,
    mut receiver: mpsc::Receiver<CompactionTask>,
    sender: mpsc::Sender<CompactionTask>,
) {
    while let Some(task) = receiver.recv().await {
        match task {
            CompactionTask::Flush(memtable) => match executor.flush_memtable(memtable).await {
                Ok(follow_up) => {
                    for task in follow_up {
                        if sender.send(task).await.is_err() {
                            log::warn!("compaction scheduler receiver dropped");
                            return;
                        }
                    }
                }
                Err(err) => {
                    log::error!("flush task failed: {err:?}");
                }
            },
            CompactionTask::CompactLevel(level) => match executor.compact_level(level).await {
                Ok(follow_up) => {
                    for task in follow_up {
                        if sender.send(task).await.is_err() {
                            log::warn!("compaction scheduler receiver dropped");
                            return;
                        }
                    }
                }
                Err(err) => {
                    log::error!("compaction task failed: {err:?}");
                }
            },
        }
    }
}
