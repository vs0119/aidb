use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, SystemTime};

use async_trait::async_trait;
use parking_lot::Mutex;
use thiserror::Error;
use tokio::task::JoinHandle;
use tokio::time::sleep;
use uuid::Uuid;

use crate::stats::{
    HistogramBuildError, HistogramBuilderStrategy, HistogramRef, StatisticsCatalog,
    StatisticsError, StatsValue,
};

#[derive(Debug, Error)]
pub enum StatsRefreshError {
    #[error("statistics catalog error: {0}")]
    Catalog(#[from] StatisticsError),
    #[error("failed to build histogram: {0}")]
    Histogram(#[from] HistogramBuildError),
    #[error("sample collection error: {0}")]
    Sample(String),
}

pub type StatsRefreshResult<T> = Result<T, StatsRefreshError>;

#[async_trait]
pub trait HistogramSampleSource: Send + Sync {
    async fn sample(&self, table: &str, column: &str) -> StatsRefreshResult<Vec<StatsValue>>;
}

#[async_trait]
pub trait StatsRefreshJob: Send + Sync {
    async fn run(&self) -> StatsRefreshResult<()>;
}

#[derive(Clone)]
pub struct StatsRefreshScheduler {
    inner: Arc<SchedulerInner>,
}

struct SchedulerInner {
    tasks: Mutex<HashMap<Uuid, ScheduledTask>>,
}

struct ScheduledTask {
    cancel_flag: Arc<AtomicBool>,
    handle: JoinHandle<()>,
}

#[derive(Clone)]
pub struct ScheduledTaskHandle {
    scheduler: StatsRefreshScheduler,
    id: Uuid,
}

impl StatsRefreshScheduler {
    pub fn new() -> Self {
        Self {
            inner: Arc::new(SchedulerInner {
                tasks: Mutex::new(HashMap::new()),
            }),
        }
    }

    pub fn schedule_job(
        &self,
        job: Arc<dyn StatsRefreshJob>,
        interval: Duration,
        initial_delay: Option<Duration>,
    ) -> ScheduledTaskHandle {
        let cancel_flag = Arc::new(AtomicBool::new(false));
        let cancel_clone = cancel_flag.clone();
        let job_clone = job.clone();
        let handle = tokio::spawn(async move {
            if let Some(delay) = initial_delay {
                sleep(delay).await;
            }
            loop {
                if cancel_clone.load(Ordering::Relaxed) {
                    break;
                }
                if let Err(err) = job_clone.run().await {
                    log::warn!(
                        target: "aidb::stats",
                        "stats_refresh_job_failed error={:?}",
                        err
                    );
                }
                if cancel_clone.load(Ordering::Relaxed) {
                    break;
                }
                sleep(interval).await;
            }
        });

        let id = Uuid::new_v4();
        self.inner.tasks.lock().insert(
            id,
            ScheduledTask {
                cancel_flag,
                handle,
            },
        );

        ScheduledTaskHandle {
            scheduler: self.clone(),
            id,
        }
    }

    pub fn schedule_histogram_refresh<C, B, S>(
        &self,
        job: HistogramRefreshJob<C, B, S>,
        interval: Duration,
        initial_delay: Option<Duration>,
    ) -> ScheduledTaskHandle
    where
        C: StatisticsCatalog + 'static,
        B: HistogramBuilderStrategy + Clone + 'static,
        S: HistogramSampleSource + 'static,
    {
        self.schedule_job(Arc::new(job), interval, initial_delay)
    }

    pub async fn shutdown(&self) {
        let tasks: Vec<ScheduledTask> = self
            .inner
            .tasks
            .lock()
            .drain()
            .map(|(_, task)| task)
            .collect();
        for task in tasks {
            task.cancel_flag.store(true, Ordering::Relaxed);
            let _ = task.handle.await;
        }
    }
}

impl ScheduledTaskHandle {
    pub async fn cancel(self) {
        if let Some(task) = self.scheduler.inner.tasks.lock().remove(&self.id) {
            task.cancel_flag.store(true, Ordering::Relaxed);
            let _ = task.handle.await;
        }
    }

    pub fn cancel_and_forget(self) {
        if let Some(task) = self.scheduler.inner.tasks.lock().remove(&self.id) {
            task.cancel_flag.store(true, Ordering::Relaxed);
            task.handle.abort();
        }
    }
}

impl Drop for SchedulerInner {
    fn drop(&mut self) {
        let tasks: Vec<ScheduledTask> = self.tasks.lock().drain().map(|(_, task)| task).collect();
        for task in tasks {
            task.cancel_flag.store(true, Ordering::Relaxed);
            task.handle.abort();
        }
    }
}

#[derive(Debug, Clone)]
pub struct HistogramRefreshConfig {
    pub table: String,
    pub column: String,
    pub bucket_count: usize,
}

pub struct HistogramRefreshJob<C, B, S>
where
    C: StatisticsCatalog,
    B: HistogramBuilderStrategy,
    S: HistogramSampleSource,
{
    catalog: Arc<C>,
    builder: B,
    source: Arc<S>,
    config: HistogramRefreshConfig,
}

impl<C, B, S> HistogramRefreshJob<C, B, S>
where
    C: StatisticsCatalog,
    B: HistogramBuilderStrategy,
    S: HistogramSampleSource,
{
    pub fn new(
        catalog: Arc<C>,
        builder: B,
        source: Arc<S>,
        config: HistogramRefreshConfig,
    ) -> Self {
        Self {
            catalog,
            builder,
            source,
            config,
        }
    }
}

#[async_trait]
impl<C, B, S> StatsRefreshJob for HistogramRefreshJob<C, B, S>
where
    C: StatisticsCatalog + 'static,
    B: HistogramBuilderStrategy + Clone + Send + Sync + 'static,
    S: HistogramSampleSource + 'static,
{
    async fn run(&self) -> StatsRefreshResult<()> {
        if self.config.bucket_count == 0 {
            return Err(StatsRefreshError::Histogram(
                HistogramBuildError::InvalidBucketCount,
            ));
        }

        let values = self
            .source
            .sample(&self.config.table, &self.config.column)
            .await?;

        let mut histogram = self.builder.build(&values, self.config.bucket_count)?;

        let existing_stats = self
            .catalog
            .get_column_stats(&self.config.table, &self.config.column)
            .await?;
        let mut column_stats = existing_stats.unwrap_or_default();

        let reuse_existing_id = column_stats
            .histogram
            .as_ref()
            .filter(|hist_ref| hist_ref.histogram_type == histogram.histogram_type)
            .map(|hist_ref| hist_ref.histogram_id);

        let histogram_id = reuse_existing_id.unwrap_or_else(Uuid::new_v4);
        histogram.histogram_id = histogram_id;
        histogram.histogram_type = self.builder.histogram_type();

        let histogram_ref = HistogramRef {
            histogram_id,
            histogram_type: histogram.histogram_type.clone(),
        };

        column_stats.histogram = Some(histogram_ref);
        column_stats.last_analyzed = Some(SystemTime::now());

        self.catalog
            .upsert_column_stats(&self.config.table, &self.config.column, column_stats)
            .await?;
        self.catalog.upsert_histogram(histogram).await?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::stats::{
        ColumnStats, EquiWidthHistogramBuilder, Histogram, HistogramBucket, HistogramRef,
        HistogramType, InMemoryStatisticsCatalog,
    };
    use std::time::SystemTime;
    use uuid::Uuid;

    struct StaticSampleSource {
        values: Vec<StatsValue>,
    }

    #[async_trait]
    impl HistogramSampleSource for StaticSampleSource {
        async fn sample(&self, _table: &str, _column: &str) -> StatsRefreshResult<Vec<StatsValue>> {
            Ok(self.values.clone())
        }
    }

    #[tokio::test]
    async fn scheduler_runs_histogram_refresh_job() {
        let catalog = Arc::new(InMemoryStatisticsCatalog::new());
        let source = Arc::new(StaticSampleSource {
            values: (0..10).map(|v| StatsValue::Int(v)).collect(),
        });
        let builder = EquiWidthHistogramBuilder::default();
        let job = HistogramRefreshJob::new(
            Arc::clone(&catalog),
            builder,
            Arc::clone(&source),
            HistogramRefreshConfig {
                table: "docs".to_string(),
                column: "rank".to_string(),
                bucket_count: 5,
            },
        );

        let scheduler = StatsRefreshScheduler::new();
        let handle = scheduler.schedule_histogram_refresh(job, Duration::from_millis(50), None);

        sleep(Duration::from_millis(120)).await;
        handle.cancel().await;

        let column_stats = catalog
            .get_column_stats("docs", "rank")
            .await
            .unwrap()
            .expect("column stats present");
        let histogram_ref = column_stats.histogram.expect("histogram reference set");
        assert_eq!(histogram_ref.histogram_type, HistogramType::EquiWidth);

        let histogram = catalog
            .get_histogram(histogram_ref.histogram_id)
            .await
            .unwrap()
            .expect("histogram stored");
        assert_eq!(histogram.histogram_type, HistogramType::EquiWidth);
        assert_eq!(histogram.buckets.len(), 5);

        scheduler.shutdown().await;
    }

    #[tokio::test]
    async fn refresh_job_reuses_existing_histogram_id() {
        let catalog = Arc::new(InMemoryStatisticsCatalog::new());
        let histogram_id = Uuid::new_v4();
        let histogram = Histogram {
            histogram_id,
            histogram_type: HistogramType::EquiWidth,
            buckets: vec![HistogramBucket {
                lower: StatsValue::Int(0),
                upper: StatsValue::Int(10),
                count: 1,
                cumulative_count: 1,
            }],
            last_refreshed: Some(SystemTime::now()),
        };
        catalog
            .upsert_histogram(histogram)
            .await
            .expect("histogram inserted");

        let column_stats = ColumnStats {
            histogram: Some(HistogramRef {
                histogram_id,
                histogram_type: HistogramType::EquiWidth,
            }),
            ..Default::default()
        };
        catalog
            .upsert_column_stats("docs", "rank", column_stats)
            .await
            .expect("column stats set");

        let source = Arc::new(StaticSampleSource {
            values: (10..20).map(|v| StatsValue::Int(v)).collect(),
        });
        let builder = EquiWidthHistogramBuilder::default();
        let job = HistogramRefreshJob::new(
            Arc::clone(&catalog),
            builder,
            Arc::clone(&source),
            HistogramRefreshConfig {
                table: "docs".to_string(),
                column: "rank".to_string(),
                bucket_count: 4,
            },
        );

        job.run().await.expect("run succeeds");

        let refreshed = catalog
            .get_column_stats("docs", "rank")
            .await
            .unwrap()
            .expect("stats exist");
        let refreshed_hist_ref = refreshed.histogram.expect("hist ref available");
        assert_eq!(refreshed_hist_ref.histogram_id, histogram_id);
    }
}
