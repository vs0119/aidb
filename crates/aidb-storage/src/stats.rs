use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::SystemTime;

use async_trait::async_trait;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use thiserror::Error;
use uuid::Uuid;

#[derive(Debug, Error)]
pub enum StatisticsError {
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("serialization error: {0}")]
    Serde(#[from] serde_json::Error),
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", content = "value")]
pub enum StatsValue {
    Int(i64),
    Float(f64),
    Text(String),
    Bool(bool),
}

impl StatsValue {
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            StatsValue::Int(i) => Some(*i as f64),
            StatsValue::Float(f) => Some(*f),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
pub struct TableStats {
    pub total_rows: Option<u64>,
    pub distinct_rows: Option<u64>,
    pub last_analyzed: Option<SystemTime>,
    pub sample_size: Option<u64>,
    #[serde(default = "default_stats_version")]
    pub stats_version: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
pub struct HistogramRef {
    pub histogram_id: Uuid,
    pub histogram_type: HistogramType,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
pub struct ColumnStats {
    pub null_count: Option<u64>,
    pub distinct_count: Option<u64>,
    pub min: Option<StatsValue>,
    pub max: Option<StatsValue>,
    pub average_width: Option<f64>,
    pub histogram: Option<HistogramRef>,
    pub last_analyzed: Option<SystemTime>,
    #[serde(default = "default_stats_version")]
    pub stats_version: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum HistogramType {
    EquiWidth,
    EquiDepth,
    TopK,
}

impl Default for HistogramType {
    fn default() -> Self {
        HistogramType::EquiWidth
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct HistogramBucket {
    pub lower: StatsValue,
    pub upper: StatsValue,
    pub count: u64,
    pub cumulative_count: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Histogram {
    pub histogram_id: Uuid,
    pub histogram_type: HistogramType,
    pub buckets: Vec<HistogramBucket>,
    pub last_refreshed: Option<SystemTime>,
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum HistogramBuildError {
    #[error("bucket count must be greater than zero")]
    InvalidBucketCount,
    #[error("no numeric values provided for histogram build")]
    NoNumericValues,
    #[error("encountered non-numeric statistic value in histogram input")]
    NonNumericValue,
    #[error("histogram has no buckets to update")]
    EmptyHistogram,
    #[error("histogram type mismatch: expected {expected:?}, got {actual:?}")]
    TypeMismatch {
        expected: HistogramType,
        actual: HistogramType,
    },
    #[error("histogram bucket boundary is non-numeric")]
    NonNumericBoundary,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HistogramMaintenanceResult {
    NoOp,
    Updated,
    NeedsRebuild,
}

pub trait HistogramBuilderStrategy: Send + Sync {
    fn histogram_type(&self) -> HistogramType;

    fn build(
        &self,
        values: &[StatsValue],
        bucket_count: usize,
    ) -> Result<Histogram, HistogramBuildError>;

    fn apply_updates(
        &self,
        histogram: &mut Histogram,
        inserts: &[StatsValue],
        deletes: &[StatsValue],
    ) -> Result<HistogramMaintenanceResult, HistogramBuildError>;
}

#[derive(Debug, Default, Clone, Copy)]
pub struct EquiWidthHistogramBuilder;

#[derive(Debug, Default, Clone, Copy)]
pub struct EquiDepthHistogramBuilder;

impl EquiWidthHistogramBuilder {
    fn numeric_values(values: &[StatsValue]) -> Result<Vec<f64>, HistogramBuildError> {
        let mut numeric = Vec::with_capacity(values.len());
        for value in values {
            if let Some(num) = value.as_f64() {
                numeric.push(num);
            } else {
                return Err(HistogramBuildError::NonNumericValue);
            }
        }
        if numeric.is_empty() {
            return Err(HistogramBuildError::NoNumericValues);
        }
        Ok(numeric)
    }

    fn boundary_to_f64(value: &StatsValue) -> Result<f64, HistogramBuildError> {
        value
            .as_f64()
            .ok_or(HistogramBuildError::NonNumericBoundary)
    }

    fn bucket_index(buckets: &[HistogramBucket], value: f64) -> Result<usize, HistogramBuildError> {
        for (idx, bucket) in buckets.iter().enumerate() {
            let upper = Self::boundary_to_f64(&bucket.upper)?;
            if idx == buckets.len() - 1 || value <= upper {
                return Ok(idx);
            }
        }
        // Safety: loop returns for last bucket
        Ok(buckets.len().saturating_sub(1))
    }

    fn recompute_cumulative(buckets: &mut [HistogramBucket]) {
        let mut cumulative = 0u64;
        for bucket in buckets.iter_mut() {
            cumulative = cumulative.saturating_add(bucket.count);
            bucket.cumulative_count = cumulative;
        }
    }
}

impl HistogramBuilderStrategy for EquiWidthHistogramBuilder {
    fn histogram_type(&self) -> HistogramType {
        HistogramType::EquiWidth
    }

    fn build(
        &self,
        values: &[StatsValue],
        bucket_count: usize,
    ) -> Result<Histogram, HistogramBuildError> {
        if bucket_count == 0 {
            return Err(HistogramBuildError::InvalidBucketCount);
        }
        let numeric = Self::numeric_values(values)?;
        let min = numeric.iter().fold(f64::INFINITY, |acc, v| acc.min(*v));
        let max = numeric.iter().fold(f64::NEG_INFINITY, |acc, v| acc.max(*v));

        let width = if (max - min).abs() < f64::EPSILON {
            0.0
        } else {
            (max - min) / bucket_count as f64
        };

        let mut buckets = Vec::with_capacity(bucket_count);
        for idx in 0..bucket_count {
            let lower = if idx == 0 {
                min
            } else {
                min + width * idx as f64
            };
            let upper = if idx == bucket_count - 1 {
                max
            } else {
                min + width * (idx as f64 + 1.0)
            };
            buckets.push(HistogramBucket {
                lower: StatsValue::Float(lower),
                upper: StatsValue::Float(upper),
                count: 0,
                cumulative_count: 0,
            });
        }

        for value in numeric {
            let bucket_idx = if width == 0.0 {
                0
            } else {
                (((value - min) / width).floor() as usize).min(bucket_count - 1)
            };
            buckets[bucket_idx].count += 1;
        }
        Self::recompute_cumulative(&mut buckets);

        Ok(Histogram {
            histogram_id: Uuid::new_v4(),
            histogram_type: HistogramType::EquiWidth,
            buckets,
            last_refreshed: Some(SystemTime::now()),
        })
    }

    fn apply_updates(
        &self,
        histogram: &mut Histogram,
        inserts: &[StatsValue],
        deletes: &[StatsValue],
    ) -> Result<HistogramMaintenanceResult, HistogramBuildError> {
        if histogram.buckets.is_empty() {
            return Err(HistogramBuildError::EmptyHistogram);
        }
        if histogram.histogram_type != HistogramType::EquiWidth {
            return Err(HistogramBuildError::TypeMismatch {
                expected: HistogramType::EquiWidth,
                actual: histogram.histogram_type.clone(),
            });
        }

        let inserts = if inserts.is_empty() {
            Vec::new()
        } else {
            Self::numeric_values(inserts)?
        };
        let deletes = if deletes.is_empty() {
            Vec::new()
        } else {
            Self::numeric_values(deletes)?
        };

        if inserts.is_empty() && deletes.is_empty() {
            return Ok(HistogramMaintenanceResult::NoOp);
        }

        let min_boundary = Self::boundary_to_f64(&histogram.buckets[0].lower)?;
        let max_boundary =
            Self::boundary_to_f64(&histogram.buckets[histogram.buckets.len() - 1].upper)?;

        let mut mutated = false;
        let mut needs_rebuild = false;

        for value in inserts {
            if value < min_boundary || value > max_boundary {
                needs_rebuild = true;
                continue;
            }
            let bucket_idx = Self::bucket_index(&histogram.buckets, value)?;
            histogram.buckets[bucket_idx].count += 1;
            mutated = true;
        }

        for value in deletes {
            if value < min_boundary || value > max_boundary {
                needs_rebuild = true;
                continue;
            }
            let bucket_idx = Self::bucket_index(&histogram.buckets, value)?;
            let prev = histogram.buckets[bucket_idx].count;
            histogram.buckets[bucket_idx].count =
                histogram.buckets[bucket_idx].count.saturating_sub(1);
            if histogram.buckets[bucket_idx].count != prev {
                mutated = true;
            }
        }

        if mutated {
            Self::recompute_cumulative(&mut histogram.buckets);
            histogram.last_refreshed = Some(SystemTime::now());
        }

        if needs_rebuild {
            Ok(HistogramMaintenanceResult::NeedsRebuild)
        } else if mutated {
            Ok(HistogramMaintenanceResult::Updated)
        } else {
            Ok(HistogramMaintenanceResult::NoOp)
        }
    }
}

impl EquiDepthHistogramBuilder {
    fn numeric_values_sorted(values: &[StatsValue]) -> Result<Vec<f64>, HistogramBuildError> {
        let mut numeric = Vec::with_capacity(values.len());
        for value in values {
            if let Some(num) = value.as_f64() {
                numeric.push(num);
            } else {
                return Err(HistogramBuildError::NonNumericValue);
            }
        }
        if numeric.is_empty() {
            return Err(HistogramBuildError::NoNumericValues);
        }
        numeric.sort_by(|a, b| a.partial_cmp(b).unwrap());
        Ok(numeric)
    }
}

impl HistogramBuilderStrategy for EquiDepthHistogramBuilder {
    fn histogram_type(&self) -> HistogramType {
        HistogramType::EquiDepth
    }

    fn build(
        &self,
        values: &[StatsValue],
        bucket_count: usize,
    ) -> Result<Histogram, HistogramBuildError> {
        if bucket_count == 0 {
            return Err(HistogramBuildError::InvalidBucketCount);
        }
        let numeric = Self::numeric_values_sorted(values)?;
        let total = numeric.len();
        let bucket_count = bucket_count.min(total).max(1);
        let mut buckets = Vec::with_capacity(bucket_count);
        let mut cumulative = 0u64;

        for bucket_idx in 0..bucket_count {
            let start = bucket_idx * total / bucket_count;
            let mut end = ((bucket_idx + 1) * total) / bucket_count;
            if end == start {
                end = (start + 1).min(total);
            }
            if bucket_idx == bucket_count - 1 {
                end = total;
            }
            let lower = numeric[start];
            let upper = numeric[end - 1];
            let count = (end - start) as u64;
            cumulative = cumulative.saturating_add(count);
            buckets.push(HistogramBucket {
                lower: StatsValue::Float(lower),
                upper: StatsValue::Float(upper),
                count,
                cumulative_count: cumulative,
            });
        }

        Ok(Histogram {
            histogram_id: Uuid::new_v4(),
            histogram_type: HistogramType::EquiDepth,
            buckets,
            last_refreshed: Some(SystemTime::now()),
        })
    }

    fn apply_updates(
        &self,
        histogram: &mut Histogram,
        inserts: &[StatsValue],
        deletes: &[StatsValue],
    ) -> Result<HistogramMaintenanceResult, HistogramBuildError> {
        if histogram.histogram_type != HistogramType::EquiDepth {
            return Err(HistogramBuildError::TypeMismatch {
                expected: HistogramType::EquiDepth,
                actual: histogram.histogram_type.clone(),
            });
        }
        if !inserts.is_empty() {
            Self::numeric_values_sorted(inserts)?;
        }
        if !deletes.is_empty() {
            Self::numeric_values_sorted(deletes)?;
        }
        if inserts.is_empty() && deletes.is_empty() {
            return Ok(HistogramMaintenanceResult::NoOp);
        }
        Ok(HistogramMaintenanceResult::NeedsRebuild)
    }
}

fn default_stats_version() -> u32 {
    1
}

#[async_trait]
pub trait StatisticsCatalog: Send + Sync {
    async fn get_table_stats(&self, table: &str) -> Result<Option<TableStats>, StatisticsError>;
    async fn get_column_stats(
        &self,
        table: &str,
        column: &str,
    ) -> Result<Option<ColumnStats>, StatisticsError>;
    async fn get_histogram(&self, histogram_id: Uuid)
        -> Result<Option<Histogram>, StatisticsError>;

    async fn upsert_table_stats(
        &self,
        table: &str,
        stats: TableStats,
    ) -> Result<(), StatisticsError>;
    async fn upsert_column_stats(
        &self,
        table: &str,
        column: &str,
        stats: ColumnStats,
    ) -> Result<(), StatisticsError>;
    async fn upsert_histogram(&self, histogram: Histogram) -> Result<(), StatisticsError>;

    async fn remove_stats_for_table(&self, table: &str) -> Result<(), StatisticsError>;
    async fn list_analyzed_tables(&self) -> Result<Vec<(String, SystemTime)>, StatisticsError>;
}

#[derive(Default, Serialize, Deserialize, Clone)]
struct StatsState {
    tables: HashMap<String, TableStats>,
    columns: HashMap<String, HashMap<String, ColumnStats>>,
    histograms: HashMap<Uuid, Histogram>,
}

impl StatsState {
    fn remove_table(&mut self, table: &str) {
        if let Some(columns) = self.columns.remove(table) {
            for stats in columns.values() {
                if let Some(hist_ref) = &stats.histogram {
                    self.histograms.remove(&hist_ref.histogram_id);
                }
            }
        }
        self.tables.remove(table);
    }
}

pub struct InMemoryStatisticsCatalog {
    state: RwLock<StatsState>,
}

impl InMemoryStatisticsCatalog {
    pub fn new() -> Self {
        Self {
            state: RwLock::new(StatsState::default()),
        }
    }
}

#[async_trait]
impl StatisticsCatalog for InMemoryStatisticsCatalog {
    async fn get_table_stats(&self, table: &str) -> Result<Option<TableStats>, StatisticsError> {
        Ok(self.state.read().tables.get(table).cloned())
    }

    async fn get_column_stats(
        &self,
        table: &str,
        column: &str,
    ) -> Result<Option<ColumnStats>, StatisticsError> {
        Ok(self
            .state
            .read()
            .columns
            .get(table)
            .and_then(|cols| cols.get(column))
            .cloned())
    }

    async fn get_histogram(
        &self,
        histogram_id: Uuid,
    ) -> Result<Option<Histogram>, StatisticsError> {
        Ok(self.state.read().histograms.get(&histogram_id).cloned())
    }

    async fn upsert_table_stats(
        &self,
        table: &str,
        stats: TableStats,
    ) -> Result<(), StatisticsError> {
        self.state.write().tables.insert(table.to_string(), stats);
        Ok(())
    }

    async fn upsert_column_stats(
        &self,
        table: &str,
        column: &str,
        stats: ColumnStats,
    ) -> Result<(), StatisticsError> {
        let mut guard = self.state.write();
        let table_key = table.to_string();
        let histogram_to_remove = guard
            .columns
            .get(&table_key)
            .and_then(|cols| cols.get(column))
            .and_then(|existing| existing.histogram.clone());
        if let Some(hist_ref) = histogram_to_remove {
            guard.histograms.remove(&hist_ref.histogram_id);
        }
        guard
            .columns
            .entry(table_key)
            .or_default()
            .insert(column.to_string(), stats);
        Ok(())
    }

    async fn upsert_histogram(&self, histogram: Histogram) -> Result<(), StatisticsError> {
        self.state
            .write()
            .histograms
            .insert(histogram.histogram_id, histogram);
        Ok(())
    }

    async fn remove_stats_for_table(&self, table: &str) -> Result<(), StatisticsError> {
        self.state.write().remove_table(table);
        Ok(())
    }

    async fn list_analyzed_tables(&self) -> Result<Vec<(String, SystemTime)>, StatisticsError> {
        let guard = self.state.read();
        let mut entries: Vec<_> = guard
            .tables
            .iter()
            .filter_map(|(table, stats)| stats.last_analyzed.map(|ts| (table.clone(), ts)))
            .collect();
        entries.sort_by_key(|(_, ts)| *ts);
        Ok(entries)
    }
}

pub struct FileStatisticsCatalog {
    path: PathBuf,
    state: RwLock<StatsState>,
}

impl FileStatisticsCatalog {
    pub fn new(path: impl AsRef<Path>) -> Result<Self, StatisticsError> {
        let path = path.as_ref().to_path_buf();
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        let state = if path.exists() {
            let data = fs::read(&path)?;
            if data.is_empty() {
                StatsState::default()
            } else {
                serde_json::from_slice(&data)?
            }
        } else {
            StatsState::default()
        };
        Ok(Self {
            path,
            state: RwLock::new(state),
        })
    }

    fn persist(&self, state: &StatsState) -> Result<(), StatisticsError> {
        let json = serde_json::to_vec_pretty(state)?;
        fs::write(&self.path, json)?;
        Ok(())
    }
}

#[async_trait]
impl StatisticsCatalog for FileStatisticsCatalog {
    async fn get_table_stats(&self, table: &str) -> Result<Option<TableStats>, StatisticsError> {
        Ok(self.state.read().tables.get(table).cloned())
    }

    async fn get_column_stats(
        &self,
        table: &str,
        column: &str,
    ) -> Result<Option<ColumnStats>, StatisticsError> {
        Ok(self
            .state
            .read()
            .columns
            .get(table)
            .and_then(|cols| cols.get(column))
            .cloned())
    }

    async fn get_histogram(
        &self,
        histogram_id: Uuid,
    ) -> Result<Option<Histogram>, StatisticsError> {
        Ok(self.state.read().histograms.get(&histogram_id).cloned())
    }

    async fn upsert_table_stats(
        &self,
        table: &str,
        stats: TableStats,
    ) -> Result<(), StatisticsError> {
        let mut guard = self.state.write();
        guard.tables.insert(table.to_string(), stats);
        self.persist(&guard)
    }

    async fn upsert_column_stats(
        &self,
        table: &str,
        column: &str,
        stats: ColumnStats,
    ) -> Result<(), StatisticsError> {
        let mut guard = self.state.write();
        let table_key = table.to_string();
        let histogram_to_remove = guard
            .columns
            .get(&table_key)
            .and_then(|cols| cols.get(column))
            .and_then(|existing| existing.histogram.clone());
        if let Some(hist_ref) = histogram_to_remove {
            guard.histograms.remove(&hist_ref.histogram_id);
        }
        guard
            .columns
            .entry(table_key)
            .or_default()
            .insert(column.to_string(), stats);
        self.persist(&guard)
    }

    async fn upsert_histogram(&self, histogram: Histogram) -> Result<(), StatisticsError> {
        let mut guard = self.state.write();
        guard.histograms.insert(histogram.histogram_id, histogram);
        self.persist(&guard)
    }

    async fn remove_stats_for_table(&self, table: &str) -> Result<(), StatisticsError> {
        let mut guard = self.state.write();
        guard.remove_table(table);
        self.persist(&guard)
    }

    async fn list_analyzed_tables(&self) -> Result<Vec<(String, SystemTime)>, StatisticsError> {
        let guard = self.state.read();
        let mut entries: Vec<_> = guard
            .tables
            .iter()
            .filter_map(|(table, stats)| stats.last_analyzed.map(|ts| (table.clone(), ts)))
            .collect();
        entries.sort_by_key(|(_, ts)| *ts);
        Ok(entries)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{Duration, SystemTime};

    #[tokio::test]
    async fn in_memory_round_trip() {
        let catalog = InMemoryStatisticsCatalog::new();
        let table = "docs";
        let column = "title";
        let now = SystemTime::now();

        let table_stats = TableStats {
            total_rows: Some(1000),
            distinct_rows: Some(900),
            last_analyzed: Some(now),
            sample_size: Some(500),
            stats_version: 1,
        };
        catalog
            .upsert_table_stats(table, table_stats.clone())
            .await
            .unwrap();

        let column_stats = ColumnStats {
            null_count: Some(10),
            distinct_count: Some(850),
            min: Some(StatsValue::Text("a".into())),
            max: Some(StatsValue::Text("z".into())),
            average_width: Some(12.5),
            histogram: None,
            last_analyzed: Some(now),
            stats_version: 1,
        };
        catalog
            .upsert_column_stats(table, column, column_stats.clone())
            .await
            .unwrap();

        let read_table = catalog.get_table_stats(table).await.unwrap();
        assert_eq!(read_table, Some(table_stats));
        let read_column = catalog.get_column_stats(table, column).await.unwrap();
        assert_eq!(read_column, Some(column_stats));

        let listed = catalog.list_analyzed_tables().await.unwrap();
        assert_eq!(listed.len(), 1);
        assert_eq!(listed[0].0, table);
    }

    #[tokio::test]
    async fn histogram_round_trip() {
        let catalog = InMemoryStatisticsCatalog::new();
        let histogram = Histogram {
            histogram_id: Uuid::new_v4(),
            histogram_type: HistogramType::EquiDepth,
            buckets: vec![HistogramBucket {
                lower: StatsValue::Int(0),
                upper: StatsValue::Int(10),
                count: 100,
                cumulative_count: 100,
            }],
            last_refreshed: Some(SystemTime::now()),
        };
        catalog.upsert_histogram(histogram.clone()).await.unwrap();
        let fetched = catalog.get_histogram(histogram.histogram_id).await.unwrap();
        assert_eq!(fetched, Some(histogram));
    }

    #[tokio::test]
    async fn remove_table_stats_cleans_histograms() {
        let catalog = InMemoryStatisticsCatalog::new();
        let histogram_id = Uuid::new_v4();
        let histogram = Histogram {
            histogram_id,
            histogram_type: HistogramType::TopK,
            buckets: vec![],
            last_refreshed: None,
        };
        catalog.upsert_histogram(histogram).await.unwrap();

        let column_stats = ColumnStats {
            histogram: Some(HistogramRef {
                histogram_id,
                histogram_type: HistogramType::TopK,
            }),
            last_analyzed: Some(SystemTime::now()),
            ..Default::default()
        };
        catalog
            .upsert_column_stats("tbl", "col", column_stats)
            .await
            .unwrap();

        catalog.remove_stats_for_table("tbl").await.unwrap();
        assert!(catalog.get_histogram(histogram_id).await.unwrap().is_none());
    }

    #[test]
    fn equi_width_builder_produces_expected_distribution() {
        let builder = EquiWidthHistogramBuilder::default();
        let values: Vec<_> = (0..20).map(|i| StatsValue::Int(i)).collect();
        let histogram = builder.build(&values, 5).expect("build succeeds");
        assert_eq!(histogram.histogram_type, HistogramType::EquiWidth);
        assert_eq!(histogram.buckets.len(), 5);
        let total: u64 = histogram.buckets.iter().map(|b| b.count).sum();
        assert_eq!(total, 20);
        // Each bucket should be non-decreasing in cumulative count.
        let mut prev = 0u64;
        for bucket in histogram.buckets.iter() {
            assert!(bucket.cumulative_count >= prev);
            prev = bucket.cumulative_count;
        }
        assert!(histogram.last_refreshed.is_some());
    }

    #[test]
    fn equi_width_incremental_updates_adjust_counts() {
        let builder = EquiWidthHistogramBuilder::default();
        let values: Vec<_> = (0..10).map(|i| StatsValue::Int(i)).collect();
        let mut histogram = builder.build(&values, 4).expect("build succeeds");

        let result = builder
            .apply_updates(&mut histogram, &[StatsValue::Int(2)], &[])
            .expect("update succeeds");
        assert_eq!(result, HistogramMaintenanceResult::Updated);
        let total: u64 = histogram.buckets.iter().map(|b| b.count).sum();
        assert_eq!(total, 11);
        let result = builder
            .apply_updates(&mut histogram, &[StatsValue::Int(-5)], &[])
            .expect("update succeeds");
        assert_eq!(result, HistogramMaintenanceResult::NeedsRebuild);
    }

    #[test]
    fn equi_depth_builder_balances_counts() {
        let builder = EquiDepthHistogramBuilder::default();
        let values: Vec<_> = (1..=12).map(|i| StatsValue::Int(i)).collect();
        let histogram = builder.build(&values, 4).expect("build succeeds");
        assert_eq!(histogram.histogram_type, HistogramType::EquiDepth);
        assert_eq!(histogram.buckets.len(), 4);
        let total: u64 = histogram.buckets.iter().map(|b| b.count).sum();
        assert_eq!(total, 12);
        for window in histogram.buckets.windows(2) {
            let left = window[0].upper.as_f64().unwrap();
            let right = window[1].lower.as_f64().unwrap();
            assert!(left <= right);
        }
    }

    #[test]
    fn equi_depth_incremental_requests_rebuild() {
        let builder = EquiDepthHistogramBuilder::default();
        let values: Vec<_> = (0..8).map(|i| StatsValue::Int(i)).collect();
        let mut histogram = builder.build(&values, 3).expect("build succeeds");
        let result = builder
            .apply_updates(&mut histogram, &[StatsValue::Int(100)], &[])
            .expect("update evaluated");
        assert_eq!(result, HistogramMaintenanceResult::NeedsRebuild);
    }

    #[tokio::test]
    async fn file_catalog_persists_state() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("stats.json");
        {
            let catalog = FileStatisticsCatalog::new(&path).unwrap();
            let table_stats = TableStats {
                total_rows: Some(42),
                last_analyzed: Some(SystemTime::now() - Duration::from_secs(60)),
                ..Default::default()
            };
            catalog
                .upsert_table_stats("numbers", table_stats.clone())
                .await
                .unwrap();
            assert_eq!(
                catalog.get_table_stats("numbers").await.unwrap(),
                Some(table_stats)
            );
        }

        let catalog = FileStatisticsCatalog::new(&path).unwrap();
        let stats = catalog.get_table_stats("numbers").await.unwrap();
        assert!(stats.is_some());
    }
}
