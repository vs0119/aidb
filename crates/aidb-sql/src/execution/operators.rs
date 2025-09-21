use super::batch::ColumnarBatch;
use crate::planner::ScanCandidates;
use crate::{compare_values, equal, full_text_matches, ColumnType, Table, Value};

pub(crate) const DEFAULT_BATCH_SIZE: usize = 1024;

#[derive(Clone)]
pub(crate) enum ScanPartition {
    Range { start: usize, end: usize },
    Indices(Vec<usize>),
}

impl ScanPartition {
    pub(crate) fn len(&self) -> usize {
        match self {
            ScanPartition::Range { start, end } => end.saturating_sub(*start),
            ScanPartition::Indices(indices) => indices.len(),
        }
    }
}

pub(crate) fn partition_scan_candidates(
    candidates: &ScanCandidates,
    total_rows: usize,
    batch_size: usize,
) -> Vec<ScanPartition> {
    let chunk = batch_size.max(1);
    match candidates {
        ScanCandidates::AllRows => {
            if total_rows == 0 {
                return Vec::new();
            }
            let mut partitions = Vec::new();
            let mut start = 0usize;
            while start < total_rows {
                let end = (start + chunk).min(total_rows);
                partitions.push(ScanPartition::Range { start, end });
                start = end;
            }
            partitions
        }
        ScanCandidates::Fixed(rows) => {
            if rows.is_empty() {
                return Vec::new();
            }
            rows.chunks(chunk)
                .map(|chunk_indices| ScanPartition::Indices(chunk_indices.to_vec()))
                .collect()
        }
    }
}

pub(crate) fn build_batch_for_partition<'a>(
    table: &'a Table,
    column_types: &[ColumnType],
    partition: &ScanPartition,
) -> ColumnarBatch<'a> {
    let mut batch = ColumnarBatch::with_capacity(column_types, partition.len());
    match partition {
        ScanPartition::Range { start, end } => {
            for row_index in *start..*end {
                let row = &table.rows[row_index];
                batch.push_row(row_index, row);
            }
        }
        ScanPartition::Indices(indices) => {
            for &row_index in indices {
                let row = &table.rows[row_index];
                batch.push_row(row_index, row);
            }
        }
    }
    batch
}

#[derive(Debug, Clone)]
pub(crate) enum BatchPredicate {
    Equals {
        column_index: usize,
        value: Value,
    },
    Between {
        column_index: usize,
        low: Value,
        high: Value,
    },
    IsNull {
        column_index: usize,
    },
    InTableColumn {
        column_index: usize,
        lookup_values: Vec<Value>,
    },
    FullText {
        column_index: usize,
        query: String,
        language: Option<String>,
    },
}

pub(crate) fn apply_predicate(batch: &mut ColumnarBatch<'_>, predicate: &BatchPredicate) {
    if batch.is_empty() {
        return;
    }
    let mask: Vec<bool> = match predicate {
        BatchPredicate::Equals {
            column_index,
            value,
        } => {
            let column = batch.column(*column_index);
            column
                .values()
                .iter()
                .map(|candidate| equal(candidate, value))
                .collect()
        }
        BatchPredicate::Between {
            column_index,
            low,
            high,
        } => {
            let column = batch.column(*column_index);
            column
                .values()
                .iter()
                .map(|candidate| {
                    if matches!(candidate, Value::Null) {
                        return false;
                    }
                    let cmp_low = compare_values(candidate, low);
                    let cmp_high = compare_values(candidate, high);
                    matches!(
                        cmp_low,
                        Some(std::cmp::Ordering::Greater) | Some(std::cmp::Ordering::Equal)
                    ) && matches!(
                        cmp_high,
                        Some(std::cmp::Ordering::Less) | Some(std::cmp::Ordering::Equal)
                    )
                })
                .collect()
        }
        BatchPredicate::IsNull { column_index } => {
            let column = batch.column(*column_index);
            column
                .values()
                .iter()
                .map(|candidate| matches!(candidate, Value::Null))
                .collect()
        }
        BatchPredicate::InTableColumn {
            column_index,
            lookup_values,
        } => {
            let column = batch.column(*column_index);
            column
                .values()
                .iter()
                .map(|candidate| {
                    if matches!(candidate, Value::Null) {
                        return false;
                    }
                    lookup_values.iter().any(|value| equal(candidate, value))
                })
                .collect()
        }
        BatchPredicate::FullText {
            column_index,
            query,
            language,
        } => {
            let column = batch.column(*column_index);
            column
                .values()
                .iter()
                .map(|candidate| full_text_matches(candidate, query, language.as_deref()))
                .collect()
        }
    };
    if mask.iter().all(|keep| *keep) {
        return;
    }
    if mask.iter().all(|keep| !*keep) {
        batch.retain_by_mask(&mask);
        return;
    }
    batch.retain_by_mask(&mask);
}

pub(crate) fn append_rows(batch: &ColumnarBatch<'_>, output: &mut Vec<Vec<Value>>) {
    if batch.is_empty() {
        return;
    }
    let column_count = batch.column_count();
    for row_idx in 0..batch.len() {
        let mut row = Vec::with_capacity(column_count);
        for column in batch.columns() {
            row.push(column.value(row_idx).clone());
        }
        output.push(row);
    }
}

pub(crate) fn collect_rows(batch: &ColumnarBatch<'_>) -> Vec<Vec<Value>> {
    let mut rows = Vec::new();
    append_rows(batch, &mut rows);
    rows
}
