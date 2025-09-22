use super::batch::{ColumnVector, ColumnarBatch};
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
    GreaterOrEqual {
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
    let mask = match predicate {
        BatchPredicate::Equals {
            column_index,
            value,
        } => equals_mask(batch.column(*column_index), value),
        BatchPredicate::GreaterOrEqual {
            column_index,
            value,
        } => greater_equal_mask(batch.column(*column_index), value),
        BatchPredicate::Between {
            column_index,
            low,
            high,
        } => between_mask(batch.column(*column_index), low, high),
        BatchPredicate::IsNull { column_index } => null_mask(batch.column(*column_index)),
        BatchPredicate::InTableColumn {
            column_index,
            lookup_values,
        } => in_list_mask(batch.column(*column_index), lookup_values),
        BatchPredicate::FullText {
            column_index,
            query,
            language,
        } => full_text_mask(batch.column(*column_index), query, language.as_deref()),
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

pub(crate) fn collect_rows(batch: &ColumnarBatch<'_>) -> Vec<Vec<Value>> {
    batch.to_rows()
}

fn equals_mask(column: &ColumnVector<'_>, value: &Value) -> Vec<bool> {
    match value {
        Value::Integer(target) => fast_equals_integer(column, *target),
        Value::Float(target) => fast_equals_float(column, *target),
        _ => column
            .values()
            .iter()
            .map(|candidate| equal(candidate, value))
            .collect(),
    }
}

fn greater_equal_mask(column: &ColumnVector<'_>, value: &Value) -> Vec<bool> {
    match value {
        Value::Integer(target) => fast_compare_integer_ge(column, *target),
        Value::Float(target) => fast_compare_float_ge(column, *target),
        _ => column
            .values()
            .iter()
            .map(|candidate| {
                if matches!(candidate, Value::Null) {
                    return false;
                }
                matches!(
                    compare_values(candidate, value),
                    Some(std::cmp::Ordering::Greater) | Some(std::cmp::Ordering::Equal)
                )
            })
            .collect(),
    }
}

fn between_mask(column: &ColumnVector<'_>, low: &Value, high: &Value) -> Vec<bool> {
    if matches!(low, Value::Integer(_)) && matches!(high, Value::Integer(_)) {
        let low = match low {
            Value::Integer(v) => *v,
            _ => unreachable!(),
        };
        let high = match high {
            Value::Integer(v) => *v,
            _ => unreachable!(),
        };
        return fast_between_integer(column, low, high);
    }
    if matches!(low, Value::Float(_)) && matches!(high, Value::Float(_)) {
        let low = match low {
            Value::Float(v) => *v,
            _ => unreachable!(),
        };
        let high = match high {
            Value::Float(v) => *v,
            _ => unreachable!(),
        };
        return fast_between_float(column, low, high);
    }
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

fn null_mask(column: &ColumnVector<'_>) -> Vec<bool> {
    column
        .values()
        .iter()
        .map(|candidate| matches!(candidate, Value::Null))
        .collect()
}

fn in_list_mask(column: &ColumnVector<'_>, lookup_values: &[Value]) -> Vec<bool> {
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

fn full_text_mask(column: &ColumnVector<'_>, query: &str, language: Option<&str>) -> Vec<bool> {
    column
        .values()
        .iter()
        .map(|candidate| full_text_matches(candidate, query, language))
        .collect()
}

fn fast_equals_integer(column: &ColumnVector<'_>, target: i64) -> Vec<bool> {
    let values = column.values();
    let mut mask = Vec::with_capacity(column.len());
    let mut chunks = values.chunks_exact(4);
    for chunk in &mut chunks {
        for value in chunk {
            match value {
                Value::Integer(i) => mask.push(*i == target),
                Value::Null => mask.push(false),
                _ => mask.push(equal(value, &Value::Integer(target))),
            }
        }
    }
    for value in chunks.remainder() {
        match value {
            Value::Integer(i) => mask.push(*i == target),
            Value::Null => mask.push(false),
            _ => mask.push(equal(value, &Value::Integer(target))),
        }
    }
    mask
}

fn fast_equals_float(column: &ColumnVector<'_>, target: f64) -> Vec<bool> {
    let values = column.values();
    let mut mask = Vec::with_capacity(column.len());
    let mut chunks = values.chunks_exact(4);
    for chunk in &mut chunks {
        for value in chunk {
            match value {
                Value::Float(f) => mask.push((*f - target).abs() <= f64::EPSILON),
                Value::Null => mask.push(false),
                _ => mask.push(equal(value, &Value::Float(target))),
            }
        }
    }
    for value in chunks.remainder() {
        match value {
            Value::Float(f) => mask.push((*f - target).abs() <= f64::EPSILON),
            Value::Null => mask.push(false),
            _ => mask.push(equal(value, &Value::Float(target))),
        }
    }
    mask
}

fn fast_compare_integer_ge(column: &ColumnVector<'_>, target: i64) -> Vec<bool> {
    let mut mask = Vec::with_capacity(column.len());
    for value in column.values() {
        match value {
            Value::Integer(i) => mask.push(*i >= target),
            Value::Null => mask.push(false),
            _ => mask.push(matches!(
                compare_values(value, &Value::Integer(target)),
                Some(std::cmp::Ordering::Greater) | Some(std::cmp::Ordering::Equal)
            )),
        }
    }
    mask
}

fn fast_compare_float_ge(column: &ColumnVector<'_>, target: f64) -> Vec<bool> {
    let mut mask = Vec::with_capacity(column.len());
    for value in column.values() {
        match value {
            Value::Float(f) => mask.push(*f >= target),
            Value::Null => mask.push(false),
            _ => mask.push(matches!(
                compare_values(value, &Value::Float(target)),
                Some(std::cmp::Ordering::Greater) | Some(std::cmp::Ordering::Equal)
            )),
        }
    }
    mask
}

fn fast_between_integer(column: &ColumnVector<'_>, low: i64, high: i64) -> Vec<bool> {
    let mut mask = Vec::with_capacity(column.len());
    for value in column.values() {
        match value {
            Value::Integer(i) => mask.push(*i >= low && *i <= high),
            Value::Null => mask.push(false),
            _ => mask.push(false),
        }
    }
    mask
}

fn fast_between_float(column: &ColumnVector<'_>, low: f64, high: f64) -> Vec<bool> {
    let mut mask = Vec::with_capacity(column.len());
    for value in column.values() {
        match value {
            Value::Float(f) => mask.push(*f >= low && *f <= high),
            Value::Null => mask.push(false),
            _ => mask.push(false),
        }
    }
    mask
}
