use std::cmp::Ordering;
use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering as AtomicOrdering};

use crate::{compare_values, ColumnType, Predicate, SqlDatabaseError, Value};

#[derive(Debug, Clone)]
pub struct ExternalColumn {
    pub name: String,
    pub ty: ColumnType,
}

#[derive(Debug, Clone, Copy)]
pub struct ExternalCostFactors {
    pub remote_io_multiplier: f64,
    pub remote_cpu_multiplier: f64,
    pub pushdown_selectivity: f64,
}

impl Default for ExternalCostFactors {
    fn default() -> Self {
        Self {
            remote_io_multiplier: 1.0,
            remote_cpu_multiplier: 1.0,
            pushdown_selectivity: 1.0,
        }
    }
}

pub struct ExternalScanRequest<'a> {
    pub predicate: Option<&'a Predicate>,
}

pub struct ExternalScanResult {
    pub rows: Vec<Vec<Value>>,
    pub pushed_down: bool,
}

pub trait ExternalSource: Send + Sync {
    fn schema(&self) -> Vec<ExternalColumn>;

    fn cost_factors(&self) -> ExternalCostFactors {
        ExternalCostFactors::default()
    }

    fn supports_predicate_pushdown(&self, _predicate: &Predicate) -> bool {
        false
    }

    fn scan(
        &self,
        request: ExternalScanRequest<'_>,
    ) -> Result<ExternalScanResult, SqlDatabaseError>;
}

pub struct ParquetConnector {
    columns: Vec<ExternalColumn>,
    column_indices: HashMap<String, usize>,
    rows: Vec<Vec<Value>>,
    cost: ExternalCostFactors,
    pushdown_calls: AtomicUsize,
}

impl ParquetConnector {
    pub fn new(
        columns: Vec<(&str, ColumnType)>,
        rows: Vec<Vec<Value>>,
    ) -> Result<Self, SqlDatabaseError> {
        if columns.is_empty() {
            return Err(SqlDatabaseError::SchemaMismatch(
                "external table requires at least one column".into(),
            ));
        }
        let mut column_defs = Vec::with_capacity(columns.len());
        let mut indices = HashMap::new();
        for (index, (name, ty)) in columns.into_iter().enumerate() {
            column_defs.push(ExternalColumn {
                name: name.to_string(),
                ty,
            });
            indices.insert(name.to_ascii_lowercase(), index);
        }
        for (row_idx, row) in rows.iter().enumerate() {
            if row.len() != column_defs.len() {
                return Err(SqlDatabaseError::SchemaMismatch(format!(
                    "row {row_idx} expected {} values but received {}",
                    column_defs.len(),
                    row.len()
                )));
            }
        }
        Ok(Self {
            columns: column_defs,
            column_indices: indices,
            rows,
            cost: ExternalCostFactors {
                remote_io_multiplier: 0.8,
                remote_cpu_multiplier: 0.7,
                pushdown_selectivity: 0.3,
            },
            pushdown_calls: AtomicUsize::new(0),
        })
    }

    pub fn pushdown_call_count(&self) -> usize {
        self.pushdown_calls.load(AtomicOrdering::SeqCst)
    }

    fn column_index(&self, name: &str) -> Result<usize, SqlDatabaseError> {
        self.column_indices
            .get(&name.to_ascii_lowercase())
            .copied()
            .ok_or_else(|| SqlDatabaseError::UnknownColumn(name.to_string()))
    }

    fn evaluate_predicate(
        &self,
        row: &[Value],
        predicate: &Predicate,
    ) -> Result<bool, SqlDatabaseError> {
        match predicate {
            Predicate::Equals { column, value } => {
                let idx = self.column_index(column)?;
                Ok(row[idx] == *value)
            }
            Predicate::GreaterOrEqual { column, value } => {
                let idx = self.column_index(column)?;
                if matches!(row[idx], Value::Null) {
                    return Ok(false);
                }
                Ok(matches!(
                    compare_values(&row[idx], value),
                    Some(Ordering::Greater) | Some(Ordering::Equal)
                ))
            }
            Predicate::Between { column, start, end } => {
                let idx = self.column_index(column)?;
                if matches!(row[idx], Value::Null) {
                    return Ok(false);
                }
                let (low, high) = match compare_values(start, end) {
                    Some(Ordering::Greater) => (end, start),
                    _ => (start, end),
                };
                let cmp_low = compare_values(&row[idx], low);
                let cmp_high = compare_values(&row[idx], high);
                Ok(
                    matches!(cmp_low, Some(Ordering::Greater) | Some(Ordering::Equal))
                        && matches!(cmp_high, Some(Ordering::Less) | Some(Ordering::Equal)),
                )
            }
            Predicate::IsNull { column } => {
                let idx = self.column_index(column)?;
                Ok(matches!(row[idx], Value::Null))
            }
            _ => Ok(false),
        }
    }
}

impl ExternalSource for ParquetConnector {
    fn schema(&self) -> Vec<ExternalColumn> {
        self.columns.clone()
    }

    fn cost_factors(&self) -> ExternalCostFactors {
        self.cost
    }

    fn supports_predicate_pushdown(&self, predicate: &Predicate) -> bool {
        let column = match predicate {
            Predicate::Equals { column, .. }
            | Predicate::GreaterOrEqual { column, .. }
            | Predicate::Between { column, .. }
            | Predicate::IsNull { column }
            | Predicate::InTableColumn { column, .. }
            | Predicate::FullText { column, .. } => column,
        };
        matches!(
            predicate,
            Predicate::Equals { .. }
                | Predicate::GreaterOrEqual { .. }
                | Predicate::Between { .. }
                | Predicate::IsNull { .. }
        ) && self
            .column_indices
            .contains_key(&column.to_ascii_lowercase())
    }

    fn scan(
        &self,
        request: ExternalScanRequest<'_>,
    ) -> Result<ExternalScanResult, SqlDatabaseError> {
        let mut rows = Vec::new();
        let mut pushed_down = false;
        match request.predicate {
            Some(predicate) if self.supports_predicate_pushdown(predicate) => {
                pushed_down = true;
                self.pushdown_calls.fetch_add(1, AtomicOrdering::SeqCst);
                for row in &self.rows {
                    if self.evaluate_predicate(row, predicate)? {
                        rows.push(row.clone());
                    }
                }
            }
            _ => rows.extend(self.rows.iter().cloned()),
        }
        Ok(ExternalScanResult { rows, pushed_down })
    }
}
