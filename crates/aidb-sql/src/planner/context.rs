#![cfg_attr(not(test), allow(dead_code))]
use std::collections::HashMap;

use crate::{ColumnStatistics, ExternalSource, SelectColumns, Table, TableStatistics, Value};

use super::cost::CostParameters;
use super::table_ref::ResolvedTable;

#[derive(Clone)]
pub struct PlanStatistics {
    observed_row_count: u64,
    average_row_width: f64,
    table_stats: Option<TableStatistics>,
    column_stats: HashMap<String, ColumnStatistics>,
}

impl PlanStatistics {
    pub fn new(
        observed_row_count: u64,
        average_row_width: f64,
        table_stats: Option<TableStatistics>,
        column_stats: Vec<ColumnStatistics>,
    ) -> Self {
        let mut map = HashMap::with_capacity(column_stats.len());
        for stat in column_stats {
            map.insert(stat.column_name.to_ascii_lowercase(), stat);
        }
        Self {
            observed_row_count,
            average_row_width,
            table_stats,
            column_stats: map,
        }
    }

    pub fn row_count(&self) -> u64 {
        self.table_stats
            .as_ref()
            .map(|stats| stats.row_count)
            .unwrap_or(self.observed_row_count)
    }

    pub fn average_row_width(&self) -> f64 {
        self.average_row_width
    }

    pub fn has_table_stats(&self) -> bool {
        self.table_stats.is_some()
    }

    pub fn column_stats(&self, column: &str) -> Option<&ColumnStatistics> {
        self.column_stats.get(&column.to_ascii_lowercase())
    }
}

#[derive(Clone)]
pub struct PlanContext<'a> {
    pub table: ResolvedTable<'a>,
    stats: PlanStatistics,
    external: Option<&'a dyn ExternalSource>,
    cost: CostParameters,
}

impl<'a> PlanContext<'a> {
    pub fn new(
        table: ResolvedTable<'a>,
        table_stats: Option<TableStatistics>,
        column_stats: Vec<ColumnStatistics>,
        external: Option<&'a dyn ExternalSource>,
    ) -> Self {
        let average_row_width = approximate_row_width(table.table);
        let stats = PlanStatistics::new(
            table.table.rows.len() as u64,
            average_row_width,
            table_stats,
            column_stats,
        );
        Self {
            table,
            stats,
            external,
            cost: CostParameters::default(),
        }
    }

    pub fn statistics(&self) -> &PlanStatistics {
        &self.stats
    }

    pub fn external(&self) -> Option<&'a dyn ExternalSource> {
        self.external
    }

    pub fn with_cost_parameters(mut self, params: CostParameters) -> Self {
        self.cost = params;
        self
    }

    pub fn cost_parameters(&self) -> CostParameters {
        self.cost
    }

    pub fn projected_row_width(&self, columns: &SelectColumns) -> f64 {
        match columns {
            SelectColumns::All => self.statistics().average_row_width(),
            SelectColumns::Some(items) => {
                if items.is_empty() {
                    return 0.0;
                }
                let total_columns = self.table.table.columns.len().max(1) as f64;
                let ratio = (items.len() as f64 / total_columns).clamp(0.0, 1.0);
                self.statistics().average_row_width() * ratio
            }
        }
    }
}

fn approximate_row_width(table: &Table) -> f64 {
    const SAMPLE_LIMIT: usize = 100;
    if table.rows.is_empty() {
        return (table.columns.len() as f64) * 8.0;
    }
    let sample_size = table.rows.len().min(SAMPLE_LIMIT);
    let total: usize = table
        .rows
        .iter()
        .take(sample_size)
        .map(|row| row.iter().map(estimate_value_size).sum::<usize>())
        .sum();
    (total as f64) / (sample_size as f64)
}

fn estimate_value_size(value: &Value) -> usize {
    match value {
        Value::Integer(_) => 8,
        Value::Float(_) => 8,
        Value::Boolean(_) => 1,
        Value::Timestamp(_) => 8,
        Value::Text(s) => s.len().max(8),
        Value::Json(_) | Value::Jsonb(_) => 64,
        Value::Xml(s) => s.len().max(16),
        Value::Geometry(_) => 64,
        Value::Null => 1,
    }
}
