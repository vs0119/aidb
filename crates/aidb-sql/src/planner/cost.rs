#![cfg_attr(not(test), allow(dead_code))]
use crate::{Predicate, SelectColumns};

use super::cardinality::{CardinalityEstimate, CardinalityEstimator, JoinPredicate};
use super::context::PlanContext;
use super::logical::{ScanCandidates, ScanOptions};

#[derive(Debug, Clone, Copy)]
pub struct CostEstimate {
    pub cardinality: CardinalityEstimate,
    pub cpu_cost: f64,
    pub io_cost: f64,
}

impl CostEstimate {
    pub fn total_cost(&self) -> f64 {
        self.cpu_cost + self.io_cost
    }
}

#[derive(Debug, Clone, Copy)]
pub struct CostParameters {
    pub page_size_bytes: f64,
    pub cpu_cost_per_row: f64,
    pub io_cost_per_page: f64,
}

impl Default for CostParameters {
    fn default() -> Self {
        Self {
            page_size_bytes: 8192.0,
            cpu_cost_per_row: 1.0,
            io_cost_per_page: 1.0,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct CostModel {
    page_size_bytes: f64,
    cpu_cost_per_row: f64,
    io_cost_per_page: f64,
}

impl Default for CostModel {
    fn default() -> Self {
        Self::from_parameters(CostParameters::default())
    }
}

impl CostModel {
    pub fn new(page_size_bytes: f64, cpu_cost_per_row: f64, io_cost_per_page: f64) -> Self {
        Self {
            page_size_bytes,
            cpu_cost_per_row,
            io_cost_per_page,
        }
    }

    pub fn from_parameters(params: CostParameters) -> Self {
        Self {
            page_size_bytes: params.page_size_bytes,
            cpu_cost_per_row: params.cpu_cost_per_row,
            io_cost_per_page: params.io_cost_per_page,
        }
    }

    pub fn estimate_scan(
        &self,
        context: &PlanContext<'_>,
        candidates: &ScanCandidates,
        options: &ScanOptions,
    ) -> CostEstimate {
        let estimator = CardinalityEstimator::new(context);
        let mut cardinality = match candidates {
            ScanCandidates::AllRows => estimator.base_cardinality(),
            ScanCandidates::Fixed(rows) => {
                let confidence = if rows.is_empty() { 0.8 } else { 0.95 };
                CardinalityEstimate::new(rows.len() as f64, confidence)
            }
        };

        if matches!(candidates, ScanCandidates::AllRows) {
            if cardinality.estimated_rows <= 0.0 {
                cardinality = CardinalityEstimate::new(1.0, cardinality.confidence.max(0.05));
            }
            if let Some(predicate) = options.pushdown_predicate.as_ref() {
                let filtered = estimator.estimate_filter(Some(predicate));
                cardinality = if filtered.estimated_rows <= 0.0 {
                    cardinality.scale(0.25, 0.1)
                } else {
                    filtered
                };
            }
        }

        let mut row_width = if let Some(columns) = options.projected_columns.as_ref() {
            context.projected_row_width(columns)
        } else {
            context.statistics().average_row_width()
        };

        if let Some(connector) = context.external() {
            let factors = connector.cost_factors();
            if let Some(pred) = options.pushdown_predicate.as_ref() {
                if connector.supports_predicate_pushdown(pred) {
                    cardinality = cardinality.scale(factors.pushdown_selectivity, 0.2);
                }
            }
            row_width *= factors.remote_io_multiplier;
            let io_cost = if row_width <= 0.0 {
                0.0
            } else {
                ((cardinality.estimated_rows * row_width) / self.page_size_bytes)
                    * self.io_cost_per_page
            };
            let cpu_cost =
                cardinality.estimated_rows * self.cpu_cost_per_row * factors.remote_cpu_multiplier;
            return CostEstimate {
                cardinality,
                cpu_cost,
                io_cost,
            };
        }

        let io_cost = if row_width <= 0.0 {
            0.0
        } else {
            ((cardinality.estimated_rows * row_width) / self.page_size_bytes)
                * self.io_cost_per_page
        };
        let cpu_cost = cardinality.estimated_rows * self.cpu_cost_per_row;

        CostEstimate {
            cardinality,
            cpu_cost,
            io_cost,
        }
    }

    pub fn estimate_filter(
        &self,
        context: &PlanContext<'_>,
        predicate: &Predicate,
        input: CostEstimate,
    ) -> CostEstimate {
        let estimator = CardinalityEstimator::new(context);
        let filtered = estimator.estimate_filter(Some(predicate));
        let estimated_rows = filtered
            .estimated_rows
            .min(input.cardinality.estimated_rows)
            .max(0.0);
        let cardinality = CardinalityEstimate::new(estimated_rows, filtered.confidence);
        CostEstimate {
            cardinality,
            cpu_cost: input.cpu_cost + estimated_rows * self.cpu_cost_per_row,
            io_cost: input.io_cost,
        }
    }

    pub fn estimate_projection(
        &self,
        input: CostEstimate,
        context: &PlanContext<'_>,
        columns: &SelectColumns,
    ) -> CostEstimate {
        let mut io_cost = input.io_cost;
        if !columns.is_all() {
            let base_width = context.statistics().average_row_width();
            let projected_width = context.projected_row_width(columns);
            if base_width > 0.0 {
                let ratio = (projected_width / base_width).clamp(0.0, 1.0);
                io_cost = input.io_cost * ratio.max(0.1);
            }
        }
        let cpu_cost =
            input.cpu_cost + input.cardinality.estimated_rows * self.cpu_cost_per_row * 0.05;
        CostEstimate {
            cardinality: input.cardinality,
            cpu_cost,
            io_cost,
        }
    }

    pub fn estimate_join(
        &self,
        left: &PlanContext<'_>,
        right: &PlanContext<'_>,
        predicate: &JoinPredicate,
    ) -> CostEstimate {
        let default_options = ScanOptions::default();
        let left_scan = self.estimate_scan(left, &ScanCandidates::AllRows, &default_options);
        let right_scan = self.estimate_scan(right, &ScanCandidates::AllRows, &default_options);
        let join_cardinality = CardinalityEstimator::estimate_join(left, right, predicate);
        let output_row_width =
            left.statistics().average_row_width() + right.statistics().average_row_width();
        let join_io = if output_row_width <= 0.0 {
            0.0
        } else {
            ((join_cardinality.estimated_rows * output_row_width) / self.page_size_bytes)
                * self.io_cost_per_page
        };
        let join_cpu = join_cardinality.estimated_rows * self.cpu_cost_per_row;
        CostEstimate {
            cardinality: join_cardinality,
            cpu_cost: left_scan.cpu_cost + right_scan.cpu_cost + join_cpu,
            io_cost: left_scan.io_cost + right_scan.io_cost + join_io * 0.5,
        }
    }
}
