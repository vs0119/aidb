#![cfg_attr(not(test), allow(dead_code))]
use crate::Predicate;

use super::cardinality::{CardinalityEstimate, CardinalityEstimator, JoinPredicate};
use super::context::PlanContext;

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
pub struct CostModel {
    page_size_bytes: f64,
    cpu_cost_per_row: f64,
}

impl Default for CostModel {
    fn default() -> Self {
        Self {
            page_size_bytes: 8192.0,
            cpu_cost_per_row: 1.0,
        }
    }
}

impl CostModel {
    pub fn new(page_size_bytes: f64, cpu_cost_per_row: f64) -> Self {
        Self {
            page_size_bytes,
            cpu_cost_per_row,
        }
    }

    pub fn estimate_scan(
        &self,
        context: &PlanContext<'_>,
        predicate: Option<&Predicate>,
    ) -> CostEstimate {
        let estimator = CardinalityEstimator::new(context);
        let mut cardinality = estimator.estimate_filter(predicate);
        let mut row_width = context.statistics().average_row_width();
        if let Some(connector) = context.external() {
            let factors = connector.cost_factors();
            if let Some(pred) = predicate {
                if connector.supports_predicate_pushdown(pred) {
                    cardinality = cardinality.scale(factors.pushdown_selectivity, 0.2);
                }
            }
            row_width *= factors.remote_io_multiplier;
            let io_cost = if row_width <= 0.0 {
                0.0
            } else {
                (cardinality.estimated_rows * row_width) / self.page_size_bytes
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
            (cardinality.estimated_rows * row_width) / self.page_size_bytes
        };
        let cpu_cost = cardinality.estimated_rows * self.cpu_cost_per_row;

        CostEstimate {
            cardinality,
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
        let left_scan = self.estimate_scan(left, None);
        let right_scan = self.estimate_scan(right, None);
        let join_cardinality = CardinalityEstimator::estimate_join(left, right, predicate);
        let output_row_width =
            left.statistics().average_row_width() + right.statistics().average_row_width();
        let join_io = if output_row_width <= 0.0 {
            0.0
        } else {
            (join_cardinality.estimated_rows * output_row_width) / self.page_size_bytes
        };
        let join_cpu = join_cardinality.estimated_rows * self.cpu_cost_per_row;
        CostEstimate {
            cardinality: join_cardinality,
            cpu_cost: left_scan.cpu_cost + right_scan.cpu_cost + join_cpu,
            io_cost: left_scan.io_cost + right_scan.io_cost + join_io * 0.5,
        }
    }
}
