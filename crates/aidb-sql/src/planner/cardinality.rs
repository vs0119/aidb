#![cfg_attr(not(test), allow(dead_code))]
use std::cmp::Ordering;

use crate::{ColumnType, Predicate, Value};

use super::context::PlanContext;

const MIN_SELECTIVITY: f64 = 1e-6;
const DEFAULT_SELECTIVITY: f64 = 0.25;
const DEFAULT_EQUALITY_SELECTIVITY: f64 = 0.1;
const DEFAULT_IN_SUBQUERY_SELECTIVITY: f64 = 0.15;
const DEFAULT_FULL_TEXT_SELECTIVITY: f64 = 0.05;

#[derive(Debug, Clone, Copy)]
pub struct CardinalityEstimate {
    pub estimated_rows: f64,
    pub confidence: f64,
}

impl CardinalityEstimate {
    pub fn new(estimated_rows: f64, confidence: f64) -> Self {
        Self {
            estimated_rows: estimated_rows.max(0.0),
            confidence: confidence.clamp(0.0, 1.0),
        }
    }

    pub fn zero() -> Self {
        Self::new(0.0, 1.0)
    }

    pub fn scale(&self, factor: f64, confidence: f64) -> Self {
        let scaled = self.estimated_rows * factor;
        Self::new(scaled, (self.confidence + confidence).min(1.0))
    }
}

pub struct CardinalityEstimator<'a> {
    context: &'a PlanContext<'a>,
}

impl<'a> CardinalityEstimator<'a> {
    pub fn new(context: &'a PlanContext<'a>) -> Self {
        Self { context }
    }

    pub fn base_cardinality(&self) -> CardinalityEstimate {
        let stats = self.context.statistics();
        let base_rows = stats.row_count() as f64;
        let confidence = if stats.has_table_stats() { 0.7 } else { 0.3 };
        CardinalityEstimate::new(base_rows, confidence)
    }

    pub fn estimate_filter(&self, predicate: Option<&Predicate>) -> CardinalityEstimate {
        let base = self.base_cardinality();
        let Some(predicate) = predicate else {
            return base;
        };
        let (selectivity, confidence) = self.selectivity_for(predicate);
        base.scale(selectivity, confidence)
    }

    pub fn estimate_join(
        left: &PlanContext<'_>,
        right: &PlanContext<'_>,
        predicate: &JoinPredicate,
    ) -> CardinalityEstimate {
        match predicate.join_type {
            JoinType::Inner => estimate_inner_join(left, right, predicate),
        }
    }

    fn selectivity_for(&self, predicate: &Predicate) -> (f64, f64) {
        match predicate {
            Predicate::Equals { column, value } => self.estimate_equality(column, value),
            Predicate::GreaterOrEqual { column, value } => {
                if let Some(max_value) = self
                    .context
                    .statistics()
                    .column_stats(column)
                    .and_then(|stats| stats.max.clone())
                {
                    self.estimate_range(column, value, &max_value)
                } else {
                    (DEFAULT_SELECTIVITY, 0.1)
                }
            }
            Predicate::Between { column, start, end } => self.estimate_range(column, start, end),
            Predicate::IsNull { column } => self.estimate_is_null(column),
            Predicate::InTableColumn { .. } => (DEFAULT_IN_SUBQUERY_SELECTIVITY, 0.1),
            Predicate::FullText { .. } => (DEFAULT_FULL_TEXT_SELECTIVITY, 0.05),
        }
    }

    fn estimate_equality(&self, column: &str, value: &Value) -> (f64, f64) {
        let stats = self.context.statistics();
        let column_type = self.resolve_column_type(column);
        let Some(column_stats) = stats.column_stats(column) else {
            return (DEFAULT_EQUALITY_SELECTIVITY, 0.1);
        };
        let non_null = stats
            .row_count()
            .saturating_sub(column_stats.null_count)
            .max(1) as f64;
        let mut selectivity = (1.0 / column_stats.distinct_count.max(1) as f64)
            .max(1.0 / non_null)
            .max(MIN_SELECTIVITY);
        let mut confidence = if stats.has_table_stats() { 0.6 } else { 0.3 };

        if let Some(col_ty) = column_type {
            if let Some(hist) = column_stats.histogram.as_ref() {
                if let Some(value_numeric) = numeric_value_for(col_ty, value) {
                    if let Some(bucket) = bucket_containing(hist, value_numeric) {
                        let bucket_span = bucket_span(col_ty, bucket).max(1.0);
                        let bucket_density = bucket.count as f64 / non_null;
                        let histogram_selectivity =
                            (bucket_density / bucket_span).max(1.0 / non_null);
                        selectivity = selectivity.max(histogram_selectivity);
                        confidence = (confidence + 0.25_f64).min(0.9_f64);
                    }
                }
            }
        }

        (selectivity.clamp(MIN_SELECTIVITY, 1.0), confidence)
    }

    fn estimate_range(&self, column: &str, start: &Value, end: &Value) -> (f64, f64) {
        let stats = self.context.statistics();
        let column_type = self.resolve_column_type(column);
        let Some(column_stats) = stats.column_stats(column) else {
            return (DEFAULT_SELECTIVITY, 0.1);
        };
        let mut low = start.clone();
        let mut high = end.clone();
        if compare_values(&low, &high) == Some(Ordering::Greater) {
            std::mem::swap(&mut low, &mut high);
        }
        let row_count = stats.row_count().max(1) as f64;
        let non_null = row_count - column_stats.null_count as f64;
        if non_null <= 0.0 {
            return (0.0, 0.9);
        }

        let mut selectivity = DEFAULT_SELECTIVITY;
        let mut confidence = if stats.has_table_stats() { 0.5 } else { 0.2 };

        if let Some(col_ty) = column_type {
            if let Some(hist) = column_stats.histogram.as_ref() {
                if let (Some(start_num), Some(end_num)) = (
                    numeric_value_for(col_ty, &low),
                    numeric_value_for(col_ty, &high),
                ) {
                    let mut acc = 0.0;
                    let left = start_num.min(end_num);
                    let right = start_num.max(end_num);
                    for bucket in &hist.buckets {
                        let overlap = overlap_amount(left, right, bucket.lower, bucket.upper);
                        if overlap <= 0.0 {
                            continue;
                        }
                        let span = bucket_span(col_ty, bucket).max(1.0);
                        let coverage = (overlap / span).clamp(0.0, 1.0);
                        acc += (bucket.count as f64 / non_null) * coverage;
                    }
                    selectivity = acc.clamp(0.0, 1.0);
                    confidence = (confidence + 0.3_f64).min(0.9_f64);
                }
            } else if let (Some(min_v), Some(max_v)) = (
                column_stats
                    .min
                    .as_ref()
                    .and_then(|v| numeric_value_for(col_ty, v)),
                column_stats
                    .max
                    .as_ref()
                    .and_then(|v| numeric_value_for(col_ty, v)),
            ) {
                let domain = (max_v - min_v).abs();
                if domain <= f64::EPSILON {
                    if let (Some(start_num), Some(end_num)) = (
                        numeric_value_for(col_ty, &low),
                        numeric_value_for(col_ty, &high),
                    ) {
                        selectivity = if start_num <= min_v && end_num >= max_v {
                            1.0
                        } else {
                            0.0
                        };
                    }
                } else if let (Some(start_num), Some(end_num)) = (
                    numeric_value_for(col_ty, &low),
                    numeric_value_for(col_ty, &high),
                ) {
                    let left = start_num.min(end_num);
                    let right = start_num.max(end_num);
                    let overlap = overlap_amount(left, right, min_v, max_v);
                    selectivity = (overlap / domain).clamp(0.0, 1.0);
                }
                confidence = (confidence + 0.2_f64).min(0.8_f64);
            }
        } else {
            selectivity = DEFAULT_SELECTIVITY;
        }

        (selectivity.clamp(0.0, 1.0), confidence)
    }

    fn estimate_is_null(&self, column: &str) -> (f64, f64) {
        let stats = self.context.statistics();
        let Some(column_stats) = stats.column_stats(column) else {
            return (MIN_SELECTIVITY, 0.05);
        };
        let row_count = stats.row_count().max(1) as f64;
        let selectivity = (column_stats.null_count as f64 / row_count).clamp(0.0, 1.0);
        let confidence = if stats.has_table_stats() { 0.5 } else { 0.2 };
        (selectivity.max(MIN_SELECTIVITY), confidence)
    }

    fn resolve_column_type(&self, column: &str) -> Option<ColumnType> {
        self.context
            .table
            .table
            .columns
            .iter()
            .find(|c| c.name.eq_ignore_ascii_case(column))
            .map(|c| c.ty)
    }
}

#[derive(Debug, Clone)]
pub struct JoinPredicate {
    pub left_column: String,
    pub right_column: String,
    pub join_type: JoinType,
}

impl JoinPredicate {
    pub fn inner(left_column: impl Into<String>, right_column: impl Into<String>) -> Self {
        Self {
            left_column: left_column.into(),
            right_column: right_column.into(),
            join_type: JoinType::Inner,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JoinType {
    Inner,
}

fn estimate_inner_join(
    left: &PlanContext<'_>,
    right: &PlanContext<'_>,
    predicate: &JoinPredicate,
) -> CardinalityEstimate {
    let left_stats = left.statistics();
    let right_stats = right.statistics();
    let left_rows = left_stats.row_count();
    let right_rows = right_stats.row_count();
    if left_rows == 0 || right_rows == 0 {
        return CardinalityEstimate::zero();
    }

    let left_col_stats = left_stats.column_stats(&predicate.left_column);
    let right_col_stats = right_stats.column_stats(&predicate.right_column);
    let left_non_null =
        left_rows.saturating_sub(left_col_stats.map_or(0, |cs| cs.null_count)) as f64;
    let right_non_null =
        right_rows.saturating_sub(right_col_stats.map_or(0, |cs| cs.null_count)) as f64;
    if left_non_null <= 0.0 || right_non_null <= 0.0 {
        return CardinalityEstimate::zero();
    }

    let left_distinct = left_col_stats
        .map(|cs| cs.distinct_count.max(1))
        .unwrap_or(left_rows.max(1));
    let right_distinct = right_col_stats
        .map(|cs| cs.distinct_count.max(1))
        .unwrap_or(right_rows.max(1));
    let max_distinct = left_distinct.max(right_distinct) as f64;
    let mut estimate = (left_non_null * right_non_null) / max_distinct.max(1.0);
    let mut confidence = 0.4;

    if let (Some(left_cs), Some(right_cs)) = (left_col_stats, right_col_stats) {
        if let (Some(left_ty), Some(right_ty)) = (
            resolve_column_type(left, &predicate.left_column),
            resolve_column_type(right, &predicate.right_column),
        ) {
            if let (Some(left_range), Some(right_range)) = (
                column_range(left_cs, left_ty),
                column_range(right_cs, right_ty),
            ) {
                let overlap =
                    overlap_amount(left_range.0, left_range.1, right_range.0, right_range.1);
                if overlap <= 0.0 {
                    return CardinalityEstimate::zero();
                }
                let left_span = (left_range.1 - left_range.0).abs().max(1.0);
                let right_span = (right_range.1 - right_range.0).abs().max(1.0);
                let left_fraction = (overlap / left_span).clamp(0.0, 1.0);
                let right_fraction = (overlap / right_span).clamp(0.0, 1.0);
                let effective_left = left_non_null * left_fraction.max(MIN_SELECTIVITY);
                let effective_right = right_non_null * right_fraction.max(MIN_SELECTIVITY);
                estimate = (effective_left * effective_right) / max_distinct.max(1.0);
                confidence = 0.75;
            }
        }
    }

    let upper_bound = (left_rows as f64) * (right_rows as f64);
    let bounded = estimate.clamp(0.0, upper_bound);
    CardinalityEstimate::new(bounded, confidence)
}

fn resolve_column_type(context: &PlanContext<'_>, column: &str) -> Option<ColumnType> {
    context
        .table
        .table
        .columns
        .iter()
        .find(|c| c.name.eq_ignore_ascii_case(column))
        .map(|c| c.ty)
}

fn column_range(
    column_stats: &crate::ColumnStatistics,
    column_type: ColumnType,
) -> Option<(f64, f64)> {
    let min = column_stats
        .min
        .as_ref()
        .and_then(|value| numeric_value_for(column_type, value))?;
    let max = column_stats
        .max
        .as_ref()
        .and_then(|value| numeric_value_for(column_type, value))?;
    Some((min.min(max), min.max(max)))
}

fn bucket_span(column_type: ColumnType, bucket: &crate::HistogramBucket) -> f64 {
    match column_type {
        ColumnType::Integer => (bucket.upper.floor() - bucket.lower.floor()).abs() + 1.0,
        ColumnType::Timestamp => (bucket.upper - bucket.lower).abs().max(1.0),
        ColumnType::Float => (bucket.upper - bucket.lower).abs().max(1.0),
        _ => 1.0,
    }
}

fn bucket_containing(
    histogram: &crate::ColumnHistogram,
    value: f64,
) -> Option<&crate::HistogramBucket> {
    histogram
        .buckets
        .iter()
        .find(|bucket| value >= bucket.lower && value <= bucket.upper)
}

fn overlap_amount(a_start: f64, a_end: f64, b_start: f64, b_end: f64) -> f64 {
    let left = a_start.max(b_start);
    let right = a_end.min(b_end);
    (right - left).max(0.0)
}

fn numeric_value_for(column_type: ColumnType, value: &Value) -> Option<f64> {
    match (column_type, value) {
        (ColumnType::Integer, Value::Integer(v)) => Some(*v as f64),
        (ColumnType::Integer, Value::Float(v)) => Some(*v),
        (ColumnType::Float, Value::Float(v)) => Some(*v),
        (ColumnType::Float, Value::Integer(v)) => Some(*v as f64),
        (ColumnType::Timestamp, Value::Timestamp(ts)) => Some(ts.timestamp_millis() as f64),
        (ColumnType::Timestamp, Value::Integer(v)) => Some(*v as f64),
        (ColumnType::Timestamp, Value::Float(v)) => Some(*v),
        _ => None,
    }
}

fn compare_values(left: &Value, right: &Value) -> Option<Ordering> {
    match (left, right) {
        (Value::Integer(a), Value::Integer(b)) => Some(a.cmp(b)),
        (Value::Float(a), Value::Float(b)) => a.partial_cmp(b),
        (Value::Integer(a), Value::Float(b)) => (*a as f64).partial_cmp(b),
        (Value::Float(a), Value::Integer(b)) => a.partial_cmp(&(*b as f64)),
        (Value::Timestamp(a), Value::Timestamp(b)) => Some(a.cmp(b)),
        (Value::Timestamp(a), Value::Integer(b)) => {
            let left = a.timestamp_millis();
            Some(left.cmp(&(*b as i64)))
        }
        (Value::Integer(a), Value::Timestamp(b)) => {
            let right = b.timestamp_millis();
            Some((*a as i64).cmp(&right))
        }
        _ => None,
    }
}
