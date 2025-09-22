use std::fmt;

use crate::{Predicate, SelectColumns, Value};

use super::cardinality::JoinPredicate;
use super::table_ref::ResolvedTable;

#[derive(Clone)]
pub struct LogicalPlan<'a> {
    pub root: LogicalExpr<'a>,
}

impl<'a> fmt::Debug for LogicalPlan<'a> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.root.fmt(f)
    }
}

#[derive(Clone)]
pub enum LogicalExpr<'a> {
    Projection(ProjectionExpr<'a>),
    Filter(FilterExpr<'a>),
    Join(JoinExpr<'a>),
    Scan(ScanExpr<'a>),
}

impl<'a> fmt::Debug for LogicalExpr<'a> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LogicalExpr::Projection(expr) => expr.fmt(f),
            LogicalExpr::Filter(expr) => expr.fmt(f),
            LogicalExpr::Join(expr) => expr.fmt(f),
            LogicalExpr::Scan(expr) => expr.fmt(f),
        }
    }
}

#[derive(Clone)]
pub struct ProjectionExpr<'a> {
    pub columns: SelectColumns,
    pub input: Box<LogicalExpr<'a>>,
}

impl<'a> fmt::Debug for ProjectionExpr<'a> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Projection")
            .field("columns", &self.columns)
            .field("input", &self.input)
            .finish()
    }
}

#[derive(Clone)]
pub struct FilterExpr<'a> {
    pub predicate: Predicate,
    pub input: Box<LogicalExpr<'a>>,
}

impl<'a> fmt::Debug for FilterExpr<'a> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Filter")
            .field("predicate", &self.predicate)
            .field("input", &self.input)
            .finish()
    }
}

#[derive(Clone)]
pub struct JoinExpr<'a> {
    pub left: Box<LogicalExpr<'a>>,
    pub right: Box<LogicalExpr<'a>>,
    pub predicate: super::cardinality::JoinPredicate,
}

impl<'a> fmt::Debug for JoinExpr<'a> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Join")
            .field("left", &self.left)
            .field("right", &self.right)
            .field("predicate", &self.predicate)
            .finish()
    }
}

#[derive(Clone)]
pub struct ScanExpr<'a> {
    pub table: ResolvedTable<'a>,
    pub candidates: ScanCandidates,
    pub options: ScanOptions,
}

impl<'a> fmt::Debug for ScanExpr<'a> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Scan")
            .field("table", &self.table)
            .field("candidates", &self.candidates)
            .field("options", &self.options)
            .finish()
    }
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct ScanOptions {
    pub pushdown_predicate: Option<Predicate>,
    pub projected_columns: Option<SelectColumns>,
    pub access_path: ScanAccessPath,
}

#[derive(Clone, Debug, PartialEq)]
pub enum ScanAccessPath {
    SeqScan,
    IndexScan(IndexScanOptions),
}

impl Default for ScanAccessPath {
    fn default() -> Self {
        ScanAccessPath::SeqScan
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct IndexScanOptions {
    pub column: String,
    pub lower_bound: Option<Value>,
    pub upper_bound: Option<Value>,
    pub lower_inclusive: bool,
    pub upper_inclusive: bool,
    pub estimated_selectivity: f64,
    pub covering: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ScanCandidates {
    AllRows,
    Fixed(Vec<usize>),
}

impl ScanCandidates {
    pub fn is_all(&self) -> bool {
        matches!(self, ScanCandidates::AllRows)
    }

    pub fn as_slice(&self) -> Option<&[usize]> {
        match self {
            ScanCandidates::AllRows => None,
            ScanCandidates::Fixed(rows) => Some(rows.as_slice()),
        }
    }
}

pub struct LogicalPlanBuilder<'a> {
    current: LogicalExpr<'a>,
}

impl<'a> LogicalPlanBuilder<'a> {
    pub fn scan(table: ResolvedTable<'a>) -> Self {
        let expr = LogicalExpr::Scan(ScanExpr {
            table,
            candidates: ScanCandidates::AllRows,
            options: ScanOptions::default(),
        });
        Self { current: expr }
    }

    pub fn filter(mut self, predicate: Predicate) -> Self {
        let input = Box::new(self.current);
        self.current = LogicalExpr::Filter(FilterExpr { predicate, input });
        self
    }

    pub fn join(left: LogicalPlan<'a>, right: LogicalPlan<'a>, predicate: JoinPredicate) -> Self {
        let expr = LogicalExpr::Join(JoinExpr {
            left: Box::new(left.root),
            right: Box::new(right.root),
            predicate,
        });
        Self { current: expr }
    }

    pub fn project(mut self, columns: SelectColumns) -> Self {
        let input = Box::new(self.current);
        self.current = LogicalExpr::Projection(ProjectionExpr { columns, input });
        self
    }

    pub fn build(self) -> LogicalPlan<'a> {
        LogicalPlan { root: self.current }
    }
}
