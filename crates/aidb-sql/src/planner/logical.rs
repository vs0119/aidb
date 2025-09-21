use std::fmt;

use crate::{Predicate, SelectColumns};

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
    Scan(ScanExpr<'a>),
}

impl<'a> fmt::Debug for LogicalExpr<'a> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LogicalExpr::Projection(expr) => expr.fmt(f),
            LogicalExpr::Filter(expr) => expr.fmt(f),
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
pub struct ScanExpr<'a> {
    pub table: ResolvedTable<'a>,
    pub candidates: ScanCandidates,
}

impl<'a> fmt::Debug for ScanExpr<'a> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Scan")
            .field("table", &self.table)
            .field("candidates", &self.candidates)
            .finish()
    }
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
        });
        Self { current: expr }
    }

    pub fn filter(mut self, predicate: Predicate) -> Self {
        let input = Box::new(self.current);
        self.current = LogicalExpr::Filter(FilterExpr { predicate, input });
        self
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
