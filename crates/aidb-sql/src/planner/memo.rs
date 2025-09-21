use crate::{Predicate, SelectColumns};

use super::logical::{
    FilterExpr, LogicalExpr, LogicalPlan, ProjectionExpr, ScanCandidates, ScanExpr, ScanOptions,
};
use super::table_ref::ResolvedTable;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GroupId(pub usize);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ExprId(pub usize);

#[derive(Debug, Clone)]
pub enum MemoExpr<'a> {
    Projection {
        columns: SelectColumns,
        input: GroupId,
    },
    Filter {
        predicate: Predicate,
        input: GroupId,
    },
    Scan {
        table: ResolvedTable<'a>,
        candidates: ScanCandidates,
        options: ScanOptions,
    },
}

#[derive(Debug, Clone)]
pub struct GroupExpression<'a> {
    pub id: ExprId,
    pub expr: MemoExpr<'a>,
}

#[derive(Debug, Clone)]
pub struct Group {
    pub expressions: Vec<ExprId>,
}

#[derive(Debug, Clone)]
pub struct Memo<'a> {
    groups: Vec<Group>,
    expressions: Vec<GroupExpression<'a>>,
    root: GroupId,
}

impl<'a> Memo<'a> {
    pub fn from_logical(plan: LogicalPlan<'a>) -> Self {
        let mut memo = Memo {
            groups: Vec::new(),
            expressions: Vec::new(),
            root: GroupId(0),
        };
        let root_group = memo.insert_expr(plan.root);
        memo.root = root_group;
        memo
    }

    fn insert_expr(&mut self, expr: LogicalExpr<'a>) -> GroupId {
        match expr {
            LogicalExpr::Projection(projection) => {
                let input = self.insert_expr(*projection.input);
                let columns = projection.columns;
                self.insert_operator(MemoExpr::Projection { columns, input })
            }
            LogicalExpr::Filter(filter) => {
                let input = self.insert_expr(*filter.input);
                let predicate = filter.predicate;
                self.insert_operator(MemoExpr::Filter { predicate, input })
            }
            LogicalExpr::Scan(scan) => self.insert_operator(MemoExpr::Scan {
                table: scan.table,
                candidates: scan.candidates,
                options: scan.options,
            }),
        }
    }

    fn insert_operator(&mut self, operator: MemoExpr<'a>) -> GroupId {
        let expr_id = ExprId(self.expressions.len());
        self.expressions.push(GroupExpression {
            id: expr_id,
            expr: operator,
        });
        let group_id = GroupId(self.groups.len());
        self.groups.push(Group {
            expressions: vec![expr_id],
        });
        group_id
    }

    pub fn group(&self, id: GroupId) -> &Group {
        &self.groups[id.0]
    }

    pub fn expression(&self, id: ExprId) -> &GroupExpression<'a> {
        &self.expressions[id.0]
    }

    pub fn expression_mut(&mut self, id: ExprId) -> &mut GroupExpression<'a> {
        &mut self.expressions[id.0]
    }

    pub fn expressions(&self) -> impl Iterator<Item = &GroupExpression<'a>> {
        self.expressions.iter()
    }

    pub fn rebuild_logical(&self) -> LogicalPlan<'a> {
        let root_expr = self.extract_group(self.root);
        LogicalPlan { root: root_expr }
    }

    fn extract_group(&self, group_id: GroupId) -> LogicalExpr<'a> {
        let group = self.group(group_id);
        let expr_id = group.expressions[0];
        let expr = self.expression(expr_id);
        match &expr.expr {
            MemoExpr::Projection { columns, input } => {
                let input_expr = self.extract_group(*input);
                LogicalExpr::Projection(ProjectionExpr {
                    columns: columns.clone(),
                    input: Box::new(input_expr),
                })
            }
            MemoExpr::Filter { predicate, input } => {
                let input_expr = self.extract_group(*input);
                LogicalExpr::Filter(FilterExpr {
                    predicate: predicate.clone(),
                    input: Box::new(input_expr),
                })
            }
            MemoExpr::Scan {
                table,
                candidates,
                options,
            } => LogicalExpr::Scan(ScanExpr {
                table: *table,
                candidates: candidates.clone(),
                options: options.clone(),
            }),
        }
    }
}
