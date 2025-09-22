use crate::{Predicate, SelectColumns};

use super::context::PlanContext;
use super::cost::{CostEstimate, CostModel};
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

impl<'a> PartialEq for MemoExpr<'a> {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (
                MemoExpr::Projection {
                    columns: left,
                    input: left_input,
                },
                MemoExpr::Projection {
                    columns: right,
                    input: right_input,
                },
            ) => left == right && left_input == right_input,
            (
                MemoExpr::Filter {
                    predicate: left_predicate,
                    input: left_input,
                },
                MemoExpr::Filter {
                    predicate: right_predicate,
                    input: right_input,
                },
            ) => left_predicate == right_predicate && left_input == right_input,
            (
                MemoExpr::Scan {
                    table: left_table,
                    candidates: left_candidates,
                    options: left_options,
                },
                MemoExpr::Scan {
                    table: right_table,
                    candidates: right_candidates,
                    options: right_options,
                },
            ) => {
                left_table.name == right_table.name
                    && left_table.source == right_table.source
                    && std::ptr::eq(left_table.table, right_table.table)
                    && left_candidates == right_candidates
                    && left_options == right_options
            }
            _ => false,
        }
    }
}

impl<'a> Eq for MemoExpr<'a> {}

#[derive(Debug, Clone)]
pub struct GroupExpression<'a> {
    pub id: ExprId,
    pub expr: MemoExpr<'a>,
    pub group: GroupId,
}

#[derive(Debug, Clone)]
pub struct Group {
    pub expressions: Vec<ExprId>,
    pub best_expr: Option<ExprId>,
    pub best_cost: Option<super::cost::CostEstimate>,
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
        let group_id = GroupId(self.groups.len());
        self.groups.push(Group {
            expressions: Vec::new(),
            best_expr: None,
            best_cost: None,
        });
        self.add_expression_to_group(group_id, operator);
        group_id
    }

    pub fn group(&self, id: GroupId) -> &Group {
        &self.groups[id.0]
    }

    pub fn group_mut(&mut self, id: GroupId) -> &mut Group {
        &mut self.groups[id.0]
    }

    pub fn expression(&self, id: ExprId) -> &GroupExpression<'a> {
        &self.expressions[id.0]
    }

    pub fn expression_mut(&mut self, id: ExprId) -> &mut GroupExpression<'a> {
        &mut self.expressions[id.0]
    }

    pub fn expression_group(&self, id: ExprId) -> GroupId {
        self.expressions[id.0].group
    }

    pub fn expressions(&self) -> impl Iterator<Item = &GroupExpression<'a>> {
        self.expressions.iter()
    }

    pub fn add_expression_to_group(
        &mut self,
        group_id: GroupId,
        expr: MemoExpr<'a>,
    ) -> Option<ExprId> {
        if self.groups[group_id.0]
            .expressions
            .iter()
            .any(|&existing| self.expressions[existing.0].expr == expr)
        {
            return None;
        }
        let expr_id = ExprId(self.expressions.len());
        self.expressions.push(GroupExpression {
            id: expr_id,
            expr,
            group: group_id,
        });
        let group = &mut self.groups[group_id.0];
        group.expressions.push(expr_id);
        group.best_expr = None;
        group.best_cost = None;
        Some(expr_id)
    }

    pub fn reset_costs(&mut self) {
        for group in &mut self.groups {
            group.best_expr = None;
            group.best_cost = None;
        }
    }

    pub fn rebuild_logical(&self) -> LogicalPlan<'a> {
        let root_expr = self.extract_group(self.root);
        LogicalPlan { root: root_expr }
    }

    fn extract_group(&self, group_id: GroupId) -> LogicalExpr<'a> {
        let group = self.group(group_id);
        let expr_id = group.best_expr.unwrap_or_else(|| group.expressions[0]);
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

    pub fn choose_best_expressions(&mut self, context: &PlanContext<'a>, cost_model: &CostModel) {
        self.reset_costs();
        self.compute_group_cost(self.root, context, cost_model);
    }

    fn compute_group_cost(
        &mut self,
        group_id: GroupId,
        context: &PlanContext<'a>,
        cost_model: &CostModel,
    ) -> CostEstimate {
        if let Some(cost) = self.groups[group_id.0].best_cost {
            return cost;
        }

        let expr_ids: Vec<_> = self.groups[group_id.0].expressions.clone();
        let mut best_expr = None;
        let mut best_cost = None;

        for expr_id in expr_ids {
            let cost = self.compute_expression_cost(expr_id, context, cost_model);
            if best_cost
                .map(|existing: CostEstimate| cost.total_cost() < existing.total_cost())
                .unwrap_or(true)
            {
                best_cost = Some(cost);
                best_expr = Some(expr_id);
            }
        }

        let group = &mut self.groups[group_id.0];
        group.best_expr = best_expr;
        group.best_cost = best_cost;
        best_cost.expect("memo group must contain at least one expression")
    }

    fn compute_expression_cost(
        &mut self,
        expr_id: ExprId,
        context: &PlanContext<'a>,
        cost_model: &CostModel,
    ) -> CostEstimate {
        let expr = self.expressions[expr_id.0].expr.clone();
        match expr {
            MemoExpr::Projection { columns, input } => {
                let input_cost = self.compute_group_cost(input, context, cost_model);
                cost_model.estimate_projection(input_cost, context, &columns)
            }
            MemoExpr::Filter { predicate, input } => {
                let input_cost = self.compute_group_cost(input, context, cost_model);
                cost_model.estimate_filter(context, &predicate, input_cost)
            }
            MemoExpr::Scan {
                candidates,
                options,
                ..
            } => cost_model.estimate_scan(context, &candidates, &options),
        }
    }
}
