use crate::{Predicate, SelectColumns};

use super::cardinality::JoinPredicate;
use super::context::PlanContext;
use super::cost::{CostEstimate, CostModel};
use super::logical::{
    FilterExpr, JoinExpr, LogicalExpr, LogicalPlan, ProjectionExpr, ScanCandidates, ScanExpr,
    ScanOptions,
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
    Join {
        left: GroupId,
        right: GroupId,
        predicate: JoinPredicate,
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
            (
                MemoExpr::Join {
                    left: left_left,
                    right: left_right,
                    predicate: left_predicate,
                },
                MemoExpr::Join {
                    left: right_left,
                    right: right_right,
                    predicate: right_predicate,
                },
            ) => {
                left_left == right_left
                    && left_right == right_right
                    && left_predicate == right_predicate
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
            LogicalExpr::Join(join) => {
                let left = self.insert_expr(*join.left);
                let right = self.insert_expr(*join.right);
                self.insert_operator(MemoExpr::Join {
                    left,
                    right,
                    predicate: join.predicate,
                })
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

    fn resolve_context_for_table<'b>(
        &self,
        table: &ResolvedTable<'a>,
        contexts: &'b [PlanContext<'a>],
    ) -> Option<&'b PlanContext<'a>> {
        let ptr = table.table as *const _ as usize;
        contexts
            .iter()
            .find(|ctx| ctx.table.table as *const _ as usize == ptr)
    }

    fn resolve_context_for_group<'b>(
        &self,
        group: GroupId,
        contexts: &'b [PlanContext<'a>],
    ) -> Option<&'b PlanContext<'a>> {
        let table = self.find_base_table(group)?;
        self.resolve_context_for_table(&table, contexts)
    }

    fn find_base_table(&self, group: GroupId) -> Option<ResolvedTable<'a>> {
        let group_ref = self.group(group);
        for &expr_id in &group_ref.expressions {
            if let Some(table) = self.expr_base_table(expr_id) {
                return Some(table);
            }
        }
        None
    }

    fn expr_base_table(&self, expr_id: ExprId) -> Option<ResolvedTable<'a>> {
        let expr = self.expression(expr_id).expr.clone();
        match expr {
            MemoExpr::Scan { table, .. } => Some(table),
            MemoExpr::Projection { input, .. } | MemoExpr::Filter { input, .. } => {
                self.find_base_table(input)
            }
            MemoExpr::Join { .. } => None,
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
            MemoExpr::Join {
                left,
                right,
                predicate,
            } => {
                let left_expr = self.extract_group(*left);
                let right_expr = self.extract_group(*right);
                LogicalExpr::Join(JoinExpr {
                    left: Box::new(left_expr),
                    right: Box::new(right_expr),
                    predicate: predicate.clone(),
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

    pub fn choose_best_expressions(
        &mut self,
        contexts: &[PlanContext<'a>],
        cost_model: &CostModel,
    ) {
        self.reset_costs();
        self.compute_group_cost(self.root, contexts, cost_model);
    }

    fn compute_group_cost(
        &mut self,
        group_id: GroupId,
        contexts: &[PlanContext<'a>],
        cost_model: &CostModel,
    ) -> CostEstimate {
        if let Some(cost) = self.groups[group_id.0].best_cost {
            return cost;
        }

        let expr_ids: Vec<_> = self.groups[group_id.0].expressions.clone();
        let mut best_expr = None;
        let mut best_cost = None;

        for expr_id in expr_ids {
            let cost = self.compute_expression_cost(expr_id, contexts, cost_model);
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
        contexts: &[PlanContext<'a>],
        cost_model: &CostModel,
    ) -> CostEstimate {
        let expr = self.expressions[expr_id.0].expr.clone();
        match expr {
            MemoExpr::Projection { columns, input } => {
                let input_cost = self.compute_group_cost(input, contexts, cost_model);
                let base_context = self
                    .resolve_context_for_group(input, contexts)
                    .unwrap_or_else(|| contexts.first().expect("at least one context"));
                cost_model.estimate_projection(input_cost, base_context, &columns)
            }
            MemoExpr::Filter { predicate, input } => {
                let input_cost = self.compute_group_cost(input, contexts, cost_model);
                let base_context = self
                    .resolve_context_for_group(input, contexts)
                    .unwrap_or_else(|| contexts.first().expect("at least one context"));
                cost_model.estimate_filter(base_context, &predicate, input_cost)
            }
            MemoExpr::Join {
                left,
                right,
                predicate,
            } => {
                let left_cost = self.compute_group_cost(left, contexts, cost_model);
                let right_cost = self.compute_group_cost(right, contexts, cost_model);
                let (left_ctx, right_ctx) = (
                    self.resolve_context_for_group(left, contexts),
                    self.resolve_context_for_group(right, contexts),
                );
                let join_cost = if let (Some(left_ctx), Some(right_ctx)) = (left_ctx, right_ctx) {
                    cost_model.estimate_join(left_ctx, right_ctx, &predicate)
                } else {
                    cost_model.estimate_join(
                        contexts.first().expect("at least one plan context"),
                        contexts.first().expect("at least one plan context"),
                        &predicate,
                    )
                };
                super::cost::CostEstimate {
                    cardinality: join_cost.cardinality,
                    cpu_cost: left_cost.cpu_cost + right_cost.cpu_cost + join_cost.cpu_cost,
                    io_cost: left_cost.io_cost + right_cost.io_cost + join_cost.io_cost,
                }
            }
            MemoExpr::Scan {
                candidates,
                options,
                table,
            } => {
                let context = self
                    .resolve_context_for_table(&table, contexts)
                    .unwrap_or_else(|| contexts.first().expect("at least one context"));
                cost_model.estimate_scan(context, &candidates, &options)
            }
        }
    }
}
