mod cardinality;
mod context;
mod cost;
mod heuristics;
mod logical;
mod memo;
mod rules;
mod table_ref;

#[allow(unused_imports)]
pub use cardinality::{CardinalityEstimate, CardinalityEstimator, JoinPredicate, JoinType};
#[allow(unused_imports)]
pub use context::{PlanContext, PlanStatistics};
#[allow(unused_imports)]
pub use cost::{CostEstimate, CostModel};
#[allow(unused_imports)]
pub use logical::{JoinExpr, LogicalExpr, LogicalPlan, LogicalPlanBuilder, ScanOptions};
pub use table_ref::{ResolvedTable, TableSource};

pub(crate) use logical::{ProjectionExpr, ScanCandidates, ScanExpr};

use crate::SqlDatabaseError;

use memo::Memo;
use rules::RewriteRule;

pub struct Planner<'a> {
    contexts: Vec<PlanContext<'a>>,
    cost_model: cost::CostModel,
    rules: Vec<Box<dyn RewriteRule<'a> + 'a>>,
}

impl<'a> Planner<'a> {
    pub fn new(context: PlanContext<'a>) -> Self {
        Self::with_contexts(vec![context])
    }

    pub fn with_contexts(contexts: Vec<PlanContext<'a>>) -> Self {
        assert!(
            !contexts.is_empty(),
            "planner requires at least one context"
        );
        let cost_model = cost::CostModel::from_parameters(contexts[0].cost_parameters());
        let rules: Vec<Box<dyn RewriteRule<'a> + 'a>> = vec![
            Box::new(heuristics::ProjectionPushdownRule::new()),
            Box::new(heuristics::FilterPushdownRule::new()),
            Box::new(heuristics::BaselineScanRule::new()),
        ];
        Self {
            contexts,
            cost_model,
            rules,
        }
    }

    pub fn optimize(&self, plan: LogicalPlan<'a>) -> Result<LogicalPlan<'a>, SqlDatabaseError> {
        let plan = self.enumerate_join_order(plan);
        let mut memo = Memo::from_logical(plan);
        loop {
            let mut any_applied = false;
            for rule in &self.rules {
                if rule.apply(&mut memo, self.primary_context())? {
                    any_applied = true;
                }
            }
            if !any_applied {
                break;
            }
        }
        memo.choose_best_expressions(&self.contexts, &self.cost_model);
        Ok(memo.rebuild_logical())
    }

    fn primary_context(&self) -> &PlanContext<'a> {
        &self.contexts[0]
    }

    fn context_for_table(&self, table: ResolvedTable<'a>) -> Option<&PlanContext<'a>> {
        let ptr = table.table as *const _ as usize;
        self.contexts
            .iter()
            .find(|ctx| ctx.table.table as *const _ as usize == ptr)
    }

    fn enumerate_join_order(&self, plan: LogicalPlan<'a>) -> LogicalPlan<'a> {
        match &plan.root {
            LogicalExpr::Join(join) => {
                let left_table = Self::extract_scan_table(join.left.as_ref());
                let right_table = Self::extract_scan_table(join.right.as_ref());
                if let (Some(left_table), Some(right_table)) = (left_table, right_table) {
                    if let (Some(left_ctx), Some(right_ctx)) = (
                        self.context_for_table(left_table),
                        self.context_for_table(right_table),
                    ) {
                        let normal_cost =
                            self.cost_model
                                .estimate_join(left_ctx, right_ctx, &join.predicate);
                        let swapped_predicate = JoinPredicate {
                            left_column: join.predicate.right_column.clone(),
                            right_column: join.predicate.left_column.clone(),
                            join_type: join.predicate.join_type,
                        };
                        let swapped_cost =
                            self.cost_model
                                .estimate_join(right_ctx, left_ctx, &swapped_predicate);
                        if swapped_cost.total_cost() + f64::EPSILON < normal_cost.total_cost() {
                            let swapped = LogicalExpr::Join(JoinExpr {
                                left: join.right.clone(),
                                right: join.left.clone(),
                                predicate: swapped_predicate,
                            });
                            return LogicalPlan { root: swapped };
                        }
                    }
                }
                plan
            }
            _ => plan,
        }
    }

    fn extract_scan_table(expr: &LogicalExpr<'a>) -> Option<ResolvedTable<'a>> {
        match expr {
            LogicalExpr::Scan(scan) => Some(scan.table),
            LogicalExpr::Filter(filter) => Self::extract_scan_table(filter.input.as_ref()),
            LogicalExpr::Projection(proj) => Self::extract_scan_table(proj.input.as_ref()),
            LogicalExpr::Join(_) => None,
        }
    }
}
