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
pub use logical::{LogicalExpr, LogicalPlan, LogicalPlanBuilder};
pub use table_ref::{ResolvedTable, TableSource};

pub(crate) use logical::{ProjectionExpr, ScanCandidates, ScanExpr};

use crate::SqlDatabaseError;

use memo::Memo;
use rules::RewriteRule;

pub struct Planner<'a> {
    context: PlanContext<'a>,
    rules: Vec<Box<dyn RewriteRule<'a> + 'a>>,
}

impl<'a> Planner<'a> {
    pub fn new(context: PlanContext<'a>) -> Self {
        let rules: Vec<Box<dyn RewriteRule<'a> + 'a>> =
            vec![Box::new(heuristics::BaselineScanRule::new())];
        Self { context, rules }
    }

    pub fn optimize(&self, plan: LogicalPlan<'a>) -> Result<LogicalPlan<'a>, SqlDatabaseError> {
        let mut memo = Memo::from_logical(plan);
        loop {
            let mut any_applied = false;
            for rule in &self.rules {
                if rule.apply(&mut memo, &self.context)? {
                    any_applied = true;
                }
            }
            if !any_applied {
                break;
            }
        }
        Ok(memo.rebuild_logical())
    }
}
