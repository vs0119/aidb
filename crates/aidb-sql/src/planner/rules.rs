use crate::SqlDatabaseError;

use super::context::PlanContext;
use super::memo::Memo;

pub trait RewriteRule<'a> {
    fn apply(
        &self,
        memo: &mut Memo<'a>,
        context: &PlanContext<'a>,
    ) -> Result<bool, SqlDatabaseError>;
}
