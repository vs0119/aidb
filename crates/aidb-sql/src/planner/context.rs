use super::table_ref::ResolvedTable;

#[derive(Clone, Copy)]
pub struct PlanContext<'a> {
    #[allow(dead_code)]
    pub table: ResolvedTable<'a>,
}

impl<'a> PlanContext<'a> {
    pub fn new(table: ResolvedTable<'a>) -> Self {
        Self { table }
    }
}
