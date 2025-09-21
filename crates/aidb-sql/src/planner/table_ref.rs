use std::fmt;

use crate::Table;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TableSource {
    Base,
    System,
    Cte,
    MaterializedView,
}

#[derive(Clone, Copy)]
pub struct ResolvedTable<'a> {
    pub name: &'a str,
    pub source: TableSource,
    pub table: &'a Table,
}

impl<'a> ResolvedTable<'a> {
    pub fn new(name: &'a str, source: TableSource, table: &'a Table) -> Self {
        Self {
            name,
            source,
            table,
        }
    }
}

impl<'a> fmt::Debug for ResolvedTable<'a> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ResolvedTable")
            .field("name", &self.name)
            .field("source", &self.source)
            .finish_non_exhaustive()
    }
}
