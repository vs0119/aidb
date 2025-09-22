use std::ops::Bound::Included;

use crate::{
    canonical_json, coerce_static_value, column_index_in_table, xml_root_name, ColumnType,
    SelectColumns, SqlDatabaseError, Value,
};

use super::context::PlanContext;
use super::logical::{ScanCandidates, ScanOptions};
use super::memo::{GroupId, Memo, MemoExpr};
use super::rules::RewriteRule;
use super::table_ref::ResolvedTable;

pub struct BaselineScanRule;

impl BaselineScanRule {
    pub fn new() -> Self {
        Self
    }
}

impl<'a> RewriteRule<'a> for BaselineScanRule {
    fn apply(
        &self,
        memo: &mut Memo<'a>,
        context: &PlanContext<'a>,
    ) -> Result<bool, SqlDatabaseError> {
        let mut changed = false;
        let expr_ids: Vec<_> = memo.expressions().map(|expr| expr.id).collect();

        for expr_id in expr_ids {
            let (predicate, child_group) = {
                let expr = memo.expression(expr_id);
                match &expr.expr {
                    MemoExpr::Filter { predicate, input } => (predicate.clone(), *input),
                    _ => continue,
                }
            };

            let scan_exprs: Vec<_> = memo.group(child_group).expressions.clone();
            for scan_expr_id in scan_exprs {
                let scan_expr = memo.expression(scan_expr_id);
                let MemoExpr::Scan {
                    table,
                    candidates,
                    options,
                } = &scan_expr.expr
                else {
                    continue;
                };

                if matches!(table.source, super::table_ref::TableSource::External) {
                    if let Some(connector) = context.external() {
                        if connector.supports_predicate_pushdown(&predicate) {
                            let mut new_options = options.clone();
                            if new_options.pushdown_predicate.as_ref() == Some(&predicate) {
                                continue;
                            }
                            new_options.pushdown_predicate = Some(predicate.clone());
                            let new_expr = MemoExpr::Scan {
                                table: *table,
                                candidates: candidates.clone(),
                                options: new_options,
                            };
                            if memo
                                .add_expression_to_group(
                                    memo.expression_group(scan_expr_id),
                                    new_expr,
                                )
                                .is_some()
                            {
                                changed = true;
                            }
                        }
                    }
                    continue;
                }

                if !candidates.is_all() {
                    continue;
                }

                let computed = compute_candidate_rows(*table, &predicate)?;
                if let Some(rows) = computed {
                    let new_expr = MemoExpr::Scan {
                        table: *table,
                        candidates: ScanCandidates::Fixed(rows),
                        options: options.clone(),
                    };
                    if memo
                        .add_expression_to_group(memo.expression_group(scan_expr_id), new_expr)
                        .is_some()
                    {
                        changed = true;
                    }
                }
            }
        }

        Ok(changed)
    }
}

pub struct FilterPushdownRule;

impl FilterPushdownRule {
    pub fn new() -> Self {
        Self
    }

    fn pushdown_into_group(
        &self,
        memo: &mut Memo<'_>,
        context: &PlanContext<'_>,
        predicate: &crate::Predicate,
        group_id: GroupId,
    ) -> Result<bool, SqlDatabaseError> {
        let expr_ids: Vec<_> = memo.group(group_id).expressions.clone();
        let mut changed = false;

        for expr_id in expr_ids {
            let expr = memo.expression(expr_id).expr.clone();
            match expr {
                MemoExpr::Projection { columns, input } => {
                    if columns.includes_column(predicate.column_name()) {
                        if self.pushdown_into_group(memo, context, predicate, input)? {
                            changed = true;
                        }
                    }
                }
                MemoExpr::Filter { input, .. } => {
                    if self.pushdown_into_group(memo, context, predicate, input)? {
                        changed = true;
                    }
                }
                MemoExpr::Scan {
                    table,
                    candidates,
                    options,
                } => {
                    if self.pushdown_to_scan(
                        memo, context, predicate, expr_id, table, candidates, options,
                    )? {
                        changed = true;
                    }
                }
                MemoExpr::Join { .. } => {}
            }
        }

        Ok(changed)
    }

    fn pushdown_to_scan<'a>(
        &self,
        memo: &mut Memo<'a>,
        context: &PlanContext<'_>,
        predicate: &crate::Predicate,
        expr_id: super::memo::ExprId,
        table: ResolvedTable<'a>,
        candidates: ScanCandidates,
        options: ScanOptions,
    ) -> Result<bool, SqlDatabaseError> {
        if !matches!(table.source, super::table_ref::TableSource::External) {
            return Ok(false);
        }

        let Some(connector) = context.external() else {
            return Ok(false);
        };
        if !connector.supports_predicate_pushdown(predicate) {
            return Ok(false);
        }
        if options.pushdown_predicate.as_ref() == Some(predicate) {
            return Ok(false);
        }
        let mut new_options = options.clone();
        new_options.pushdown_predicate = Some(predicate.clone());
        let new_expr = MemoExpr::Scan {
            table,
            candidates,
            options: new_options,
        };
        Ok(memo
            .add_expression_to_group(memo.expression_group(expr_id), new_expr)
            .is_some())
    }
}

impl<'a> RewriteRule<'a> for FilterPushdownRule {
    fn apply(
        &self,
        memo: &mut Memo<'a>,
        context: &PlanContext<'a>,
    ) -> Result<bool, SqlDatabaseError> {
        let expr_ids: Vec<_> = memo.expressions().map(|expr| expr.id).collect();
        let mut changed = false;

        for expr_id in expr_ids {
            let (predicate, child_group) = {
                let expr = memo.expression(expr_id);
                match &expr.expr {
                    MemoExpr::Filter { predicate, input } => (predicate.clone(), *input),
                    _ => continue,
                }
            };
            if self.pushdown_into_group(memo, context, &predicate, child_group)? {
                changed = true;
            }
        }

        Ok(changed)
    }
}

pub struct ProjectionPushdownRule;

impl ProjectionPushdownRule {
    pub fn new() -> Self {
        Self
    }

    fn pushdown_columns(
        &self,
        memo: &mut Memo<'_>,
        columns: &SelectColumns,
        group_id: GroupId,
    ) -> bool {
        let expr_ids: Vec<_> = memo.group(group_id).expressions.clone();
        let mut changed = false;

        for expr_id in expr_ids {
            let expr = memo.expression(expr_id).expr.clone();
            match expr {
                MemoExpr::Filter { input, .. } => {
                    if self.pushdown_columns(memo, columns, input) {
                        changed = true;
                    }
                }
                MemoExpr::Projection { input, .. } => {
                    if self.pushdown_columns(memo, columns, input) {
                        changed = true;
                    }
                }
                MemoExpr::Scan {
                    table,
                    candidates,
                    options,
                } => {
                    if let Some(existing) = options.projected_columns.as_ref() {
                        if existing == columns {
                            continue;
                        }
                    }
                    if columns.is_all() {
                        continue;
                    }
                    let mut new_options = options.clone();
                    new_options.projected_columns = Some(columns.clone());
                    let new_expr = MemoExpr::Scan {
                        table,
                        candidates,
                        options: new_options,
                    };
                    if memo
                        .add_expression_to_group(memo.expression_group(expr_id), new_expr)
                        .is_some()
                    {
                        changed = true;
                    }
                }
                MemoExpr::Join { .. } => {}
            }
        }

        changed
    }
}

impl<'a> RewriteRule<'a> for ProjectionPushdownRule {
    fn apply(
        &self,
        memo: &mut Memo<'a>,
        _context: &PlanContext<'a>,
    ) -> Result<bool, SqlDatabaseError> {
        let expr_ids: Vec<_> = memo.expressions().map(|expr| expr.id).collect();
        let mut changed = false;

        for expr_id in expr_ids {
            let (columns, child_group) = {
                let expr = memo.expression(expr_id);
                match &expr.expr {
                    MemoExpr::Projection { columns, input } => (columns.clone(), *input),
                    _ => continue,
                }
            };

            if self.pushdown_columns(memo, &columns, child_group) {
                changed = true;
            }
        }

        Ok(changed)
    }
}

fn compute_candidate_rows(
    table_ref: ResolvedTable<'_>,
    predicate: &crate::Predicate,
) -> Result<Option<Vec<usize>>, SqlDatabaseError> {
    if let Some(candidate) = partition_prune(table_ref, predicate)? {
        return Ok(Some(candidate));
    }

    if let Some(candidate) = secondary_index_lookup(table_ref, predicate)? {
        return Ok(Some(candidate));
    }

    Ok(None)
}

fn partition_prune(
    table_ref: ResolvedTable<'_>,
    predicate: &crate::Predicate,
) -> Result<Option<Vec<usize>>, SqlDatabaseError> {
    let table = table_ref.table;
    let partitioning = match &table.partitioning {
        Some(p) => p,
        None => return Ok(None),
    };

    if !partitioning.matches_column(predicate.column_name()) {
        return Ok(None);
    }

    match predicate {
        crate::Predicate::Equals { value, .. } => {
            let idx = partitioning.column_index;
            let column_type = table.columns[idx].ty;
            let coerced = coerce_static_value(value, column_type)?;
            if let Some(key) = partitioning.partition_key(&coerced) {
                let rows = table.partitions.get(&key).cloned().unwrap_or_default();
                return Ok(Some(rows));
            }
            Ok(Some(Vec::new()))
        }
        crate::Predicate::Between { start, end, .. } => {
            let idx = partitioning.column_index;
            let column_type = table.columns[idx].ty;
            let start_val = coerce_static_value(start, column_type)?;
            let end_val = coerce_static_value(end, column_type)?;
            if let (Some(start_key), Some(end_key)) = (
                partitioning.partition_key(&start_val),
                partitioning.partition_key(&end_val),
            ) {
                let (lower, upper) = if start_key <= end_key {
                    (start_key, end_key)
                } else {
                    (end_key, start_key)
                };
                let mut indices = Vec::new();
                for rows in table
                    .partitions
                    .range((Included(lower), Included(upper)))
                    .map(|(_, v)| v)
                {
                    indices.extend(rows.iter().copied());
                }
                return Ok(Some(indices));
            }
            Ok(Some(Vec::new()))
        }
        _ => Ok(None),
    }
}

fn secondary_index_lookup(
    table_ref: ResolvedTable<'_>,
    predicate: &crate::Predicate,
) -> Result<Option<Vec<usize>>, SqlDatabaseError> {
    let table = table_ref.table;
    let (column, value) = match predicate {
        crate::Predicate::Equals { column, value } => (column, value),
        _ => return Ok(None),
    };

    let idx = column_index_in_table(table, column)?;
    let column_type = table.columns[idx].ty;

    match column_type {
        ColumnType::Json => {
            if let Some(index) = table.json_indexes.get(&idx) {
                let coerced = coerce_static_value(value, ColumnType::Json)?;
                if let Value::Json(json) = coerced {
                    let key = canonical_json(&json);
                    let rows = index.map.get(&key).cloned().unwrap_or_default();
                    return Ok(Some(rows));
                }
            }
        }
        ColumnType::Jsonb => {
            if let Some(index) = table.jsonb_indexes.get(&idx) {
                let coerced = coerce_static_value(value, ColumnType::Jsonb)?;
                if let Value::Jsonb(json) = coerced {
                    let key = canonical_json(&json);
                    let rows = index.map.get(&key).cloned().unwrap_or_default();
                    return Ok(Some(rows));
                }
            }
        }
        ColumnType::Xml => {
            if let Some(index) = table.xml_indexes.get(&idx) {
                let coerced = coerce_static_value(value, ColumnType::Xml)?;
                if let Value::Xml(xml) = coerced {
                    if let Some(root) = xml_root_name(&xml) {
                        let rows = index.map.get(&root).cloned().unwrap_or_default();
                        return Ok(Some(rows));
                    }
                    return Ok(Some(Vec::new()));
                }
            }
        }
        ColumnType::Geometry => {
            if let Some(index) = table.spatial_indexes.get(&idx) {
                let coerced = coerce_static_value(value, ColumnType::Geometry)?;
                if let Value::Geometry(geom) = coerced {
                    let mut rows = Vec::new();
                    let envelope = geom.to_aabb();
                    for item in index.tree.locate_in_envelope_intersecting(&envelope) {
                        rows.push(item.row);
                    }
                    rows.sort_unstable();
                    rows.dedup();
                    return Ok(Some(rows));
                }
            }
        }
        _ => {}
    }

    Ok(None)
}
