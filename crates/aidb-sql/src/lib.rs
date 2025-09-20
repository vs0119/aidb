use std::cmp::Ordering;
use std::collections::{BTreeMap, HashMap};
use std::ops::Bound::Included;

use chrono::Datelike;
use chrono::{DateTime, NaiveDate, NaiveDateTime, TimeZone, Utc};

use thiserror::Error;

#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    Integer(i64),
    Float(f64),
    Text(String),
    Boolean(bool),
    Timestamp(DateTime<Utc>),
    Null,
}

#[derive(Debug, Clone)]
struct Column {
    name: String,
    ty: ColumnType,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ColumnType {
    Integer,
    Float,
    Text,
    Boolean,
    Timestamp,
}

#[derive(Debug, Default)]
struct Table {
    columns: Vec<Column>,
    rows: Vec<Vec<Value>>,
    partitioning: Option<TimePartitioning>,
    partitions: BTreeMap<PartitionKey, Vec<usize>>,
}

#[derive(Debug, Default)]
struct Graph {
    nodes: HashMap<String, GraphNode>,
    edges: Vec<GraphEdge>,
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
struct GraphNode {
    id: String,
    properties: HashMap<String, Value>,
}

#[derive(Debug, Clone)]
struct GraphEdge {
    from: String,
    to: String,
    label: Option<String>,
    properties: HashMap<String, Value>,
}

#[derive(Debug, Clone)]
struct TimePartitioning {
    column_index: usize,
    column_name: String,
    granularity: PartitionGranularity,
}

impl TimePartitioning {
    fn matches_column(&self, column: &str) -> bool {
        self.column_name.eq_ignore_ascii_case(column)
    }

    fn partition_key(&self, value: &Value) -> Option<PartitionKey> {
        match value {
            Value::Timestamp(timestamp) => Some(self.partition_key_from_datetime(timestamp)),
            _ => None,
        }
    }

    fn partition_key_from_datetime(&self, timestamp: &DateTime<Utc>) -> PartitionKey {
        match self.granularity {
            PartitionGranularity::Day => {
                let days = timestamp.date_naive().num_days_from_ce();
                PartitionKey(days as i64)
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct PartitionKey(i64);

#[derive(Debug, Clone, Copy)]
enum PartitionGranularity {
    Day,
}

#[derive(Debug, Clone)]
struct PartitioningInfo {
    column: String,
    granularity: PartitionGranularity,
}

#[derive(Debug)]
pub struct SqlDatabase {
    tables: HashMap<String, Table>,
    graphs: HashMap<String, Graph>,
}

#[derive(Debug, Error)]
pub enum SqlDatabaseError {
    #[error("parse error: {0}")]
    Parse(String),
    #[error("table '{0}' already exists")]
    TableExists(String),
    #[error("table '{0}' does not exist")]
    UnknownTable(String),
    #[error("unknown column '{0}'")]
    UnknownColumn(String),
    #[error("schema mismatch: {0}")]
    SchemaMismatch(String),
    #[error("partition column '{0}' requires TIMESTAMP type")]
    InvalidPartitionColumn(String),
    #[error("graph '{0}' already exists")]
    GraphExists(String),
    #[error("graph '{0}' does not exist")]
    UnknownGraph(String),
    #[error("node '{1}' already exists in graph '{0}'")]
    NodeExists(String, String),
    #[error("graph '{0}' does not contain node '{1}'")]
    UnknownGraphNode(String, String),
    #[error("unsupported statement")]
    Unsupported,
}

#[derive(Debug, Clone, PartialEq)]
pub enum QueryResult {
    None,
    Inserted(usize),
    Rows {
        columns: Vec<String>,
        rows: Vec<Vec<Value>>,
    },
}

impl SqlDatabase {
    pub fn new() -> Self {
        Self {
            tables: HashMap::new(),
            graphs: HashMap::new(),
        }
    }

    pub fn execute(&mut self, sql: &str) -> Result<QueryResult, SqlDatabaseError> {
        let stmt = parse_statement(sql)?;
        match stmt {
            Statement::CreateTable {
                name,
                columns,
                partitioning,
            } => {
                self.exec_create_table(name, columns, partitioning)?;
                Ok(QueryResult::None)
            }
            Statement::Insert {
                table,
                columns,
                rows,
            } => {
                let inserted = self.exec_insert(table, columns, rows)?;
                Ok(QueryResult::Inserted(inserted))
            }
            Statement::Select {
                table,
                columns,
                predicate,
            } => self.exec_select(table, columns, predicate),
            Statement::CreateGraph { name } => {
                self.exec_create_graph(name)?;
                Ok(QueryResult::None)
            }
            Statement::CreateNode {
                graph,
                id,
                properties,
            } => {
                self.exec_create_node(graph, id, properties)?;
                Ok(QueryResult::None)
            }
            Statement::CreateEdge {
                graph,
                from,
                to,
                label,
                properties,
            } => {
                self.exec_create_edge(graph, from, to, label, properties)?;
                Ok(QueryResult::None)
            }
            Statement::MatchGraph {
                graph,
                from,
                to,
                label,
                property_filter,
            } => self.exec_match_graph(graph, from, to, label, property_filter),
        }
    }

    fn exec_create_table(
        &mut self,
        name: String,
        columns: Vec<(String, ColumnType)>,
        partitioning: Option<PartitioningInfo>,
    ) -> Result<(), SqlDatabaseError> {
        if self.tables.contains_key(&name) {
            return Err(SqlDatabaseError::TableExists(name));
        }
        if columns.is_empty() {
            return Err(SqlDatabaseError::SchemaMismatch(
                "CREATE TABLE requires at least one column".into(),
            ));
        }
        let mut table = Table::default();
        for (name, ty) in columns {
            table.columns.push(Column { name, ty });
        }
        if let Some(info) = partitioning {
            let column_index = table
                .columns
                .iter()
                .position(|c| c.name.eq_ignore_ascii_case(&info.column))
                .ok_or_else(|| SqlDatabaseError::UnknownColumn(info.column.clone()))?;
            if table.columns[column_index].ty != ColumnType::Timestamp {
                return Err(SqlDatabaseError::InvalidPartitionColumn(info.column));
            }
            let column_name = table.columns[column_index].name.clone();
            table.partitioning = Some(TimePartitioning {
                column_index,
                column_name,
                granularity: info.granularity,
            });
        }
        self.tables.insert(name, table);
        Ok(())
    }

    fn exec_insert(
        &mut self,
        table_name: String,
        columns: Option<Vec<String>>,
        rows: Vec<Vec<Value>>,
    ) -> Result<usize, SqlDatabaseError> {
        let table = self
            .tables
            .get_mut(&table_name)
            .ok_or_else(|| SqlDatabaseError::UnknownTable(table_name.clone()))?;
        let column_mapping = if let Some(columns) = columns {
            let mut mapping = Vec::with_capacity(columns.len());
            for column in columns {
                let idx = table
                    .columns
                    .iter()
                    .position(|c| c.name.eq_ignore_ascii_case(&column))
                    .ok_or_else(|| SqlDatabaseError::UnknownColumn(column.clone()))?;
                mapping.push(idx);
            }
            mapping
        } else {
            (0..table.columns.len()).collect()
        };

        let mut inserted = 0;
        for row in rows {
            if row.len() != column_mapping.len() {
                return Err(SqlDatabaseError::SchemaMismatch(format!(
                    "expected {} values but got {}",
                    column_mapping.len(),
                    row.len()
                )));
            }
            let mut new_row = vec![Value::Null; table.columns.len()];
            for (value, &col_idx) in row.into_iter().zip(column_mapping.iter()) {
                let coerced = coerce_value(value, table.columns[col_idx].ty)?;
                new_row[col_idx] = coerced;
            }
            let row_index = table.rows.len();
            table.rows.push(new_row);
            if let Some(partitioning) = &table.partitioning {
                if let Some(value) = table.rows[row_index].get(partitioning.column_index) {
                    if let Some(key) = partitioning.partition_key(value) {
                        table.partitions.entry(key).or_default().push(row_index);
                    }
                }
            }
            inserted += 1;
        }
        Ok(inserted)
    }

    fn exec_select(
        &self,
        table_name: String,
        columns: SelectColumns,
        predicate: Option<Predicate>,
    ) -> Result<QueryResult, SqlDatabaseError> {
        let table = self
            .tables
            .get(&table_name)
            .ok_or_else(|| SqlDatabaseError::UnknownTable(table_name.clone()))?;

        let mut candidate_indices: Option<Vec<usize>> = None;
        if let Some(predicate) = &predicate {
            if let Some(partitioning) = &table.partitioning {
                if partitioning.matches_column(predicate.column_name()) {
                    match predicate {
                        Predicate::Equals { value, .. } => {
                            let idx = partitioning.column_index;
                            let column_type = table.columns[idx].ty;
                            let coerced = coerce_static_value(value, column_type)?;
                            if let Some(key) = partitioning.partition_key(&coerced) {
                                if let Some(part_rows) = table.partitions.get(&key) {
                                    candidate_indices = Some(part_rows.clone());
                                } else {
                                    candidate_indices = Some(Vec::new());
                                }
                            } else {
                                candidate_indices = Some(Vec::new());
                            }
                        }
                        Predicate::Between { start, end, .. } => {
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
                                candidate_indices = Some(indices);
                            } else {
                                candidate_indices = Some(Vec::new());
                            }
                        }
                    }
                }
            }
        }
        let iter: Box<dyn Iterator<Item = usize>> = match candidate_indices {
            Some(indices) => Box::new(indices.into_iter()),
            None => Box::new(0..table.rows.len()),
        };
        let mut matching_indices = Vec::new();
        for row_index in iter {
            let row = &table.rows[row_index];
            if let Some(predicate) = &predicate {
                if !self.evaluate_predicate(table, predicate, row)? {
                    continue;
                }
            }
            matching_indices.push(row_index);
        }

        match columns {
            SelectColumns::All => {
                let column_names = table.columns.iter().map(|c| c.name.clone()).collect();
                let rows = matching_indices
                    .into_iter()
                    .map(|idx| table.rows[idx].clone())
                    .collect();
                Ok(QueryResult::Rows {
                    columns: column_names,
                    rows,
                })
            }
            SelectColumns::Some(items) => {
                let mut projections = Vec::new();
                let mut window_specs = Vec::new();

                for item in items {
                    match item {
                        SelectItem::Column(name) => {
                            let idx = self.column_index(table, &name)?;
                            let output_name = table.columns[idx].name.clone();
                            projections.push(PreparedProjection {
                                output_name,
                                kind: ProjectionKind::Column { index: idx },
                            });
                        }
                        SelectItem::WindowFunction(spec) => {
                            let window_index = window_specs.len();
                            let output_name = spec
                                .alias
                                .clone()
                                .unwrap_or_else(|| spec.function.default_alias().to_string());
                            window_specs.push(spec);
                            projections.push(PreparedProjection {
                                output_name,
                                kind: ProjectionKind::Window {
                                    index: window_index,
                                },
                            });
                        }
                    }
                }

                let mut window_values = Vec::new();
                for spec in &window_specs {
                    let values = self.compute_window_function(table, &matching_indices, spec)?;
                    window_values.push(values);
                }

                let mut rows = Vec::new();
                for (position, &row_index) in matching_indices.iter().enumerate() {
                    let row = &table.rows[row_index];
                    let mut output_row = Vec::with_capacity(projections.len());
                    for projection in &projections {
                        match &projection.kind {
                            ProjectionKind::Column { index } => {
                                output_row.push(row[*index].clone());
                            }
                            ProjectionKind::Window { index } => {
                                output_row.push(window_values[*index][position].clone());
                            }
                        }
                    }
                    rows.push(output_row);
                }

                let columns = projections
                    .into_iter()
                    .map(|projection| projection.output_name)
                    .collect();

                Ok(QueryResult::Rows { columns, rows })
            }
        }
    }

    fn exec_create_graph(&mut self, name: String) -> Result<(), SqlDatabaseError> {
        if self.graphs.contains_key(&name) {
            return Err(SqlDatabaseError::GraphExists(name));
        }
        self.graphs.insert(name, Graph::default());
        Ok(())
    }

    fn exec_create_node(
        &mut self,
        graph: String,
        id: String,
        properties: HashMap<String, Value>,
    ) -> Result<(), SqlDatabaseError> {
        let graph_entry = self
            .graphs
            .get_mut(&graph)
            .ok_or_else(|| SqlDatabaseError::UnknownGraph(graph.clone()))?;
        if graph_entry.nodes.contains_key(&id) {
            return Err(SqlDatabaseError::NodeExists(graph, id));
        }
        graph_entry
            .nodes
            .insert(id.clone(), GraphNode { id, properties });
        Ok(())
    }

    fn exec_create_edge(
        &mut self,
        graph: String,
        from: String,
        to: String,
        label: Option<String>,
        properties: HashMap<String, Value>,
    ) -> Result<(), SqlDatabaseError> {
        let graph_entry = self
            .graphs
            .get_mut(&graph)
            .ok_or_else(|| SqlDatabaseError::UnknownGraph(graph.clone()))?;
        if !graph_entry.nodes.contains_key(&from) {
            return Err(SqlDatabaseError::UnknownGraphNode(graph.clone(), from));
        }
        if !graph_entry.nodes.contains_key(&to) {
            return Err(SqlDatabaseError::UnknownGraphNode(graph.clone(), to));
        }
        graph_entry.edges.push(GraphEdge {
            from,
            to,
            label,
            properties,
        });
        Ok(())
    }

    fn exec_match_graph(
        &self,
        graph: String,
        from: Option<String>,
        to: Option<String>,
        label: Option<String>,
        property_filter: Option<HashMap<String, Value>>,
    ) -> Result<QueryResult, SqlDatabaseError> {
        let graph_entry = self
            .graphs
            .get(&graph)
            .ok_or_else(|| SqlDatabaseError::UnknownGraph(graph.clone()))?;
        let mut rows = Vec::new();
        for edge in &graph_entry.edges {
            if let Some(from_id) = &from {
                if !edge.from.eq_ignore_ascii_case(from_id) {
                    continue;
                }
            }
            if let Some(to_id) = &to {
                if !edge.to.eq_ignore_ascii_case(to_id) {
                    continue;
                }
            }
            if let Some(label_filter) = &label {
                match &edge.label {
                    Some(edge_label) if edge_label.eq_ignore_ascii_case(label_filter) => {}
                    _ => continue,
                }
            }
            if let Some(filter) = &property_filter {
                let mut matches = true;
                for (key, value) in filter {
                    match edge.properties.get(key) {
                        Some(existing) if equal(existing, value) => {}
                        _ => {
                            matches = false;
                            break;
                        }
                    }
                }
                if !matches {
                    continue;
                }
            }
            let mut row = Vec::new();
            row.push(Value::Text(edge.from.clone()));
            row.push(Value::Text(edge.to.clone()));
            row.push(match &edge.label {
                Some(label) => Value::Text(label.clone()),
                None => Value::Null,
            });
            row.push(Value::Text(format_properties(&edge.properties)));
            rows.push(row);
        }
        Ok(QueryResult::Rows {
            columns: vec![
                "from".into(),
                "to".into(),
                "label".into(),
                "properties".into(),
            ],
            rows,
        })
    }

    fn evaluate_predicate(
        &self,
        table: &Table,
        predicate: &Predicate,
        row: &[Value],
    ) -> Result<bool, SqlDatabaseError> {
        match predicate {
            Predicate::Equals { column, value } => {
                let idx = self.column_index(table, column)?;
                let column_type = table.columns[idx].ty;
                let coerced = coerce_static_value(value, column_type)?;
                Ok(equal(&row[idx], &coerced))
            }
            Predicate::Between { column, start, end } => {
                let idx = self.column_index(table, column)?;
                if matches!(row[idx], Value::Null) {
                    return Ok(false);
                }
                let column_type = table.columns[idx].ty;
                let start_val = coerce_static_value(start, column_type)?;
                let end_val = coerce_static_value(end, column_type)?;
                let (low, high) = match compare_values(&start_val, &end_val) {
                    Some(Ordering::Greater) => (&end_val, &start_val),
                    _ => (&start_val, &end_val),
                };
                let cmp_low = compare_values(&row[idx], low);
                let cmp_high = compare_values(&row[idx], high);
                Ok(
                    matches!(cmp_low, Some(Ordering::Greater) | Some(Ordering::Equal))
                        && matches!(cmp_high, Some(Ordering::Less) | Some(Ordering::Equal)),
                )
            }
        }
    }

    fn compute_window_function(
        &self,
        table: &Table,
        row_indices: &[usize],
        spec: &WindowFunctionExpr,
    ) -> Result<Vec<Value>, SqlDatabaseError> {
        match spec.function {
            WindowFunctionType::RowNumber => self.compute_row_number(
                table,
                row_indices,
                spec.partition_by.as_ref(),
                &spec.order_by,
            ),
        }
    }

    fn compute_row_number(
        &self,
        table: &Table,
        row_indices: &[usize],
        partition_by: Option<&String>,
        order_by: &str,
    ) -> Result<Vec<Value>, SqlDatabaseError> {
        let order_index = self.column_index(table, order_by)?;
        let partition_index = match partition_by {
            Some(column) => Some(self.column_index(table, column)?),
            None => None,
        };

        let mut partitions: HashMap<String, Vec<(usize, usize)>> = HashMap::new();
        for (position, &row_index) in row_indices.iter().enumerate() {
            let key = if let Some(part_idx) = partition_index {
                partition_key_string(&table.rows[row_index][part_idx])
            } else {
                "__all__".to_string()
            };
            partitions
                .entry(key)
                .or_default()
                .push((position, row_index));
        }

        let mut results = vec![Value::Null; row_indices.len()];
        for group in partitions.values_mut() {
            group.sort_by(|a, b| {
                let left = &table.rows[a.1][order_index];
                let right = &table.rows[b.1][order_index];
                match compare_values(left, right) {
                    Some(Ordering::Less) => Ordering::Less,
                    Some(Ordering::Greater) => Ordering::Greater,
                    Some(Ordering::Equal) | None => a.0.cmp(&b.0),
                }
            });
            for (rank, (position, _)) in group.iter().enumerate() {
                results[*position] = Value::Integer((rank + 1) as i64);
            }
        }

        Ok(results)
    }

    fn column_index(&self, table: &Table, name: &str) -> Result<usize, SqlDatabaseError> {
        table
            .columns
            .iter()
            .position(|c| c.name.eq_ignore_ascii_case(name))
            .ok_or_else(|| SqlDatabaseError::UnknownColumn(name.into()))
    }
}

#[derive(Debug)]
enum Statement {
    CreateTable {
        name: String,
        columns: Vec<(String, ColumnType)>,
        partitioning: Option<PartitioningInfo>,
    },
    Insert {
        table: String,
        columns: Option<Vec<String>>,
        rows: Vec<Vec<Value>>,
    },
    Select {
        table: String,
        columns: SelectColumns,
        predicate: Option<Predicate>,
    },
    CreateGraph {
        name: String,
    },
    CreateNode {
        graph: String,
        id: String,
        properties: HashMap<String, Value>,
    },
    CreateEdge {
        graph: String,
        from: String,
        to: String,
        label: Option<String>,
        properties: HashMap<String, Value>,
    },
    MatchGraph {
        graph: String,
        from: Option<String>,
        to: Option<String>,
        label: Option<String>,
        property_filter: Option<HashMap<String, Value>>,
    },
}

#[derive(Debug)]
enum SelectColumns {
    All,
    Some(Vec<SelectItem>),
}

#[derive(Debug)]
enum SelectItem {
    Column(String),
    WindowFunction(WindowFunctionExpr),
}

#[derive(Debug)]
struct PreparedProjection {
    output_name: String,
    kind: ProjectionKind,
}

#[derive(Debug)]
enum ProjectionKind {
    Column { index: usize },
    Window { index: usize },
}

#[derive(Debug, Clone)]
struct WindowFunctionExpr {
    function: WindowFunctionType,
    partition_by: Option<String>,
    order_by: String,
    alias: Option<String>,
}

#[derive(Debug, Clone)]
enum WindowFunctionType {
    RowNumber,
}

impl WindowFunctionType {
    fn default_alias(&self) -> &'static str {
        match self {
            WindowFunctionType::RowNumber => "row_number",
        }
    }
}

#[derive(Debug, Clone)]
enum Predicate {
    Equals {
        column: String,
        value: Value,
    },
    Between {
        column: String,
        start: Value,
        end: Value,
    },
}

impl Predicate {
    fn column_name(&self) -> &str {
        match self {
            Predicate::Equals { column, .. } => column,
            Predicate::Between { column, .. } => column,
        }
    }
}

fn parse_statement(sql: &str) -> Result<Statement, SqlDatabaseError> {
    let trimmed = sql.trim();
    if trimmed.is_empty() {
        return Err(SqlDatabaseError::Parse("empty SQL statement".into()));
    }
    let trimmed = trimmed.trim_end_matches(';').trim();
    if let Some(rest) = strip_keyword_ci(trimmed, "CREATE") {
        let rest = rest.trim_start();
        if let Some(rest) = strip_keyword_ci(rest, "TABLE") {
            parse_create_table(rest)
        } else if let Some(rest) = strip_keyword_ci(rest, "GRAPH") {
            parse_create_graph(rest)
        } else if let Some(rest) = strip_keyword_ci(rest, "NODE") {
            parse_create_node(rest)
        } else if let Some(rest) = strip_keyword_ci(rest, "EDGE") {
            parse_create_edge(rest)
        } else {
            Err(SqlDatabaseError::Parse(
                "unsupported CREATE statement".into(),
            ))
        }
    } else if let Some(rest) = strip_keyword_ci(trimmed, "INSERT") {
        let rest = expect_keyword_ci(rest, "INTO")?;
        parse_insert(rest)
    } else if let Some(rest) = strip_keyword_ci(trimmed, "SELECT") {
        parse_select(rest)
    } else if let Some(rest) = strip_keyword_ci(trimmed, "MATCH") {
        let rest = expect_keyword_ci(rest, "GRAPH")?;
        parse_match_graph(rest)
    } else {
        Err(SqlDatabaseError::Unsupported)
    }
}

fn parse_create_table(input: &str) -> Result<Statement, SqlDatabaseError> {
    let (name, rest) = parse_identifier(input)?;
    let (columns_raw, remainder) = take_parenthesized(rest)?;
    let parts = split_comma(&columns_raw)?;
    let mut columns = Vec::new();
    for part in parts {
        let mut iter = part.split_whitespace();
        let name = iter
            .next()
            .ok_or_else(|| SqlDatabaseError::Parse("missing column name".into()))?;
        let ty = iter
            .next()
            .ok_or_else(|| SqlDatabaseError::Parse("missing column type".into()))?;
        if iter.next().is_some() {
            return Err(SqlDatabaseError::Parse(
                "only single-word column types are supported".into(),
            ));
        }
        let ty = parse_column_type(ty)?;
        columns.push((name.to_string(), ty));
    }
    let mut partitioning = None;
    let mut remainder_trimmed = remainder.trim_start();
    if !remainder_trimmed.is_empty() {
        let (info, rest) = parse_partitioning_clause(remainder_trimmed)?;
        partitioning = Some(info);
        remainder_trimmed = rest.trim_start();
    }
    ensure_no_trailing_tokens(remainder_trimmed)?;
    Ok(Statement::CreateTable {
        name,
        columns,
        partitioning,
    })
}

fn parse_insert(input: &str) -> Result<Statement, SqlDatabaseError> {
    let (table, rest) = parse_identifier(input)?;
    let mut columns = None;
    let rest_trimmed = rest.trim_start();
    let mut remainder = rest_trimmed;
    if remainder.starts_with('(') {
        let (cols_raw, rest_after) = take_parenthesized(remainder)?;
        let cols = split_comma(&cols_raw)?
            .into_iter()
            .map(|s| s.trim().to_string())
            .collect::<Vec<_>>();
        columns = Some(cols);
        remainder = rest_after;
    }
    let mut remainder = expect_keyword_ci(remainder, "VALUES")?;
    let mut rows = Vec::new();
    loop {
        let (values_raw, rest) = take_parenthesized(remainder)?;
        let values = split_comma(&values_raw)?
            .into_iter()
            .map(|s| parse_value(&s))
            .collect::<Result<Vec<_>, _>>()?;
        rows.push(values);
        remainder = rest.trim_start();
        if remainder.starts_with(',') {
            remainder = &remainder[1..];
            continue;
        }
        ensure_no_trailing_tokens(remainder)?;
        break;
    }
    Ok(Statement::Insert {
        table,
        columns,
        rows,
    })
}

fn parse_partitioning_clause(input: &str) -> Result<(PartitioningInfo, &str), SqlDatabaseError> {
    let rest = expect_keyword_ci(input, "PARTITION")?;
    let rest = expect_keyword_ci(rest, "BY")?;
    let rest = rest.trim_start();
    if let Some(rest) = strip_keyword_ci(rest, "DAY") {
        let (column_raw, remainder) = take_parenthesized(rest)?;
        let column = column_raw.trim();
        if column.is_empty() {
            return Err(SqlDatabaseError::Parse(
                "PARTITION BY DAY requires a column name".into(),
            ));
        }
        if column.contains(',') {
            return Err(SqlDatabaseError::Parse(
                "only a single column can be used for time partitioning".into(),
            ));
        }
        Ok((
            PartitioningInfo {
                column: column.to_string(),
                granularity: PartitionGranularity::Day,
            },
            remainder,
        ))
    } else {
        Err(SqlDatabaseError::Parse(
            "unsupported time partition granularity".into(),
        ))
    }
}

fn parse_literal_token(input: &str) -> Result<(Value, &str), SqlDatabaseError> {
    let trimmed = input.trim_start();
    if trimmed.is_empty() {
        return Err(SqlDatabaseError::Parse("missing literal".into()));
    }
    if trimmed.starts_with('\'') {
        let mut escaped = false;
        for (idx, byte) in trimmed.as_bytes().iter().enumerate().skip(1) {
            let ch = *byte as char;
            if ch == '\\' && !escaped {
                escaped = true;
                continue;
            }
            if ch == '\'' && !escaped {
                let token = &trimmed[..=idx];
                let rest = &trimmed[idx + 1..];
                let value = parse_value(token)?;
                return Ok((value, rest));
            }
            escaped = false;
        }
        Err(SqlDatabaseError::Parse(
            "unterminated string literal".into(),
        ))
    } else if trimmed.starts_with('"') {
        for (idx, byte) in trimmed.as_bytes().iter().enumerate().skip(1) {
            let ch = *byte as char;
            if ch == '"' {
                let token = &trimmed[..=idx];
                let rest = &trimmed[idx + 1..];
                let value = parse_value(token)?;
                return Ok((value, rest));
            }
        }
        Err(SqlDatabaseError::Parse(
            "unterminated string literal".into(),
        ))
    } else {
        let mut end = trimmed.len();
        for (idx, ch) in trimmed.char_indices() {
            if ch.is_whitespace() {
                end = idx;
                break;
            }
        }
        let token = &trimmed[..end];
        let rest = &trimmed[end..];
        let value = parse_value(token)?;
        Ok((value, rest))
    }
}

fn parse_properties_clause(
    input: &str,
) -> Result<(HashMap<String, Value>, &str), SqlDatabaseError> {
    let rest = expect_keyword_ci(input, "WITH")?;
    let rest = expect_keyword_ci(rest, "PROPERTIES")?;
    let (raw, remainder) = take_parenthesized(rest)?;
    let props = parse_properties_map(&raw)?;
    Ok((props, remainder))
}

fn parse_properties_map(raw: &str) -> Result<HashMap<String, Value>, SqlDatabaseError> {
    let mut props = HashMap::new();
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return Ok(props);
    }
    for part in split_comma(trimmed)? {
        let (key, value_str) = part
            .split_once('=')
            .ok_or_else(|| SqlDatabaseError::Parse("invalid property assignment".into()))?;
        let key = key.trim();
        if key.is_empty() {
            return Err(SqlDatabaseError::Parse(
                "property keys cannot be empty".into(),
            ));
        }
        let value = parse_value(value_str.trim())?;
        props.insert(key.to_string(), value);
    }
    Ok(props)
}

fn value_to_string(value: Value, context: &str) -> Result<String, SqlDatabaseError> {
    match value {
        Value::Text(s) => Ok(s),
        Value::Integer(i) => Ok(i.to_string()),
        Value::Float(f) => Ok(f.to_string()),
        Value::Boolean(b) => Ok(b.to_string()),
        Value::Timestamp(ts) => Ok(ts.to_rfc3339()),
        Value::Null => Err(SqlDatabaseError::Parse(
            format!("{context} cannot be NULL",),
        )),
    }
}

fn parse_create_graph(input: &str) -> Result<Statement, SqlDatabaseError> {
    let (name, rest) = parse_identifier(input)?;
    ensure_no_trailing_tokens(rest)?;
    Ok(Statement::CreateGraph { name })
}

fn parse_create_node(input: &str) -> Result<Statement, SqlDatabaseError> {
    let (graph, rest) = parse_identifier(input)?;
    let (id_value, mut remainder) = {
        let rest = expect_keyword_ci(rest, "ID")?;
        parse_literal_token(rest)?
    };
    let id = value_to_string(id_value, "node id")?;
    let mut properties = HashMap::new();
    remainder = remainder.trim_start();
    if !remainder.is_empty() {
        let (props, rest) = parse_properties_clause(remainder)?;
        properties = props;
        remainder = rest;
    }
    ensure_no_trailing_tokens(remainder)?;
    Ok(Statement::CreateNode {
        graph,
        id,
        properties,
    })
}

fn parse_create_edge(input: &str) -> Result<Statement, SqlDatabaseError> {
    let (graph, rest) = parse_identifier(input)?;
    let rest = expect_keyword_ci(rest, "FROM")?;
    let (from_value, rest) = parse_literal_token(rest)?;
    let from = value_to_string(from_value, "edge source")?;
    let rest = expect_keyword_ci(rest, "TO")?;
    let (to_value, mut remainder) = parse_literal_token(rest)?;
    let to = value_to_string(to_value, "edge target")?;
    let mut label = None;
    remainder = remainder.trim_start();
    if let Some(rest) = strip_keyword_ci(remainder, "LABEL") {
        let (label_value, rest_after) = parse_literal_token(rest)?;
        label = Some(value_to_string(label_value, "edge label")?);
        remainder = rest_after;
    }
    remainder = remainder.trim_start();
    let mut properties = HashMap::new();
    if !remainder.is_empty() {
        let (props, rest_after) = parse_properties_clause(remainder)?;
        properties = props;
        remainder = rest_after;
    }
    ensure_no_trailing_tokens(remainder)?;
    Ok(Statement::CreateEdge {
        graph,
        from,
        to,
        label,
        properties,
    })
}

fn parse_match_graph(input: &str) -> Result<Statement, SqlDatabaseError> {
    let (graph, mut remainder) = parse_identifier(input)?;
    let mut from = None;
    let mut to = None;
    let mut label = None;
    let mut property_filter = None;
    loop {
        let trimmed = remainder.trim_start();
        if trimmed.is_empty() {
            remainder = trimmed;
            break;
        }
        if let Some(rest) = strip_keyword_ci(trimmed, "FROM") {
            if from.is_some() {
                return Err(SqlDatabaseError::Parse(
                    "FROM clause specified multiple times".into(),
                ));
            }
            let (value, rest_after) = parse_literal_token(rest)?;
            from = Some(value_to_string(value, "match source")?);
            remainder = rest_after;
            continue;
        }
        if let Some(rest) = strip_keyword_ci(trimmed, "TO") {
            if to.is_some() {
                return Err(SqlDatabaseError::Parse(
                    "TO clause specified multiple times".into(),
                ));
            }
            let (value, rest_after) = parse_literal_token(rest)?;
            to = Some(value_to_string(value, "match destination")?);
            remainder = rest_after;
            continue;
        }
        if let Some(rest) = strip_keyword_ci(trimmed, "LABEL") {
            if label.is_some() {
                return Err(SqlDatabaseError::Parse(
                    "LABEL clause specified multiple times".into(),
                ));
            }
            let (value, rest_after) = parse_literal_token(rest)?;
            label = Some(value_to_string(value, "match label")?);
            remainder = rest_after;
            continue;
        }
        if let Some(_) = strip_keyword_ci(trimmed, "WITH") {
            if property_filter.is_some() {
                return Err(SqlDatabaseError::Parse(
                    "properties filter specified multiple times".into(),
                ));
            }
            let (props, rest_after) = parse_properties_clause(trimmed)?;
            property_filter = Some(props);
            remainder = rest_after;
            continue;
        }
        return Err(SqlDatabaseError::Parse(
            "unexpected token in MATCH GRAPH statement".into(),
        ));
    }
    ensure_no_trailing_tokens(remainder)?;
    Ok(Statement::MatchGraph {
        graph,
        from,
        to,
        label,
        property_filter,
    })
}

fn parse_select(input: &str) -> Result<Statement, SqlDatabaseError> {
    let (columns_raw, rest) = split_keyword_ci(input, "FROM")
        .ok_or_else(|| SqlDatabaseError::Parse("missing FROM clause".into()))?;
    let columns_raw = columns_raw.trim();
    let select_columns = if columns_raw == "*" {
        SelectColumns::All
    } else {
        let mut items = Vec::new();
        for part in split_comma(columns_raw)? {
            items.push(parse_select_item(&part)?);
        }
        SelectColumns::Some(items)
    };
    let (table, remainder) = parse_identifier(rest)?;
    let remainder = remainder.trim();
    let predicate = if remainder.is_empty() {
        None
    } else {
        let rest = expect_keyword_ci(remainder, "WHERE")?;
        let (column, rest) = parse_identifier(rest)?;
        let rest = rest.trim_start();
        if let Some(rest) = strip_keyword_ci(rest, "BETWEEN") {
            let (start, rest) = parse_literal_token(rest)?;
            let rest = expect_keyword_ci(rest, "AND")?;
            let (end, rest) = parse_literal_token(rest)?;
            ensure_no_trailing_tokens(rest)?;
            Some(Predicate::Between { column, start, end })
        } else {
            if !rest.starts_with('=') {
                return Err(SqlDatabaseError::Parse(
                    "expected '=' in WHERE clause".into(),
                ));
            }
            let (value, rest) = parse_literal_token(&rest[1..])?;
            ensure_no_trailing_tokens(rest)?;
            Some(Predicate::Equals { column, value })
        }
    };
    Ok(Statement::Select {
        table,
        columns: select_columns,
        predicate,
    })
}

fn parse_select_item(input: &str) -> Result<SelectItem, SqlDatabaseError> {
    let trimmed = input.trim();
    if trimmed.is_empty() {
        return Err(SqlDatabaseError::Parse("empty select item".into()));
    }

    if let Some(window) = parse_window_function(trimmed)? {
        return Ok(SelectItem::WindowFunction(window));
    }

    if split_alias_keyword(trimmed).is_some() {
        return Err(SqlDatabaseError::Parse(
            "aliases are only supported for window functions".into(),
        ));
    }

    let (column, rest) = parse_identifier(trimmed)?;
    ensure_no_trailing_tokens(rest)?;
    Ok(SelectItem::Column(column))
}

fn parse_window_function(input: &str) -> Result<Option<WindowFunctionExpr>, SqlDatabaseError> {
    let mut alias = None;
    let mut expression = input.trim().to_string();
    if let Some((before, after)) = split_alias_keyword(input) {
        let alias_part = after.trim_start();
        if alias_part.is_empty() {
            return Err(SqlDatabaseError::Parse("missing alias name".into()));
        }
        let (alias_ident, rest) = parse_identifier(alias_part)?;
        ensure_no_trailing_tokens(rest)?;
        alias = Some(alias_ident);
        expression = before.trim_end().to_string();
    }

    if let Some(rest) = strip_keyword_ci(&expression, "ROW_NUMBER") {
        let mut remainder = rest.trim_start();
        if !remainder.starts_with('(') {
            return Err(SqlDatabaseError::Parse(
                "expected '()' after ROW_NUMBER".into(),
            ));
        }
        remainder = remainder[1..].trim_start();
        if !remainder.starts_with(')') {
            return Err(SqlDatabaseError::Parse(
                "ROW_NUMBER does not accept arguments".into(),
            ));
        }
        remainder = &remainder[1..];
        remainder = expect_keyword_ci(remainder, "OVER")?;
        let remainder = remainder.trim_start();
        let (window_raw, rest_after) = take_parenthesized(remainder)?;
        ensure_no_trailing_tokens(rest_after)?;

        let mut spec_remainder = window_raw.as_str();
        spec_remainder = spec_remainder.trim_start();
        if spec_remainder.is_empty() {
            return Err(SqlDatabaseError::Parse(
                "OVER clause requires ORDER BY".into(),
            ));
        }
        let mut partition_by = None;
        if let Some(rest) = strip_keyword_ci(spec_remainder, "PARTITION") {
            let rest = expect_keyword_ci(rest, "BY")?;
            let (column, rest_after) = parse_identifier(rest)?;
            partition_by = Some(column);
            spec_remainder = rest_after.trim_start();
        }
        let rest = expect_keyword_ci(spec_remainder, "ORDER")?;
        let rest = expect_keyword_ci(rest, "BY")?;
        let (order_by, rest_after) = parse_identifier(rest)?;
        ensure_no_trailing_tokens(rest_after)?;

        return Ok(Some(WindowFunctionExpr {
            function: WindowFunctionType::RowNumber,
            partition_by,
            order_by,
            alias,
        }));
    }

    if alias.is_some() {
        return Err(SqlDatabaseError::Parse(
            "aliases are only supported for window functions".into(),
        ));
    }

    Ok(None)
}

fn parse_column_type(input: &str) -> Result<ColumnType, SqlDatabaseError> {
    match input.to_ascii_uppercase().as_str() {
        "INT" | "INTEGER" | "BIGINT" | "SMALLINT" | "TINYINT" => Ok(ColumnType::Integer),
        "FLOAT" | "REAL" | "DOUBLE" | "NUMERIC" | "DECIMAL" => Ok(ColumnType::Float),
        "TEXT" | "STRING" | "VARCHAR" | "CHAR" => Ok(ColumnType::Text),
        "BOOL" | "BOOLEAN" => Ok(ColumnType::Boolean),
        "TIMESTAMP" | "DATETIME" => Ok(ColumnType::Timestamp),
        other => Err(SqlDatabaseError::Parse(format!(
            "unsupported column type '{other}'"
        ))),
    }
}

fn parse_value(token: &str) -> Result<Value, SqlDatabaseError> {
    let trimmed = token.trim();
    if trimmed.is_empty() {
        return Err(SqlDatabaseError::Parse("missing value".into()));
    }
    if trimmed.starts_with('"') && trimmed.ends_with('"') && trimmed.len() >= 2 {
        return Ok(Value::Text(trimmed[1..trimmed.len() - 1].to_string()));
    }
    if trimmed.starts_with('\'') && trimmed.ends_with('\'') && trimmed.len() >= 2 {
        let inner = trimmed[1..trimmed.len() - 1].replace("\\'", "'");
        return Ok(Value::Text(inner));
    }
    match trimmed.to_ascii_lowercase().as_str() {
        "true" => return Ok(Value::Boolean(true)),
        "false" => return Ok(Value::Boolean(false)),
        "null" => return Ok(Value::Null),
        _ => {}
    }
    if let Ok(int) = trimmed.parse::<i64>() {
        return Ok(Value::Integer(int));
    }
    if let Ok(float) = trimmed.parse::<f64>() {
        return Ok(Value::Float(float));
    }
    Err(SqlDatabaseError::Parse(format!(
        "unable to parse literal '{trimmed}'"
    )))
}

fn strip_keyword_ci<'a>(input: &'a str, keyword: &str) -> Option<&'a str> {
    let trimmed = input.trim_start();
    if trimmed.len() < keyword.len() {
        return None;
    }
    if trimmed[..keyword.len()].eq_ignore_ascii_case(keyword) {
        Some(&trimmed[keyword.len()..])
    } else {
        None
    }
}

fn expect_keyword_ci<'a>(input: &'a str, keyword: &str) -> Result<&'a str, SqlDatabaseError> {
    strip_keyword_ci(input, keyword)
        .ok_or_else(|| SqlDatabaseError::Parse(format!("expected keyword '{keyword}'")))
}

fn parse_identifier(input: &str) -> Result<(String, &str), SqlDatabaseError> {
    let trimmed = input.trim_start();
    if trimmed.is_empty() {
        return Err(SqlDatabaseError::Parse("missing identifier".into()));
    }
    let offset = input.len() - trimmed.len();
    let mut end = trimmed.len();
    for (idx, ch) in trimmed.char_indices() {
        if idx == 0 {
            if !is_identifier_start(ch) {
                return Err(SqlDatabaseError::Parse("invalid identifier".into()));
            }
        } else if !is_identifier_part(ch) {
            end = idx;
            break;
        }
    }
    let ident = &trimmed[..end];
    if ident.is_empty() {
        return Err(SqlDatabaseError::Parse("invalid identifier".into()));
    }
    let rest_index = offset + end;
    Ok((ident.to_string(), &input[rest_index..]))
}

fn is_identifier_start(ch: char) -> bool {
    ch.is_ascii_alphabetic() || ch == '_'
}

fn is_identifier_part(ch: char) -> bool {
    ch.is_ascii_alphanumeric() || ch == '_'
}

fn take_parenthesized(input: &str) -> Result<(String, &str), SqlDatabaseError> {
    let trimmed = input.trim_start();
    let offset = input.len() - trimmed.len();
    let mut chars = trimmed.char_indices();
    let first = chars
        .next()
        .ok_or_else(|| SqlDatabaseError::Parse("expected '('".into()))?;
    if first.1 != '(' {
        return Err(SqlDatabaseError::Parse("expected '('".into()));
    }
    let mut depth = 1usize;
    let mut in_string = false;
    for (idx, ch) in chars {
        match ch {
            '\'' if !in_string => {
                in_string = true;
            }
            '\'' if in_string => {
                in_string = false;
            }
            '(' if !in_string => depth += 1,
            ')' if !in_string => {
                depth -= 1;
                if depth == 0 {
                    let inner = &trimmed[1..idx];
                    let rest_index = offset + idx + 1;
                    return Ok((inner.to_string(), &input[rest_index..]));
                }
            }
            _ => {}
        }
    }
    Err(SqlDatabaseError::Parse("unclosed '('".into()))
}

fn ensure_no_trailing_tokens(input: &str) -> Result<(), SqlDatabaseError> {
    if !input.trim().is_empty() {
        return Err(SqlDatabaseError::Parse(
            "unexpected tokens after statement".into(),
        ));
    }
    Ok(())
}

fn split_comma(input: &str) -> Result<Vec<String>, SqlDatabaseError> {
    let mut parts = Vec::new();
    let mut current = String::new();
    let mut depth = 0usize;
    let mut in_string = false;
    let mut chars = input.chars().peekable();
    while let Some(ch) = chars.next() {
        match ch {
            '\'' => {
                current.push(ch);
                in_string = !in_string;
            }
            '(' if !in_string => {
                depth += 1;
                current.push(ch);
            }
            ')' if !in_string => {
                if depth == 0 {
                    return Err(SqlDatabaseError::Parse("unmatched ')'".into()));
                }
                depth -= 1;
                current.push(ch);
            }
            ',' if !in_string && depth == 0 => {
                parts.push(current.trim().to_string());
                current.clear();
            }
            _ => current.push(ch),
        }
    }
    if depth != 0 || in_string {
        return Err(SqlDatabaseError::Parse("unterminated expression".into()));
    }
    if !current.trim().is_empty() {
        parts.push(current.trim().to_string());
    }
    Ok(parts)
}

fn split_keyword_ci<'a>(input: &'a str, keyword: &str) -> Option<(String, &'a str)> {
    let mut in_string = false;
    let upper = keyword.to_ascii_uppercase();
    let bytes = input.as_bytes();
    let key_bytes = upper.as_bytes();
    let mut i = 0;
    while i + key_bytes.len() <= bytes.len() {
        let ch = bytes[i] as char;
        if ch == '\'' {
            in_string = !in_string;
            i += 1;
            continue;
        }
        if !in_string {
            let candidate = &input[i..i + key_bytes.len()];
            if candidate.eq_ignore_ascii_case(keyword) {
                let before = input[..i].to_string();
                let after = &input[i + key_bytes.len()..];
                return Some((before, after));
            }
        }
        i += 1;
    }
    None
}

fn split_alias_keyword(input: &str) -> Option<(String, &str)> {
    let mut in_string = false;
    let mut depth = 0usize;
    let bytes = input.as_bytes();
    let mut i = 0usize;
    while i + 2 <= bytes.len() {
        let ch = bytes[i] as char;
        match ch {
            '\'' => {
                in_string = !in_string;
                i += 1;
                continue;
            }
            '(' if !in_string => depth += 1,
            ')' if !in_string && depth > 0 => depth -= 1,
            _ => {}
        }
        if !in_string && depth == 0 && input[i..i + 2].eq_ignore_ascii_case("AS") {
            if i == 0 || !bytes[i - 1].is_ascii_whitespace() {
                i += 1;
                continue;
            }
            let after = &input[i + 2..];
            if after
                .chars()
                .next()
                .map(|c| c.is_whitespace())
                .unwrap_or(false)
            {
                return Some((input[..i].to_string(), after));
            }
        }
        i += 1;
    }
    None
}

fn partition_key_string(value: &Value) -> String {
    match value {
        Value::Null => "__null__".to_string(),
        Value::Integer(i) => format!("I:{i}"),
        Value::Float(f) => format!("F:{f}"),
        Value::Text(s) => format!("T:{s}"),
        Value::Boolean(b) => format!("B:{b}"),
        Value::Timestamp(ts) => format!("TS:{}", ts.to_rfc3339()),
    }
}

fn compare_values(left: &Value, right: &Value) -> Option<Ordering> {
    match (left, right) {
        (Value::Null, _) | (_, Value::Null) => None,
        (Value::Integer(a), Value::Integer(b)) => Some(a.cmp(b)),
        (Value::Float(a), Value::Float(b)) => a.partial_cmp(b),
        (Value::Integer(a), Value::Float(b)) => (*a as f64).partial_cmp(b),
        (Value::Float(a), Value::Integer(b)) => a.partial_cmp(&(*b as f64)),
        (Value::Text(a), Value::Text(b)) => Some(a.cmp(b)),
        (Value::Boolean(a), Value::Boolean(b)) => Some(a.cmp(b)),
        (Value::Timestamp(a), Value::Timestamp(b)) => Some(a.cmp(b)),
        _ => None,
    }
}

fn coerce_value(value: Value, target: ColumnType) -> Result<Value, SqlDatabaseError> {
    match target {
        ColumnType::Integer => match value {
            Value::Integer(i) => Ok(Value::Integer(i)),
            Value::Float(f) => {
                if (f - f.trunc()).abs() < f64::EPSILON {
                    Ok(Value::Integer(f as i64))
                } else {
                    Err(SqlDatabaseError::SchemaMismatch(
                        "cannot store non integral float in INTEGER column".into(),
                    ))
                }
            }
            Value::Text(s) => s.parse::<i64>().map(Value::Integer).map_err(|_| {
                SqlDatabaseError::SchemaMismatch("failed to parse string as INTEGER".into())
            }),
            Value::Boolean(b) => Ok(Value::Integer(if b { 1 } else { 0 })),
            Value::Null => Ok(Value::Null),
            Value::Timestamp(_) => Err(SqlDatabaseError::SchemaMismatch(
                "cannot coerce TIMESTAMP to INTEGER".into(),
            )),
        },
        ColumnType::Float => match value {
            Value::Integer(i) => Ok(Value::Float(i as f64)),
            Value::Float(f) => Ok(Value::Float(f)),
            Value::Text(s) => s.parse::<f64>().map(Value::Float).map_err(|_| {
                SqlDatabaseError::SchemaMismatch("failed to parse string as FLOAT".into())
            }),
            Value::Boolean(b) => Ok(Value::Float(if b { 1.0 } else { 0.0 })),
            Value::Null => Ok(Value::Null),
            Value::Timestamp(_) => Err(SqlDatabaseError::SchemaMismatch(
                "cannot coerce TIMESTAMP to FLOAT".into(),
            )),
        },
        ColumnType::Text => match value {
            Value::Text(s) => Ok(Value::Text(s)),
            Value::Integer(i) => Ok(Value::Text(i.to_string())),
            Value::Float(f) => Ok(Value::Text(f.to_string())),
            Value::Boolean(b) => Ok(Value::Text(b.to_string())),
            Value::Null => Ok(Value::Null),
            Value::Timestamp(ts) => Ok(Value::Text(ts.to_rfc3339())),
        },
        ColumnType::Boolean => match value {
            Value::Boolean(b) => Ok(Value::Boolean(b)),
            Value::Integer(i) => Ok(Value::Boolean(i != 0)),
            Value::Float(f) => Ok(Value::Boolean(f != 0.0)),
            Value::Text(s) => match s.to_ascii_lowercase().as_str() {
                "true" | "t" | "1" => Ok(Value::Boolean(true)),
                "false" | "f" | "0" => Ok(Value::Boolean(false)),
                _ => Err(SqlDatabaseError::SchemaMismatch(
                    "failed to parse string as BOOLEAN".into(),
                )),
            },
            Value::Null => Ok(Value::Null),
            Value::Timestamp(_) => Err(SqlDatabaseError::SchemaMismatch(
                "cannot coerce TIMESTAMP to BOOLEAN".into(),
            )),
        },
        ColumnType::Timestamp => match value {
            Value::Timestamp(ts) => Ok(Value::Timestamp(ts)),
            Value::Text(s) => parse_timestamp_string(&s)
                .map(Value::Timestamp)
                .ok_or_else(|| {
                    SqlDatabaseError::SchemaMismatch("failed to parse string as TIMESTAMP".into())
                }),
            Value::Integer(i) => timestamp_from_integer(i)
                .map(Value::Timestamp)
                .ok_or_else(|| {
                    SqlDatabaseError::SchemaMismatch(
                        "failed to convert integer to TIMESTAMP".into(),
                    )
                }),
            Value::Float(f) => timestamp_from_float(f)
                .map(Value::Timestamp)
                .ok_or_else(|| {
                    SqlDatabaseError::SchemaMismatch("failed to convert float to TIMESTAMP".into())
                }),
            Value::Boolean(_) => Err(SqlDatabaseError::SchemaMismatch(
                "cannot coerce BOOLEAN to TIMESTAMP".into(),
            )),
            Value::Null => Ok(Value::Null),
        },
    }
}

fn coerce_static_value(value: &Value, target: ColumnType) -> Result<Value, SqlDatabaseError> {
    coerce_value(value.clone(), target)
}

fn equal(left: &Value, right: &Value) -> bool {
    match (left, right) {
        (Value::Null, _) | (_, Value::Null) => false,
        (Value::Integer(a), Value::Integer(b)) => a == b,
        (Value::Float(a), Value::Float(b)) => (a - b).abs() < f64::EPSILON,
        (Value::Integer(a), Value::Float(b)) => (*a as f64 - b).abs() < f64::EPSILON,
        (Value::Float(a), Value::Integer(b)) => (a - *b as f64).abs() < f64::EPSILON,
        (Value::Text(a), Value::Text(b)) => a == b,
        (Value::Boolean(a), Value::Boolean(b)) => a == b,
        (Value::Boolean(a), Value::Integer(b)) => (*a && *b != 0) || (!*a && *b == 0),
        (Value::Integer(a), Value::Boolean(b)) => (*a != 0 && *b) || (*a == 0 && !*b),
        (Value::Boolean(a), Value::Float(b)) => (*a && *b != 0.0) || (!*a && *b == 0.0),
        (Value::Float(a), Value::Boolean(b)) => (*a != 0.0 && *b) || (*a == 0.0 && !*b),
        (Value::Text(a), Value::Boolean(b)) => match a.to_ascii_lowercase().as_str() {
            "true" | "t" | "1" => *b,
            "false" | "f" | "0" => !*b,
            _ => false,
        },
        (Value::Boolean(a), Value::Text(b)) => match b.to_ascii_lowercase().as_str() {
            "true" | "t" | "1" => *a,
            "false" | "f" | "0" => !*a,
            _ => false,
        },
        (Value::Text(a), Value::Integer(b)) => a.parse::<i64>().map(|v| v == *b).unwrap_or(false),
        (Value::Integer(a), Value::Text(b)) => b.parse::<i64>().map(|v| v == *a).unwrap_or(false),
        (Value::Text(a), Value::Float(b)) => a
            .parse::<f64>()
            .map(|v| (v - b).abs() < f64::EPSILON)
            .unwrap_or(false),
        (Value::Float(a), Value::Text(b)) => b
            .parse::<f64>()
            .map(|v| (a - v).abs() < f64::EPSILON)
            .unwrap_or(false),
        (Value::Timestamp(a), Value::Timestamp(b)) => a == b,
        (Value::Timestamp(a), Value::Text(b)) => parse_timestamp_string(b)
            .map(|parsed| parsed == *a)
            .unwrap_or(false),
        (Value::Text(a), Value::Timestamp(b)) => parse_timestamp_string(a)
            .map(|parsed| parsed == *b)
            .unwrap_or(false),
        (Value::Timestamp(a), Value::Integer(b)) => timestamp_from_integer(*b)
            .map(|parsed| parsed == *a)
            .unwrap_or(false),
        (Value::Integer(a), Value::Timestamp(b)) => timestamp_from_integer(*a)
            .map(|parsed| parsed == *b)
            .unwrap_or(false),
        (Value::Timestamp(a), Value::Float(b)) => timestamp_from_float(*b)
            .map(|parsed| parsed == *a)
            .unwrap_or(false),
        (Value::Float(a), Value::Timestamp(b)) => timestamp_from_float(*a)
            .map(|parsed| parsed == *b)
            .unwrap_or(false),
        _ => false,
    }
}

fn format_properties(props: &HashMap<String, Value>) -> String {
    if props.is_empty() {
        return "{}".into();
    }
    let mut entries: Vec<_> = props.iter().collect();
    entries.sort_by(|a, b| a.0.cmp(b.0));
    let parts = entries
        .into_iter()
        .map(|(key, value)| format!("{key}: {}", format_value(value)))
        .collect::<Vec<_>>();
    format!("{{{}}}", parts.join(", "))
}

fn format_value(value: &Value) -> String {
    match value {
        Value::Integer(i) => i.to_string(),
        Value::Float(f) => f.to_string(),
        Value::Text(s) => format!("\"{s}\""),
        Value::Boolean(b) => b.to_string(),
        Value::Timestamp(ts) => ts.to_rfc3339(),
        Value::Null => "NULL".into(),
    }
}

fn parse_timestamp_string(value: &str) -> Option<DateTime<Utc>> {
    if let Ok(dt) = DateTime::parse_from_rfc3339(value) {
        return Some(dt.with_timezone(&Utc));
    }
    if let Ok(dt) = NaiveDateTime::parse_from_str(value, "%Y-%m-%d %H:%M:%S") {
        return Some(Utc.from_utc_datetime(&dt));
    }
    if let Ok(dt) = NaiveDateTime::parse_from_str(value, "%Y-%m-%d %H:%M:%S%.f") {
        return Some(Utc.from_utc_datetime(&dt));
    }
    if let Ok(date) = NaiveDate::parse_from_str(value, "%Y-%m-%d") {
        if let Some(dt) = date.and_hms_opt(0, 0, 0) {
            return Some(Utc.from_utc_datetime(&dt));
        }
    }
    None
}

fn timestamp_from_integer(value: i64) -> Option<DateTime<Utc>> {
    Utc.timestamp_opt(value, 0).single()
}

fn timestamp_from_float(value: f64) -> Option<DateTime<Utc>> {
    if !value.is_finite() {
        return None;
    }
    let seconds = value.trunc() as i64;
    Utc.timestamp_opt(seconds, 0).single()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn create_insert_select_flow() {
        let mut db = SqlDatabase::new();
        db.execute("CREATE TABLE users (id INT, name TEXT, active BOOLEAN);")
            .unwrap();
        db.execute("INSERT INTO users VALUES (1, 'Alice', true);")
            .unwrap();
        db.execute("INSERT INTO users VALUES (2, 'Bob', false);")
            .unwrap();

        let result = db
            .execute("SELECT id, name FROM users WHERE active = true;")
            .unwrap();
        match result {
            QueryResult::Rows { columns, rows } => {
                assert_eq!(columns, vec!["id".to_string(), "name".to_string()]);
                assert_eq!(rows.len(), 1);
                assert_eq!(
                    rows[0],
                    vec![Value::Integer(1), Value::Text("Alice".into())]
                );
            }
            other => panic!("unexpected result {other:?}"),
        }
    }

    #[test]
    fn insert_with_columns_subset() {
        let mut db = SqlDatabase::new();
        db.execute("CREATE TABLE metrics (name TEXT, value FLOAT, updated BOOLEAN);")
            .unwrap();
        db.execute("INSERT INTO metrics (name, value) VALUES ('latency', 2.5);")
            .unwrap();
        let result = db.execute("SELECT * FROM metrics;").unwrap();
        match result {
            QueryResult::Rows { columns, rows } => {
                assert_eq!(columns, vec!["name", "value", "updated"]);
                assert_eq!(rows.len(), 1);
                assert_eq!(rows[0][0], Value::Text("latency".into()));
                assert_eq!(rows[0][1], Value::Float(2.5));
                assert_eq!(rows[0][2], Value::Null);
            }
            _ => panic!("unexpected result"),
        }
    }

    #[test]
    fn time_series_partition_and_range_query() {
        let mut db = SqlDatabase::new();
        db.execute("CREATE TABLE readings (ts TIMESTAMP, value INT) PARTITION BY DAY(ts);")
            .unwrap();
        db.execute("INSERT INTO readings VALUES ('2023-01-01T12:00:00Z', 10);")
            .unwrap();
        db.execute("INSERT INTO readings VALUES ('2023-01-02T08:00:00Z', 20);")
            .unwrap();
        db.execute("INSERT INTO readings VALUES ('2023-01-03T09:30:00Z', 30);")
            .unwrap();

        let result = db
            .execute(
                "SELECT value FROM readings WHERE ts BETWEEN '2023-01-01T00:00:00Z' AND '2023-01-02T23:59:59Z';",
            )
            .unwrap();
        match result {
            QueryResult::Rows { rows, .. } => {
                assert_eq!(rows.len(), 2);
                assert_eq!(rows[0], vec![Value::Integer(10)]);
                assert_eq!(rows[1], vec![Value::Integer(20)]);
            }
            _ => panic!("unexpected result"),
        }
    }

    #[test]
    fn graph_creation_and_matching() {
        let mut db = SqlDatabase::new();
        db.execute("CREATE GRAPH social;").unwrap();
        db.execute("CREATE NODE social ID 'alice' WITH PROPERTIES (role='admin');")
            .unwrap();
        db.execute("CREATE NODE social ID 'bob';").unwrap();
        db.execute(
            "CREATE EDGE social FROM 'alice' TO 'bob' LABEL 'FOLLOWS' WITH PROPERTIES (since='2023-01-01');",
        )
        .unwrap();

        let result = db
            .execute("MATCH GRAPH social FROM 'alice' LABEL 'FOLLOWS';")
            .unwrap();
        match result {
            QueryResult::Rows { columns, rows } => {
                assert_eq!(columns, vec!["from", "to", "label", "properties"]);
                assert_eq!(rows.len(), 1);
                assert_eq!(rows[0][0], Value::Text("alice".into()));
                assert_eq!(rows[0][1], Value::Text("bob".into()));
                assert_eq!(rows[0][2], Value::Text("FOLLOWS".into()));
            }
            _ => panic!("unexpected result"),
        }

        let filtered = db
            .execute("MATCH GRAPH social FROM 'alice' WITH PROPERTIES (since='2023-01-01');")
            .unwrap();
        match filtered {
            QueryResult::Rows { rows, .. } => {
                assert_eq!(rows.len(), 1);
            }
            _ => panic!("unexpected result"),
        }
    }

    #[test]
    fn insert_multiple_rows() {
        let mut db = SqlDatabase::new();
        db.execute("CREATE TABLE counts (id INT);").unwrap();
        db.execute("INSERT INTO counts VALUES (1), (2), (3);")
            .unwrap();
        let result = db.execute("SELECT * FROM counts;").unwrap();
        match result {
            QueryResult::Rows { rows, .. } => {
                assert_eq!(rows.len(), 3);
                assert_eq!(rows[0][0], Value::Integer(1));
                assert_eq!(rows[1][0], Value::Integer(2));
                assert_eq!(rows[2][0], Value::Integer(3));
            }
            _ => panic!("unexpected result"),
        }
    }

    #[test]
    fn schema_mismatch_errors() {
        let mut db = SqlDatabase::new();
        db.execute("CREATE TABLE events (id INT, note TEXT);")
            .unwrap();
        let err = db
            .execute("INSERT INTO events VALUES ('abc', 'note');")
            .unwrap_err();
        match err {
            SqlDatabaseError::SchemaMismatch(_) => {}
            other => panic!("unexpected error {other:?}"),
        }
    }

    #[test]
    fn row_number_window_function_basic() {
        let mut db = SqlDatabase::new();
        db.execute("CREATE TABLE metrics (id INT);").unwrap();
        db.execute("INSERT INTO metrics VALUES (2), (1), (3);")
            .unwrap();

        let result = db
            .execute("SELECT id, ROW_NUMBER() OVER (ORDER BY id) AS rn FROM metrics;")
            .unwrap();
        match result {
            QueryResult::Rows { columns, rows } => {
                assert_eq!(columns, vec!["id", "rn"]);
                assert_eq!(rows.len(), 3);
                assert_eq!(rows[0], vec![Value::Integer(2), Value::Integer(2)]);
                assert_eq!(rows[1], vec![Value::Integer(1), Value::Integer(1)]);
                assert_eq!(rows[2], vec![Value::Integer(3), Value::Integer(3)]);
            }
            _ => panic!("unexpected result"),
        }
    }

    #[test]
    fn row_number_with_partitioning() {
        let mut db = SqlDatabase::new();
        db.execute("CREATE TABLE sales (region TEXT, amount INT);")
            .unwrap();
        db.execute("INSERT INTO sales VALUES ('east', 100), ('east', 50), ('west', 75);")
            .unwrap();

        let result = db
            .execute(
                "SELECT region, ROW_NUMBER() OVER (PARTITION BY region ORDER BY amount) AS rn FROM sales;",
            )
            .unwrap();
        match result {
            QueryResult::Rows { columns, rows } => {
                assert_eq!(columns, vec!["region", "rn"]);
                assert_eq!(rows.len(), 3);
                assert_eq!(rows[0], vec![Value::Text("east".into()), Value::Integer(2)]);
                assert_eq!(rows[1], vec![Value::Text("east".into()), Value::Integer(1)]);
                assert_eq!(rows[2], vec![Value::Text("west".into()), Value::Integer(1)]);
            }
            _ => panic!("unexpected result"),
        }
    }

    #[test]
    fn row_number_default_alias() {
        let mut db = SqlDatabase::new();
        db.execute("CREATE TABLE items (id INT);").unwrap();
        db.execute("INSERT INTO items VALUES (1);").unwrap();

        let result = db
            .execute("SELECT ROW_NUMBER() OVER (ORDER BY id) FROM items;")
            .unwrap();
        match result {
            QueryResult::Rows { columns, rows } => {
                assert_eq!(columns, vec!["row_number"]);
                assert_eq!(rows, vec![vec![Value::Integer(1)]]);
            }
            _ => panic!("unexpected result"),
        }
    }
}
