use std::collections::HashMap;

use thiserror::Error;

#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    Integer(i64),
    Float(f64),
    Text(String),
    Boolean(bool),
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
}

#[derive(Debug, Default)]
struct Table {
    columns: Vec<Column>,
    rows: Vec<Vec<Value>>,
}

#[derive(Debug)]
pub struct SqlDatabase {
    tables: HashMap<String, Table>,
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
        }
    }

    pub fn execute(&mut self, sql: &str) -> Result<QueryResult, SqlDatabaseError> {
        let stmt = parse_statement(sql)?;
        match stmt {
            Statement::CreateTable { name, columns } => {
                self.exec_create_table(name, columns)?;
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
        }
    }

    fn exec_create_table(
        &mut self,
        name: String,
        columns: Vec<(String, ColumnType)>,
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
            table.rows.push(new_row);
            inserted += 1;
        }
        Ok(inserted)
    }

    fn exec_select(
        &self,
        table_name: String,
        columns: SelectColumns,
        predicate: Option<(String, Value)>,
    ) -> Result<QueryResult, SqlDatabaseError> {
        let table = self
            .tables
            .get(&table_name)
            .ok_or_else(|| SqlDatabaseError::UnknownTable(table_name.clone()))?;

        let projection = match columns {
            SelectColumns::All => (0..table.columns.len()).collect::<Vec<_>>(),
            SelectColumns::Some(cols) => cols
                .into_iter()
                .map(|name| self.column_index(table, &name))
                .collect::<Result<Vec<_>, _>>()?,
        };

        let mut rows = Vec::new();
        for row in &table.rows {
            if let Some((column, value)) = &predicate {
                let idx = self.column_index(table, column)?;
                if !equal(&row[idx], value) {
                    continue;
                }
            }
            rows.push(projection.iter().map(|&idx| row[idx].clone()).collect());
        }

        let column_names = projection
            .iter()
            .map(|&idx| table.columns[idx].name.clone())
            .collect();

        Ok(QueryResult::Rows {
            columns: column_names,
            rows,
        })
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
    },
    Insert {
        table: String,
        columns: Option<Vec<String>>,
        rows: Vec<Vec<Value>>,
    },
    Select {
        table: String,
        columns: SelectColumns,
        predicate: Option<(String, Value)>,
    },
}

#[derive(Debug)]
enum SelectColumns {
    All,
    Some(Vec<String>),
}

fn parse_statement(sql: &str) -> Result<Statement, SqlDatabaseError> {
    let trimmed = sql.trim();
    if trimmed.is_empty() {
        return Err(SqlDatabaseError::Parse("empty SQL statement".into()));
    }
    let trimmed = trimmed.trim_end_matches(';').trim();
    if let Some(rest) = strip_keyword_ci(trimmed, "CREATE") {
        let rest = expect_keyword_ci(rest, "TABLE")?;
        parse_create_table(rest)
    } else if let Some(rest) = strip_keyword_ci(trimmed, "INSERT") {
        let rest = expect_keyword_ci(rest, "INTO")?;
        parse_insert(rest)
    } else if let Some(rest) = strip_keyword_ci(trimmed, "SELECT") {
        parse_select(rest)
    } else {
        Err(SqlDatabaseError::Unsupported)
    }
}

fn parse_create_table(input: &str) -> Result<Statement, SqlDatabaseError> {
    let (name, rest) = parse_identifier(input)?;
    let (columns_raw, remainder) = take_parenthesized(rest)?;
    ensure_no_trailing_tokens(remainder)?;
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
    Ok(Statement::CreateTable { name, columns })
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

fn parse_select(input: &str) -> Result<Statement, SqlDatabaseError> {
    let (columns_raw, rest) = split_keyword_ci(input, "FROM")
        .ok_or_else(|| SqlDatabaseError::Parse("missing FROM clause".into()))?;
    let columns_raw = columns_raw.trim();
    let select_columns = if columns_raw == "*" {
        SelectColumns::All
    } else {
        let cols = split_comma(columns_raw)?
            .into_iter()
            .map(|s| s.trim().to_string())
            .collect();
        SelectColumns::Some(cols)
    };
    let (table, remainder) = parse_identifier(rest)?;
    let remainder = remainder.trim();
    let predicate = if remainder.is_empty() {
        None
    } else {
        let rest = expect_keyword_ci(remainder, "WHERE")?;
        let (column, rest) = parse_identifier(rest)?;
        let rest = rest.trim_start();
        if !rest.starts_with('=') {
            return Err(SqlDatabaseError::Parse(
                "expected '=' in WHERE clause".into(),
            ));
        }
        let value_str = rest[1..].trim();
        if value_str.is_empty() {
            return Err(SqlDatabaseError::Parse(
                "missing literal in WHERE clause".into(),
            ));
        }
        let value = parse_value(value_str)?;
        Some((column, value))
    };
    Ok(Statement::Select {
        table,
        columns: select_columns,
        predicate,
    })
}

fn parse_column_type(input: &str) -> Result<ColumnType, SqlDatabaseError> {
    match input.to_ascii_uppercase().as_str() {
        "INT" | "INTEGER" | "BIGINT" | "SMALLINT" | "TINYINT" => Ok(ColumnType::Integer),
        "FLOAT" | "REAL" | "DOUBLE" | "NUMERIC" | "DECIMAL" => Ok(ColumnType::Float),
        "TEXT" | "STRING" | "VARCHAR" | "CHAR" => Ok(ColumnType::Text),
        "BOOL" | "BOOLEAN" => Ok(ColumnType::Boolean),
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
        },
        ColumnType::Float => match value {
            Value::Integer(i) => Ok(Value::Float(i as f64)),
            Value::Float(f) => Ok(Value::Float(f)),
            Value::Text(s) => s.parse::<f64>().map(Value::Float).map_err(|_| {
                SqlDatabaseError::SchemaMismatch("failed to parse string as FLOAT".into())
            }),
            Value::Boolean(b) => Ok(Value::Float(if b { 1.0 } else { 0.0 })),
            Value::Null => Ok(Value::Null),
        },
        ColumnType::Text => match value {
            Value::Text(s) => Ok(Value::Text(s)),
            Value::Integer(i) => Ok(Value::Text(i.to_string())),
            Value::Float(f) => Ok(Value::Text(f.to_string())),
            Value::Boolean(b) => Ok(Value::Text(b.to_string())),
            Value::Null => Ok(Value::Null),
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
        },
    }
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
    }
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
}
