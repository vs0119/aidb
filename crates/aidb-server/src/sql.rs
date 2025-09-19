use std::collections::HashMap;

use aidb_core::{Id, JsonValue};
use uuid::Uuid;

#[derive(Debug, Clone)]
pub enum SqlStatement {
    CreateCollection(CreateCollectionStatement),
    Insert(InsertStatement),
    Search(SearchStatement),
}

#[derive(Debug, Clone)]
pub struct CreateCollectionStatement {
    pub name: String,
    pub dim: usize,
    pub metric: Option<String>,
    pub wal_dir: Option<String>,
    pub index: Option<String>,
    pub hnsw: Option<HnswParams>,
}

#[derive(Debug, Clone, Default)]
pub struct HnswParams {
    pub m: Option<usize>,
    pub ef_construction: Option<usize>,
    pub ef_search: Option<usize>,
}

impl HnswParams {
    fn is_empty(&self) -> bool {
        self.m.is_none() && self.ef_construction.is_none() && self.ef_search.is_none()
    }
}

#[derive(Debug, Clone)]
pub struct InsertStatement {
    pub collection: String,
    pub id: Option<Id>,
    pub vector: Vec<f32>,
    pub payload: Option<JsonValue>,
}

#[derive(Debug, Clone)]
pub struct SearchStatement {
    pub collection: String,
    pub vector: Vec<f32>,
    pub top_k: usize,
    pub filter: Option<HashMap<String, JsonValue>>,
}

#[derive(Debug)]
pub struct SqlParseError {
    msg: String,
}

impl SqlParseError {
    fn new(msg: impl Into<String>) -> Self {
        Self { msg: msg.into() }
    }
}

impl std::fmt::Display for SqlParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.msg)
    }
}

impl std::error::Error for SqlParseError {}

pub fn parse(input: &str) -> Result<SqlStatement, SqlParseError> {
    let trimmed = input.trim();
    if trimmed.is_empty() {
        return Err(SqlParseError::new("empty SQL statement"));
    }
    let trimmed = trimmed.trim_end_matches(';').trim();
    if let Some(rest) = strip_keyword(trimmed, "CREATE") {
        let rest = expect_keyword(rest, "COLLECTION")?;
        parse_create_collection(rest)
    } else if let Some(rest) = strip_keyword(trimmed, "INSERT") {
        let rest = expect_keyword(rest, "INTO")?;
        parse_insert(rest)
    } else if let Some(rest) = strip_keyword(trimmed, "SEARCH") {
        parse_search(rest)
    } else {
        Err(SqlParseError::new("unsupported SQL statement"))
    }
}

fn parse_create_collection(input: &str) -> Result<SqlStatement, SqlParseError> {
    let (name, rest) = parse_identifier(input)?;
    let (params, remainder) = take_parenthesized(rest)?;
    ensure_no_trailing_tokens(remainder)?;
    let assignments = parse_assignments(&params)?;

    let mut dim: Option<usize> = None;
    let mut metric: Option<String> = None;
    let mut wal_dir: Option<String> = None;
    let mut index: Option<String> = None;
    let mut hnsw = HnswParams::default();

    for (key, value) in assignments {
        match key.as_str() {
            "DIM" => dim = Some(parse_usize(&value)?),
            "METRIC" => metric = Some(parse_string_value(&value)?.to_lowercase()),
            "WAL_DIR" => wal_dir = Some(parse_string_value(&value)?),
            "INDEX" => index = Some(parse_string_value(&value)?.to_lowercase()),
            "M" => hnsw.m = Some(parse_usize(&value)?),
            "EF_CONSTRUCTION" => hnsw.ef_construction = Some(parse_usize(&value)?),
            "EF_SEARCH" => hnsw.ef_search = Some(parse_usize(&value)?),
            other => return Err(SqlParseError::new(format!("unknown parameter '{other}'"))),
        }
    }

    let dim = dim.ok_or_else(|| SqlParseError::new("DIM is required"))?;
    let hnsw = if hnsw.is_empty() { None } else { Some(hnsw) };

    Ok(SqlStatement::CreateCollection(CreateCollectionStatement {
        name,
        dim,
        metric,
        wal_dir,
        index,
        hnsw,
    }))
}

fn parse_insert(input: &str) -> Result<SqlStatement, SqlParseError> {
    let (collection, rest) = parse_identifier(input)?;
    let rest = expect_keyword(rest, "VALUES")?;
    let (params, remainder) = take_parenthesized(rest)?;
    ensure_no_trailing_tokens(remainder)?;
    let assignments = parse_assignments(&params)?;

    let mut id: Option<Id> = None;
    let mut vector: Option<Vec<f32>> = None;
    let mut payload: Option<JsonValue> = None;

    for (key, value) in assignments {
        match key.as_str() {
            "ID" => id = Some(parse_uuid(&value)?),
            "VECTOR" => vector = Some(parse_vector(&value)?),
            "PAYLOAD" => payload = Some(parse_json_value(&value)?),
            other => return Err(SqlParseError::new(format!("unknown field '{other}'"))),
        }
    }

    let vector = vector.ok_or_else(|| SqlParseError::new("VECTOR is required"))?;

    Ok(SqlStatement::Insert(InsertStatement {
        collection,
        id,
        vector,
        payload,
    }))
}

fn parse_search(input: &str) -> Result<SqlStatement, SqlParseError> {
    let (collection, rest) = parse_identifier(input)?;
    let (params, remainder) = take_parenthesized(rest)?;
    ensure_no_trailing_tokens(remainder)?;
    let assignments = parse_assignments(&params)?;

    let mut vector: Option<Vec<f32>> = None;
    let mut top_k: Option<usize> = None;
    let mut filter: Option<HashMap<String, JsonValue>> = None;

    for (key, value) in assignments {
        match key.as_str() {
            "VECTOR" => vector = Some(parse_vector(&value)?),
            "TOPK" | "TOP_K" => top_k = Some(parse_usize(&value)?),
            "FILTER" => filter = Some(parse_filter(&value)?),
            other => return Err(SqlParseError::new(format!("unknown field '{other}'"))),
        }
    }

    let vector = vector.ok_or_else(|| SqlParseError::new("VECTOR is required"))?;
    let top_k = top_k.ok_or_else(|| SqlParseError::new("TOPK is required"))?;

    Ok(SqlStatement::Search(SearchStatement {
        collection,
        vector,
        top_k,
        filter,
    }))
}

fn parse_assignments(input: &str) -> Result<Vec<(String, String)>, SqlParseError> {
    let mut out = Vec::new();
    for part in split_top_level(input) {
        if part.is_empty() {
            continue;
        }
        let Some(idx) = part.find('=') else {
            return Err(SqlParseError::new("expected '=' in assignment"));
        };
        let key = part[..idx].trim();
        let value = part[idx + 1..].trim();
        if key.is_empty() || value.is_empty() {
            return Err(SqlParseError::new("invalid assignment"));
        }
        out.push((key.to_ascii_uppercase(), value.to_string()));
    }
    Ok(out)
}

fn parse_identifier(input: &str) -> Result<(String, &str), SqlParseError> {
    let trimmed = input.trim_start();
    if trimmed.is_empty() {
        return Err(SqlParseError::new("expected identifier"));
    }
    if trimmed.starts_with('"') {
        let rest = &trimmed[1..];
        if let Some(end) = rest.find('"') {
            let name = &rest[..end];
            let remainder = &rest[end + 1..];
            return Ok((name.to_string(), remainder));
        } else {
            return Err(SqlParseError::new("unterminated identifier"));
        }
    }
    let ident: String = trimmed
        .chars()
        .take_while(|ch| ch.is_alphanumeric() || *ch == '_' || *ch == '-')
        .collect();
    if ident.is_empty() {
        return Err(SqlParseError::new("expected identifier"));
    }
    let len = ident.len();
    Ok((ident, &trimmed[len..]))
}

fn take_parenthesized(input: &str) -> Result<(String, &str), SqlParseError> {
    let trimmed = input.trim_start();
    if !trimmed.starts_with('(') {
        return Err(SqlParseError::new("expected '('"));
    }
    let mut depth = 0usize;
    let mut in_single = false;
    let mut in_double = false;
    let mut start_inner = None;
    for (idx, ch) in trimmed.char_indices() {
        match ch {
            '\'' if !in_double => in_single = !in_single,
            '"' if !in_single => in_double = !in_double,
            '(' if !in_single && !in_double => {
                depth += 1;
                if depth == 1 {
                    start_inner = Some(idx + ch.len_utf8());
                }
            }
            ')' if !in_single && !in_double => {
                if depth == 0 {
                    return Err(SqlParseError::new("unexpected ')'"));
                }
                depth -= 1;
                if depth == 0 {
                    let start = start_inner.ok_or_else(|| SqlParseError::new("expected '('"))?;
                    let inside = &trimmed[start..idx];
                    let remainder = &trimmed[idx + ch.len_utf8()..];
                    return Ok((inside.trim().to_string(), remainder));
                }
            }
            _ => {}
        }
    }
    Err(SqlParseError::new("unterminated parenthesis"))
}

fn split_top_level(input: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut depth_paren = 0usize;
    let mut depth_brace = 0usize;
    let mut depth_bracket = 0usize;
    let mut in_single = false;
    let mut in_double = false;
    let mut start = 0usize;
    let bytes = input.as_bytes();
    let mut idx = 0usize;
    while idx < bytes.len() {
        let ch = bytes[idx] as char;
        match ch {
            '\'' if !in_double => in_single = !in_single,
            '"' if !in_single => in_double = !in_double,
            '(' if !in_single && !in_double => depth_paren += 1,
            ')' if !in_single && !in_double => {
                if depth_paren > 0 {
                    depth_paren -= 1;
                }
            }
            '{' if !in_single && !in_double => depth_brace += 1,
            '}' if !in_single && !in_double => {
                if depth_brace > 0 {
                    depth_brace -= 1;
                }
            }
            '[' if !in_single && !in_double => depth_bracket += 1,
            ']' if !in_single && !in_double => {
                if depth_bracket > 0 {
                    depth_bracket -= 1;
                }
            }
            ',' if !in_single
                && !in_double
                && depth_paren == 0
                && depth_brace == 0
                && depth_bracket == 0 =>
            {
                let part = input[start..idx].trim();
                if !part.is_empty() {
                    out.push(part.to_string());
                }
                start = idx + 1;
            }
            _ => {}
        }
        idx += 1;
    }
    let tail = input[start..].trim();
    if !tail.is_empty() {
        out.push(tail.to_string());
    }
    out
}

fn strip_keyword<'a>(input: &'a str, keyword: &str) -> Option<&'a str> {
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

fn expect_keyword<'a>(input: &'a str, keyword: &str) -> Result<&'a str, SqlParseError> {
    strip_keyword(input, keyword).ok_or_else(|| SqlParseError::new(format!("expected '{keyword}'")))
}

fn ensure_no_trailing_tokens(input: &str) -> Result<(), SqlParseError> {
    if input.trim().is_empty() {
        Ok(())
    } else {
        Err(SqlParseError::new("unexpected tokens after statement"))
    }
}

fn parse_string_value(value: &str) -> Result<String, SqlParseError> {
    let trimmed = value.trim();
    if trimmed.starts_with('"') || trimmed.starts_with('\'') {
        if trimmed.len() < 2 {
            return Err(SqlParseError::new("unterminated string"));
        }
        let last = trimmed
            .chars()
            .last()
            .ok_or_else(|| SqlParseError::new("unterminated string"))?;
        if last != trimmed.chars().next().unwrap() {
            return Err(SqlParseError::new("unterminated string"));
        }
        Ok(trimmed[1..trimmed.len() - 1].to_string())
    } else {
        Ok(trimmed.to_string())
    }
}

fn parse_usize(value: &str) -> Result<usize, SqlParseError> {
    let text = parse_string_value(value)?;
    text.parse::<usize>()
        .map_err(|_| SqlParseError::new("expected integer value"))
}

fn parse_uuid(value: &str) -> Result<Id, SqlParseError> {
    let text = parse_string_value(value)?;
    Uuid::parse_str(&text).map_err(|_| SqlParseError::new("invalid UUID"))
}

fn parse_vector(value: &str) -> Result<Vec<f32>, SqlParseError> {
    let text = parse_string_value(value)?;
    serde_json::from_str::<Vec<f32>>(&text)
        .map_err(|e| SqlParseError::new(format!("invalid vector: {e}")))
}

fn parse_json_value(value: &str) -> Result<JsonValue, SqlParseError> {
    let text = parse_string_value(value)?;
    serde_json::from_str(&text).map_err(|e| SqlParseError::new(format!("invalid JSON: {e}")))
}

fn parse_filter(value: &str) -> Result<HashMap<String, JsonValue>, SqlParseError> {
    let json = parse_json_value(value)?;
    match json {
        JsonValue::Object(map) => Ok(map.into_iter().collect()),
        _ => Err(SqlParseError::new("FILTER must be a JSON object")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;

    #[test]
    fn parse_create_collection_basic() {
        let stmt = parse(
            "CREATE COLLECTION docs (DIM = 4, METRIC = 'cosine', INDEX = 'hnsw', WAL_DIR = './data', M = 16, EF_CONSTRUCTION = 200, EF_SEARCH = 64)",
        )
        .unwrap();
        match stmt {
            SqlStatement::CreateCollection(c) => {
                assert_eq!(c.name, "docs");
                assert_eq!(c.dim, 4);
                assert_eq!(c.metric.as_deref(), Some("cosine"));
                assert_eq!(c.index.as_deref(), Some("hnsw"));
                assert_eq!(c.wal_dir.as_deref(), Some("./data"));
                let h = c.hnsw.unwrap();
                assert_eq!(h.m, Some(16));
                assert_eq!(h.ef_construction, Some(200));
                assert_eq!(h.ef_search, Some(64));
            }
            _ => panic!("expected create"),
        }
    }

    #[test]
    fn parse_insert_statement() {
        let stmt = parse(
            "INSERT INTO docs VALUES (ID = '550e8400-e29b-41d4-a716-446655440000', VECTOR = [1.0, 0.5], PAYLOAD = {\"source\": \"a\"})",
        )
        .unwrap();
        match stmt {
            SqlStatement::Insert(i) => {
                assert_eq!(i.collection, "docs");
                assert_eq!(
                    i.id.unwrap(),
                    Uuid::parse_str("550e8400-e29b-41d4-a716-446655440000").unwrap()
                );
                assert_eq!(i.vector, vec![1.0, 0.5]);
                let payload = i.payload.unwrap();
                assert_eq!(payload["source"], Value::String("a".to_string()));
            }
            _ => panic!("expected insert"),
        }
    }

    #[test]
    fn parse_search_statement() {
        let stmt =
            parse("SEARCH docs (VECTOR = [0.1, 0.2], TOPK = 5, FILTER = {\"source\": \"a\"})")
                .unwrap();
        match stmt {
            SqlStatement::Search(s) => {
                assert_eq!(s.collection, "docs");
                assert_eq!(s.vector, vec![0.1, 0.2]);
                assert_eq!(s.top_k, 5);
                let filter = s.filter.unwrap();
                assert_eq!(filter["source"], Value::String("a".to_string()));
            }
            _ => panic!("expected search"),
        }
    }

    #[test]
    fn parse_rejects_unknown_statement() {
        assert!(parse("DROP TABLE docs").is_err());
    }
}
