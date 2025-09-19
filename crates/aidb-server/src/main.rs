use std::collections::HashMap;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;

use aidb_core::{Id, JsonValue, MetadataFilter, Metric, Vector};
use aidb_index_bf::BruteForceIndex;
use aidb_index_hnsw::{HnswIndex, HnswParams};
use aidb_storage::{Collection, Engine};
use axum::{
    extract::Path,
    extract::State,
    http::StatusCode,
    routing::{delete, get, post},
    Json, Router,
};
use chrono::{DateTime, SecondsFormat, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

mod sql;

#[derive(serde::Serialize, serde::Deserialize)]
enum AnyIndex {
    Bruteforce(BruteForceIndex),
    Hnsw(HnswIndex),
}

impl aidb_core::VectorIndex for AnyIndex {
    fn add(&mut self, id: Id, vector: Vector, payload: Option<JsonValue>) {
        match self {
            AnyIndex::Bruteforce(i) => i.add(id, vector, payload),
            AnyIndex::Hnsw(i) => i.add(id, vector, payload),
        }
    }
    fn remove(&mut self, id: &Id) -> bool {
        match self {
            AnyIndex::Bruteforce(i) => i.remove(id),
            AnyIndex::Hnsw(i) => i.remove(id),
        }
    }
    fn search(
        &self,
        vector: &[f32],
        top_k: usize,
        metric: Metric,
        filter: Option<&aidb_core::MetadataFilter>,
    ) -> Vec<aidb_core::SearchResult> {
        match self {
            AnyIndex::Bruteforce(i) => i.search(vector, top_k, metric, filter),
            AnyIndex::Hnsw(i) => i.search(vector, top_k, metric, filter),
        }
    }
    fn len(&self) -> usize {
        match self {
            AnyIndex::Bruteforce(i) => i.len(),
            AnyIndex::Hnsw(i) => i.len(),
        }
    }
    fn dim(&self) -> usize {
        match self {
            AnyIndex::Bruteforce(i) => i.dim(),
            AnyIndex::Hnsw(i) => i.dim(),
        }
    }
}

#[derive(Clone)]
struct AppState {
    engine: Arc<Engine<AnyIndex>>,
}

fn format_system_time(ts: std::time::SystemTime) -> Option<String> {
    Some(DateTime::<Utc>::from(ts).to_rfc3339_opts(SecondsFormat::Secs, true))
}

#[derive(Deserialize)]
struct CreateCollectionReq {
    name: String,
    dim: usize,
    metric: Option<String>,
    wal_dir: Option<String>,
    index: Option<String>,
    hnsw: Option<HnswParamsReq>,
}

#[derive(Deserialize)]
struct UpsertPointReq {
    id: Option<Uuid>,
    vector: Vector,
    payload: Option<JsonValue>,
}

#[derive(Deserialize, Clone, Copy)]
struct HnswParamsReq {
    m: Option<usize>,
    ef_construction: Option<usize>,
    ef_search: Option<usize>,
}

#[derive(Deserialize)]
struct SearchReq {
    vector: Vector,
    top_k: usize,
    #[allow(dead_code)]
    filter: Option<HashMap<String, JsonValue>>,
}

#[derive(Deserialize)]
struct BatchSearchReq {
    queries: Vec<Vector>,
    top_k: usize,
    #[allow(dead_code)]
    filter: Option<HashMap<String, JsonValue>>,
}

#[derive(Serialize)]
struct CreateCollectionResp {
    ok: bool,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    env_logger::init();
    let engine = Arc::new(Engine::<AnyIndex>::new());

    let app = Router::new()
        .route("/health", get(health))
        .route("/metrics", get(metrics))
        .route(
            "/collections",
            post(create_collection).get(list_collections),
        )
        .route("/collections/:name/points", post(upsert_point))
        .route("/collections/:name/points:batch", post(upsert_points_batch))
        .route("/collections/:name/search", post(search_points))
        .route("/collections/:name/search:batch", post(search_points_batch))
        .route("/sql", post(exec_sql))
        .route("/collections/:name", get(collection_info))
        .route("/collections/:name/snapshot", post(snapshot_collection))
        .route("/collections/:name/compact", post(compact_collection))
        .route("/collections/:name/params", post(update_params))
        .route("/collections/:name/points/:id", delete(delete_point))
        .with_state(AppState { engine });

    let addr: SocketAddr = "0.0.0.0:8080".parse().unwrap();
    println!("aidb-server listening on {addr}");
    let listener = tokio::net::TcpListener::bind(&addr).await?;
    axum::serve(listener, app).await?;
    Ok(())
}

async fn health() -> &'static str {
    "ok"
}

async fn metrics(
    State(state): State<AppState>,
) -> Result<(StatusCode, String), (StatusCode, String)> {
    let map = state.engine.collections.read();
    let mut out = String::new();
    out.push_str("# TYPE aidb_wal_size_bytes gauge\n");
    out.push_str("# TYPE aidb_wal_seconds_since_truncate gauge\n");
    out.push_str("# TYPE aidb_wal_bytes_since_truncate counter\n");
    for (name, coll) in map.iter() {
        let stats = coll.wal_stats();
        let since = stats
            .last_truncate
            .and_then(|ts| ts.elapsed().ok())
            .map(|d| d.as_secs_f64())
            .unwrap_or(f64::INFINITY);
        let escaped = name.replace('"', "\\\"");
        out.push_str(&format!(
            "aidb_wal_size_bytes{{collection=\"{}\"}} {}\n",
            escaped, stats.size_bytes
        ));
        out.push_str(&format!(
            "aidb_wal_seconds_since_truncate{{collection=\"{}\"}} {}\n",
            escaped, since
        ));
        out.push_str(&format!(
            "aidb_wal_bytes_since_truncate{{collection=\"{}\"}} {}\n",
            escaped, stats.bytes_since_truncate
        ));
    }
    Ok((StatusCode::OK, out))
}

async fn create_collection(
    State(state): State<AppState>,
    Json(req): Json<CreateCollectionReq>,
) -> Result<Json<CreateCollectionResp>, (StatusCode, String)> {
    create_collection_internal(&state, req)?;
    Ok(Json(CreateCollectionResp { ok: true }))
}

fn create_collection_internal(
    state: &AppState,
    req: CreateCollectionReq,
) -> Result<(), (StatusCode, String)> {
    let metric = match req.metric.as_deref() {
        Some("cosine") | None => Metric::Cosine,
        Some("euclidean") => Metric::Euclidean,
        Some(other) => return Err((StatusCode::BAD_REQUEST, format!("unknown metric {other}"))),
    };
    let wal_dir = req.wal_dir.unwrap_or_else(|| "./data".to_string());
    let wal_path = PathBuf::from(wal_dir).join(format!("{}.wal", req.name));
    let index_kind = req.index.as_deref().unwrap_or("hnsw");
    let coll = match index_kind {
        "bruteforce" => {
            let index = AnyIndex::Bruteforce(BruteForceIndex::new(req.dim));
            Collection::new(&req.name, req.dim, metric, index, wal_path).map_err(int_err)?
        }
        "hnsw" => {
            let hp = req.hnsw.unwrap_or(HnswParamsReq {
                m: None,
                ef_construction: None,
                ef_search: None,
            });
            let params = HnswParams {
                m: hp.m.unwrap_or(16),
                ef_construction: hp.ef_construction.unwrap_or(200),
                ef_search: hp.ef_search.unwrap_or(50),
            };
            let index = HnswIndex::try_new(req.dim, metric, params)
                .map_err(|e| (StatusCode::BAD_REQUEST, e.to_string()))?;
            let index = AnyIndex::Hnsw(index);
            Collection::new(&req.name, req.dim, metric, index, wal_path).map_err(int_err)?
        }
        other => return Err((StatusCode::BAD_REQUEST, format!("unknown index {other}"))),
    };
    coll.recover().map_err(int_err)?;
    state.engine.insert_collection(coll);
    Ok(())
}

async fn upsert_point(
    State(state): State<AppState>,
    Path(name): Path<String>,
    Json(req): Json<UpsertPointReq>,
) -> Result<Json<Id>, (StatusCode, String)> {
    let id = upsert_point_internal(&state, &name, req)?;
    Ok(Json(id))
}

fn upsert_point_internal(
    state: &AppState,
    name: &str,
    req: UpsertPointReq,
) -> Result<Id, (StatusCode, String)> {
    let id = req.id.unwrap_or_else(Uuid::new_v4);
    let coll = get_collection(state, name)?;
    coll.upsert(id, req.vector, req.payload).map_err(int_err)?;
    Ok(id)
}

#[derive(Deserialize)]
struct UpsertPointsBatchReq {
    points: Vec<UpsertPointReq>,
}

async fn upsert_points_batch(
    State(state): State<AppState>,
    Path(name): Path<String>,
    Json(req): Json<UpsertPointsBatchReq>,
) -> Result<Json<Vec<Id>>, (StatusCode, String)> {
    let mut ids = Vec::with_capacity(req.points.len());
    for p in req.points {
        let id = upsert_point_internal(&state, &name, p)?;
        ids.push(id);
    }
    Ok(Json(ids))
}

async fn search_points(
    State(state): State<AppState>,
    Path(name): Path<String>,
    Json(req): Json<SearchReq>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    let coll = get_collection(&state, &name)?;
    let filter = build_filter(&req.filter);
    let res = search_collection(&coll, &req.vector, req.top_k, filter)?;
    Ok(Json(serde_json::json!({ "results": res })))
}

async fn search_points_batch(
    State(state): State<AppState>,
    Path(name): Path<String>,
    Json(req): Json<BatchSearchReq>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    let coll = get_collection(&state, &name)?;
    let filter = build_filter(&req.filter);
    let mut results = Vec::with_capacity(req.queries.len());
    for (idx, query) in req.queries.iter().enumerate() {
        if query.len() != coll.dim() {
            return Err((
                StatusCode::BAD_REQUEST,
                format!(
                    "query at index {idx} has dimension {}, expected {}",
                    query.len(),
                    coll.dim()
                ),
            ));
        }
        results.push(coll.search(query, req.top_k, filter.as_ref()));
    }
    Ok(Json(serde_json::json!({ "results": results })))
}

async fn delete_point(
    State(state): State<AppState>,
    Path((name, id)): Path<(String, Uuid)>,
) -> Result<Json<bool>, (StatusCode, String)> {
    let coll = get_collection(&state, &name)?;
    let ok = coll.remove(id).map_err(int_err)?;
    Ok(Json(ok))
}

#[derive(Serialize)]
struct CollectionInfoResp {
    name: String,
    dim: usize,
    metric: String,
    len: usize,
    wal_size_bytes: u64,
    wal_last_truncate: Option<String>,
    wal_bytes_since_truncate: u64,
}

async fn collection_info(
    State(state): State<AppState>,
    Path(name): Path<String>,
) -> Result<Json<CollectionInfoResp>, (StatusCode, String)> {
    let coll = get_collection(&state, &name)?;
    let metric = match coll.metric() {
        Metric::Cosine => "cosine",
        Metric::Euclidean => "euclidean",
    };
    let stats = coll.wal_stats();
    Ok(Json(CollectionInfoResp {
        name,
        dim: coll.dim(),
        metric: metric.to_string(),
        len: coll.len(),
        wal_size_bytes: stats.size_bytes,
        wal_last_truncate: stats.last_truncate.and_then(format_system_time),
        wal_bytes_since_truncate: stats.bytes_since_truncate,
    }))
}

async fn list_collections(
    State(state): State<AppState>,
) -> Result<Json<Vec<CollectionInfoResp>>, (StatusCode, String)> {
    let map = state.engine.collections.read();
    let mut out = Vec::new();
    for (name, coll) in map.iter() {
        let metric = match coll.metric() {
            Metric::Cosine => "cosine",
            Metric::Euclidean => "euclidean",
        };
        let stats = coll.wal_stats();
        out.push(CollectionInfoResp {
            name: name.clone(),
            dim: coll.dim(),
            metric: metric.to_string(),
            len: coll.len(),
            wal_size_bytes: stats.size_bytes,
            wal_last_truncate: stats.last_truncate.and_then(format_system_time),
            wal_bytes_since_truncate: stats.bytes_since_truncate,
        });
    }
    Ok(Json(out))
}

#[derive(Deserialize)]
struct SnapshotReq {
    path: Option<String>,
}

async fn snapshot_collection(
    State(state): State<AppState>,
    Path(name): Path<String>,
    Json(req): Json<SnapshotReq>,
) -> Result<Json<bool>, (StatusCode, String)> {
    let coll = get_collection(&state, &name)?;
    let path = req
        .path
        .unwrap_or_else(|| format!("./data/{}.snapshot", name));
    // Works because BruteForceIndex implements serde
    coll.snapshot_to(&path).map_err(int_err)?;
    Ok(Json(true))
}

fn int_err<E: std::fmt::Display>(e: E) -> (StatusCode, String) {
    (StatusCode::INTERNAL_SERVER_ERROR, e.to_string())
}

#[derive(Deserialize)]
struct UpdateParamsReq {
    ef_search: Option<usize>,
}

async fn compact_collection(
    State(state): State<AppState>,
    Path(name): Path<String>,
) -> Result<Json<bool>, (StatusCode, String)> {
    let coll = get_collection(&state, &name)?;
    let ok = coll.with_index_mut(|idx| match idx {
        AnyIndex::Hnsw(h) => {
            h.rebuild();
            true
        }
        AnyIndex::Bruteforce(_) => false,
    });
    Ok(Json(ok))
}

async fn update_params(
    State(state): State<AppState>,
    Path(name): Path<String>,
    Json(req): Json<UpdateParamsReq>,
) -> Result<Json<bool>, (StatusCode, String)> {
    let coll = get_collection(&state, &name)?;
    let ok = coll.with_index_mut(|idx| match idx {
        AnyIndex::Hnsw(h) => {
            if let Some(ef) = req.ef_search {
                h.set_ef_search(ef);
            }
            true
        }
        AnyIndex::Bruteforce(_) => false,
    });
    Ok(Json(ok))
}

fn get_collection(
    state: &AppState,
    name: &str,
) -> Result<Arc<Collection<AnyIndex>>, (StatusCode, String)> {
    let map = state.engine.collections.read();
    map.get(name)
        .cloned()
        .ok_or_else(|| (StatusCode::NOT_FOUND, "collection not found".into()))
}

fn build_filter(filter: &Option<HashMap<String, JsonValue>>) -> Option<MetadataFilter> {
    filter.as_ref().and_then(|map| {
        if map.is_empty() {
            None
        } else {
            Some(MetadataFilter {
                equals: map.iter().map(|(k, v)| (k.clone(), v.clone())).collect(),
            })
        }
    })
}

fn build_filter_owned(filter: Option<HashMap<String, JsonValue>>) -> Option<MetadataFilter> {
    filter.and_then(|map| {
        if map.is_empty() {
            None
        } else {
            Some(MetadataFilter {
                equals: map.into_iter().collect(),
            })
        }
    })
}

fn search_collection(
    coll: &Arc<Collection<AnyIndex>>,
    vector: &[f32],
    top_k: usize,
    filter: Option<MetadataFilter>,
) -> Result<Vec<aidb_core::SearchResult>, (StatusCode, String)> {
    if vector.len() != coll.dim() {
        return Err((
            StatusCode::BAD_REQUEST,
            format!(
                "expected vector dimension {} but got {}",
                coll.dim(),
                vector.len()
            ),
        ));
    }
    let filter_ref = filter.as_ref();
    Ok(coll.search(vector, top_k, filter_ref))
}

#[derive(Deserialize)]
struct SqlReq {
    query: String,
}

#[derive(Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum SqlResp {
    Create {
        ok: bool,
    },
    Insert {
        id: Id,
    },
    Search {
        results: Vec<aidb_core::SearchResult>,
    },
}

async fn exec_sql(
    State(state): State<AppState>,
    Json(req): Json<SqlReq>,
) -> Result<Json<SqlResp>, (StatusCode, String)> {
    let stmt = sql::parse(&req.query).map_err(|e| (StatusCode::BAD_REQUEST, e.to_string()))?;
    match stmt {
        sql::SqlStatement::CreateCollection(stmt) => {
            let hnsw = stmt.hnsw.map(|h| HnswParamsReq {
                m: h.m,
                ef_construction: h.ef_construction,
                ef_search: h.ef_search,
            });
            let req = CreateCollectionReq {
                name: stmt.name,
                dim: stmt.dim,
                metric: stmt.metric,
                wal_dir: stmt.wal_dir,
                index: stmt.index,
                hnsw,
            };
            create_collection_internal(&state, req)?;
            Ok(Json(SqlResp::Create { ok: true }))
        }
        sql::SqlStatement::Insert(stmt) => {
            let req = UpsertPointReq {
                id: stmt.id,
                vector: stmt.vector,
                payload: stmt.payload,
            };
            let id = upsert_point_internal(&state, &stmt.collection, req)?;
            Ok(Json(SqlResp::Insert { id }))
        }
        sql::SqlStatement::Search(stmt) => {
            let coll = get_collection(&state, &stmt.collection)?;
            let filter = build_filter_owned(stmt.filter);
            let results = search_collection(&coll, &stmt.vector, stmt.top_k, filter)?;
            Ok(Json(SqlResp::Search { results }))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use aidb_core::Metric;
    use aidb_index_bf::BruteForceIndex;

    fn temp_path() -> PathBuf {
        std::env::temp_dir().join(format!("aidb-metrics-{}", uuid::Uuid::new_v4()))
    }

    #[tokio::test]
    async fn metrics_endpoint_outputs_prometheus_format() {
        let tmp = temp_path();
        let _ = std::fs::create_dir_all(&tmp);
        let wal_path = tmp.join("col.wal");
        let index = AnyIndex::Bruteforce(BruteForceIndex::new(4));
        let collection = Collection::new("col", 4, Metric::Cosine, index, &wal_path).unwrap();
        collection
            .upsert(uuid::Uuid::new_v4(), vec![0.1, 0.2, 0.3, 0.4], None)
            .unwrap();

        let engine = Arc::new(Engine::<AnyIndex>::new());
        engine.insert_collection(collection);
        let state = AppState { engine };

        let (status, body) = metrics(State(state)).await.unwrap();
        assert_eq!(status, StatusCode::OK);
        assert!(body.contains("aidb_wal_size_bytes{collection=\"col\"}"));
        assert!(body.contains("aidb_wal_seconds_since_truncate{collection=\"col\"}"));
        assert!(body.contains("aidb_wal_bytes_since_truncate{collection=\"col\"}"));

        let _ = std::fs::remove_dir_all(&tmp);
    }
}
