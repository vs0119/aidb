use chrono::Datelike;
use chrono::{DateTime, Duration, NaiveDate, NaiveDateTime, TimeZone, Utc};
use roxmltree::Document;
use rstar::{PointDistance, RTree, RTreeObject, AABB};
use serde_json::Value as JsonValue;
use std::cell::RefCell;
use std::cmp::Ordering;
use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};
use std::fmt;
use std::sync::{Arc, Mutex};

use thiserror::Error;

mod execution;
mod external;
mod jit;
mod planner;

use execution::TaskScheduler;

pub use external::{ExternalCostFactors, ExternalSource, ParquetConnector};
pub use planner::JoinPredicate;

#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub enum Value {
    Integer(i64),
    Float(f64),
    Text(String),
    Boolean(bool),
    Timestamp(DateTime<Utc>),
    Json(JsonValue),
    Jsonb(JsonValue),
    Xml(String),
    Geometry(Geometry),
    Null,
}

#[derive(Debug, Clone)]
pub(crate) struct Column {
    pub(crate) name: String,
    pub(crate) ty: ColumnType,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColumnType {
    Integer,
    Float,
    Text,
    Boolean,
    Timestamp,
    Json,
    Jsonb,
    Xml,
    Geometry,
}

fn column_type_name(ty: ColumnType) -> &'static str {
    match ty {
        ColumnType::Integer => "INTEGER",
        ColumnType::Float => "FLOAT",
        ColumnType::Text => "TEXT",
        ColumnType::Boolean => "BOOLEAN",
        ColumnType::Timestamp => "TIMESTAMP",
        ColumnType::Json => "JSON",
        ColumnType::Jsonb => "JSONB",
        ColumnType::Xml => "XML",
        ColumnType::Geometry => "GEOMETRY",
    }
}

#[derive(Debug, Clone)]
pub(crate) struct Table {
    pub(crate) columns: Vec<Column>,
    pub(crate) rows: Vec<Vec<Value>>,
    pub(crate) partitioning: Option<TimePartitioning>,
    pub(crate) partitions: BTreeMap<PartitionKey, Vec<usize>>,
    pub(crate) json_indexes: HashMap<usize, JsonIndex>,
    pub(crate) jsonb_indexes: HashMap<usize, JsonIndex>,
    pub(crate) xml_indexes: HashMap<usize, XmlIndex>,
    pub(crate) spatial_indexes: HashMap<usize, SpatialIndex>,
}

unsafe impl Send for Table {}
unsafe impl Sync for Table {}

impl Default for Table {
    fn default() -> Self {
        Self {
            columns: Vec::new(),
            rows: Vec::new(),
            partitioning: None,
            partitions: BTreeMap::new(),
            json_indexes: HashMap::new(),
            jsonb_indexes: HashMap::new(),
            xml_indexes: HashMap::new(),
            spatial_indexes: HashMap::new(),
        }
    }
}

#[derive(Clone, Copy)]
struct TableHandle {
    ptr: usize,
    _marker: std::marker::PhantomData<*const Table>,
}

unsafe impl Send for TableHandle {}
unsafe impl Sync for TableHandle {}

impl TableHandle {
    fn new(table: &Table) -> Self {
        Self {
            ptr: table as *const Table as usize,
            _marker: std::marker::PhantomData,
        }
    }

    unsafe fn get(&self) -> &Table {
        &*(self.ptr as *const Table)
    }
}

impl Table {
    fn with_columns(columns: Vec<(String, ColumnType)>) -> Self {
        let mut table = Table::default();
        for (name, ty) in columns {
            table.columns.push(Column { name, ty });
        }
        initialize_secondary_indexes(&mut table);
        table
    }
}

#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub enum Geometry {
    Point {
        x: f64,
        y: f64,
    },
    BoundingBox {
        min_x: f64,
        min_y: f64,
        max_x: f64,
        max_y: f64,
    },
}

impl Geometry {
    fn to_aabb(&self) -> AABB<[f64; 2]> {
        match self {
            Geometry::Point { x, y } => AABB::from_point([*x, *y]),
            Geometry::BoundingBox {
                min_x,
                min_y,
                max_x,
                max_y,
            } => AABB::from_corners([*min_x, *min_y], [*max_x, *max_y]),
        }
    }
}

#[derive(Debug, Clone, Default)]
pub(crate) struct JsonIndex {
    pub(crate) map: HashMap<String, Vec<usize>>,
}

#[derive(Debug, Clone, Default)]
pub(crate) struct XmlIndex {
    pub(crate) map: HashMap<String, Vec<usize>>,
}

#[derive(Debug, Clone, Default)]
pub(crate) struct SpatialIndex {
    pub(crate) tree: RTree<SpatialIndexItem>,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct SpatialIndexItem {
    row: usize,
    bbox: AABB<[f64; 2]>,
}

impl RTreeObject for SpatialIndexItem {
    type Envelope = AABB<[f64; 2]>;

    fn envelope(&self) -> Self::Envelope {
        self.bbox
    }
}

impl PointDistance for SpatialIndexItem {
    fn distance_2(&self, point: &[f64; 2]) -> f64 {
        self.bbox.distance_2(point)
    }
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
pub(crate) struct TimePartitioning {
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

const TABLE_STATS_TABLE: &str = "__aidb_table_stats";
const COLUMN_STATS_TABLE: &str = "__aidb_column_stats";
const DEFAULT_HISTOGRAM_BUCKETS: usize = 10;

#[derive(Debug, Clone, PartialEq)]
pub struct TableStatistics {
    pub table_name: String,
    pub row_count: u64,
    pub analyzed_at: DateTime<Utc>,
    pub stats_version: u32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ColumnStatistics {
    pub table_name: String,
    pub column_name: String,
    pub data_type: String,
    pub null_count: u64,
    pub distinct_count: u64,
    pub min: Option<Value>,
    pub max: Option<Value>,
    pub histogram: Option<ColumnHistogram>,
    pub analyzed_at: DateTime<Utc>,
    pub stats_version: u32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ColumnHistogram {
    pub buckets: Vec<HistogramBucket>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct HistogramBucket {
    pub lower: f64,
    pub upper: f64,
    pub count: u64,
}

#[derive(Debug)]
pub struct SqlDatabase {
    tables: HashMap<String, Table>,
    system_tables: HashMap<String, Table>,
    external_tables: HashMap<String, ExternalTableEntry>,
    materialized_views: HashMap<String, MaterializedView>,
    materialized_view_dependencies: HashMap<String, HashSet<String>>,
    pending_view_refreshes: VecDeque<MaterializedViewRefreshRequest>,
    graphs: HashMap<String, Graph>,
    table_stats_catalog: HashMap<String, TableStatistics>,
    column_stats_catalog: HashMap<(String, String), ColumnStatistics>,
    plan_cache: RefCell<HashMap<PlanCacheKey, CachedSelectResult>>,
    plan_cache_index: RefCell<HashMap<String, HashSet<PlanCacheKey>>>,
    plan_cache_stats: RefCell<CacheStats>,
    execution_config: ExecutionConfig,
    scheduler: Option<Arc<TaskScheduler>>,
    execution_stats: Arc<Mutex<ExecutionStats>>,
    jit_manager: RefCell<jit::JitManager>,
}

struct ExternalTableEntry {
    schema: Table,
    connector: Arc<dyn ExternalSource>,
}

impl fmt::Debug for ExternalTableEntry {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ExternalTableEntry")
            .field("columns", &self.schema.columns.len())
            .finish()
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CacheStats {
    pub hits: usize,
    pub misses: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct PlanCacheKey {
    shape: String,
    params_fingerprint: String,
}

impl PlanCacheKey {
    fn new(shape: String, params: Vec<String>) -> Self {
        Self {
            shape,
            params_fingerprint: Self::normalize_params(params),
        }
    }

    fn normalize_params(params: Vec<String>) -> String {
        if params.is_empty() {
            return String::new();
        }
        let mut fingerprint = String::new();
        use std::fmt::Write;
        for param in params {
            let _ = write!(fingerprint, "{:08x}:{param};", param.len());
        }
        fingerprint
    }
}

#[derive(Debug, Clone)]
struct CachedBatchSlice {
    #[allow(dead_code)]
    start: usize,
    #[allow(dead_code)]
    len: usize,
}

#[derive(Debug, Clone)]
struct CachedSelectResult {
    result: QueryResult,
    #[allow(dead_code)]
    matching_indices: Option<Vec<usize>>,
    #[allow(dead_code)]
    batch_slices: Option<Vec<CachedBatchSlice>>,
}

#[derive(Debug, Clone)]
struct VectorizedSelectResult {
    result: QueryResult,
    batch_slices: Vec<CachedBatchSlice>,
}

#[derive(Debug, Clone)]
enum SelectExecutionOutcome {
    Row {
        result: QueryResult,
        matching_indices: Vec<usize>,
    },
    Vectorized(VectorizedSelectResult),
}

#[derive(Debug, Clone)]
struct PlanCacheMetadata {
    key: PlanCacheKey,
    dependent_tables: Vec<String>,
}

#[derive(Debug, Error)]
pub enum SqlDatabaseError {
    #[error("parse error: {0}")]
    Parse(String),
    #[error("table '{0}' already exists")]
    TableExists(String),
    #[error("table '{0}' does not exist")]
    UnknownTable(String),
    #[error("materialized view '{0}' already exists")]
    MaterializedViewExists(String),
    #[error("materialized view '{0}' does not exist")]
    UnknownMaterializedView(String),
    #[error("cannot modify materialized view '{0}' directly")]
    MaterializedViewModification(String),
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
    #[error("cannot modify system catalog table '{0}'")]
    SystemTableModification(String),
    #[error("unsupported statement")]
    Unsupported,
    #[error("JIT compilation failed: {0}")]
    JitCompilation(String),
    #[error("JIT operation not supported")]
    JitUnsupported,
    #[error("column '{0}' not found")]
    ColumnNotFound(String),
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExecutionMode {
    Auto,
    Vectorized,
    Row,
}

#[derive(Debug, Default, Clone)]
pub struct ExecutionStats {
    pub scan_input_rows: usize,
    pub scan_output_rows: usize,
    pub scan_partitions: usize,
    pub scan_skew_detected: bool,
    pub join_build_rows: usize,
    pub join_probe_rows: usize,
    pub join_output_rows: usize,
    pub join_skew_detected: bool,
    pub join_strategy_switches: usize,
}

#[derive(Debug, Clone)]
pub struct ExecutionConfig {
    pub batch_size: usize,
    pub max_parallelism: usize,
    pub skew_threshold: usize,
}

impl ExecutionConfig {
    pub fn new(batch_size: usize, max_parallelism: usize) -> Self {
        let batch = batch_size.max(1);
        let parallel = max_parallelism.max(1);
        Self {
            batch_size: batch,
            max_parallelism: parallel,
            skew_threshold: batch * 4,
        }
    }

    pub fn with_parallelism(max_parallelism: usize) -> Self {
        Self::new(execution::DEFAULT_BATCH_SIZE, max_parallelism)
    }
}

impl Default for ExecutionConfig {
    fn default() -> Self {
        let parallelism = num_cpus::get().max(1);
        Self {
            batch_size: execution::DEFAULT_BATCH_SIZE,
            max_parallelism: parallelism,
            skew_threshold: execution::DEFAULT_BATCH_SIZE * 4,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MaterializedViewRefreshMode {
    Synchronous,
    Asynchronous,
}

#[derive(Debug)]
struct MaterializedView {
    name: String,
    definition: SelectStatement,
    refresh_mode: MaterializedViewRefreshMode,
    table: Table,
    last_refreshed_at: Option<DateTime<Utc>>,
    pending_async_refresh: bool,
}

#[derive(Debug)]
struct MaterializedViewRefreshRequest {
    view: String,
    due: DateTime<Utc>,
}

const ASYNC_REFRESH_DELAY: Duration = Duration::milliseconds(50);

impl SqlDatabase {
    pub fn new() -> Self {
        let execution_config = ExecutionConfig::default();
        let scheduler = SqlDatabase::build_scheduler(execution_config.max_parallelism);
        let stats = Arc::new(Mutex::new(ExecutionStats::default()));
        let mut db = Self {
            tables: HashMap::new(),
            system_tables: HashMap::new(),
            external_tables: HashMap::new(),
            materialized_views: HashMap::new(),
            materialized_view_dependencies: HashMap::new(),
            pending_view_refreshes: VecDeque::new(),
            graphs: HashMap::new(),
            table_stats_catalog: HashMap::new(),
            column_stats_catalog: HashMap::new(),
            plan_cache: RefCell::new(HashMap::new()),
            plan_cache_index: RefCell::new(HashMap::new()),
            plan_cache_stats: RefCell::new(CacheStats::default()),
            execution_config,
            scheduler,
            execution_stats: stats,
            jit_manager: RefCell::new(jit::JitManager::new()),
        };
        db.initialize_system_catalog();
        db
    }

    pub fn register_external_table(
        &mut self,
        name: &str,
        connector: Arc<dyn ExternalSource>,
    ) -> Result<(), SqlDatabaseError> {
        if self.tables.contains_key(name)
            || self.system_tables.contains_key(name)
            || self.materialized_views.contains_key(name)
            || self.external_tables.contains_key(name)
        {
            return Err(SqlDatabaseError::TableExists(name.to_string()));
        }
        let columns = connector
            .schema()
            .into_iter()
            .map(|col| (col.name, col.ty))
            .collect();
        let schema = Table::with_columns(columns);
        self.external_tables
            .insert(name.to_string(), ExternalTableEntry { schema, connector });
        self.invalidate_cache_for_table(name);
        Ok(())
    }

    fn build_scheduler(parallelism: usize) -> Option<Arc<TaskScheduler>> {
        if parallelism > 1 {
            Some(Arc::new(TaskScheduler::new(parallelism)))
        } else {
            None
        }
    }

    pub fn execution_config(&self) -> &ExecutionConfig {
        &self.execution_config
    }

    pub fn set_execution_config(&mut self, config: ExecutionConfig) {
        if config.max_parallelism != self.execution_config.max_parallelism {
            self.scheduler = SqlDatabase::build_scheduler(config.max_parallelism);
        }
        self.execution_config = config;
    }

    pub fn set_skew_threshold(&mut self, threshold: usize) {
        let mut config = self.execution_config.clone();
        config.skew_threshold = threshold.max(1);
        self.set_execution_config(config);
    }

    pub fn reset_execution_stats(&self) {
        if let Ok(mut stats) = self.execution_stats.lock() {
            *stats = ExecutionStats::default();
        }
    }

    pub fn run_pending_refreshes(&mut self) -> Result<(), SqlDatabaseError> {
        self.process_scheduled_refreshes(true)
    }

    fn process_scheduled_refreshes(&mut self, force: bool) -> Result<(), SqlDatabaseError> {
        let now = Utc::now();
        let mut remaining = VecDeque::new();
        while let Some(request) = self.pending_view_refreshes.pop_front() {
            if force || request.due <= now {
                self.refresh_materialized_view_now(&request.view)?;
            } else {
                remaining.push_back(request);
            }
        }
        self.pending_view_refreshes = remaining;
        Ok(())
    }

    pub fn last_execution_stats(&self) -> ExecutionStats {
        self.execution_stats
            .lock()
            .map(|s| s.clone())
            .unwrap_or_default()
    }

    pub fn set_parallelism(&mut self, parallelism: usize) {
        let mut config = self.execution_config.clone();
        config.max_parallelism = parallelism.max(1);
        self.set_execution_config(config);
    }

    fn record_scan_stats(&self, input: usize, output: usize, partitions: usize, skew: bool) {
        if let Ok(mut stats) = self.execution_stats.lock() {
            stats.scan_input_rows = input;
            stats.scan_output_rows = output;
            stats.scan_partitions = partitions;
            stats.scan_skew_detected = skew;
        }
    }

    pub fn hash_join(
        &self,
        left_table: &str,
        right_table: &str,
        predicate: planner::JoinPredicate,
    ) -> Result<QueryResult, SqlDatabaseError> {
        let left = self
            .tables
            .get(left_table)
            .ok_or_else(|| SqlDatabaseError::UnknownTable(left_table.to_string()))?;
        let right = self
            .tables
            .get(right_table)
            .ok_or_else(|| SqlDatabaseError::UnknownTable(right_table.to_string()))?;

        let rows = execution::parallel_hash_join(
            left,
            right,
            &predicate,
            self.scheduler.as_deref(),
            self.execution_config.batch_size,
            self.execution_config.skew_threshold,
            Some(&self.execution_stats),
        )?;

        let mut columns = Vec::with_capacity(left.columns.len() + right.columns.len());
        columns.extend(
            left.columns
                .iter()
                .map(|c| format!("{left_table}.{}", c.name)),
        );
        columns.extend(
            right
                .columns
                .iter()
                .map(|c| format!("{right_table}.{}", c.name)),
        );

        Ok(QueryResult::Rows { columns, rows })
    }

    fn initialize_system_catalog(&mut self) {
        let table_stats = Table::with_columns(vec![
            ("table_name".to_string(), ColumnType::Text),
            ("row_count".to_string(), ColumnType::Integer),
            ("analyzed_at".to_string(), ColumnType::Timestamp),
            ("stats_version".to_string(), ColumnType::Integer),
        ]);
        let column_stats = Table::with_columns(vec![
            ("table_name".to_string(), ColumnType::Text),
            ("column_name".to_string(), ColumnType::Text),
            ("data_type".to_string(), ColumnType::Text),
            ("null_count".to_string(), ColumnType::Integer),
            ("distinct_count".to_string(), ColumnType::Integer),
            ("min_value".to_string(), ColumnType::Text),
            ("max_value".to_string(), ColumnType::Text),
            ("analyzed_at".to_string(), ColumnType::Timestamp),
            ("stats_version".to_string(), ColumnType::Integer),
        ]);
        self.system_tables
            .insert(TABLE_STATS_TABLE.to_string(), table_stats);
        self.system_tables
            .insert(COLUMN_STATS_TABLE.to_string(), column_stats);
    }

    fn refresh_stats_catalog_tables(&mut self) {
        let mut updated_tables: Vec<&'static str> = Vec::new();
        if let Some(table) = self.system_tables.get_mut(TABLE_STATS_TABLE) {
            table.rows.clear();
            let mut entries: Vec<_> = self.table_stats_catalog.values().cloned().collect();
            entries.sort_by(|a, b| a.table_name.cmp(&b.table_name));
            for stat in entries {
                table.rows.push(vec![
                    Value::Text(stat.table_name.clone()),
                    Value::Integer(clamp_u64_to_i64(stat.row_count)),
                    Value::Timestamp(stat.analyzed_at),
                    Value::Integer(stat.stats_version as i64),
                ]);
            }
            updated_tables.push(TABLE_STATS_TABLE);
        }
        if let Some(table) = self.system_tables.get_mut(COLUMN_STATS_TABLE) {
            table.rows.clear();
            let mut entries: Vec<_> = self.column_stats_catalog.values().cloned().collect();
            entries.sort_by(|a, b| match a.table_name.cmp(&b.table_name) {
                Ordering::Equal => a.column_name.cmp(&b.column_name),
                other => other,
            });
            for stat in entries {
                table.rows.push(vec![
                    Value::Text(stat.table_name.clone()),
                    Value::Text(stat.column_name.clone()),
                    Value::Text(stat.data_type.clone()),
                    Value::Integer(clamp_u64_to_i64(stat.null_count)),
                    Value::Integer(clamp_u64_to_i64(stat.distinct_count)),
                    stat.min
                        .as_ref()
                        .map(|v| Value::Text(value_to_plain_string(v)))
                        .unwrap_or(Value::Null),
                    stat.max
                        .as_ref()
                        .map(|v| Value::Text(value_to_plain_string(v)))
                        .unwrap_or(Value::Null),
                    Value::Timestamp(stat.analyzed_at),
                    Value::Integer(stat.stats_version as i64),
                ]);
            }
            updated_tables.push(COLUMN_STATS_TABLE);
        }
        for table in updated_tables {
            self.invalidate_cache_for_table(table);
        }
    }

    fn is_system_table(&self, name: &str) -> bool {
        self.system_tables.contains_key(name)
    }

    pub fn table_statistics(&self) -> Vec<TableStatistics> {
        let mut stats: Vec<_> = self.table_stats_catalog.values().cloned().collect();
        stats.sort_by(|a, b| a.table_name.cmp(&b.table_name));
        stats
    }

    pub fn get_table_statistics(&self, table: &str) -> Option<TableStatistics> {
        self.table_stats_catalog.get(table).cloned()
    }

    pub fn column_statistics(&self, table: &str) -> Vec<ColumnStatistics> {
        let mut stats: Vec<_> = self
            .column_stats_catalog
            .values()
            .filter(|stat| stat.table_name == table)
            .cloned()
            .collect();
        stats.sort_by(|a, b| a.column_name.cmp(&b.column_name));
        stats
    }

    pub fn get_column_statistics(&self, table: &str, column: &str) -> Option<ColumnStatistics> {
        let key = (table.to_string(), column.to_string());
        self.column_stats_catalog.get(&key).cloned()
    }

    pub fn all_column_statistics(&self) -> Vec<ColumnStatistics> {
        let mut stats: Vec<_> = self.column_stats_catalog.values().cloned().collect();
        stats.sort_by(|a, b| match a.table_name.cmp(&b.table_name) {
            Ordering::Equal => a.column_name.cmp(&b.column_name),
            other => other,
        });
        stats
    }

    pub fn cache_stats(&self) -> CacheStats {
        *self.plan_cache_stats.borrow()
    }

    pub fn notify_materialized_view_update(&mut self, name: &str) {
        self.invalidate_cache_for_table(name);
    }

    fn register_materialized_view_dependencies(&mut self, view: &str, dependencies: &[String]) {
        for dependency in dependencies {
            self.materialized_view_dependencies
                .entry(dependency.clone())
                .or_default()
                .insert(view.to_string());
        }
    }

    fn collect_materialized_view_dependencies(
        &self,
        select: &SelectStatement,
    ) -> Result<Vec<String>, SqlDatabaseError> {
        if self.tables.contains_key(&select.table)
            || self.materialized_views.contains_key(&select.table)
            || self.system_tables.contains_key(&select.table)
        {
            Ok(vec![select.table.clone()])
        } else {
            Err(SqlDatabaseError::UnknownTable(select.table.clone()))
        }
    }

    fn schedule_materialized_view_refresh(&mut self, name: &str) -> Result<(), SqlDatabaseError> {
        let view = self
            .materialized_views
            .get_mut(name)
            .ok_or_else(|| SqlDatabaseError::UnknownMaterializedView(name.to_string()))?;
        if view.pending_async_refresh {
            return Ok(());
        }
        view.pending_async_refresh = true;
        let due = Utc::now() + ASYNC_REFRESH_DELAY;
        self.pending_view_refreshes
            .push_back(MaterializedViewRefreshRequest {
                view: name.to_string(),
                due,
            });
        Ok(())
    }

    fn on_relation_changed(&mut self, name: &str) -> Result<(), SqlDatabaseError> {
        if let Some(dependents) = self.materialized_view_dependencies.get(name).cloned() {
            for dependent in dependents {
                let mode = match self.materialized_views.get(&dependent) {
                    Some(view) => view.refresh_mode,
                    None => continue,
                };
                match mode {
                    MaterializedViewRefreshMode::Synchronous => {
                        self.refresh_materialized_view_now(&dependent)?;
                    }
                    MaterializedViewRefreshMode::Asynchronous => {
                        self.schedule_materialized_view_refresh(&dependent)?;
                    }
                }
            }
        }
        Ok(())
    }

    fn find_covering_materialized_view(
        &self,
        select: &SelectStatement,
    ) -> Option<&MaterializedView> {
        self.materialized_views
            .values()
            .find(|view| view.definition == *select)
    }

    fn lookup_relation<'a, V>(map: &'a HashMap<String, V>, name: &str) -> Option<(&'a str, &'a V)> {
        map.iter()
            .find(|(key, _)| key.eq_ignore_ascii_case(name))
            .map(|(key, value)| (key.as_str(), value))
    }

    fn refresh_materialized_view_now(&mut self, name: &str) -> Result<(), SqlDatabaseError> {
        let mut view = self
            .materialized_views
            .remove(name)
            .ok_or_else(|| SqlDatabaseError::UnknownMaterializedView(name.to_string()))?;
        let definition = view.definition.clone();
        let result =
            match self.exec_select_statement(&HashMap::new(), &definition, ExecutionMode::Row) {
                Ok(result) => result,
                Err(err) => {
                    view.pending_async_refresh = false;
                    self.materialized_views.insert(name.to_string(), view);
                    return Err(err);
                }
            };
        let (columns, rows) = match result {
            QueryResult::Rows { columns, rows } => (columns, rows),
            _ => {
                view.pending_async_refresh = false;
                self.materialized_views.insert(name.to_string(), view);
                return Err(SqlDatabaseError::Unsupported);
            }
        };
        view.table = table_from_rows(columns, rows);
        view.last_refreshed_at = Some(Utc::now());
        view.pending_async_refresh = false;
        let name_owned = name.to_string();
        self.materialized_views.insert(name_owned.clone(), view);
        self.notify_materialized_view_update(&name_owned);
        self.on_relation_changed(&name_owned)?;
        Ok(())
    }

    pub fn execute(&mut self, sql: &str) -> Result<QueryResult, SqlDatabaseError> {
        self.execute_with_mode(sql, ExecutionMode::Auto)
    }

    pub fn execute_with_mode(
        &mut self,
        sql: &str,
        mode: ExecutionMode,
    ) -> Result<QueryResult, SqlDatabaseError> {
        self.process_scheduled_refreshes(false)?;
        let stmt = parse_statement(sql)?;
        self.reset_execution_stats();
        match stmt {
            Statement::CreateTable {
                name,
                columns,
                partitioning,
            } => {
                self.exec_create_table(name, columns, partitioning)?;
                Ok(QueryResult::None)
            }
            Statement::CreateMaterializedView {
                name,
                query,
                refresh,
            } => {
                self.exec_create_materialized_view(name, query, refresh)?;
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
            Statement::Select { ctes, body } => self.exec_select(ctes, body, mode),
            Statement::Merge { ctes, merge } => self.exec_merge(ctes, merge),
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
            Statement::Analyze { table } => {
                self.exec_analyze(table)?;
                Ok(QueryResult::None)
            }
            Statement::RefreshMaterializedView { name, strategy } => {
                self.exec_refresh_materialized_view(name, strategy)?;
                Ok(QueryResult::None)
            }
            Statement::ShowTables => self.exec_show_tables(),
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
        if self.is_system_table(&name) {
            return Err(SqlDatabaseError::SystemTableModification(name));
        }
        if self.tables.contains_key(&name) {
            return Err(SqlDatabaseError::TableExists(name));
        }
        if self.materialized_views.contains_key(&name) {
            return Err(SqlDatabaseError::MaterializedViewExists(name));
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
        initialize_secondary_indexes(&mut table);
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
        let table_name = name.clone();
        self.tables.insert(name, table);
        self.invalidate_cache_for_table(&table_name);
        Ok(())
    }

    fn exec_create_materialized_view(
        &mut self,
        name: String,
        query: SelectStatement,
        refresh: MaterializedViewRefreshMode,
    ) -> Result<(), SqlDatabaseError> {
        if self.is_system_table(&name) {
            return Err(SqlDatabaseError::SystemTableModification(name));
        }
        if self.tables.contains_key(&name) {
            return Err(SqlDatabaseError::TableExists(name));
        }
        if self.materialized_views.contains_key(&name) {
            return Err(SqlDatabaseError::MaterializedViewExists(name));
        }
        let dependencies = self.collect_materialized_view_dependencies(&query)?;
        let result = self.exec_select_statement(&HashMap::new(), &query, ExecutionMode::Row)?;
        let (columns, rows) = match result {
            QueryResult::Rows { columns, rows } => (columns, rows),
            _ => return Err(SqlDatabaseError::Unsupported),
        };
        let table = table_from_rows(columns, rows);
        let view = MaterializedView {
            name: name.clone(),
            definition: query,
            refresh_mode: refresh,
            table,
            last_refreshed_at: Some(Utc::now()),
            pending_async_refresh: false,
        };
        self.materialized_views.insert(name.clone(), view);
        self.register_materialized_view_dependencies(&name, &dependencies);
        self.notify_materialized_view_update(&name);
        Ok(())
    }

    fn exec_insert(
        &mut self,
        table_name: String,
        columns: Option<Vec<String>>,
        rows: Vec<Vec<Value>>,
    ) -> Result<usize, SqlDatabaseError> {
        if self.is_system_table(&table_name) {
            return Err(SqlDatabaseError::SystemTableModification(table_name));
        }
        if self.materialized_views.contains_key(&table_name) {
            return Err(SqlDatabaseError::MaterializedViewModification(table_name));
        }
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
            update_indexes_for_row(table, row_index);
            inserted += 1;
        }
        self.invalidate_cache_for_table(&table_name);
        self.on_relation_changed(&table_name)?;
        Ok(inserted)
    }

    fn exec_refresh_materialized_view(
        &mut self,
        name: String,
        strategy: MaterializedViewRefreshMode,
    ) -> Result<(), SqlDatabaseError> {
        match strategy {
            MaterializedViewRefreshMode::Synchronous => {
                self.refresh_materialized_view_now(&name)?;
            }
            MaterializedViewRefreshMode::Asynchronous => {
                self.schedule_materialized_view_refresh(&name)?;
            }
        }
        Ok(())
    }

    fn exec_analyze(&mut self, target: Option<String>) -> Result<(), SqlDatabaseError> {
        let target_tables = if let Some(name) = target {
            if self.is_system_table(&name) {
                return Err(SqlDatabaseError::SystemTableModification(name));
            }
            if !self.tables.contains_key(&name) {
                return Err(SqlDatabaseError::UnknownTable(name));
            }
            vec![name]
        } else {
            self.tables.keys().cloned().collect()
        };

        for table_name in target_tables {
            let table = self
                .tables
                .get(&table_name)
                .ok_or_else(|| SqlDatabaseError::UnknownTable(table_name.clone()))?;
            let analyzed_at = Utc::now();
            let version = self
                .table_stats_catalog
                .get(&table_name)
                .map(|stat| stat.stats_version + 1)
                .unwrap_or(1);

            let table_stat = TableStatistics {
                table_name: table_name.clone(),
                row_count: table.rows.len() as u64,
                analyzed_at,
                stats_version: version,
            };
            self.table_stats_catalog
                .insert(table_name.clone(), table_stat);

            self.column_stats_catalog
                .retain(|(tbl, _), _| tbl != &table_name);

            for (idx, column) in table.columns.iter().enumerate() {
                let column_stat = compute_column_statistics(
                    &table_name,
                    column,
                    idx,
                    &table.rows,
                    analyzed_at,
                    version,
                );
                self.column_stats_catalog
                    .insert((table_name.clone(), column.name.clone()), column_stat);
            }
        }

        self.refresh_stats_catalog_tables();
        Ok(())
    }

    fn exec_show_tables(&self) -> Result<QueryResult, SqlDatabaseError> {
        let mut entries: Vec<(String, i64)> = self
            .tables
            .iter()
            .map(|(name, table)| (name.clone(), table.rows.len() as i64))
            .collect();
        entries.extend(
            self.materialized_views
                .iter()
                .map(|(name, view)| (name.clone(), view.table.rows.len() as i64)),
        );
        entries.sort_by(|a, b| a.0.cmp(&b.0));

        let mut rows = Vec::new();
        for (name, count) in entries {
            rows.push(vec![
                Value::Text(name),
                Value::Integer(count),
                Value::Null,
                Value::Null,
            ]);
        }

        let columns = vec![
            "name".to_string(),
            "row_count".to_string(),
            "created_at".to_string(),
            "size".to_string(),
        ];

        Ok(QueryResult::Rows { columns, rows })
    }

    fn exec_select(
        &self,
        ctes: Vec<CommonTableExpression>,
        body: SelectStatement,
        mode: ExecutionMode,
    ) -> Result<QueryResult, SqlDatabaseError> {
        let cte_tables = self.materialize_ctes(&ctes, mode)?;
        self.exec_select_statement(&cte_tables, &body, mode)
    }

    fn exec_select_statement(
        &self,
        cte_tables: &HashMap<String, Table>,
        select: &SelectStatement,
        mode: ExecutionMode,
    ) -> Result<QueryResult, SqlDatabaseError> {
        if let Some(table) = cte_tables.get(&select.table) {
            let resolved = planner::ResolvedTable::new(
                select.table.as_str(),
                planner::TableSource::Cte,
                table,
            );
            self.exec_select_from_table(
                resolved,
                &select.columns,
                select.predicate.as_ref(),
                cte_tables,
                mode,
            )
        } else if let Some((view_name, view)) =
            Self::lookup_relation(&self.materialized_views, &select.table)
        {
            let resolved = planner::ResolvedTable::new(
                view_name,
                planner::TableSource::MaterializedView,
                &view.table,
            );
            self.exec_select_from_table(
                resolved,
                &select.columns,
                select.predicate.as_ref(),
                cte_tables,
                mode,
            )
        } else if let Some(view) = self.find_covering_materialized_view(select) {
            let resolved = planner::ResolvedTable::new(
                view.name.as_str(),
                planner::TableSource::MaterializedView,
                &view.table,
            );
            self.exec_select_from_table(
                resolved,
                &select.columns,
                select.predicate.as_ref(),
                cte_tables,
                mode,
            )
        } else if let Some((table_name, entry)) =
            Self::lookup_relation(&self.external_tables, &select.table)
        {
            let resolved = planner::ResolvedTable::new(
                table_name,
                planner::TableSource::External,
                &entry.schema,
            );
            self.exec_select_from_table(
                resolved,
                &select.columns,
                select.predicate.as_ref(),
                cte_tables,
                mode,
            )
        } else if let Some((table_name, table)) = Self::lookup_relation(&self.tables, &select.table)
        {
            let resolved =
                planner::ResolvedTable::new(table_name, planner::TableSource::Base, table);
            self.exec_select_from_table(
                resolved,
                &select.columns,
                select.predicate.as_ref(),
                cte_tables,
                mode,
            )
        } else if let Some((table_name, table)) =
            Self::lookup_relation(&self.system_tables, &select.table)
        {
            let resolved =
                planner::ResolvedTable::new(table_name, planner::TableSource::System, table);
            self.exec_select_from_table(
                resolved,
                &select.columns,
                select.predicate.as_ref(),
                cte_tables,
                mode,
            )
        } else {
            Err(SqlDatabaseError::UnknownTable(select.table.clone()))
        }
    }

    fn exec_select_from_table(
        &self,
        table_ref: planner::ResolvedTable,
        columns: &SelectColumns,
        predicate: Option<&Predicate>,
        cte_tables: &HashMap<String, Table>,
        mode: ExecutionMode,
    ) -> Result<QueryResult, SqlDatabaseError> {
        let table_stats = self.get_table_statistics(table_ref.name);
        let column_stats = self.column_statistics(table_ref.name);
        let external = if matches!(table_ref.source, planner::TableSource::External) {
            self.external_tables
                .get(table_ref.name)
                .map(|entry| entry.connector.as_ref())
        } else {
            None
        };
        let predicate = if matches!(table_ref.source, planner::TableSource::MaterializedView) {
            None
        } else {
            predicate
        };
        let context = planner::PlanContext::new(table_ref, table_stats, column_stats, external);
        let mut builder = planner::LogicalPlanBuilder::scan(table_ref);
        if let Some(pred) = predicate.cloned() {
            builder = builder.filter(pred);
        }
        let plan = builder.project(columns.clone()).build();
        let planner = planner::Planner::new(context);
        let optimized = planner.optimize(plan)?;
        self.execute_select_plan(&optimized.root, cte_tables, mode)
    }

    fn execute_select_plan(
        &self,
        expr: &planner::LogicalExpr,
        cte_tables: &HashMap<String, Table>,
        mode: ExecutionMode,
    ) -> Result<QueryResult, SqlDatabaseError> {
        let cache_metadata = Self::plan_cache_metadata(expr, cte_tables);
        if let Some(meta) = cache_metadata.as_ref() {
            if let Some(entry) = self.plan_cache.borrow().get(&meta.key) {
                self.plan_cache_stats.borrow_mut().hits += 1;
                return Ok(entry.result.clone());
            }
        }

        let (projection, predicate, scan) = match expr {
            planner::LogicalExpr::Projection(projection) => match projection.input.as_ref() {
                planner::LogicalExpr::Filter(filter) => match filter.input.as_ref() {
                    planner::LogicalExpr::Scan(scan) => (projection, Some(&filter.predicate), scan),
                    _ => return Err(SqlDatabaseError::Unsupported),
                },
                planner::LogicalExpr::Scan(scan) => (projection, None, scan),
                _ => return Err(SqlDatabaseError::Unsupported),
            },
            _ => return Err(SqlDatabaseError::Unsupported),
        };
        let vectorized_result = match mode {
            ExecutionMode::Auto | ExecutionMode::Vectorized => {
                self.try_vectorized_select(projection, predicate, scan, cte_tables)?
            }
            ExecutionMode::Row => None,
        };

        let outcome = match mode {
            ExecutionMode::Auto => {
                if let Some(result) = vectorized_result {
                    SelectExecutionOutcome::Vectorized(result)
                } else {
                    let (result, matching_indices) =
                        self.execute_select_plan_row(projection, predicate, scan, cte_tables)?;
                    SelectExecutionOutcome::Row {
                        result,
                        matching_indices,
                    }
                }
            }
            ExecutionMode::Vectorized => {
                if let Some(result) = vectorized_result {
                    SelectExecutionOutcome::Vectorized(result)
                } else {
                    return Err(SqlDatabaseError::Unsupported);
                }
            }
            ExecutionMode::Row => {
                let (result, matching_indices) =
                    self.execute_select_plan_row(projection, predicate, scan, cte_tables)?;
                SelectExecutionOutcome::Row {
                    result,
                    matching_indices,
                }
            }
        };

        let cache_entry = if cache_metadata.is_some() {
            Some(match &outcome {
                SelectExecutionOutcome::Row {
                    result,
                    matching_indices,
                } => CachedSelectResult {
                    result: result.clone(),
                    matching_indices: Some(matching_indices.clone()),
                    batch_slices: None,
                },
                SelectExecutionOutcome::Vectorized(vectorized) => CachedSelectResult {
                    result: vectorized.result.clone(),
                    matching_indices: None,
                    batch_slices: Some(vectorized.batch_slices.clone()),
                },
            })
        } else {
            None
        };

        if let Some(metadata) = cache_metadata {
            if let Some(entry) = cache_entry {
                self.plan_cache_stats.borrow_mut().misses += 1;
                self.store_plan_cache(metadata, entry);
            }
        }

        match outcome {
            SelectExecutionOutcome::Row { result, .. } => Ok(result),
            SelectExecutionOutcome::Vectorized(vectorized) => Ok(vectorized.result),
        }
    }

    fn execute_select_plan_row(
        &self,
        projection: &planner::ProjectionExpr,
        predicate: Option<&Predicate>,
        scan: &planner::ScanExpr,
        cte_tables: &HashMap<String, Table>,
    ) -> Result<(QueryResult, Vec<usize>), SqlDatabaseError> {
        if matches!(scan.table.source, planner::TableSource::External) {
            return self.execute_external_select(projection, predicate, scan, cte_tables);
        }
        let table = scan.table.table;
        let iter: Box<dyn Iterator<Item = usize>> = match scan.candidates.as_slice() {
            Some(rows) => Box::new(rows.iter().copied()),
            None => Box::new(0..table.rows.len()),
        };

        let mut matching_indices = Vec::new();
        for row_index in iter {
            let row = &table.rows[row_index];
            if let Some(predicate) = predicate {
                if !self.evaluate_predicate(table, predicate, row, cte_tables)? {
                    continue;
                }
            }
            matching_indices.push(row_index);
        }

        let result = match &projection.columns {
            SelectColumns::All => {
                let column_names = table.columns.iter().map(|c| c.name.clone()).collect();
                let rows = matching_indices
                    .iter()
                    .map(|&idx| table.rows[idx].clone())
                    .collect();
                QueryResult::Rows {
                    columns: column_names,
                    rows,
                }
            }
            SelectColumns::Some(items) => {
                let mut projections = Vec::new();
                let mut window_specs = Vec::new();

                for item in items {
                    match item {
                        SelectItem::Column(name) => {
                            let idx = self.column_index(table, name.as_str())?;
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
                            window_specs.push(spec.clone());
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

                let mut rows = Vec::with_capacity(matching_indices.len());
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

                QueryResult::Rows { columns, rows }
            }
        };

        Ok((result, matching_indices))
    }

    fn execute_external_select(
        &self,
        projection: &planner::ProjectionExpr,
        predicate: Option<&Predicate>,
        scan: &planner::ScanExpr,
        cte_tables: &HashMap<String, Table>,
    ) -> Result<(QueryResult, Vec<usize>), SqlDatabaseError> {
        let entry = self
            .external_tables
            .get(scan.table.name)
            .ok_or_else(|| SqlDatabaseError::UnknownTable(scan.table.name.to_string()))?;
        let connector = entry.connector.as_ref();
        let request = external::ExternalScanRequest {
            predicate: scan.options.pushdown_predicate.as_ref(),
            projected_columns: scan.options.projected_columns.as_ref(),
        };
        let result = connector.scan(request)?;
        let total_rows = result.rows.len();
        let mut filtered_rows = Vec::new();
        let mut matching_indices = Vec::new();
        for (idx, row) in result.rows.into_iter().enumerate() {
            if let Some(pred) = predicate {
                if !self.evaluate_predicate(&entry.schema, pred, &row, cte_tables)? {
                    continue;
                }
            }
            matching_indices.push(idx);
            filtered_rows.push(row);
        }

        self.record_scan_stats(total_rows, filtered_rows.len(), 1, false);

        let output = match &projection.columns {
            SelectColumns::All => {
                let columns = entry
                    .schema
                    .columns
                    .iter()
                    .map(|c| c.name.clone())
                    .collect();
                QueryResult::Rows {
                    columns,
                    rows: filtered_rows,
                }
            }
            SelectColumns::Some(items) => {
                let mut indices = Vec::with_capacity(items.len());
                let mut columns = Vec::with_capacity(items.len());
                for item in items {
                    match item {
                        SelectItem::Column(name) => {
                            let idx = self.column_index(&entry.schema, name)?;
                            indices.push(idx);
                            columns.push(entry.schema.columns[idx].name.clone());
                        }
                        SelectItem::WindowFunction(_) => {
                            return Err(SqlDatabaseError::Unsupported);
                        }
                    }
                }
                let mut rows = Vec::with_capacity(filtered_rows.len());
                for row in filtered_rows {
                    let mut projected = Vec::with_capacity(indices.len());
                    for idx in &indices {
                        projected.push(row[*idx].clone());
                    }
                    rows.push(projected);
                }
                QueryResult::Rows { columns, rows }
            }
        };

        Ok((output, matching_indices))
    }

    fn try_vectorized_select(
        &self,
        projection: &planner::ProjectionExpr,
        predicate: Option<&Predicate>,
        scan: &planner::ScanExpr,
        cte_tables: &HashMap<String, Table>,
    ) -> Result<Option<VectorizedSelectResult>, SqlDatabaseError> {
        if matches!(scan.table.source, planner::TableSource::External) {
            return Ok(None);
        }
        use execution::{
            apply_predicate, build_batch_for_partition, collect_rows, partition_scan_candidates,
        };

        #[derive(Default, Clone)]
        struct PartitionOutcome {
            rows: Vec<Vec<Value>>,
            input_rows: usize,
            output_rows: usize,
            skew: bool,
        }

        let table = scan.table.table;

        let (projection_indices, output_columns) = match &projection.columns {
            SelectColumns::All => {
                let indices: Vec<usize> = (0..table.columns.len()).collect();
                let columns = table.columns.iter().map(|c| c.name.clone()).collect();
                (indices, columns)
            }
            SelectColumns::Some(items) => {
                let mut indices = Vec::with_capacity(items.len());
                let mut columns = Vec::with_capacity(items.len());
                for item in items {
                    match item {
                        SelectItem::Column(name) => {
                            let idx = self.column_index(table, name.as_str())?;
                            indices.push(idx);
                            columns.push(table.columns[idx].name.clone());
                        }
                        SelectItem::WindowFunction(_) => {
                            return Ok(None);
                        }
                    }
                }
                (indices, columns)
            }
        };

        let prepared_predicate = match predicate {
            Some(pred) => Some(self.prepare_batch_predicate(table, pred, cte_tables)?),
            None => None,
        };

        let chunk_size = self.execution_config.batch_size.max(1);
        let partitions = partition_scan_candidates(&scan.candidates, table.rows.len(), chunk_size);

        if partitions.is_empty() {
            self.record_scan_stats(0, 0, 0, false);
            return Ok(Some(VectorizedSelectResult {
                result: QueryResult::Rows {
                    columns: output_columns,
                    rows: Vec::new(),
                },
                batch_slices: Vec::new(),
            }));
        }

        let threshold = self.execution_config.skew_threshold.max(1);
        let partition_count = partitions.len();

        let mut total_input = 0usize;
        let mut total_output = 0usize;
        let mut skew_detected = false;

        let parallelism = self.execution_config.max_parallelism;
        let use_parallel = parallelism > 1 && partition_count > 1 && self.scheduler.is_some();

        if !use_parallel {
            let mut rows = Vec::new();
            let mut batch_slices = Vec::new();
            let column_types: Vec<ColumnType> = table.columns.iter().map(|c| c.ty).collect();
            for partition in &partitions {
                let mut batch = build_batch_for_partition(table, &column_types, partition);
                let input_rows = batch.len();
                total_input += input_rows;
                if input_rows >= threshold {
                    skew_detected = true;
                }
                if let Some(prepared) = prepared_predicate.as_ref() {
                    apply_predicate(&mut batch, prepared);
                    if batch.is_empty() {
                        continue;
                    }
                }
                let output_rows = batch.len();
                total_output += output_rows;
                if output_rows == 0 {
                    continue;
                }
                let projected = batch.project(&projection_indices);
                let mut partition_rows = collect_rows(&projected);
                if partition_rows.is_empty() {
                    continue;
                }
                let start = rows.len();
                let len = partition_rows.len();
                rows.append(&mut partition_rows);
                batch_slices.push(CachedBatchSlice { start, len });
            }
            self.record_scan_stats(total_input, total_output, partition_count, skew_detected);
            return Ok(Some(VectorizedSelectResult {
                result: QueryResult::Rows {
                    columns: output_columns,
                    rows,
                },
                batch_slices,
            }));
        }

        let column_types: Vec<ColumnType> = table.columns.iter().map(|c| c.ty).collect();
        let column_types = Arc::new(column_types);
        let projection = Arc::new(projection_indices.clone());
        let predicate = prepared_predicate.clone();
        let results = Arc::new(Mutex::new(vec![
            PartitionOutcome::default();
            partition_count
        ]));
        let threshold_arc = Arc::new(threshold);
        let table_handle = TableHandle::new(table);

        let scheduler = self.scheduler.as_ref().expect("scheduler not initialized");
        let tasks = partitions.into_iter().enumerate().map(|(idx, partition)| {
            let column_types = Arc::clone(&column_types);
            let projection = Arc::clone(&projection);
            let results = Arc::clone(&results);
            let predicate = predicate.clone();
            let threshold = Arc::clone(&threshold_arc);
            move || {
                let table_ref = unsafe { table_handle.get() };
                let mut batch =
                    build_batch_for_partition(table_ref, column_types.as_slice(), &partition);
                let mut outcome = PartitionOutcome {
                    rows: Vec::new(),
                    input_rows: batch.len(),
                    output_rows: 0,
                    skew: batch.len() >= *threshold,
                };
                if let Some(prepared) = predicate.as_ref() {
                    apply_predicate(&mut batch, prepared);
                    if batch.is_empty() {
                        let mut guard = results.lock().unwrap();
                        guard[idx] = outcome;
                        return;
                    }
                }
                outcome.output_rows = batch.len();
                if outcome.output_rows > 0 {
                    let projected = batch.project(projection.as_slice());
                    outcome.rows = collect_rows(&projected);
                }
                let mut guard = results.lock().unwrap();
                guard[idx] = outcome;
            }
        });

        scheduler.execute(tasks);

        let mut rows = Vec::new();
        let mut batch_slices = Vec::new();
        if let Ok(mut guard) = results.lock() {
            for outcome in guard.iter_mut() {
                total_input += outcome.input_rows;
                total_output += outcome.output_rows;
                if outcome.skew {
                    skew_detected = true;
                }
                if !outcome.rows.is_empty() {
                    let start = rows.len();
                    let len = outcome.rows.len();
                    rows.append(&mut outcome.rows);
                    batch_slices.push(CachedBatchSlice { start, len });
                }
            }
        }

        self.record_scan_stats(total_input, total_output, partition_count, skew_detected);
        Ok(Some(VectorizedSelectResult {
            result: QueryResult::Rows {
                columns: output_columns,
                rows,
            },
            batch_slices,
        }))
    }
    fn prepare_batch_predicate(
        &self,
        table: &Table,
        predicate: &Predicate,
        cte_tables: &HashMap<String, Table>,
    ) -> Result<execution::BatchPredicate, SqlDatabaseError> {
        use execution::BatchPredicate;
        match predicate {
            Predicate::Equals { column, value } => {
                let idx = self.column_index(table, column)?;
                let column_type = table.columns[idx].ty;
                let coerced = coerce_static_value(value, column_type)?;
                Ok(BatchPredicate::Equals {
                    column_index: idx,
                    value: coerced,
                })
            }
            Predicate::GreaterOrEqual { column, value } => {
                let idx = self.column_index(table, column)?;
                let column_type = table.columns[idx].ty;
                let coerced = coerce_static_value(value, column_type)?;
                Ok(BatchPredicate::GreaterOrEqual {
                    column_index: idx,
                    value: coerced,
                })
            }
            Predicate::Between { column, start, end } => {
                use std::cmp::Ordering;
                let idx = self.column_index(table, column)?;
                let column_type = table.columns[idx].ty;
                let start_val = coerce_static_value(start, column_type)?;
                let end_val = coerce_static_value(end, column_type)?;
                let (low, high) = match compare_values(&start_val, &end_val) {
                    Some(Ordering::Greater) => (end_val, start_val),
                    _ => (start_val, end_val),
                };
                Ok(BatchPredicate::Between {
                    column_index: idx,
                    low,
                    high,
                })
            }
            Predicate::IsNull { column } => {
                let idx = self.column_index(table, column)?;
                Ok(BatchPredicate::IsNull { column_index: idx })
            }
            Predicate::InTableColumn {
                column,
                table: other_table,
                table_column,
            } => {
                let idx = self.column_index(table, column)?;
                if let Some((_, entry)) = Self::lookup_relation(&self.external_tables, other_table)
                {
                    let other_idx = entry
                        .schema
                        .columns
                        .iter()
                        .position(|c| c.name.eq_ignore_ascii_case(table_column))
                        .ok_or_else(|| SqlDatabaseError::UnknownColumn(table_column.clone()))?;
                    let scan = entry.connector.scan(external::ExternalScanRequest {
                        predicate: None,
                        projected_columns: None,
                    })?;
                    let lookup_values = scan
                        .rows
                        .into_iter()
                        .map(|row| row[other_idx].clone())
                        .collect();
                    return Ok(BatchPredicate::InTableColumn {
                        column_index: idx,
                        lookup_values,
                    });
                }

                let lookup_table = if let Some(cte_table) = cte_tables.get(other_table) {
                    cte_table
                } else if let Some((_, base)) = Self::lookup_relation(&self.tables, other_table) {
                    base
                } else if let Some((_, system)) =
                    Self::lookup_relation(&self.system_tables, other_table)
                {
                    system
                } else {
                    return Err(SqlDatabaseError::UnknownTable(other_table.clone()));
                };
                let other_idx = lookup_table
                    .columns
                    .iter()
                    .position(|c| c.name.eq_ignore_ascii_case(table_column))
                    .ok_or_else(|| SqlDatabaseError::UnknownColumn(table_column.clone()))?;
                let lookup_values = lookup_table
                    .rows
                    .iter()
                    .map(|row| row[other_idx].clone())
                    .collect();
                Ok(BatchPredicate::InTableColumn {
                    column_index: idx,
                    lookup_values,
                })
            }
            Predicate::FullText {
                column,
                query,
                language,
            } => {
                let idx = self.column_index(table, column)?;
                if table.columns[idx].ty != ColumnType::Text {
                    return Err(SqlDatabaseError::SchemaMismatch(format!(
                        "column '{column}' must be TEXT for full-text search"
                    )));
                }
                Ok(BatchPredicate::FullText {
                    column_index: idx,
                    query: query.clone(),
                    language: language.clone(),
                })
            }
        }
    }

    fn materialize_ctes(
        &self,
        ctes: &[CommonTableExpression],
        mode: ExecutionMode,
    ) -> Result<HashMap<String, Table>, SqlDatabaseError> {
        let mut tables = HashMap::new();
        for cte in ctes {
            if tables.contains_key(&cte.name) {
                return Err(SqlDatabaseError::Parse(format!(
                    "duplicate CTE name '{}'",
                    cte.name
                )));
            }
            let table = match &cte.body {
                CteBody::NonRecursive(select) => {
                    let result = self.exec_select_statement(&tables, select, mode)?;
                    match result {
                        QueryResult::Rows { columns, rows } => table_from_rows(columns, rows),
                        _ => return Err(SqlDatabaseError::Unsupported),
                    }
                }
                CteBody::Recursive { anchor, recursive } => {
                    let anchor_result = self.exec_select_statement(&tables, anchor, mode)?;
                    let (anchor_columns, anchor_rows) = match anchor_result {
                        QueryResult::Rows { columns, rows } => (columns, rows),
                        _ => return Err(SqlDatabaseError::Unsupported),
                    };
                    let column_names = anchor_columns.clone();
                    let mut all_rows = anchor_rows;
                    let mut seen: HashSet<String> = HashSet::new();
                    for row in &all_rows {
                        seen.insert(row_signature(row));
                    }
                    let mut working_table = table_from_rows(column_names.clone(), all_rows.clone());
                    loop {
                        let mut context = tables.clone();
                        context.insert(cte.name.clone(), working_table.clone());
                        let recursive_result =
                            self.exec_select_statement(&context, recursive, mode)?;
                        let (rec_columns, rec_rows) = match recursive_result {
                            QueryResult::Rows { columns, rows } => (columns, rows),
                            _ => return Err(SqlDatabaseError::Unsupported),
                        };
                        if rec_columns.len() != column_names.len() {
                            return Err(SqlDatabaseError::SchemaMismatch(
                                "recursive CTE result column count mismatch".into(),
                            ));
                        }
                        let mut added_new = false;
                        for row in rec_rows {
                            if row.len() != column_names.len() {
                                return Err(SqlDatabaseError::SchemaMismatch(
                                    "recursive CTE produced row with incorrect column count".into(),
                                ));
                            }
                            let signature = row_signature(&row);
                            if seen.insert(signature) {
                                all_rows.push(row);
                                added_new = true;
                            }
                        }
                        if !added_new {
                            break;
                        }
                        working_table = table_from_rows(column_names.clone(), all_rows.clone());
                    }
                    table_from_rows(column_names, all_rows)
                }
            };
            tables.insert(cte.name.clone(), table);
        }
        Ok(tables)
    }

    fn store_plan_cache(&self, metadata: PlanCacheMetadata, entry: CachedSelectResult) {
        let PlanCacheMetadata {
            key,
            dependent_tables,
        } = metadata;
        {
            let mut cache = self.plan_cache.borrow_mut();
            cache.insert(key.clone(), entry);
        }
        let mut index = self.plan_cache_index.borrow_mut();
        for table in dependent_tables {
            index.entry(table).or_default().insert(key.clone());
        }
    }

    fn invalidate_cache_for_table(&self, table: &str) {
        let mut index = self.plan_cache_index.borrow_mut();
        if let Some(keys) = index.remove(table) {
            let key_list: Vec<_> = keys.into_iter().collect();
            {
                let mut cache = self.plan_cache.borrow_mut();
                for key in &key_list {
                    cache.remove(key);
                }
            }
            for entries in index.values_mut() {
                for key in &key_list {
                    entries.remove(key);
                }
            }
        }
    }

    fn plan_cache_metadata(
        expr: &planner::LogicalExpr,
        cte_tables: &HashMap<String, Table>,
    ) -> Option<PlanCacheMetadata> {
        let mut params = Vec::new();
        let (shape, mut dependencies) =
            Self::logical_expr_signature(expr, &mut params, cte_tables)?;
        dependencies.sort();
        dependencies.dedup();
        if dependencies.is_empty() {
            return None;
        }
        let key = PlanCacheKey::new(shape, params);
        Some(PlanCacheMetadata {
            key,
            dependent_tables: dependencies,
        })
    }

    fn logical_expr_signature(
        expr: &planner::LogicalExpr,
        params: &mut Vec<String>,
        cte_tables: &HashMap<String, Table>,
    ) -> Option<(String, Vec<String>)> {
        match expr {
            planner::LogicalExpr::Projection(projection) => {
                let projection_shape = Self::projection_signature(&projection.columns);
                let (input_shape, dependencies) =
                    Self::logical_expr_signature(projection.input.as_ref(), params, cte_tables)?;
                let shape = format!("proj:{projection_shape}|{input_shape}");
                Some((shape, dependencies))
            }
            planner::LogicalExpr::Filter(filter) => {
                let (predicate_shape, mut predicate_deps) =
                    Self::predicate_signature(&filter.predicate, params, cte_tables)?;
                let (input_shape, mut dependencies) =
                    Self::logical_expr_signature(filter.input.as_ref(), params, cte_tables)?;
                dependencies.append(&mut predicate_deps);
                let shape = format!("filter:{predicate_shape}|{input_shape}");
                Some((shape, dependencies))
            }
            planner::LogicalExpr::Join(_) => None,
            planner::LogicalExpr::Scan(scan) => match scan.table.source {
                planner::TableSource::Cte => None,
                planner::TableSource::External => None,
                planner::TableSource::Base
                | planner::TableSource::System
                | planner::TableSource::MaterializedView => {
                    let source = match scan.table.source {
                        planner::TableSource::Base => "base",
                        planner::TableSource::System => "system",
                        planner::TableSource::MaterializedView => "materialized",
                        planner::TableSource::External => unreachable!(),
                        planner::TableSource::Cte => unreachable!(),
                    };
                    let mut shape = format!("scan:{source}:{}", scan.table.name);
                    match scan.candidates.as_slice() {
                        Some(rows) => {
                            shape.push_str(&format!(":fixed({})", rows.len()));
                            let candidate_repr = if rows.is_empty() {
                                String::new()
                            } else {
                                rows.iter()
                                    .map(|idx| idx.to_string())
                                    .collect::<Vec<_>>()
                                    .join(",")
                            };
                            params.push(format!("candidates:{candidate_repr}"));
                        }
                        None => shape.push_str(":all"),
                    }
                    Some((shape, vec![scan.table.name.to_string()]))
                }
            },
        }
    }

    fn projection_signature(columns: &SelectColumns) -> String {
        match columns {
            SelectColumns::All => "all".to_string(),
            SelectColumns::Some(items) => {
                let mut parts = Vec::new();
                for item in items {
                    match item {
                        SelectItem::Column(name) => {
                            parts.push(format!("col:{}", name.to_ascii_lowercase()));
                        }
                        SelectItem::WindowFunction(spec) => {
                            parts.push(Self::window_function_signature(spec));
                        }
                    }
                }
                format!("list[{}]", parts.join(","))
            }
        }
    }

    fn window_function_signature(spec: &WindowFunctionExpr) -> String {
        let mut repr = match spec.function {
            WindowFunctionType::RowNumber => "win:row_number".to_string(),
        };
        if let Some(partition) = &spec.partition_by {
            repr.push_str(&format!(":partition={}", partition.to_ascii_lowercase()));
        } else {
            repr.push_str(":partition=");
        }
        repr.push_str(&format!(":order={}", spec.order_by.to_ascii_lowercase()));
        if let Some(alias) = &spec.alias {
            repr.push_str(&format!(":alias={alias}"));
        }
        repr
    }

    fn predicate_signature(
        predicate: &Predicate,
        params: &mut Vec<String>,
        cte_tables: &HashMap<String, Table>,
    ) -> Option<(String, Vec<String>)> {
        match predicate {
            Predicate::Equals { column, value } => {
                params.push(value_fingerprint(value));
                Some((format!("eq:{}", column.to_ascii_lowercase()), Vec::new()))
            }
            Predicate::GreaterOrEqual { column, value } => {
                params.push(value_fingerprint(value));
                Some((format!("ge:{}", column.to_ascii_lowercase()), Vec::new()))
            }
            Predicate::Between { column, start, end } => {
                let mut start_key = value_fingerprint(start);
                let mut end_key = value_fingerprint(end);
                if matches!(compare_values(start, end), Some(Ordering::Greater)) {
                    std::mem::swap(&mut start_key, &mut end_key);
                }
                params.push(start_key);
                params.push(end_key);
                Some((
                    format!("between:{}", column.to_ascii_lowercase()),
                    Vec::new(),
                ))
            }
            Predicate::IsNull { column } => Some((
                format!("is_null:{}", column.to_ascii_lowercase()),
                Vec::new(),
            )),
            Predicate::InTableColumn {
                column,
                table,
                table_column,
            } => {
                if cte_tables.contains_key(table) {
                    return None;
                }
                let shape = format!(
                    "in:{}->{}.{}",
                    column.to_ascii_lowercase(),
                    table.to_ascii_lowercase(),
                    table_column.to_ascii_lowercase()
                );
                Some((shape, vec![table.clone()]))
            }
            Predicate::FullText {
                column,
                query,
                language,
            } => {
                params.push(format!("query={query}"));
                params.push(format!("lang={}", language.clone().unwrap_or_default()));
                Some((
                    format!("full_text:{}", column.to_ascii_lowercase()),
                    Vec::new(),
                ))
            }
        }
    }

    fn exec_merge(
        &mut self,
        ctes: Vec<CommonTableExpression>,
        merge: MergeStatement,
    ) -> Result<QueryResult, SqlDatabaseError> {
        let cte_tables = self.materialize_ctes(&ctes, ExecutionMode::Auto)?;

        let clone_table = |table: &Table| {
            let mut clone = Table {
                columns: table.columns.clone(),
                rows: table.rows.clone(),
                ..Table::default()
            };
            initialize_secondary_indexes(&mut clone);
            rebuild_secondary_indexes(&mut clone);
            clone
        };

        let source_table = if let Some(table) = cte_tables.get(&merge.source.name) {
            clone_table(table)
        } else {
            let table = self
                .tables
                .get(&merge.source.name)
                .ok_or_else(|| SqlDatabaseError::UnknownTable(merge.source.name.clone()))?;
            clone_table(table)
        };

        let target_name = merge.target.name.clone();
        if self.materialized_views.contains_key(&target_name) {
            return Err(SqlDatabaseError::MaterializedViewModification(target_name));
        }
        let (target_on_idx, matched_assignments, insert_info) = {
            let table = self
                .tables
                .get(&target_name)
                .ok_or_else(|| SqlDatabaseError::UnknownTable(target_name.clone()))?;
            let target_on_idx = column_index_in_table(table, &merge.condition.target_column)?;
            let matched_assignments = if let Some(update) = &merge.when_matched {
                let mut assignments = Vec::new();
                for assignment in &update.assignments {
                    let idx = column_index_in_table(table, &assignment.column)?;
                    assignments.push((idx, &assignment.value));
                }
                Some(assignments)
            } else {
                None
            };
            let insert_info = if let Some(insert) = &merge.when_not_matched {
                let mut indices = Vec::new();
                for column in &insert.columns {
                    let idx = column_index_in_table(table, column)?;
                    indices.push(idx);
                }
                Some((indices, insert.values.iter().collect::<Vec<_>>()))
            } else {
                None
            };
            (target_on_idx, matched_assignments, insert_info)
        };

        let target_table = self
            .tables
            .get_mut(&target_name)
            .ok_or_else(|| SqlDatabaseError::UnknownTable(target_name.clone()))?;

        let source_on_idx = column_index_in_table(&source_table, &merge.condition.source_column)?;

        let mut modified = false;

        for source_row in &source_table.rows {
            let source_key = &source_row[source_on_idx];
            let mut matched_index = None;
            for (idx, target_row) in target_table.rows.iter().enumerate() {
                if equal(&target_row[target_on_idx], source_key) {
                    matched_index = Some(idx);
                    break;
                }
            }

            if let Some(idx) = matched_index {
                if let Some(assignments) = &matched_assignments {
                    let computed_values = {
                        let current_row = &target_table.rows[idx];
                        let mut values = Vec::new();
                        for (column_index, value_expr) in assignments {
                            let value = evaluate_merge_value(
                                &merge,
                                value_expr,
                                target_table,
                                Some(current_row),
                                &source_table,
                                source_row,
                            )?;
                            let coerced =
                                coerce_value(value, target_table.columns[*column_index].ty)?;
                            values.push((*column_index, coerced));
                        }
                        values
                    };
                    let row = &mut target_table.rows[idx];
                    for (column_index, value) in computed_values {
                        row[column_index] = value;
                    }
                    modified = true;
                }
            } else if let Some((column_indices, value_exprs)) = &insert_info {
                let mut new_row = vec![Value::Null; target_table.columns.len()];
                for (column_index, value_expr) in
                    column_indices.iter().zip(value_exprs.iter().copied())
                {
                    let value = evaluate_merge_value(
                        &merge,
                        value_expr,
                        target_table,
                        None,
                        &source_table,
                        source_row,
                    )?;
                    let coerced = coerce_value(value, target_table.columns[*column_index].ty)?;
                    new_row[*column_index] = coerced;
                }
                target_table.rows.push(new_row);
                modified = true;
            }
        }

        if modified && target_table.partitioning.is_some() {
            rebuild_partitions(target_table);
        }
        if modified {
            rebuild_secondary_indexes(target_table);
            self.invalidate_cache_for_table(&target_name);
            self.on_relation_changed(&target_name)?;
        }

        Ok(QueryResult::None)
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
        cte_tables: &HashMap<String, Table>,
    ) -> Result<bool, SqlDatabaseError> {
        match predicate {
            Predicate::Equals { column, value } => {
                let idx = self.column_index(table, column)?;
                let column_type = table.columns[idx].ty;
                let coerced = coerce_static_value(value, column_type)?;
                Ok(equal(&row[idx], &coerced))
            }
            Predicate::GreaterOrEqual { column, value } => {
                let idx = self.column_index(table, column)?;
                if matches!(row[idx], Value::Null) {
                    return Ok(false);
                }
                let column_type = table.columns[idx].ty;
                let coerced = coerce_static_value(value, column_type)?;
                Ok(matches!(
                    compare_values(&row[idx], &coerced),
                    Some(Ordering::Greater) | Some(Ordering::Equal)
                ))
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
            Predicate::IsNull { column } => {
                let idx = self.column_index(table, column)?;
                Ok(matches!(row[idx], Value::Null))
            }
            Predicate::InTableColumn {
                column,
                table: other_table,
                table_column,
            } => {
                let idx = self.column_index(table, column)?;
                if matches!(row[idx], Value::Null) {
                    return Ok(false);
                }
                let value = &row[idx];
                let lookup_table = if let Some(cte_table) = cte_tables.get(other_table) {
                    cte_table
                } else if let Some((_, table)) = Self::lookup_relation(&self.tables, other_table) {
                    table
                } else if let Some((_, table)) =
                    Self::lookup_relation(&self.system_tables, other_table)
                {
                    table
                } else if let Some((_, entry)) =
                    Self::lookup_relation(&self.external_tables, other_table)
                {
                    let scan = entry.connector.scan(external::ExternalScanRequest {
                        predicate: None,
                        projected_columns: None,
                    })?;
                    let other_idx = entry
                        .schema
                        .columns
                        .iter()
                        .position(|c| c.name.eq_ignore_ascii_case(table_column))
                        .ok_or_else(|| SqlDatabaseError::UnknownColumn(table_column.clone()))?;
                    return Ok(scan.rows.iter().any(|r| equal(&r[other_idx], value)));
                } else {
                    return Err(SqlDatabaseError::UnknownTable(other_table.clone()));
                };
                let other_idx = lookup_table
                    .columns
                    .iter()
                    .position(|c| c.name.eq_ignore_ascii_case(table_column))
                    .ok_or_else(|| SqlDatabaseError::UnknownColumn(table_column.clone()))?;
                Ok(lookup_table
                    .rows
                    .iter()
                    .any(|r| equal(&r[other_idx], value)))
            }
            Predicate::FullText {
                column,
                query,
                language,
            } => {
                let idx = self.column_index(table, column)?;
                if table.columns[idx].ty != ColumnType::Text {
                    return Err(SqlDatabaseError::SchemaMismatch(format!(
                        "column '{column}' must be TEXT for full-text search"
                    )));
                }
                Ok(full_text_matches(&row[idx], query, language.as_deref()))
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

#[derive(Debug, Clone, PartialEq)]
struct SelectStatement {
    table: String,
    columns: SelectColumns,
    predicate: Option<Predicate>,
}

#[derive(Debug)]
struct CommonTableExpression {
    name: String,
    body: CteBody,
}

#[derive(Debug)]
enum CteBody {
    NonRecursive(SelectStatement),
    Recursive {
        anchor: SelectStatement,
        recursive: SelectStatement,
    },
}

#[derive(Debug)]
struct MergeStatement {
    target: TableReference,
    source: TableReference,
    condition: MergeCondition,
    when_matched: Option<MergeUpdate>,
    when_not_matched: Option<MergeInsert>,
}

#[derive(Debug)]
struct MergeCondition {
    target_column: String,
    source_column: String,
}

#[derive(Debug)]
struct MergeUpdate {
    assignments: Vec<MergeAssignment>,
}

#[derive(Debug)]
struct MergeAssignment {
    column: String,
    value: MergeValue,
}

#[derive(Debug)]
struct MergeInsert {
    columns: Vec<String>,
    values: Vec<MergeValue>,
}

#[derive(Debug)]
enum MergeValue {
    Literal(Value),
    Column {
        table: Option<String>,
        column: String,
    },
}

#[derive(Debug)]
struct TableReference {
    name: String,
    alias: Option<String>,
}

impl TableReference {
    fn matches(&self, identifier: &str) -> bool {
        if self.name.eq_ignore_ascii_case(identifier) {
            return true;
        }
        if let Some(alias) = &self.alias {
            if alias.eq_ignore_ascii_case(identifier) {
                return true;
            }
        }
        false
    }
}

#[derive(Debug)]
struct QualifiedColumn {
    table: String,
    column: String,
}

#[derive(Debug)]
enum Statement {
    CreateTable {
        name: String,
        columns: Vec<(String, ColumnType)>,
        partitioning: Option<PartitioningInfo>,
    },
    CreateMaterializedView {
        name: String,
        query: SelectStatement,
        refresh: MaterializedViewRefreshMode,
    },
    Insert {
        table: String,
        columns: Option<Vec<String>>,
        rows: Vec<Vec<Value>>,
    },
    Select {
        ctes: Vec<CommonTableExpression>,
        body: SelectStatement,
    },
    Merge {
        ctes: Vec<CommonTableExpression>,
        merge: MergeStatement,
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
    Analyze {
        table: Option<String>,
    },
    RefreshMaterializedView {
        name: String,
        strategy: MaterializedViewRefreshMode,
    },
    ShowTables,
    MatchGraph {
        graph: String,
        from: Option<String>,
        to: Option<String>,
        label: Option<String>,
        property_filter: Option<HashMap<String, Value>>,
    },
}

#[derive(Debug, Clone, PartialEq)]
enum SelectColumns {
    All,
    Some(Vec<SelectItem>),
}

impl SelectColumns {
    pub(crate) fn is_all(&self) -> bool {
        matches!(self, SelectColumns::All)
    }

    pub(crate) fn includes_column(&self, column: &str) -> bool {
        match self {
            SelectColumns::All => true,
            SelectColumns::Some(items) => items.iter().any(|item| match item {
                SelectItem::Column(name) => name.eq_ignore_ascii_case(column),
                SelectItem::WindowFunction(_) => false,
            }),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
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

#[derive(Debug, Clone, PartialEq)]
struct WindowFunctionExpr {
    function: WindowFunctionType,
    partition_by: Option<String>,
    order_by: String,
    alias: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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

#[derive(Debug, Clone, PartialEq)]
pub enum Predicate {
    Equals {
        column: String,
        value: Value,
    },
    GreaterOrEqual {
        column: String,
        value: Value,
    },
    Between {
        column: String,
        start: Value,
        end: Value,
    },
    IsNull {
        column: String,
    },
    InTableColumn {
        column: String,
        table: String,
        table_column: String,
    },
    FullText {
        column: String,
        query: String,
        language: Option<String>,
    },
}

impl Predicate {
    fn column_name(&self) -> &str {
        match self {
            Predicate::Equals { column, .. } => column,
            Predicate::GreaterOrEqual { column, .. } => column,
            Predicate::Between { column, .. } => column,
            Predicate::IsNull { column } => column,
            Predicate::InTableColumn { column, .. } => column,
            Predicate::FullText { column, .. } => column,
        }
    }
}

fn parse_statement(sql: &str) -> Result<Statement, SqlDatabaseError> {
    let trimmed = sql.trim();
    if trimmed.is_empty() {
        return Err(SqlDatabaseError::Parse("empty SQL statement".into()));
    }
    let trimmed = trimmed.trim_end_matches(';').trim();
    if let Some(rest) = strip_keyword_ci(trimmed, "WITH") {
        let rest = rest.trim_start();
        let (allow_recursive, rest) = if let Some(after) = strip_keyword_ci(rest, "RECURSIVE") {
            (true, after)
        } else {
            (false, rest)
        };
        let (ctes, remainder) = parse_common_table_expressions(rest, allow_recursive)?;
        let remainder = remainder.trim_start();
        if let Some(rest_select) = strip_keyword_ci(remainder, "SELECT") {
            let select = parse_select_statement(rest_select)?;
            Ok(Statement::Select { ctes, body: select })
        } else if let Some(rest_merge) = strip_keyword_ci(remainder, "MERGE") {
            let merge = parse_merge_statement(rest_merge)?;
            Ok(Statement::Merge { ctes, merge })
        } else {
            Err(SqlDatabaseError::Parse(
                "WITH clause must be followed by SELECT or MERGE".into(),
            ))
        }
    } else if let Some(rest) = strip_keyword_ci(trimmed, "CREATE") {
        let rest = rest.trim_start();
        if let Some(rest) = strip_keyword_ci(rest, "TABLE") {
            parse_create_table(rest)
        } else if let Some(rest) = strip_keyword_ci(rest, "MATERIALIZED") {
            let rest = expect_keyword_ci(rest, "VIEW")?;
            parse_create_materialized_view(rest)
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
        let select = parse_select_statement(rest)?;
        Ok(Statement::Select {
            ctes: Vec::new(),
            body: select,
        })
    } else if let Some(rest) = strip_keyword_ci(trimmed, "MERGE") {
        let merge = parse_merge_statement(rest)?;
        Ok(Statement::Merge {
            ctes: Vec::new(),
            merge,
        })
    } else if let Some(rest) = strip_keyword_ci(trimmed, "ANALYZE") {
        parse_analyze(rest)
    } else if let Some(rest) = strip_keyword_ci(trimmed, "REFRESH") {
        parse_refresh(rest)
    } else if let Some(rest) = strip_keyword_ci(trimmed, "SHOW") {
        parse_show(rest)
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

fn parse_create_materialized_view(input: &str) -> Result<Statement, SqlDatabaseError> {
    let (name, rest) = parse_identifier(input)?;
    let rest = rest.trim_start();
    let (refresh, rest) = if let Some(after_refresh) = strip_keyword_ci(rest, "REFRESH") {
        let after_refresh = after_refresh.trim_start();
        if let Some(after_sync) = strip_keyword_ci(after_refresh, "SYNCHRONOUS") {
            (MaterializedViewRefreshMode::Synchronous, after_sync)
        } else if let Some(after_async) = strip_keyword_ci(after_refresh, "ASYNCHRONOUS") {
            (MaterializedViewRefreshMode::Asynchronous, after_async)
        } else {
            return Err(SqlDatabaseError::Parse(
                "expected SYNCHRONOUS or ASYNCHRONOUS after REFRESH".into(),
            ));
        }
    } else {
        (MaterializedViewRefreshMode::Synchronous, rest)
    };
    let rest = expect_keyword_ci(rest, "AS")?;
    let rest = rest.trim_start();
    let select_body = strip_keyword_ci(rest, "SELECT")
        .ok_or_else(|| SqlDatabaseError::Parse("materialized view requires SELECT".into()))?;
    let query = parse_select_statement(select_body)?;
    Ok(Statement::CreateMaterializedView {
        name,
        query,
        refresh,
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
            .map(|s| parse_literal(&s))
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

fn parse_analyze(input: &str) -> Result<Statement, SqlDatabaseError> {
    let trimmed = input.trim();
    if trimmed.is_empty() {
        return Ok(Statement::Analyze { table: None });
    }
    let (table, rest) = parse_identifier(trimmed)?;
    ensure_no_trailing_tokens(rest)?;
    Ok(Statement::Analyze { table: Some(table) })
}

fn parse_refresh(input: &str) -> Result<Statement, SqlDatabaseError> {
    let rest = input.trim_start();
    if let Some(after_materialized) = strip_keyword_ci(rest, "MATERIALIZED") {
        let rest = expect_keyword_ci(after_materialized, "VIEW")?;
        parse_refresh_materialized_view(rest)
    } else {
        Err(SqlDatabaseError::Parse(
            "REFRESH supports MATERIALIZED VIEW only".into(),
        ))
    }
}

fn parse_refresh_materialized_view(input: &str) -> Result<Statement, SqlDatabaseError> {
    let (name, rest) = parse_identifier(input)?;
    let rest = rest.trim();
    let strategy = if rest.is_empty() {
        MaterializedViewRefreshMode::Synchronous
    } else if let Some(after_sync) = strip_keyword_ci(rest, "SYNCHRONOUS") {
        ensure_no_trailing_tokens(after_sync)?;
        MaterializedViewRefreshMode::Synchronous
    } else if let Some(after_async) = strip_keyword_ci(rest, "ASYNCHRONOUS") {
        ensure_no_trailing_tokens(after_async)?;
        MaterializedViewRefreshMode::Asynchronous
    } else {
        return Err(SqlDatabaseError::Parse(
            "expected SYNCHRONOUS or ASYNCHRONOUS".into(),
        ));
    };
    Ok(Statement::RefreshMaterializedView { name, strategy })
}

fn parse_show(input: &str) -> Result<Statement, SqlDatabaseError> {
    let trimmed = input.trim();
    if let Some(rest) = strip_keyword_ci(trimmed, "TABLES") {
        ensure_no_trailing_tokens(rest)?;
        Ok(Statement::ShowTables)
    } else {
        Err(SqlDatabaseError::Parse("unsupported SHOW statement".into()))
    }
}

fn parse_literal_token(input: &str) -> Result<(Value, &str), SqlDatabaseError> {
    let trimmed = input.trim_start();
    if trimmed.is_empty() {
        return Err(SqlDatabaseError::Parse("missing literal".into()));
    }
    if let Some(rest) = strip_keyword_ci(trimmed, "JSONB") {
        let (inner, remainder) = parse_literal_token(rest)?;
        if matches!(inner, Value::Null) {
            return Ok((Value::Null, remainder));
        }
        let json = value_into_json(inner)?;
        return Ok((Value::Jsonb(normalize_json(json)), remainder));
    }
    if let Some(rest) = strip_keyword_ci(trimmed, "JSON") {
        let (inner, remainder) = parse_literal_token(rest)?;
        if matches!(inner, Value::Null) {
            return Ok((Value::Null, remainder));
        }
        let json = value_into_json(inner)?;
        return Ok((Value::Json(json), remainder));
    }
    if let Some(rest) = strip_keyword_ci(trimmed, "XML") {
        let (inner, remainder) = parse_literal_token(rest)?;
        if matches!(inner, Value::Null) {
            return Ok((Value::Null, remainder));
        }
        let xml = value_into_xml(inner)?;
        return Ok((Value::Xml(xml), remainder));
    }
    if let Some(rest) = strip_keyword_ci(trimmed, "GEOMETRY") {
        let (inner, remainder) = parse_literal_token(rest)?;
        if matches!(inner, Value::Null) {
            return Ok((Value::Null, remainder));
        }
        let geometry = value_into_geometry(inner)?;
        return Ok((Value::Geometry(geometry), remainder));
    }
    if let Some(geometry) = parse_geometry_literal(trimmed) {
        let rest = &trimmed[trimmed.len()..];
        return Ok((Value::Geometry(geometry), rest));
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

fn parse_literal(input: &str) -> Result<Value, SqlDatabaseError> {
    let (value, rest) = parse_literal_token(input)?;
    if !rest.trim().is_empty() {
        return Err(SqlDatabaseError::Parse(
            "unexpected trailing tokens after literal".into(),
        ));
    }
    Ok(value)
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
        let value = parse_literal(value_str.trim())?;
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
        Value::Json(v) | Value::Jsonb(v) => serde_json::to_string(&v)
            .map_err(|_| SqlDatabaseError::Parse(format!("failed to serialize {context}"))),
        Value::Xml(s) => Ok(s),
        Value::Geometry(g) => Ok(geometry_to_string(&g)),
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

fn parse_common_table_expressions(
    input: &str,
    allow_recursive: bool,
) -> Result<(Vec<CommonTableExpression>, &str), SqlDatabaseError> {
    let mut remainder = input;
    let mut ctes = Vec::new();
    loop {
        let (name, rest) = parse_identifier(remainder)?;
        let rest = expect_keyword_ci(rest, "AS")?;
        let rest = rest.trim_start();
        if !rest.starts_with('(') {
            return Err(SqlDatabaseError::Parse(
                "CTE definition must use parentheses".into(),
            ));
        }
        let (body, rest_after) = take_parenthesized(rest)?;
        let body_trimmed = body.trim();
        let cte_body = if allow_recursive {
            if let Some((anchor_raw, recursive_raw)) = split_recursive_union(body_trimmed) {
                let anchor = parse_cte_select(&anchor_raw)?;
                let recursive = parse_cte_select(&recursive_raw)?;
                CteBody::Recursive { anchor, recursive }
            } else {
                CteBody::NonRecursive(parse_cte_select(body_trimmed)?)
            }
        } else {
            if split_recursive_union(body_trimmed).is_some() {
                return Err(SqlDatabaseError::Parse(
                    "UNION ALL in CTE requires WITH RECURSIVE".into(),
                ));
            }
            CteBody::NonRecursive(parse_cte_select(body_trimmed)?)
        };
        ctes.push(CommonTableExpression {
            name,
            body: cte_body,
        });
        remainder = rest_after.trim_start();
        if remainder.starts_with(',') {
            remainder = remainder[1..].trim_start();
            continue;
        }
        break;
    }
    Ok((ctes, remainder))
}

fn parse_cte_select(body: &str) -> Result<SelectStatement, SqlDatabaseError> {
    if let Some(select_body) = strip_keyword_ci(body.trim(), "SELECT") {
        parse_select_statement(select_body)
    } else {
        Err(SqlDatabaseError::Parse(
            "CTE body must be a SELECT statement".into(),
        ))
    }
}

fn split_recursive_union(body: &str) -> Option<(String, String)> {
    let lower = body.to_ascii_lowercase();
    let mut search_start = 0usize;
    while let Some(relative_idx) = lower[search_start..].find("union all") {
        let idx = search_start + relative_idx;
        let before = &body[..idx];
        let mut depth = 0i32;
        for ch in before.chars() {
            match ch {
                '(' => depth += 1,
                ')' => depth -= 1,
                _ => {}
            }
        }
        if depth == 0 {
            let anchor = body[..idx].trim().to_string();
            let recursive = body[idx + "union all".len()..].trim().to_string();
            if anchor.is_empty() || recursive.is_empty() {
                return None;
            }
            return Some((anchor, recursive));
        }
        search_start = idx + "union all".len();
    }
    None
}

fn parse_select_statement(input: &str) -> Result<SelectStatement, SqlDatabaseError> {
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
        let (column, rest_after_column) = parse_identifier(rest)?;
        let mut rest = rest_after_column.trim_start();
        if let Some(after_between) = strip_keyword_ci(rest, "BETWEEN") {
            let (start, rest_between) = parse_literal_token(after_between)?;
            let rest_between = expect_keyword_ci(rest_between, "AND")?;
            let (end, rest_final) = parse_literal_token(rest_between)?;
            ensure_no_trailing_tokens(rest_final)?;
            Some(Predicate::Between {
                column: column.clone(),
                start,
                end,
            })
        } else if let Some(after_is) = strip_keyword_ci(rest, "IS") {
            let rest_is = strip_keyword_ci(after_is, "NULL")
                .ok_or_else(|| SqlDatabaseError::Parse("expected NULL after IS".into()))?;
            ensure_no_trailing_tokens(rest_is)?;
            Some(Predicate::IsNull {
                column: column.clone(),
            })
        } else if let Some(after_in) = strip_keyword_ci(rest, "IN") {
            let (other_table, rest_after_table) = parse_identifier(after_in)?;
            let rest_after_table = rest_after_table.trim_start();
            if !rest_after_table.starts_with('.') {
                return Err(SqlDatabaseError::Parse(
                    "expected '.' in IN table reference".into(),
                ));
            }
            let (other_column, rest_after_column_name) = parse_identifier(&rest_after_table[1..])?;
            ensure_no_trailing_tokens(rest_after_column_name)?;
            Some(Predicate::InTableColumn {
                column: column.clone(),
                table: other_table,
                table_column: other_column,
            })
        } else if rest.starts_with("@@") {
            let (query_value, rest_after_query) = parse_literal_token(&rest[2..])?;
            let query = value_to_string(query_value, "full-text query")?;
            let remainder = rest_after_query.trim_start();
            let mut language = None;
            if !remainder.is_empty() {
                let rest_after_lang_keyword = expect_keyword_ci(remainder, "LANGUAGE")?;
                let (language_value, rest_after_language) =
                    parse_literal_token(rest_after_lang_keyword)?;
                let lang = value_to_string(language_value, "language specifier")?;
                ensure_no_trailing_tokens(rest_after_language)?;
                language = Some(lang);
            } else {
                ensure_no_trailing_tokens(remainder)?;
            }
            Some(Predicate::FullText {
                column: column.clone(),
                query,
                language,
            })
        } else {
            rest = rest.trim_start();
            if let Some(after_ge) = rest.strip_prefix(">=") {
                let (value, rest_after_value) = parse_literal_token(after_ge)?;
                ensure_no_trailing_tokens(rest_after_value)?;
                Some(Predicate::GreaterOrEqual { column, value })
            } else if !rest.starts_with('=') {
                return Err(SqlDatabaseError::Parse(
                    "expected '=' in WHERE clause".into(),
                ));
            } else {
                let (value, rest_after_value) = parse_literal_token(&rest[1..])?;
                ensure_no_trailing_tokens(rest_after_value)?;
                Some(Predicate::Equals { column, value })
            }
        }
    };
    Ok(SelectStatement {
        table,
        columns: select_columns,
        predicate,
    })
}

fn parse_merge_statement(input: &str) -> Result<MergeStatement, SqlDatabaseError> {
    let rest = expect_keyword_ci(input, "INTO")?;
    let (target, rest) = parse_table_reference(rest)?;
    let rest = expect_keyword_ci(rest, "USING")?;
    let (source, rest) = parse_table_reference(rest)?;
    let rest = expect_keyword_ci(rest, "ON")?;
    let (left, rest) = parse_qualified_column(rest)?;
    let rest = rest.trim_start();
    if !rest.starts_with('=') {
        return Err(SqlDatabaseError::Parse("ON clause must contain '='".into()));
    }
    let (right, mut remainder) = parse_qualified_column(&rest[1..])?;
    let (target_column, source_column) =
        if target.matches(&left.table) && source.matches(&right.table) {
            (left.column, right.column)
        } else if target.matches(&right.table) && source.matches(&left.table) {
            (right.column, left.column)
        } else {
            return Err(SqlDatabaseError::Parse(
                "ON clause must compare target and source columns".into(),
            ));
        };

    let mut when_matched = None;
    let mut when_not_matched = None;
    loop {
        let trimmed = remainder.trim_start();
        if trimmed.is_empty() {
            remainder = trimmed;
            break;
        }
        if let Some(rest) = strip_keyword_ci(trimmed, "WHEN") {
            let rest = rest.trim_start();
            if let Some(rest) = strip_keyword_ci(rest, "MATCHED") {
                if when_matched.is_some() {
                    return Err(SqlDatabaseError::Parse(
                        "multiple WHEN MATCHED clauses".into(),
                    ));
                }
                let rest = expect_keyword_ci(rest, "THEN")?;
                let (update, rest_after) = parse_merge_update_clause(rest, &target, &source)?;
                when_matched = Some(update);
                remainder = rest_after;
                continue;
            } else if let Some(rest) = strip_keyword_ci(rest, "NOT") {
                let rest = expect_keyword_ci(rest, "MATCHED")?;
                if when_not_matched.is_some() {
                    return Err(SqlDatabaseError::Parse(
                        "multiple WHEN NOT MATCHED clauses".into(),
                    ));
                }
                let rest = expect_keyword_ci(rest, "THEN")?;
                let (insert, rest_after) = parse_merge_insert_clause(rest, &target, &source)?;
                when_not_matched = Some(insert);
                remainder = rest_after;
                continue;
            } else {
                return Err(SqlDatabaseError::Parse("unsupported WHEN clause".into()));
            }
        }
        remainder = trimmed;
        break;
    }

    ensure_no_trailing_tokens(remainder)?;
    Ok(MergeStatement {
        target,
        source,
        condition: MergeCondition {
            target_column,
            source_column,
        },
        when_matched,
        when_not_matched,
    })
}

fn parse_table_reference(input: &str) -> Result<(TableReference, &str), SqlDatabaseError> {
    let (name, rest) = parse_identifier(input)?;
    let rest = rest.trim_start();
    if let Some(rest) = strip_keyword_ci(rest, "AS") {
        let (alias, rest_after) = parse_identifier(rest)?;
        Ok((
            TableReference {
                name,
                alias: Some(alias),
            },
            rest_after,
        ))
    } else {
        Ok((TableReference { name, alias: None }, rest))
    }
}

fn parse_qualified_column(input: &str) -> Result<(QualifiedColumn, &str), SqlDatabaseError> {
    let (table, rest) = parse_identifier(input.trim_start())?;
    let rest = rest.trim_start();
    if !rest.starts_with('.') {
        return Err(SqlDatabaseError::Parse(
            "qualified column must include '.'".into(),
        ));
    }
    let (column, rest_after) = parse_identifier(&rest[1..])?;
    Ok((QualifiedColumn { table, column }, rest_after))
}

fn parse_merge_update_clause<'a>(
    input: &'a str,
    target: &TableReference,
    source: &TableReference,
) -> Result<(MergeUpdate, &'a str), SqlDatabaseError> {
    let rest = expect_keyword_ci(input, "UPDATE")?;
    let rest = expect_keyword_ci(rest, "SET")?;
    let (assignments, remainder) = parse_merge_assignments(rest, target, source)?;
    Ok((MergeUpdate { assignments }, remainder))
}

fn parse_merge_insert_clause<'a>(
    input: &'a str,
    target: &TableReference,
    source: &TableReference,
) -> Result<(MergeInsert, &'a str), SqlDatabaseError> {
    let rest = expect_keyword_ci(input, "INSERT")?;
    let rest = rest.trim_start();
    if !rest.starts_with('(') {
        return Err(SqlDatabaseError::Parse(
            "INSERT clause requires column list".into(),
        ));
    }
    let (columns_raw, rest_after) = take_parenthesized(rest)?;
    let mut columns = Vec::new();
    for part in split_comma(&columns_raw)? {
        let trimmed = part.trim();
        let (column, remainder) = parse_target_column(trimmed, target)?;
        ensure_no_trailing_tokens(remainder)?;
        columns.push(column);
    }
    let rest = expect_keyword_ci(rest_after, "VALUES")?;
    let (values_raw, remainder) = take_parenthesized(rest)?;
    let parts = split_comma(&values_raw)?;
    if columns.len() != parts.len() {
        return Err(SqlDatabaseError::Parse(
            "INSERT column count must match values".into(),
        ));
    }
    let mut values = Vec::new();
    for part in parts {
        let (value, remainder) = parse_merge_value(&part, target, source)?;
        ensure_no_trailing_tokens(remainder)?;
        values.push(value);
    }
    Ok((MergeInsert { columns, values }, remainder))
}

fn parse_merge_assignments<'a>(
    input: &'a str,
    target: &TableReference,
    source: &TableReference,
) -> Result<(Vec<MergeAssignment>, &'a str), SqlDatabaseError> {
    let mut assignments = Vec::new();
    let mut remainder = input;
    loop {
        let (column, rest) = parse_target_column(remainder, target)?;
        let rest = rest.trim_start();
        if !rest.starts_with('=') {
            return Err(SqlDatabaseError::Parse("expected '=' in assignment".into()));
        }
        let (value, rest_after) = parse_merge_value(&rest[1..], target, source)?;
        assignments.push(MergeAssignment { column, value });
        remainder = rest_after.trim_start();
        if remainder.starts_with(',') {
            remainder = remainder[1..].trim_start();
            continue;
        }
        break;
    }
    Ok((assignments, remainder))
}

fn parse_target_column<'a>(
    input: &'a str,
    target: &TableReference,
) -> Result<(String, &'a str), SqlDatabaseError> {
    let trimmed = input.trim_start();
    let (identifier, rest) = parse_identifier(trimmed)?;
    let rest_trimmed = rest.trim_start();
    if rest_trimmed.starts_with('.') {
        let (column, rest_after) = parse_identifier(&rest_trimmed[1..])?;
        if !target.matches(&identifier) {
            return Err(SqlDatabaseError::Parse(format!(
                "unknown table '{identifier}' in MERGE assignment"
            )));
        }
        Ok((column, rest_after))
    } else {
        Ok((identifier, rest_trimmed))
    }
}

fn parse_merge_value<'a>(
    input: &'a str,
    target: &TableReference,
    source: &TableReference,
) -> Result<(MergeValue, &'a str), SqlDatabaseError> {
    let trimmed = input.trim_start();
    if trimmed.is_empty() {
        return Err(SqlDatabaseError::Parse("missing value".into()));
    }
    let first = trimmed.chars().next().unwrap();
    if first == '\'' || first == '"' || first.is_ascii_digit() || first == '-' {
        let (value, rest) = parse_literal_token(trimmed)?;
        return Ok((MergeValue::Literal(value), rest));
    }
    let (ident, rest) = parse_identifier(trimmed)?;
    let rest_trimmed = rest.trim_start();
    if rest_trimmed.starts_with('.') {
        let (column, rest_after) = parse_identifier(&rest_trimmed[1..])?;
        if target.matches(&ident) || source.matches(&ident) {
            return Ok((
                MergeValue::Column {
                    table: Some(ident),
                    column,
                },
                rest_after,
            ));
        } else {
            return Err(SqlDatabaseError::Parse(format!(
                "unknown table '{ident}' in MERGE expression",
            )));
        }
    }
    let upper = ident.to_ascii_uppercase();
    if matches!(upper.as_str(), "NULL" | "TRUE" | "FALSE") {
        let value = parse_literal(ident.as_str())?;
        return Ok((MergeValue::Literal(value), rest_trimmed));
    }
    Ok((
        MergeValue::Column {
            table: None,
            column: ident,
        },
        rest_trimmed,
    ))
}

pub(crate) fn column_index_in_table(table: &Table, name: &str) -> Result<usize, SqlDatabaseError> {
    table
        .columns
        .iter()
        .position(|c| c.name.eq_ignore_ascii_case(name))
        .ok_or_else(|| SqlDatabaseError::UnknownColumn(name.into()))
}

fn table_from_rows(columns: Vec<String>, rows: Vec<Vec<Value>>) -> Table {
    let mut table = Table::default();
    let mut column_defs = Vec::new();
    for (index, name) in columns.into_iter().enumerate() {
        let ty = infer_column_type(&rows, index);
        column_defs.push(Column { name, ty });
    }
    table.columns = column_defs;
    table.rows = rows;
    initialize_secondary_indexes(&mut table);
    rebuild_secondary_indexes(&mut table);
    table
}

fn infer_column_type(rows: &[Vec<Value>], index: usize) -> ColumnType {
    for row in rows {
        if let Some(value) = row.get(index) {
            match value {
                Value::Integer(_) => return ColumnType::Integer,
                Value::Float(_) => return ColumnType::Float,
                Value::Text(_) => return ColumnType::Text,
                Value::Boolean(_) => return ColumnType::Boolean,
                Value::Timestamp(_) => return ColumnType::Timestamp,
                Value::Json(_) => return ColumnType::Json,
                Value::Jsonb(_) => return ColumnType::Jsonb,
                Value::Xml(_) => return ColumnType::Xml,
                Value::Geometry(_) => return ColumnType::Geometry,
                Value::Null => continue,
            }
        }
    }
    ColumnType::Text
}

fn rebuild_partitions(table: &mut Table) {
    table.partitions.clear();
    if let Some(partitioning) = &table.partitioning {
        for (idx, row) in table.rows.iter().enumerate() {
            if let Some(value) = row.get(partitioning.column_index) {
                if let Some(key) = partitioning.partition_key(value) {
                    table.partitions.entry(key).or_default().push(idx);
                }
            }
        }
    }
}

fn clamp_u64_to_i64(value: u64) -> i64 {
    if value > i64::MAX as u64 {
        i64::MAX
    } else {
        value as i64
    }
}

fn value_fingerprint(value: &Value) -> String {
    match value {
        Value::Null => "null".to_string(),
        Value::Integer(i) => format!("i:{i}"),
        Value::Float(f) => format!("f:{:x}", f.to_bits()),
        Value::Text(s) => format!("t:{s}"),
        Value::Boolean(b) => format!("b:{}", *b as u8),
        Value::Timestamp(ts) => {
            let nanos = ts.timestamp_nanos_opt().unwrap_or_else(|| {
                ts.timestamp().saturating_mul(1_000_000_000)
                    + i64::from(ts.timestamp_subsec_nanos())
            });
            format!("ts:{nanos}")
        }
        Value::Json(v) => format!("j:{}", canonical_json(v)),
        Value::Jsonb(v) => format!("jb:{}", canonical_json(v)),
        Value::Xml(s) => format!("x:{s}"),
        Value::Geometry(g) => format!("g:{}", geometry_to_string(g)),
    }
}

fn value_to_plain_string(value: &Value) -> String {
    match value {
        Value::Text(s) => s.clone(),
        Value::Integer(i) => i.to_string(),
        Value::Float(f) => f.to_string(),
        Value::Boolean(b) => b.to_string(),
        Value::Timestamp(ts) => ts.to_rfc3339(),
        Value::Json(v) | Value::Jsonb(v) => serde_json::to_string(v).unwrap_or_default(),
        Value::Xml(s) => s.clone(),
        Value::Geometry(g) => geometry_to_string(g),
        Value::Null => "NULL".to_string(),
    }
}

fn distinct_key_for_value(value: &Value) -> Option<String> {
    match value {
        Value::Null => None,
        Value::Integer(i) => Some(format!("i:{i}")),
        Value::Float(f) => Some(format!("f:{:x}", f.to_bits())),
        Value::Text(s) => Some(format!("t:{s}")),
        Value::Boolean(b) => Some(format!("b:{}", *b as u8)),
        Value::Timestamp(ts) => {
            let nanos = ts.timestamp_nanos_opt().unwrap_or_else(|| {
                ts.timestamp().saturating_mul(1_000_000_000)
                    + i64::from(ts.timestamp_subsec_nanos())
            });
            Some(format!("ts:{}", nanos))
        }
        Value::Json(v) => Some(format!("j:{}", canonical_json(v))),
        Value::Jsonb(v) => Some(format!("jb:{}", canonical_json(v))),
        Value::Xml(s) => Some(format!("x:{s}")),
        Value::Geometry(g) => Some(format!("g:{}", geometry_to_string(g))),
    }
}

fn compare_stat_values(column_type: ColumnType, left: &Value, right: &Value) -> Option<Ordering> {
    match column_type {
        ColumnType::Integer => match (left, right) {
            (Value::Integer(a), Value::Integer(b)) => Some(a.cmp(b)),
            _ => None,
        },
        ColumnType::Float => match (left, right) {
            (Value::Float(a), Value::Float(b)) => a.partial_cmp(b),
            _ => None,
        },
        ColumnType::Text => match (left, right) {
            (Value::Text(a), Value::Text(b)) => Some(a.cmp(b)),
            _ => None,
        },
        ColumnType::Boolean => match (left, right) {
            (Value::Boolean(a), Value::Boolean(b)) => Some((*a as u8).cmp(&(*b as u8))),
            _ => None,
        },
        ColumnType::Timestamp => match (left, right) {
            (Value::Timestamp(a), Value::Timestamp(b)) => Some(a.cmp(b)),
            _ => None,
        },
        ColumnType::Json | ColumnType::Jsonb | ColumnType::Geometry => None,
        ColumnType::Xml => match (left, right) {
            (Value::Xml(a), Value::Xml(b)) => Some(a.cmp(b)),
            _ => None,
        },
    }
}

fn update_min_max(
    column_type: ColumnType,
    min_value: &mut Option<Value>,
    max_value: &mut Option<Value>,
    candidate: &Value,
) {
    if compare_stat_values(column_type, candidate, candidate).is_none() {
        return;
    }

    match min_value {
        Some(current_min) => {
            if let Some(Ordering::Less) = compare_stat_values(column_type, candidate, current_min) {
                *min_value = Some(candidate.clone());
            }
        }
        None => *min_value = Some(candidate.clone()),
    }

    match max_value {
        Some(current_max) => {
            if let Some(Ordering::Greater) =
                compare_stat_values(column_type, candidate, current_max)
            {
                *max_value = Some(candidate.clone());
            }
        }
        None => *max_value = Some(candidate.clone()),
    }
}

fn stats_numeric_value(column_type: ColumnType, value: &Value) -> Option<f64> {
    match column_type {
        ColumnType::Integer => match value {
            Value::Integer(v) => Some(*v as f64),
            Value::Float(v) => Some(*v),
            _ => None,
        },
        ColumnType::Float => match value {
            Value::Float(v) => Some(*v),
            Value::Integer(v) => Some(*v as f64),
            _ => None,
        },
        ColumnType::Timestamp => match value {
            Value::Timestamp(ts) => Some(ts.timestamp_millis() as f64),
            _ => None,
        },
        _ => None,
    }
}

fn build_histogram_from_numeric(values: &mut Vec<f64>) -> Option<ColumnHistogram> {
    if values.is_empty() {
        return None;
    }
    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
    let len = values.len();
    let bucket_count = DEFAULT_HISTOGRAM_BUCKETS.min(len);
    if bucket_count == 0 {
        return None;
    }
    let mut buckets = Vec::with_capacity(bucket_count);
    for bucket_idx in 0..bucket_count {
        let start = (bucket_idx * len) / bucket_count;
        let mut end = ((bucket_idx + 1) * len) / bucket_count;
        if end == start {
            end = (start + 1).min(len);
        }
        let slice = &values[start..end];
        if slice.is_empty() {
            continue;
        }
        let lower = slice.first().copied().unwrap_or(0.0);
        let upper = slice.last().copied().unwrap_or(lower);
        buckets.push(HistogramBucket {
            lower,
            upper,
            count: slice.len() as u64,
        });
    }
    if buckets.is_empty() {
        None
    } else {
        Some(ColumnHistogram { buckets })
    }
}

fn compute_column_statistics(
    table_name: &str,
    column: &Column,
    column_index: usize,
    rows: &[Vec<Value>],
    analyzed_at: DateTime<Utc>,
    stats_version: u32,
) -> ColumnStatistics {
    let mut null_count = 0u64;
    let mut distinct_values: HashSet<String> = HashSet::new();
    let mut min_value: Option<Value> = None;
    let mut max_value: Option<Value> = None;
    let mut numeric_values: Vec<f64> = Vec::new();

    for row in rows {
        if let Some(value) = row.get(column_index) {
            if matches!(value, Value::Null) {
                null_count += 1;
                continue;
            }
            if let Some(key) = distinct_key_for_value(value) {
                distinct_values.insert(key);
            }
            update_min_max(column.ty, &mut min_value, &mut max_value, value);
            if let Some(num) = stats_numeric_value(column.ty, value) {
                numeric_values.push(num);
            }
        }
    }

    let histogram = build_histogram_from_numeric(&mut numeric_values);

    ColumnStatistics {
        table_name: table_name.to_string(),
        column_name: column.name.clone(),
        data_type: column_type_name(column.ty).to_string(),
        null_count,
        distinct_count: distinct_values.len() as u64,
        min: min_value,
        max: max_value,
        histogram,
        analyzed_at,
        stats_version,
    }
}

fn initialize_secondary_indexes(table: &mut Table) {
    table.json_indexes.clear();
    table.jsonb_indexes.clear();
    table.xml_indexes.clear();
    table.spatial_indexes.clear();
    for (idx, column) in table.columns.iter().enumerate() {
        match column.ty {
            ColumnType::Json => {
                table.json_indexes.insert(idx, JsonIndex::default());
            }
            ColumnType::Jsonb => {
                table.jsonb_indexes.insert(idx, JsonIndex::default());
            }
            ColumnType::Xml => {
                table.xml_indexes.insert(idx, XmlIndex::default());
            }
            ColumnType::Geometry => {
                table.spatial_indexes.insert(idx, SpatialIndex::default());
            }
            _ => {}
        }
    }
}

fn rebuild_secondary_indexes(table: &mut Table) {
    for index in table.json_indexes.values_mut() {
        index.map.clear();
    }
    for index in table.jsonb_indexes.values_mut() {
        index.map.clear();
    }
    for index in table.xml_indexes.values_mut() {
        index.map.clear();
    }
    for index in table.spatial_indexes.values_mut() {
        index.tree = RTree::new();
    }
    for row_index in 0..table.rows.len() {
        update_indexes_for_row(table, row_index);
    }
}

fn update_indexes_for_row(table: &mut Table, row_index: usize) {
    let row = &table.rows[row_index];
    for (col_idx, index) in table.json_indexes.iter_mut() {
        if let Some(Value::Json(value)) = row.get(*col_idx) {
            let key = canonical_json(value);
            index.map.entry(key).or_default().push(row_index);
        }
    }
    for (col_idx, index) in table.jsonb_indexes.iter_mut() {
        if let Some(Value::Jsonb(value)) = row.get(*col_idx) {
            let key = canonical_json(value);
            index.map.entry(key).or_default().push(row_index);
        }
    }
    for (col_idx, index) in table.xml_indexes.iter_mut() {
        if let Some(Value::Xml(xml)) = row.get(*col_idx) {
            if let Some(root) = xml_root_name(xml) {
                index.map.entry(root).or_default().push(row_index);
            }
        }
    }
    for (col_idx, index) in table.spatial_indexes.iter_mut() {
        if let Some(Value::Geometry(geom)) = row.get(*col_idx) {
            let bbox = geom.to_aabb();
            index.tree.insert(SpatialIndexItem {
                row: row_index,
                bbox,
            });
        }
    }
}

pub(crate) fn canonical_json(value: &JsonValue) -> String {
    match value {
        JsonValue::Null => "null".to_string(),
        JsonValue::Bool(b) => b.to_string(),
        JsonValue::Number(num) => num.to_string(),
        JsonValue::String(s) => serde_json::to_string(s).unwrap_or_default(),
        JsonValue::Array(values) => {
            let items: Vec<_> = values.iter().map(canonical_json).collect();
            format!("[{}]", items.join(","))
        }
        JsonValue::Object(map) => {
            let mut items: Vec<_> = map.iter().collect();
            items.sort_by(|a, b| a.0.cmp(b.0));
            let mut parts = Vec::with_capacity(items.len());
            for (key, value) in items {
                let key_serialized = serde_json::to_string(key).unwrap_or_default();
                parts.push(format!("{key_serialized}:{}", canonical_json(value)));
            }
            format!("{{{}}}", parts.join(","))
        }
    }
}

fn normalize_json(value: JsonValue) -> JsonValue {
    match value {
        JsonValue::Array(values) => {
            JsonValue::Array(values.into_iter().map(normalize_json).collect())
        }
        JsonValue::Object(map) => {
            let mut items: Vec<_> = map.into_iter().collect();
            items.sort_by(|a, b| a.0.cmp(&b.0));
            let normalized = items
                .into_iter()
                .map(|(key, value)| (key, normalize_json(value)))
                .collect();
            JsonValue::Object(normalized)
        }
        other => other,
    }
}

pub(crate) fn xml_root_name(xml: &str) -> Option<String> {
    Document::parse(xml)
        .ok()
        .map(|doc| doc.root_element().tag_name().name().to_string())
}

fn value_into_json(value: Value) -> Result<JsonValue, SqlDatabaseError> {
    match value {
        Value::Json(v) | Value::Jsonb(v) => Ok(v),
        Value::Text(text) => serde_json::from_str(&text)
            .map_err(|_| SqlDatabaseError::SchemaMismatch("failed to parse string as JSON".into())),
        Value::Null => Ok(JsonValue::Null),
        other => Err(SqlDatabaseError::SchemaMismatch(format!(
            "cannot interpret {other:?} as JSON"
        ))),
    }
}

fn value_into_xml(value: Value) -> Result<String, SqlDatabaseError> {
    match value {
        Value::Xml(xml) => {
            Document::parse(&xml)
                .map_err(|_| SqlDatabaseError::SchemaMismatch("invalid XML literal".into()))?;
            Ok(xml)
        }
        Value::Text(text) => {
            Document::parse(&text)
                .map_err(|_| SqlDatabaseError::SchemaMismatch("invalid XML literal".into()))?;
            Ok(text)
        }
        Value::Null => Ok(String::new()),
        other => Err(SqlDatabaseError::SchemaMismatch(format!(
            "cannot interpret {other:?} as XML"
        ))),
    }
}

fn value_into_geometry(value: Value) -> Result<Geometry, SqlDatabaseError> {
    match value {
        Value::Geometry(g) => Ok(g),
        Value::Text(text) => parse_geometry_literal(&text)
            .ok_or_else(|| SqlDatabaseError::SchemaMismatch("invalid geometry literal".into())),
        Value::Json(json) | Value::Jsonb(json) => parse_geometry_from_json(&json)
            .ok_or_else(|| SqlDatabaseError::SchemaMismatch("invalid geometry JSON".into())),
        Value::Null => Err(SqlDatabaseError::SchemaMismatch(
            "geometry literals cannot be NULL".into(),
        )),
        other => Err(SqlDatabaseError::SchemaMismatch(format!(
            "cannot interpret {other:?} as geometry"
        ))),
    }
}

fn parse_geometry_literal(text: &str) -> Option<Geometry> {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return None;
    }
    let upper = trimmed.to_ascii_uppercase();
    if upper.starts_with("POINT(") && trimmed.ends_with(')') {
        let inner = &trimmed[6..trimmed.len() - 1];
        let parts: Vec<_> = inner
            .split(|c: char| c == ',' || c.is_whitespace())
            .filter(|part| !part.is_empty())
            .collect();
        if parts.len() != 2 {
            return None;
        }
        let x = parts[0].parse::<f64>().ok()?;
        let y = parts[1].parse::<f64>().ok()?;
        return Some(Geometry::Point { x, y });
    }
    if trimmed.starts_with('[') {
        if let Ok(array) = serde_json::from_str::<Vec<f64>>(trimmed) {
            if array.len() == 4 {
                return Some(Geometry::BoundingBox {
                    min_x: array[0],
                    min_y: array[1],
                    max_x: array[2],
                    max_y: array[3],
                });
            }
        }
    }
    None
}

fn parse_geometry_from_json(json: &JsonValue) -> Option<Geometry> {
    match json {
        JsonValue::Object(map) => {
            let ty = map.get("type")?.as_str()?;
            match ty.to_ascii_uppercase().as_str() {
                "POINT" => {
                    let coords = map.get("coordinates")?.as_array()?;
                    if coords.len() == 2 {
                        let x = coords[0].as_f64()?;
                        let y = coords[1].as_f64()?;
                        Some(Geometry::Point { x, y })
                    } else {
                        None
                    }
                }
                "BBOX" => {
                    let coords = map.get("coordinates")?.as_array()?;
                    if coords.len() == 4 {
                        let min_x = coords[0].as_f64()?;
                        let min_y = coords[1].as_f64()?;
                        let max_x = coords[2].as_f64()?;
                        let max_y = coords[3].as_f64()?;
                        Some(Geometry::BoundingBox {
                            min_x,
                            min_y,
                            max_x,
                            max_y,
                        })
                    } else {
                        None
                    }
                }
                _ => None,
            }
        }
        JsonValue::Array(values) if values.len() == 4 => Some(Geometry::BoundingBox {
            min_x: values[0].as_f64()?,
            min_y: values[1].as_f64()?,
            max_x: values[2].as_f64()?,
            max_y: values[3].as_f64()?,
        }),
        _ => None,
    }
}

fn row_signature(row: &[Value]) -> String {
    let mut signature = String::new();
    for value in row {
        match value {
            Value::Integer(i) => {
                signature.push_str("i:");
                signature.push_str(&i.to_string());
            }
            Value::Float(f) => {
                signature.push_str("f:");
                signature.push_str(&format!("{:.12}", f));
            }
            Value::Text(s) => {
                signature.push_str("t:");
                signature.push_str(s);
            }
            Value::Boolean(b) => {
                signature.push_str("b:");
                signature.push_str(if *b { "1" } else { "0" });
            }
            Value::Timestamp(ts) => {
                signature.push_str("ts:");
                signature.push_str(&ts.to_rfc3339());
            }
            Value::Json(v) => {
                signature.push_str("j:");
                signature.push_str(&canonical_json(v));
            }
            Value::Jsonb(v) => {
                signature.push_str("jb:");
                signature.push_str(&canonical_json(v));
            }
            Value::Xml(s) => {
                signature.push_str("x:");
                signature.push_str(s);
            }
            Value::Geometry(g) => {
                signature.push_str("g:");
                signature.push_str(&geometry_to_string(g));
            }
            Value::Null => {
                signature.push_str("n:");
            }
        }
        signature.push('|');
    }
    signature
}

pub(crate) fn full_text_matches(value: &Value, query: &str, language: Option<&str>) -> bool {
    let document = match value {
        Value::Text(text) => text,
        _ => return false,
    };
    let doc_tokens = analyze_text(document, language);
    if doc_tokens.is_empty() {
        return false;
    }
    let query_tokens = analyze_text(query, language);
    if query_tokens.is_empty() {
        return false;
    }
    let doc_set: HashSet<String> = doc_tokens.into_iter().collect();
    query_tokens
        .into_iter()
        .all(|token| doc_set.contains(&token))
}

const ENGLISH_STOPWORDS: &[&str] = &[
    "the", "and", "or", "a", "an", "of", "to", "in", "on", "for", "with", "is", "it", "this",
    "that", "by", "from", "as", "at", "be", "are", "was", "were",
];

const SPANISH_STOPWORDS: &[&str] = &[
    "el", "la", "los", "las", "y", "de", "del", "que", "un", "una", "en", "por",
];

fn analyze_text(text: &str, language: Option<&str>) -> Vec<String> {
    let mut tokens = Vec::new();
    for part in text.split(|c: char| !c.is_alphanumeric()) {
        let trimmed = part.trim();
        if trimmed.is_empty() {
            continue;
        }
        tokens.push(trimmed.to_lowercase());
    }
    match language {
        Some(lang) if lang.eq_ignore_ascii_case("english") => {
            tokens.into_iter().filter_map(apply_english_rules).collect()
        }
        Some(lang) if lang.eq_ignore_ascii_case("spanish") => {
            tokens.into_iter().filter_map(apply_spanish_rules).collect()
        }
        _ => tokens,
    }
}

fn apply_english_rules(token: String) -> Option<String> {
    if ENGLISH_STOPWORDS.iter().any(|&stop| stop == token) {
        return None;
    }
    Some(stem_english(&token))
}

fn stem_english(token: &str) -> String {
    if token.ends_with("ies") && token.len() > 4 {
        let stem = &token[..token.len() - 3];
        return format!("{stem}y");
    }
    if token.ends_with("ing") && token.len() > 4 {
        return token[..token.len() - 3].to_string();
    }
    if token.ends_with("ed") && token.len() > 3 {
        return token[..token.len() - 2].to_string();
    }
    if token.ends_with("es") && token.len() > 4 {
        return token[..token.len() - 2].to_string();
    }
    if token.ends_with('s') && token.len() > 3 {
        return token[..token.len() - 1].to_string();
    }
    token.to_string()
}

fn apply_spanish_rules(token: String) -> Option<String> {
    if SPANISH_STOPWORDS.iter().any(|&stop| stop == token) {
        return None;
    }
    Some(stem_spanish(&token))
}

fn stem_spanish(token: &str) -> String {
    if token.ends_with("mente") && token.len() > 6 {
        return token[..token.len() - 5].to_string();
    }
    if token.ends_with("aciones") && token.len() > 8 {
        return token[..token.len() - 6].to_string();
    }
    if token.ends_with("es") && token.len() > 4 {
        return token[..token.len() - 2].to_string();
    }
    if token.ends_with('s') && token.len() > 4 {
        return token[..token.len() - 1].to_string();
    }
    token.to_string()
}

fn evaluate_merge_value(
    merge: &MergeStatement,
    value: &MergeValue,
    target_table: &Table,
    target_row: Option<&[Value]>,
    source_table: &Table,
    source_row: &[Value],
) -> Result<Value, SqlDatabaseError> {
    match value {
        MergeValue::Literal(v) => Ok(v.clone()),
        MergeValue::Column { table, column } => {
            if let Some(table_name) = table {
                if merge.target.matches(table_name) {
                    let row = target_row.ok_or_else(|| {
                        SqlDatabaseError::SchemaMismatch(
                            "cannot reference target columns in INSERT".into(),
                        )
                    })?;
                    let idx = column_index_in_table(target_table, column)?;
                    Ok(row[idx].clone())
                } else if merge.source.matches(table_name) {
                    let idx = column_index_in_table(source_table, column)?;
                    Ok(source_row[idx].clone())
                } else {
                    Err(SqlDatabaseError::UnknownColumn(format!(
                        "{table_name}.{column}"
                    )))
                }
            } else {
                let idx = column_index_in_table(source_table, column)?;
                Ok(source_row[idx].clone())
            }
        }
    }
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
        "JSON" => Ok(ColumnType::Json),
        "JSONB" => Ok(ColumnType::Jsonb),
        "XML" => Ok(ColumnType::Xml),
        "GEOMETRY" | "SPATIAL" => Ok(ColumnType::Geometry),
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
    if trimmed.starts_with('{') || trimmed.starts_with('[') {
        if let Ok(json) = serde_json::from_str::<JsonValue>(trimmed) {
            return Ok(Value::Json(json));
        }
    }
    if trimmed.starts_with('<') {
        if Document::parse(trimmed).is_ok() {
            return Ok(Value::Xml(trimmed.to_string()));
        }
    }
    if let Some(geometry) = parse_geometry_literal(trimmed) {
        return Ok(Value::Geometry(geometry));
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
        Value::Json(v) => format!("J:{}", canonical_json(v)),
        Value::Jsonb(v) => format!("JB:{}", canonical_json(v)),
        Value::Xml(s) => format!("X:{s}"),
        Value::Geometry(g) => format!("G:{}", geometry_to_string(g)),
    }
}

pub(crate) fn compare_values(left: &Value, right: &Value) -> Option<Ordering> {
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
            Value::Json(_) | Value::Jsonb(_) | Value::Xml(_) | Value::Geometry(_) => Err(
                SqlDatabaseError::SchemaMismatch("cannot coerce complex types to INTEGER".into()),
            ),
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
            Value::Json(_) | Value::Jsonb(_) | Value::Xml(_) | Value::Geometry(_) => Err(
                SqlDatabaseError::SchemaMismatch("cannot coerce complex types to FLOAT".into()),
            ),
        },
        ColumnType::Text => match value {
            Value::Text(s) => Ok(Value::Text(s)),
            Value::Integer(i) => Ok(Value::Text(i.to_string())),
            Value::Float(f) => Ok(Value::Text(f.to_string())),
            Value::Boolean(b) => Ok(Value::Text(b.to_string())),
            Value::Null => Ok(Value::Null),
            Value::Timestamp(ts) => Ok(Value::Text(ts.to_rfc3339())),
            Value::Json(v) | Value::Jsonb(v) => {
                serde_json::to_string(&v).map(Value::Text).map_err(|_| {
                    SqlDatabaseError::SchemaMismatch(
                        "failed to serialize JSON value as TEXT".into(),
                    )
                })
            }
            Value::Xml(s) => Ok(Value::Text(s)),
            Value::Geometry(g) => Ok(Value::Text(geometry_to_string(&g))),
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
            Value::Json(_) | Value::Jsonb(_) | Value::Xml(_) | Value::Geometry(_) => Err(
                SqlDatabaseError::SchemaMismatch("cannot coerce complex types to BOOLEAN".into()),
            ),
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
            Value::Json(_) | Value::Jsonb(_) | Value::Xml(_) | Value::Geometry(_) => Err(
                SqlDatabaseError::SchemaMismatch("cannot coerce complex types to TIMESTAMP".into()),
            ),
        },
        ColumnType::Json => match value {
            Value::Json(v) => Ok(Value::Json(v)),
            Value::Jsonb(v) => Ok(Value::Json(v)),
            Value::Text(s) => serde_json::from_str(&s).map(Value::Json).map_err(|_| {
                SqlDatabaseError::SchemaMismatch("failed to parse string as JSON".into())
            }),
            Value::Null => Ok(Value::Null),
            other => Err(SqlDatabaseError::SchemaMismatch(format!(
                "cannot coerce {other:?} to JSON"
            ))),
        },
        ColumnType::Jsonb => match value {
            Value::Json(v) => Ok(Value::Jsonb(normalize_json(v))),
            Value::Jsonb(v) => Ok(Value::Jsonb(normalize_json(v))),
            Value::Text(s) => serde_json::from_str(&s)
                .map(normalize_json)
                .map(Value::Jsonb)
                .map_err(|_| {
                    SqlDatabaseError::SchemaMismatch("failed to parse string as JSONB".into())
                }),
            Value::Null => Ok(Value::Null),
            other => Err(SqlDatabaseError::SchemaMismatch(format!(
                "cannot coerce {other:?} to JSONB"
            ))),
        },
        ColumnType::Xml => match value {
            Value::Xml(xml) => {
                Document::parse(&xml)
                    .map_err(|_| SqlDatabaseError::SchemaMismatch("invalid XML literal".into()))?;
                Ok(Value::Xml(xml))
            }
            Value::Text(s) => {
                Document::parse(&s)
                    .map_err(|_| SqlDatabaseError::SchemaMismatch("invalid XML literal".into()))?;
                Ok(Value::Xml(s))
            }
            Value::Null => Ok(Value::Null),
            other => Err(SqlDatabaseError::SchemaMismatch(format!(
                "cannot coerce {other:?} to XML"
            ))),
        },
        ColumnType::Geometry => match value {
            Value::Geometry(g) => Ok(Value::Geometry(g)),
            Value::Text(s) => parse_geometry_literal(&s)
                .map(Value::Geometry)
                .ok_or_else(|| SqlDatabaseError::SchemaMismatch("invalid geometry literal".into())),
            Value::Json(json) => parse_geometry_from_json(&json)
                .map(Value::Geometry)
                .ok_or_else(|| SqlDatabaseError::SchemaMismatch("invalid geometry JSON".into())),
            Value::Jsonb(json) => parse_geometry_from_json(&json)
                .map(Value::Geometry)
                .ok_or_else(|| SqlDatabaseError::SchemaMismatch("invalid geometry JSON".into())),
            Value::Null => Ok(Value::Null),
            other => Err(SqlDatabaseError::SchemaMismatch(format!(
                "cannot coerce {other:?} to GEOMETRY"
            ))),
        },
    }
}

pub(crate) fn coerce_static_value(
    value: &Value,
    target: ColumnType,
) -> Result<Value, SqlDatabaseError> {
    coerce_value(value.clone(), target)
}

pub(crate) fn equal(left: &Value, right: &Value) -> bool {
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
        (Value::Json(a), Value::Json(b)) => canonical_json(a) == canonical_json(b),
        (Value::Json(a), Value::Jsonb(b)) | (Value::Jsonb(b), Value::Json(a)) => {
            canonical_json(a) == canonical_json(b)
        }
        (Value::Jsonb(a), Value::Jsonb(b)) => canonical_json(a) == canonical_json(b),
        (Value::Json(a), Value::Text(b)) | (Value::Text(b), Value::Json(a)) => {
            serde_json::from_str::<JsonValue>(b)
                .map(|parsed| canonical_json(&parsed) == canonical_json(a))
                .unwrap_or(false)
        }
        (Value::Jsonb(a), Value::Text(b)) | (Value::Text(b), Value::Jsonb(a)) => {
            serde_json::from_str::<JsonValue>(b)
                .map(|parsed| canonical_json(&parsed) == canonical_json(a))
                .unwrap_or(false)
        }
        (Value::Xml(a), Value::Xml(b)) => a == b,
        (Value::Xml(a), Value::Text(b)) | (Value::Text(b), Value::Xml(a)) => a == b,
        (Value::Geometry(a), Value::Geometry(b)) => a == b,
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
        Value::Json(v) | Value::Jsonb(v) => serde_json::to_string(v).unwrap_or_default(),
        Value::Xml(s) => s.clone(),
        Value::Geometry(g) => geometry_to_string(g),
        Value::Null => "NULL".into(),
    }
}

fn geometry_to_string(geometry: &Geometry) -> String {
    match geometry {
        Geometry::Point { x, y } => format!("POINT({x} {y})"),
        Geometry::BoundingBox {
            min_x,
            min_y,
            max_x,
            max_y,
        } => format!("[{min_x}, {min_y}, {max_x}, {max_y}]"),
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
    use super::planner::{
        CardinalityEstimator, CostModel, JoinPredicate, LogicalExpr, LogicalPlanBuilder,
        PlanContext, Planner, ResolvedTable, ScanCandidates, TableSource,
    };
    use super::*;
    use serde_json::json;

    fn row_values(result: &QueryResult) -> &Vec<Vec<Value>> {
        match result {
            QueryResult::Rows { rows, .. } => rows,
            _ => panic!("expected rows result"),
        }
    }

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
    fn json_and_jsonb_support() {
        let mut db = SqlDatabase::new();
        db.execute("CREATE TABLE docs (id INT, data JSON, meta JSONB);")
            .unwrap();
        db.execute("INSERT INTO docs VALUES (1, JSON '{\"k\":1}', JSONB '{\"sorted\":true}');")
            .unwrap();
        db.execute("INSERT INTO docs VALUES (2, '{\"k\":2}', JSONB '{\"sorted\":false}');")
            .unwrap();

        let result = db
            .execute("SELECT id FROM docs WHERE data = JSON '{\"k\":1}';")
            .unwrap();
        match result {
            QueryResult::Rows { rows, .. } => {
                assert_eq!(rows, vec![vec![Value::Integer(1)]]);
            }
            other => panic!("unexpected result {other:?}"),
        }

        let result = db
            .execute("SELECT id FROM docs WHERE meta = JSONB '{\"sorted\":true}';")
            .unwrap();
        match result {
            QueryResult::Rows { rows, .. } => {
                assert_eq!(rows, vec![vec![Value::Integer(1)]]);
            }
            other => panic!("unexpected result {other:?}"),
        }
    }

    #[test]
    fn xml_support() {
        let mut db = SqlDatabase::new();
        db.execute("CREATE TABLE xml_docs (id INT, body XML);")
            .unwrap();
        db.execute("INSERT INTO xml_docs VALUES (1, XML '<root><item/></root>');")
            .unwrap();
        db.execute("INSERT INTO xml_docs VALUES (2, '<alt></alt>');")
            .unwrap();

        let result = db
            .execute("SELECT id FROM xml_docs WHERE body = XML '<root><item/></root>';")
            .unwrap();
        match result {
            QueryResult::Rows { rows, .. } => {
                assert_eq!(rows, vec![vec![Value::Integer(1)]]);
            }
            other => panic!("unexpected result {other:?}"),
        }
    }

    #[test]
    fn geometry_support_with_rtree() {
        let mut db = SqlDatabase::new();
        db.execute("CREATE TABLE shapes (id INT, area GEOMETRY);")
            .unwrap();
        db.execute("INSERT INTO shapes VALUES (1, GEOMETRY 'POINT(1 2)');")
            .unwrap();
        db.execute("INSERT INTO shapes VALUES (2, GEOMETRY '[0, 0, 5, 5]');")
            .unwrap();
        db.execute("INSERT INTO shapes VALUES (3, POINT(1 2));")
            .unwrap();

        let result = db
            .execute("SELECT id FROM shapes WHERE area = GEOMETRY 'POINT(1 2)';")
            .unwrap();
        match result {
            QueryResult::Rows { mut rows, .. } => {
                rows.sort_by_key(|row| match &row[0] {
                    Value::Integer(i) => *i,
                    _ => 0,
                });
                assert_eq!(rows, vec![vec![Value::Integer(1)], vec![Value::Integer(3)]]);
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

    #[test]
    fn select_with_cte() {
        let mut db = SqlDatabase::new();
        db.execute("CREATE TABLE base (id INT, value INT);")
            .unwrap();
        db.execute("INSERT INTO base VALUES (1, 10), (2, 20);")
            .unwrap();

        let result = db
            .execute(
                "WITH filtered AS (SELECT id, value FROM base WHERE value = 20) SELECT id FROM filtered;",
            )
            .unwrap();
        match result {
            QueryResult::Rows { columns, rows } => {
                assert_eq!(columns, vec!["id"]);
                assert_eq!(rows, vec![vec![Value::Integer(2)]]);
            }
            _ => panic!("unexpected result"),
        }
    }

    #[test]
    fn merge_upserts_rows() {
        let mut db = SqlDatabase::new();
        db.execute("CREATE TABLE target (id INT, amount INT);")
            .unwrap();
        db.execute("INSERT INTO target VALUES (1, 10);").unwrap();
        db.execute("CREATE TABLE updates (id INT, amount INT);")
            .unwrap();
        db.execute("INSERT INTO updates VALUES (1, 100), (2, 200);")
            .unwrap();

        db.execute(
            "MERGE INTO target USING updates ON target.id = updates.id \
             WHEN MATCHED THEN UPDATE SET amount = updates.amount \
             WHEN NOT MATCHED THEN INSERT (id, amount) VALUES (updates.id, updates.amount);",
        )
        .unwrap();

        let result = db.execute("SELECT * FROM target;").unwrap();
        match result {
            QueryResult::Rows { columns, rows } => {
                assert_eq!(columns, vec!["id", "amount"]);
                assert_eq!(rows.len(), 2);
                assert_eq!(rows[0], vec![Value::Integer(1), Value::Integer(100)]);
                assert_eq!(rows[1], vec![Value::Integer(2), Value::Integer(200)]);
            }
            _ => panic!("unexpected result"),
        }
    }

    #[test]
    fn merge_with_cte_source() {
        let mut db = SqlDatabase::new();
        db.execute("CREATE TABLE dest (id INT, name TEXT);")
            .unwrap();
        db.execute("INSERT INTO dest VALUES (1, 'old');").unwrap();
        db.execute("CREATE TABLE staging (id INT, name TEXT);")
            .unwrap();
        db.execute("INSERT INTO staging VALUES (1, 'new'), (3, 'third');")
            .unwrap();

        db.execute(
            "WITH candidate AS (SELECT id, name FROM staging) \
             MERGE INTO dest USING candidate ON dest.id = candidate.id \
             WHEN MATCHED THEN UPDATE SET name = candidate.name \
             WHEN NOT MATCHED THEN INSERT (id, name) VALUES (candidate.id, candidate.name);",
        )
        .unwrap();

        let result = db.execute("SELECT * FROM dest;").unwrap();
        match result {
            QueryResult::Rows { columns, rows } => {
                assert_eq!(columns, vec!["id", "name"]);
                assert_eq!(rows.len(), 2);
                assert_eq!(rows[0], vec![Value::Integer(1), Value::Text("new".into())]);
                assert_eq!(
                    rows[1],
                    vec![Value::Integer(3), Value::Text("third".into())]
                );
            }
            _ => panic!("unexpected result"),
        }
    }

    #[test]
    fn recursive_cte_traverses_hierarchy() {
        let mut db = SqlDatabase::new();
        db.execute("CREATE TABLE categories (id INT, parent_id INT, name TEXT);")
            .unwrap();
        db.execute(
            "INSERT INTO categories VALUES (1, NULL, 'root'), (2, 1, 'child'), (3, 2, 'grandchild'), (4, NULL, 'standalone');",
        )
        .unwrap();

        let result = db
            .execute(
                "WITH RECURSIVE tree AS (SELECT id, parent_id, name FROM categories WHERE parent_id IS NULL UNION ALL SELECT id, parent_id, name FROM categories WHERE parent_id IN tree.id) SELECT name FROM tree;",
            )
            .unwrap();

        match result {
            QueryResult::Rows { columns, rows } => {
                assert_eq!(columns, vec!["name".to_string()]);
                assert_eq!(rows.len(), 4);
                let mut names = rows
                    .into_iter()
                    .map(|row| match row.into_iter().next().unwrap() {
                        Value::Text(s) => s,
                        other => panic!("unexpected value {other:?}"),
                    })
                    .collect::<Vec<_>>();
                names.sort();
                assert_eq!(names, vec!["child", "grandchild", "root", "standalone"]);
            }
            _ => panic!("unexpected result"),
        }
    }

    #[test]
    fn full_text_search_basic() {
        let mut db = SqlDatabase::new();
        db.execute("CREATE TABLE docs (id INT, content TEXT);")
            .unwrap();
        db.execute(
            "INSERT INTO docs VALUES (1, 'Rust systems programming'), (2, 'Distributed databases and storage');",
        )
        .unwrap();

        let result = db
            .execute("SELECT id FROM docs WHERE content @@ 'systems programming';")
            .unwrap();
        match result {
            QueryResult::Rows { rows, .. } => {
                assert_eq!(rows.len(), 1);
                assert_eq!(rows[0][0], Value::Integer(1));
            }
            _ => panic!("unexpected result"),
        }
    }

    #[test]
    fn full_text_language_specific_analyzer() {
        let mut db = SqlDatabase::new();
        db.execute("CREATE TABLE articles (id INT, content TEXT);")
            .unwrap();
        db.execute(
            "INSERT INTO articles VALUES (1, 'The future of systems design'), (2, 'Rust excels at systems work');",
        )
        .unwrap();

        let default = db
            .execute("SELECT id FROM articles WHERE content @@ 'the systems';")
            .unwrap();
        match default {
            QueryResult::Rows { rows, .. } => {
                assert_eq!(rows.len(), 1);
                assert_eq!(rows[0][0], Value::Integer(1));
            }
            _ => panic!("unexpected result"),
        }

        let english = db
            .execute("SELECT id FROM articles WHERE content @@ 'the systems' LANGUAGE 'english';")
            .unwrap();
        match english {
            QueryResult::Rows { rows, .. } => {
                assert_eq!(rows.len(), 2);
                let mut ids = rows
                    .into_iter()
                    .map(|row| match row.get(0) {
                        Some(Value::Integer(id)) => *id,
                        _ => panic!("unexpected id"),
                    })
                    .collect::<Vec<_>>();
                ids.sort();
                assert_eq!(ids, vec![1, 2]);
            }
            _ => panic!("unexpected result"),
        }
    }

    #[test]
    fn full_text_spanish_analyzer() {
        let mut db = SqlDatabase::new();
        db.execute("CREATE TABLE textos (id INT, contenido TEXT);")
            .unwrap();
        db.execute(
            "INSERT INTO textos VALUES (1, 'Los sistemas distribuidos'), (2, 'Arquitectura de sistemas complejos');",
        )
        .unwrap();

        let result = db
            .execute("SELECT id FROM textos WHERE contenido @@ 'los sistemas' LANGUAGE 'spanish';")
            .unwrap();
        match result {
            QueryResult::Rows { rows, .. } => {
                assert_eq!(rows.len(), 2);
                let mut ids = rows
                    .into_iter()
                    .map(|row| match row.into_iter().next().unwrap() {
                        Value::Integer(id) => id,
                        _ => panic!("unexpected id"),
                    })
                    .collect::<Vec<_>>();
                ids.sort();
                assert_eq!(ids, vec![1, 2]);
            }
            _ => panic!("unexpected result"),
        }
    }

    #[test]
    fn materialized_view_synchronous_refresh() {
        let mut db = SqlDatabase::new();
        db.execute("CREATE TABLE users (id INT, active BOOLEAN);")
            .unwrap();
        db.execute("INSERT INTO users VALUES (1, true);").unwrap();
        db.execute(
            "CREATE MATERIALIZED VIEW active_users REFRESH SYNCHRONOUS AS SELECT id FROM users WHERE active = true;",
        )
        .unwrap();
        db.execute("INSERT INTO users VALUES (2, true);").unwrap();

        let result = db
            .execute("SELECT id FROM users WHERE active = true;")
            .unwrap();
        let mut ids: Vec<i64> = row_values(&result)
            .iter()
            .map(|row| match row.get(0) {
                Some(Value::Integer(id)) => *id,
                other => panic!("unexpected value {other:?}"),
            })
            .collect();
        ids.sort();
        assert_eq!(ids, vec![1, 2]);

        let direct = db.execute("SELECT id FROM active_users;").unwrap();
        let mut direct_ids: Vec<i64> = row_values(&direct)
            .iter()
            .map(|row| match row.get(0) {
                Some(Value::Integer(id)) => *id,
                other => panic!("unexpected value {other:?}"),
            })
            .collect();
        direct_ids.sort();
        assert_eq!(direct_ids, vec![1, 2]);
    }

    #[test]
    fn materialized_view_asynchronous_refresh_and_rewrite() {
        let mut db = SqlDatabase::new();
        db.execute("CREATE TABLE events (id INT, kind TEXT);")
            .unwrap();
        db.execute("INSERT INTO events VALUES (1, 'login');")
            .unwrap();
        db.execute(
            "CREATE MATERIALIZED VIEW login_events REFRESH ASYNCHRONOUS AS SELECT id FROM events WHERE kind = 'login';",
        )
        .unwrap();
        db.execute("INSERT INTO events VALUES (2, 'login');")
            .unwrap();

        let initial = db
            .execute("SELECT id FROM events WHERE kind = 'login';")
            .unwrap();
        let ids: Vec<i64> = row_values(&initial)
            .iter()
            .map(|row| match row.get(0) {
                Some(Value::Integer(id)) => *id,
                other => panic!("unexpected value {other:?}"),
            })
            .collect();
        assert_eq!(ids, vec![1]);

        db.run_pending_refreshes().unwrap();

        let refreshed = db
            .execute("SELECT id FROM events WHERE kind = 'login';")
            .unwrap();
        let mut refreshed_ids: Vec<i64> = row_values(&refreshed)
            .iter()
            .map(|row| match row.get(0) {
                Some(Value::Integer(id)) => *id,
                other => panic!("unexpected value {other:?}"),
            })
            .collect();
        refreshed_ids.sort();
        assert_eq!(refreshed_ids, vec![1, 2]);
    }

    #[test]
    fn analyze_populates_catalogs() {
        let mut db = SqlDatabase::new();
        db.execute("CREATE TABLE metrics (id INT, score FLOAT, label TEXT, created TIMESTAMP);")
            .unwrap();
        db.execute(
            "INSERT INTO metrics VALUES (1, 0.5, 'alpha', '2024-01-01T00:00:00Z'), (2, 0.7, 'beta', '2024-01-02T00:00:00Z'), (3, 0.7, 'beta', '2024-01-02T00:00:00Z');",
        )
        .unwrap();

        db.execute("ANALYZE metrics;").unwrap();

        let table_stats = db.get_table_statistics("metrics").expect("table stats");
        assert_eq!(table_stats.row_count, 3);
        assert_eq!(table_stats.stats_version, 1);

        let score_stats = db
            .get_column_statistics("metrics", "score")
            .expect("score stats");
        assert_eq!(score_stats.null_count, 0);
        assert_eq!(score_stats.distinct_count, 2);
        assert_eq!(score_stats.min, Some(Value::Float(0.5)));
        assert_eq!(score_stats.max, Some(Value::Float(0.7)));
        assert_eq!(score_stats.stats_version, 1);
        assert_eq!(score_stats.analyzed_at, table_stats.analyzed_at);

        let label_stats = db
            .get_column_statistics("metrics", "label")
            .expect("label stats");
        assert_eq!(label_stats.min, Some(Value::Text("alpha".into())));
        assert_eq!(label_stats.max, Some(Value::Text("beta".into())));

        let catalog = db
            .execute("SELECT table_name, row_count FROM __aidb_table_stats;")
            .unwrap();
        match catalog {
            QueryResult::Rows { rows, .. } => {
                assert!(rows.contains(&vec![Value::Text("metrics".into()), Value::Integer(3)]));
            }
            other => panic!("unexpected result {other:?}"),
        }

        let column_catalog = db
            .execute("SELECT table_name, column_name, distinct_count FROM __aidb_column_stats;")
            .unwrap();
        match column_catalog {
            QueryResult::Rows { rows, .. } => {
                assert!(rows.contains(&vec![
                    Value::Text("metrics".into()),
                    Value::Text("score".into()),
                    Value::Integer(2),
                ]));
            }
            other => panic!("unexpected result {other:?}"),
        }
    }

    #[test]
    fn analyze_increments_version_on_reanalyze() {
        let mut db = SqlDatabase::new();
        db.execute("CREATE TABLE readings (id INT, value FLOAT);")
            .unwrap();
        db.execute("INSERT INTO readings VALUES (1, 10.0), (2, 11.0);")
            .unwrap();
        db.execute("ANALYZE readings;").unwrap();
        let first = db.get_table_statistics("readings").expect("initial stats");

        db.execute("INSERT INTO readings VALUES (3, 12.5);")
            .unwrap();
        db.execute("ANALYZE readings;").unwrap();

        let second = db.get_table_statistics("readings").expect("updated stats");
        assert_eq!(second.stats_version, first.stats_version + 1);
        assert!(second.analyzed_at >= first.analyzed_at);

        let id_stats = db
            .get_column_statistics("readings", "id")
            .expect("id stats");
        assert_eq!(id_stats.max, Some(Value::Integer(3)));
        assert_eq!(id_stats.distinct_count, 3);
        assert_eq!(id_stats.stats_version, second.stats_version);
    }

    #[test]
    fn analyze_all_tables_without_argument() {
        let mut db = SqlDatabase::new();
        db.execute("CREATE TABLE a (id INT);").unwrap();
        db.execute("CREATE TABLE b (name TEXT);").unwrap();
        db.execute("INSERT INTO a VALUES (1), (2);").unwrap();
        db.execute("INSERT INTO b VALUES ('x');").unwrap();

        db.execute("ANALYZE;").unwrap();

        assert!(db.get_table_statistics("a").is_some());
        assert!(db.get_table_statistics("b").is_some());

        let a_id = db.get_column_statistics("a", "id").expect("stats for a.id");
        assert_eq!(a_id.distinct_count, 2);
    }

    #[test]
    fn analyze_unknown_table_returns_error() {
        let mut db = SqlDatabase::new();
        let err = db.execute("ANALYZE missing;");
        assert!(matches!(err, Err(SqlDatabaseError::UnknownTable(table)) if table == "missing"));
    }

    #[test]
    fn system_catalog_is_read_only() {
        let mut db = SqlDatabase::new();
        let err = db.execute(
            "INSERT INTO __aidb_table_stats VALUES ('demo', 1, '2024-01-01T00:00:00Z', 1);",
        );
        assert!(
            matches!(err, Err(SqlDatabaseError::SystemTableModification(name)) if name == "__aidb_table_stats")
        );
    }

    #[test]
    fn join_cardinality_estimator_matches_observed() {
        let mut db = SqlDatabase::new();
        db.execute("CREATE TABLE customers (id INT, region TEXT);")
            .unwrap();
        db.execute("CREATE TABLE orders (id INT, customer_id INT, total FLOAT);")
            .unwrap();
        db.execute(
            "INSERT INTO customers VALUES (1, 'north'), (2, 'south'), (3, 'north'), (4, 'west'), (5, 'south');",
        )
        .unwrap();
        db.execute(
            "INSERT INTO orders VALUES \
            (10, 1, 120.5), (11, 1, 80.0), (12, 2, 55.0), (13, 2, 75.0), (14, 2, 110.0), (15, 3, 20.0),\
            (16, 4, 42.0), (17, NULL, 15.0), (18, 6, 9.5);",
        )
        .unwrap();
        db.execute("ANALYZE customers;").unwrap();
        db.execute("ANALYZE orders;").unwrap();

        let customers_ctx = plan_context_for_table(&db, "customers");
        let orders_ctx = plan_context_for_table(&db, "orders");
        let predicate = JoinPredicate::inner("id", "customer_id");
        let estimate = CardinalityEstimator::estimate_join(&customers_ctx, &orders_ctx, &predicate);

        let actual = actual_join_cardinality(
            db.tables.get("customers").unwrap(),
            "id",
            db.tables.get("orders").unwrap(),
            "customer_id",
        ) as f64;

        let tolerance = actual.max(1.0) * 0.25;
        let diff = (estimate.estimated_rows - actual).abs();
        assert!(
            diff <= tolerance,
            "join cardinality estimate {estimate:?} diverges from observed {actual} (diff {diff})",
        );
        assert!(
            estimate.confidence >= 0.7,
            "expected high confidence, got {}",
            estimate.confidence
        );
    }

    #[test]
    fn join_estimator_handles_missing_statistics() {
        let mut db = SqlDatabase::new();
        db.execute("CREATE TABLE inventory (sku INT, qty INT);")
            .unwrap();
        db.execute("CREATE TABLE shipments (sku INT, shipped INT);")
            .unwrap();
        db.execute("INSERT INTO inventory VALUES (100, 20), (101, 15), (102, 5), (103, 12);")
            .unwrap();
        db.execute("INSERT INTO shipments VALUES (100, 3), (101, 7), (101, 4), (105, 9);")
            .unwrap();

        let inventory_ctx = plan_context_for_table(&db, "inventory");
        let shipments_ctx = plan_context_for_table(&db, "shipments");
        let predicate = JoinPredicate::inner("sku", "sku");
        let estimate =
            CardinalityEstimator::estimate_join(&inventory_ctx, &shipments_ctx, &predicate);

        assert!(
            estimate.estimated_rows > 0.0,
            "fallback estimation should be positive"
        );
        assert!(
            estimate.confidence <= 0.5,
            "fallback estimation should reflect low confidence, got {}",
            estimate.confidence
        );
    }

    #[test]
    fn cost_model_uses_cardinality_estimates() {
        let mut db = SqlDatabase::new();
        db.execute("CREATE TABLE authors (id INT, name TEXT);")
            .unwrap();
        db.execute("CREATE TABLE books (id INT, author_id INT, title TEXT);")
            .unwrap();
        db.execute("INSERT INTO authors VALUES (1, 'Ada'), (2, 'Grace'), (3, 'Edsger');")
            .unwrap();
        db.execute(
            "INSERT INTO books VALUES (10, 1, 'Programming'), (11, 1, 'Algorithms'), \
             (12, 2, 'Compilers'), (13, 2, 'COBOL'), (14, 2, 'Computing Machinery'), (15, 4, 'Ghost');",
        )
        .unwrap();
        db.execute("ANALYZE authors;").unwrap();
        db.execute("ANALYZE books;").unwrap();

        let authors_ctx = plan_context_for_table(&db, "authors");
        let books_ctx = plan_context_for_table(&db, "books");
        let predicate = Predicate::Equals {
            column: "author_id".into(),
            value: Value::Integer(2),
        };
        let cost_model = CostModel::new(8192.0, 1.0, 1.0);
        let mut scan_options = planner::ScanOptions::default();
        scan_options.pushdown_predicate = Some(predicate.clone());
        let scan_cost =
            cost_model.estimate_scan(&books_ctx, &planner::ScanCandidates::AllRows, &scan_options);
        assert!(scan_cost.cardinality.estimated_rows > 0.0);

        let join_predicate = JoinPredicate::inner("id", "author_id");
        let join_cost = cost_model.estimate_join(&authors_ctx, &books_ctx, &join_predicate);
        assert!(
            join_cost.cardinality.estimated_rows >= scan_cost.cardinality.estimated_rows,
            "join cardinality should dominate scan"
        );
        assert!(
            join_cost.cpu_cost > scan_cost.cpu_cost,
            "join CPU cost should exceed filtered scan cost"
        );
        assert!(
            join_cost.total_cost() > 0.0,
            "total cost should be positive"
        );
    }

    fn optimized_plan_for<'a>(
        db: &'a SqlDatabase,
        table_name: &'a str,
        predicate: Option<Predicate>,
        columns: SelectColumns,
    ) -> planner::LogicalPlan<'a> {
        let table = db
            .tables
            .get(table_name)
            .unwrap_or_else(|| panic!("table {table_name} not found"));
        let resolved = ResolvedTable::new(table_name, TableSource::Base, table);
        let table_stats = db.get_table_statistics(table_name);
        let column_stats = db.column_statistics(table_name);
        let context = PlanContext::new(resolved, table_stats, column_stats, None);
        let mut builder = LogicalPlanBuilder::scan(resolved);
        if let Some(pred) = predicate {
            builder = builder.filter(pred);
        }
        let plan = builder.project(columns).build();
        Planner::new(context)
            .optimize(plan)
            .expect("plan optimization should succeed")
    }

    fn plan_context_for_table<'a>(db: &'a SqlDatabase, table_name: &'a str) -> PlanContext<'a> {
        let table = db
            .tables
            .get(table_name)
            .unwrap_or_else(|| panic!("table {table_name} not found"));
        let resolved = ResolvedTable::new(table_name, TableSource::Base, table);
        let table_stats = db.get_table_statistics(table_name);
        let column_stats = db.column_statistics(table_name);
        PlanContext::new(resolved, table_stats, column_stats, None)
    }

    fn actual_join_cardinality(
        left: &super::Table,
        left_column: &str,
        right: &super::Table,
        right_column: &str,
    ) -> usize {
        let left_idx = left
            .columns
            .iter()
            .position(|c| c.name.eq_ignore_ascii_case(left_column))
            .expect("left join column");
        let right_idx = right
            .columns
            .iter()
            .position(|c| c.name.eq_ignore_ascii_case(right_column))
            .expect("right join column");
        let mut count = 0usize;
        for left_row in &left.rows {
            let left_value = &left_row[left_idx];
            if matches!(left_value, Value::Null) {
                continue;
            }
            for right_row in &right.rows {
                let right_value = &right_row[right_idx];
                if matches!(right_value, Value::Null) {
                    continue;
                }
                if left_value == right_value {
                    count += 1;
                }
            }
        }
        count
    }

    fn plan_components<'a>(
        plan: &'a planner::LogicalPlan<'a>,
    ) -> (
        &'a SelectColumns,
        Option<&'a Predicate>,
        &'a ScanCandidates,
        &'a planner::ScanOptions,
    ) {
        match &plan.root {
            LogicalExpr::Projection(projection) => match projection.input.as_ref() {
                LogicalExpr::Filter(filter) => match filter.input.as_ref() {
                    LogicalExpr::Scan(scan) => (
                        &projection.columns,
                        Some(&filter.predicate),
                        &scan.candidates,
                        &scan.options,
                    ),
                    _ => panic!("unexpected logical plan shape"),
                },
                LogicalExpr::Scan(scan) => {
                    (&projection.columns, None, &scan.candidates, &scan.options)
                }
                _ => panic!("unexpected logical plan shape"),
            },
            _ => panic!("unexpected logical plan root"),
        }
    }

    #[test]
    fn planner_partition_pruning_selects_partition_rows() {
        let mut db = SqlDatabase::new();
        db.execute(
            "CREATE TABLE events (id INT, event_time TIMESTAMP) PARTITION BY DAY(event_time);",
        )
        .unwrap();
        db.execute("INSERT INTO events VALUES (1, '2024-01-01T00:00:00Z');")
            .unwrap();
        db.execute("INSERT INTO events VALUES (2, '2024-01-02T00:00:00Z');")
            .unwrap();

        let predicate = Predicate::Equals {
            column: "event_time".to_string(),
            value: Value::Timestamp(
                Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0)
                    .single()
                    .expect("valid timestamp"),
            ),
        };

        let plan = optimized_plan_for(&db, "events", Some(predicate), SelectColumns::All);
        let (_, _, candidates, _) = plan_components(&plan);

        match candidates {
            ScanCandidates::Fixed(rows) => assert_eq!(rows.as_slice(), &[0]),
            other => panic!("expected partition pruning, got {other:?}"),
        }

        let result = db
            .execute("SELECT id FROM events WHERE event_time = '2024-01-01T00:00:00Z';")
            .unwrap();
        match result {
            QueryResult::Rows { rows, .. } => {
                assert_eq!(rows, vec![vec![Value::Integer(1)]]);
            }
            _ => panic!("unexpected result"),
        }
    }

    #[test]
    fn planner_json_index_selects_index_hits() {
        let mut db = SqlDatabase::new();
        db.execute("CREATE TABLE docs (id INT, data JSON);")
            .unwrap();
        db.execute("INSERT INTO docs VALUES (1, JSON '{\"k\":1}');")
            .unwrap();
        db.execute("INSERT INTO docs VALUES (2, JSON '{\"k\":2}');")
            .unwrap();

        let predicate = Predicate::Equals {
            column: "data".to_string(),
            value: Value::Json(json!({ "k": 1 })),
        };

        let plan = optimized_plan_for(
            &db,
            "docs",
            Some(predicate),
            SelectColumns::Some(vec![SelectItem::Column("id".into())]),
        );
        let (_, _, candidates, _) = plan_components(&plan);

        match candidates {
            ScanCandidates::Fixed(rows) => assert_eq!(rows.as_slice(), &[0]),
            other => panic!("expected json index usage, got {other:?}"),
        }

        let result = db
            .execute("SELECT id FROM docs WHERE data = JSON '{\"k\":1}';")
            .unwrap();
        match result {
            QueryResult::Rows { rows, .. } => {
                assert_eq!(rows, vec![vec![Value::Integer(1)]]);
            }
            _ => panic!("unexpected result"),
        }
    }

    #[test]
    fn planner_projection_pushdown_tracks_required_columns() {
        let mut db = SqlDatabase::new();
        db.execute("CREATE TABLE metrics (id INT, score INT, extra INT);")
            .unwrap();
        db.execute("INSERT INTO metrics VALUES (1, 10, 5);")
            .unwrap();

        let plan = optimized_plan_for(
            &db,
            "metrics",
            None,
            SelectColumns::Some(vec![
                SelectItem::Column("id".into()),
                SelectItem::Column("score".into()),
            ]),
        );
        let (_, _, _, options) = plan_components(&plan);

        match options.projected_columns.as_ref() {
            Some(SelectColumns::Some(items)) => {
                assert_eq!(items.len(), 2, "expected projected columns pushdown");
            }
            other => panic!("expected scan options to include projection pushdown, got {other:?}"),
        }
    }

    #[test]
    fn planner_without_predicate_scans_all_rows() {
        let mut db = SqlDatabase::new();
        db.execute("CREATE TABLE metrics (id INT, score INT);")
            .unwrap();
        db.execute("INSERT INTO metrics VALUES (1, 10), (2, 20);")
            .unwrap();

        let plan = optimized_plan_for(&db, "metrics", None, SelectColumns::All);
        let (_, predicate, candidates, _) = plan_components(&plan);

        assert!(predicate.is_none());
        match candidates {
            ScanCandidates::AllRows => {}
            other => panic!("expected full scan, got {other:?}"),
        }

        let result = db.execute("SELECT * FROM metrics;").unwrap();
        match result {
            QueryResult::Rows { rows, .. } => {
                assert_eq!(rows.len(), 2);
            }
            _ => panic!("unexpected result"),
        }
    }

    #[test]
    fn select_cache_hits_and_invalidation_on_insert() {
        let mut db = SqlDatabase::new();
        db.execute("CREATE TABLE users (id INT, name TEXT);")
            .unwrap();
        db.execute("INSERT INTO users VALUES (1, 'Alice');")
            .unwrap();

        assert_eq!(db.cache_stats().hits, 0);
        assert_eq!(db.cache_stats().misses, 0);

        let first = db
            .execute("SELECT id, name FROM users WHERE id = 1;")
            .unwrap();
        assert_eq!(row_values(&first).len(), 1);
        assert_eq!(db.cache_stats().misses, 1);
        assert_eq!(db.cache_stats().hits, 0);

        let second = db
            .execute("SELECT id, name FROM users WHERE id = 1;")
            .unwrap();
        assert_eq!(second, first);
        assert_eq!(db.cache_stats().hits, 1);

        db.execute("INSERT INTO users VALUES (2, 'Bob');").unwrap();
        let stats_after_insert = db.cache_stats();
        assert_eq!(stats_after_insert.hits, 1);

        let third = db
            .execute("SELECT id, name FROM users WHERE id = 1;")
            .unwrap();
        assert_eq!(third, first);
        let stats_after_third = db.cache_stats();
        assert_eq!(stats_after_third.hits, 1);
        assert_eq!(stats_after_third.misses, stats_after_insert.misses + 1);
    }

    #[test]
    fn vectorized_selects_are_cached() {
        let mut db = SqlDatabase::new();
        db.execute("CREATE TABLE metrics (id INT, score INT);")
            .unwrap();
        db.execute("INSERT INTO metrics VALUES (1, 5), (2, 10), (3, 15), (4, 20);")
            .unwrap();

        let first = db
            .execute_with_mode(
                "SELECT id FROM metrics WHERE score >= 10;",
                ExecutionMode::Vectorized,
            )
            .unwrap();
        assert_eq!(row_values(&first).len(), 3);
        let stats_after_first = db.cache_stats();
        assert_eq!(stats_after_first.misses, 1);

        let second = db
            .execute_with_mode(
                "SELECT id FROM metrics WHERE score >= 10;",
                ExecutionMode::Vectorized,
            )
            .unwrap();
        assert_eq!(second, first);
        let stats_after_second = db.cache_stats();
        assert_eq!(stats_after_second.hits, stats_after_first.hits + 1);
    }

    #[test]
    fn normalized_plan_signatures_ignore_identifier_case() {
        let mut db = SqlDatabase::new();
        db.execute("CREATE TABLE accounts (id INT, name TEXT);")
            .unwrap();
        db.execute("INSERT INTO accounts VALUES (1, 'alice');")
            .unwrap();

        let first = db
            .execute("SELECT name FROM accounts WHERE id = 1;")
            .unwrap();
        assert_eq!(row_values(&first).len(), 1);
        assert_eq!(db.cache_stats().misses, 1);

        let second = db
            .execute("select NAME from ACCOUNTS where ID = 1;")
            .unwrap();
        assert_eq!(second, first);
        assert_eq!(db.cache_stats().hits, 1);
    }

    #[test]
    fn cache_invalidation_for_stats_tables_and_mv_hook() {
        let mut db = SqlDatabase::new();
        db.execute("CREATE TABLE items (id INT, name TEXT);")
            .unwrap();
        db.execute("INSERT INTO items VALUES (1, 'alpha');")
            .unwrap();

        let first = db.execute("SELECT * FROM __aidb_table_stats;").unwrap();
        assert!(row_values(&first).is_empty());
        let after_first = db.cache_stats();
        assert_eq!(after_first.misses, 1);

        let second = db.execute("SELECT * FROM __aidb_table_stats;").unwrap();
        assert_eq!(row_values(&second), row_values(&first));
        let after_second = db.cache_stats();
        assert_eq!(after_second.hits, after_first.hits + 1);

        db.execute("ANALYZE items;").unwrap();
        let stats_after_analyze = db.cache_stats();

        let third = db.execute("SELECT * FROM __aidb_table_stats;").unwrap();
        assert!(!row_values(&third).is_empty());
        let after_third = db.cache_stats();
        assert_eq!(after_third.misses, stats_after_analyze.misses + 1);

        db.notify_materialized_view_update("__aidb_table_stats");
        let before_manual_invalidation = db.cache_stats();

        let fourth = db.execute("SELECT * FROM __aidb_table_stats;").unwrap();
        assert_eq!(row_values(&fourth), row_values(&third));
        let after_fourth = db.cache_stats();
        assert_eq!(after_fourth.misses, before_manual_invalidation.misses + 1);
    }
}
