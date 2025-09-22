use super::operators::{partition_scan_candidates, ScanPartition};
use super::scheduler::TaskScheduler;
use crate::planner::{JoinPredicate, JoinType, ScanCandidates};
use crate::{
    canonical_json, column_index_in_table, equal, ExecutionStats, Geometry, SqlDatabaseError,
    Table, Value,
};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

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

#[derive(Default, Clone)]
struct JoinPartitionOutcome {
    rows: Vec<Vec<Value>>,
    input_rows: usize,
    output_rows: usize,
}

fn partition_probe_rows(total_rows: usize, batch_size: usize) -> Vec<ScanPartition> {
    if total_rows == 0 || batch_size == 0 {
        return Vec::new();
    }

    let mut partitions = Vec::new();
    let mut start = 0;

    while start < total_rows {
        let end = (start + batch_size).min(total_rows);
        partitions.push(ScanPartition::Range { start, end });
        start = end;
    }

    partitions
}

fn value_hash_key(value: &Value) -> Option<String> {
    match value {
        Value::Null => None,
        Value::Integer(i) => Some(format!("i:{i}")),
        Value::Float(f) => Some(format!("f:{:016x}", f.to_bits())),
        Value::Text(s) => Some(format!("s:{s}")),
        Value::Boolean(b) => Some(format!("b:{}", if *b { 1 } else { 0 })),
        Value::Timestamp(ts) => Some(format!(
            "t:{}",
            ts.timestamp_nanos_opt().unwrap_or_default()
        )),
        Value::Json(v) => Some(format!("j:{}", canonical_json(v))),
        Value::Jsonb(v) => Some(format!("jb:{}", canonical_json(v))),
        Value::Xml(s) => Some(format!("x:{s}")),
        Value::Geometry(g) => Some(match g {
            Geometry::Point { x, y } => format!("gp:{x}:{y}"),
            Geometry::BoundingBox {
                min_x,
                min_y,
                max_x,
                max_y,
            } => format!("gb:{min_x}:{min_y}:{max_x}:{max_y}"),
        }),
    }
}

fn run_hash_partition(
    partition: &ScanPartition,
    build_ref: TableHandle,
    probe_ref: TableHandle,
    build_is_left: bool,
    _build_index: usize,
    probe_index: usize,
    hash_map: &HashMap<String, Vec<usize>>,
) -> JoinPartitionOutcome {
    let build_table = unsafe { build_ref.get() };
    let probe_table = unsafe { probe_ref.get() };
    let mut outcome = JoinPartitionOutcome::default();
    outcome.input_rows = partition.len();

    match partition {
        ScanPartition::Range { start, end } => {
            for probe_idx in *start..*end {
                let probe_row = &probe_table.rows[probe_idx];
                if let Some(key) = value_hash_key(&probe_row[probe_index]) {
                    if let Some(matches) = hash_map.get(&key) {
                        for &build_row_idx in matches {
                            let build_row = &build_table.rows[build_row_idx];
                            let (left_row, right_row) = if build_is_left {
                                (build_row, probe_row)
                            } else {
                                (probe_row, build_row)
                            };
                            let mut combined = Vec::with_capacity(left_row.len() + right_row.len());
                            combined.extend(left_row.iter().cloned());
                            combined.extend(right_row.iter().cloned());
                            outcome.rows.push(combined);
                        }
                    }
                }
            }
        }
        ScanPartition::Indices(indices) => {
            for &probe_idx in indices {
                let probe_row = &probe_table.rows[probe_idx];
                if let Some(key) = value_hash_key(&probe_row[probe_index]) {
                    if let Some(matches) = hash_map.get(&key) {
                        for &build_row_idx in matches {
                            let build_row = &build_table.rows[build_row_idx];
                            let (left_row, right_row) = if build_is_left {
                                (build_row, probe_row)
                            } else {
                                (probe_row, build_row)
                            };
                            let mut combined = Vec::with_capacity(left_row.len() + right_row.len());
                            combined.extend(left_row.iter().cloned());
                            combined.extend(right_row.iter().cloned());
                            outcome.rows.push(combined);
                        }
                    }
                }
            }
        }
    }

    outcome.output_rows = outcome.rows.len();
    outcome
}

fn run_nested_partition(
    partition: &ScanPartition,
    build_ref: TableHandle,
    probe_ref: TableHandle,
    build_is_left: bool,
    build_index: usize,
    probe_index: usize,
) -> JoinPartitionOutcome {
    let build_table = unsafe { build_ref.get() };
    let probe_table = unsafe { probe_ref.get() };
    let mut outcome = JoinPartitionOutcome::default();
    outcome.input_rows = partition.len();

    match partition {
        ScanPartition::Range { start, end } => {
            for probe_idx in *start..*end {
                let probe_row = &probe_table.rows[probe_idx];
                for build_row in &build_table.rows {
                    if equal(&build_row[build_index], &probe_row[probe_index]) {
                        let (left_row, right_row) = if build_is_left {
                            (build_row, probe_row)
                        } else {
                            (probe_row, build_row)
                        };
                        let mut combined = Vec::with_capacity(left_row.len() + right_row.len());
                        combined.extend(left_row.iter().cloned());
                        combined.extend(right_row.iter().cloned());
                        outcome.rows.push(combined);
                    }
                }
            }
        }
        ScanPartition::Indices(indices) => {
            for &probe_idx in indices {
                let probe_row = &probe_table.rows[probe_idx];
                for build_row in &build_table.rows {
                    if equal(&build_row[build_index], &probe_row[probe_index]) {
                        let (left_row, right_row) = if build_is_left {
                            (build_row, probe_row)
                        } else {
                            (probe_row, build_row)
                        };
                        let mut combined = Vec::with_capacity(left_row.len() + right_row.len());
                        combined.extend(left_row.iter().cloned());
                        combined.extend(right_row.iter().cloned());
                        outcome.rows.push(combined);
                    }
                }
            }
        }
    }

    outcome.output_rows = outcome.rows.len();
    outcome
}

pub(crate) fn parallel_hash_join(
    left: &Table,
    right: &Table,
    predicate: &JoinPredicate,
    scheduler: Option<&TaskScheduler>,
    batch_size: usize,
    skew_threshold: usize,
    stats: Option<&Arc<Mutex<ExecutionStats>>>,
) -> Result<Vec<Vec<Value>>, SqlDatabaseError> {
    if predicate.join_type != JoinType::Inner {
        return Err(SqlDatabaseError::Unsupported);
    }

    let left_column = column_index_in_table(left, &predicate.left_column)?;
    let right_column = column_index_in_table(right, &predicate.right_column)?;

    let (build_table, build_index, probe_table, probe_index, build_is_left) =
        if left.rows.len() <= right.rows.len() {
            (left, left_column, right, right_column, true)
        } else {
            (right, right_column, left, left_column, false)
        };

    let mut hash_map: HashMap<String, Vec<usize>> = HashMap::with_capacity(build_table.rows.len());
    for (row_idx, row) in build_table.rows.iter().enumerate() {
        if let Some(key) = value_hash_key(&row[build_index]) {
            hash_map.entry(key).or_default().push(row_idx);
        }
    }

    if hash_map.is_empty() {
        if let Some(stats) = stats {
            if let Ok(mut s) = stats.lock() {
                s.join_build_rows = build_table.rows.len();
                s.join_probe_rows = probe_table.rows.len();
                s.join_output_rows = 0;
                s.join_skew_detected = false;
                s.join_strategy_switches = 0;
            }
        }
        return Ok(Vec::new());
    }

    let partitions = partition_scan_candidates(
        &ScanCandidates::AllRows,
        probe_table.rows.len(),
        batch_size.max(1),
    );
    if partitions.is_empty() {
        return Ok(Vec::new());
    }

    let max_bucket = hash_map.values().map(|v| v.len()).max().unwrap_or(0);
    let use_skew_fallback = max_bucket > skew_threshold.max(1);

    let hash_map = Arc::new(hash_map);
    let build_ref = TableHandle::new(build_table);
    let probe_ref = TableHandle::new(probe_table);
    let results_main = Arc::new(Mutex::new(vec![
        JoinPartitionOutcome::default();
        partitions.len()
    ]));

    if scheduler.is_some() && partitions.len() > 1 {
        let scheduler = scheduler.expect("scheduler required for parallel execution");
        let results_handle = Arc::clone(&results_main);
        let hash_map_handle = Arc::clone(&hash_map);
        let build_ref_copy = build_ref;
        let probe_ref_copy = probe_ref;
        let tasks = partitions
            .into_iter()
            .enumerate()
            .map(move |(idx, partition)| {
                let results = Arc::clone(&results_handle);
                let hash_map = Arc::clone(&hash_map_handle);
                move || {
                    let outcome = if use_skew_fallback {
                        run_nested_partition(
                            &partition,
                            build_ref_copy,
                            probe_ref_copy,
                            build_is_left,
                            build_index,
                            probe_index,
                        )
                    } else {
                        run_hash_partition(
                            &partition,
                            build_ref_copy,
                            probe_ref_copy,
                            build_is_left,
                            build_index,
                            probe_index,
                            hash_map.as_ref(),
                        )
                    };
                    let mut guard = results.lock().unwrap();
                    guard[idx] = outcome;
                }
            });
        scheduler.execute(tasks);
    } else {
        let mut guard = results_main.lock().unwrap();
        for (idx, partition) in partitions.iter().enumerate() {
            guard[idx] = if use_skew_fallback {
                run_nested_partition(
                    partition,
                    build_ref,
                    probe_ref,
                    build_is_left,
                    build_index,
                    probe_index,
                )
            } else {
                run_hash_partition(
                    partition,
                    build_ref,
                    probe_ref,
                    build_is_left,
                    build_index,
                    probe_index,
                    hash_map.as_ref(),
                )
            };
        }
    }

    let mut total_output = 0usize;
    let mut rows = Vec::new();
    if let Ok(mut outcomes) = results_main.lock() {
        for outcome in outcomes.iter_mut() {
            total_output += outcome.output_rows;
            rows.append(&mut outcome.rows);
        }
    }

    if let Some(stats) = stats {
        if let Ok(mut s) = stats.lock() {
            s.join_build_rows = build_table.rows.len();
            s.join_probe_rows = probe_table.rows.len();
            s.join_output_rows = total_output;
            s.join_skew_detected = use_skew_fallback;
            s.join_strategy_switches = if use_skew_fallback { 1 } else { 0 };
        }
    }

    Ok(rows)
}
