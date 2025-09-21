use std::time::Instant;

use aidb_sql::SqlDatabase;
use serde::Serialize;

#[derive(Serialize)]
pub struct SqlJitBenchmarkReport {
    pub rows: usize,
    pub iterations: usize,
    pub warmup_iterations: usize,
    pub baseline_ms: f64,
    pub jit_ms: f64,
    pub speedup: f64,
}

pub fn run_sql_jit_benchmark(row_count: usize, iterations: usize) -> SqlJitBenchmarkReport {
    let warmup_iterations = 5;
    let query = "SELECT id, category FROM metrics WHERE value BETWEEN 50 AND 500;";

    let mut baseline_db = build_database(row_count);
    baseline_db.set_jit_enabled(false);
    for _ in 0..warmup_iterations {
        baseline_db
            .execute(query)
            .expect("execute baseline warmup query");
    }
    let baseline_start = Instant::now();
    for _ in 0..iterations {
        baseline_db
            .execute(query)
            .expect("execute baseline benchmark query");
    }
    let baseline_ms = baseline_start.elapsed().as_secs_f64() * 1_000.0;

    let mut jit_db = build_database(row_count);
    jit_db.set_jit_enabled(true);
    for _ in 0..warmup_iterations {
        jit_db.execute(query).expect("execute jit warmup query");
    }
    let jit_start = Instant::now();
    for _ in 0..iterations {
        jit_db.execute(query).expect("execute jit benchmark query");
    }
    let jit_ms = jit_start.elapsed().as_secs_f64() * 1_000.0;

    let speedup = if jit_ms > 0.0 {
        baseline_ms / jit_ms
    } else {
        0.0
    };

    SqlJitBenchmarkReport {
        rows: row_count,
        iterations,
        warmup_iterations,
        baseline_ms,
        jit_ms,
        speedup,
    }
}

fn build_database(row_count: usize) -> SqlDatabase {
    let mut db = SqlDatabase::new();
    db.execute("CREATE TABLE metrics (id INT, value INT, category INT);")
        .expect("create table");
    for i in 0..row_count {
        let value = (i as i64 % 1_000) + 100;
        let category = (i as i64 % 10) + 1;
        let stmt = format!(
            "INSERT INTO metrics VALUES ({}, {}, {});",
            i as i64, value, category
        );
        db.execute(&stmt).expect("insert row");
    }
    db
}
