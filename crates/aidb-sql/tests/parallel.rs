use aidb_sql::JoinPredicate;
use aidb_sql::{ExecutionConfig, ExecutionMode, QueryResult, SqlDatabase};

fn extract_row_count(result: &QueryResult) -> usize {
    match result {
        QueryResult::Rows { rows, .. } => rows.len(),
        _ => 0,
    }
}

#[test]
fn parallel_scan_matches_row_execution() {
    let mut db = SqlDatabase::new();
    db.set_execution_config(ExecutionConfig::with_parallelism(4));

    db.execute("CREATE TABLE metrics (id INTEGER, region TEXT);")
        .unwrap();
    for i in 0..10_000 {
        let region = if i % 2 == 0 { "east" } else { "west" };
        db.execute(&format!(
            "INSERT INTO metrics VALUES ({}, '{}');",
            i, region
        ))
        .unwrap();
    }

    let sql = "SELECT id FROM metrics WHERE id BETWEEN 100 AND 9000";
    let row = db
        .execute_with_mode(sql, ExecutionMode::Row)
        .expect("row execution should succeed");
    let auto = db.execute(sql).expect("auto execution should succeed");
    let vectorized = db
        .execute_with_mode(sql, ExecutionMode::Vectorized)
        .expect("vectorized execution should succeed");

    assert_eq!(extract_row_count(&row), 8901);
    assert_eq!(row, auto);
    assert_eq!(row, vectorized);
}

#[test]
fn parallel_hash_join_matches_sequential() {
    let mut db = SqlDatabase::new();
    db.execute("CREATE TABLE users (id INTEGER, name TEXT);")
        .unwrap();
    db.execute("CREATE TABLE events (user_id INTEGER, action TEXT);")
        .unwrap();

    for i in 0..2_000 {
        db.execute(&format!("INSERT INTO users VALUES ({}, 'user{}');", i, i))
            .unwrap();
        if i % 3 == 0 {
            db.execute(&format!("INSERT INTO events VALUES ({}, 'click');", i))
                .unwrap();
        }
    }

    let predicate = JoinPredicate::inner("id", "user_id");

    db.set_parallelism(1);
    let sequential = db
        .hash_join("users", "events", JoinPredicate::inner("id", "user_id"))
        .expect("sequential hash join");

    db.set_parallelism(4);
    let parallel = db
        .hash_join("users", "events", predicate)
        .expect("parallel hash join");

    assert_eq!(sequential, parallel);
    if let QueryResult::Rows { rows, .. } = sequential {
        assert_eq!(rows.len(), 667);
    } else {
        panic!("unexpected result variant");
    }
}
