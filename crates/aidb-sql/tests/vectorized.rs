use aidb_sql::{ExecutionMode, SqlDatabase, SqlDatabaseError};

fn setup_db() -> SqlDatabase {
    let mut db = SqlDatabase::new();

    db.execute("CREATE TABLE metrics (id INTEGER, amount FLOAT, category TEXT, flag BOOLEAN)")
        .unwrap();
    db.execute("INSERT INTO metrics (id, amount, category, flag) VALUES (1, 10, 'A', true)")
        .unwrap();
    db.execute("INSERT INTO metrics (id, amount, category, flag) VALUES (2, 25, 'B', false)")
        .unwrap();
    db.execute("INSERT INTO metrics (id, amount, category, flag) VALUES (3, 40, 'C', true)")
        .unwrap();
    db.execute("INSERT INTO metrics (id, amount, category, flag) VALUES (4, 15, NULL, false)")
        .unwrap();

    db.execute("CREATE TABLE categories (name TEXT)").unwrap();
    db.execute("INSERT INTO categories (name) VALUES ('A'), ('C')")
        .unwrap();

    db.execute("CREATE TABLE documents (id INTEGER, body TEXT)")
        .unwrap();
    db.execute(
        "INSERT INTO documents (id, body) VALUES \
         (1, 'Rust database systems'),\
         (2, 'Vectorized execution rocks'),\
         (3, 'Learning databases by building one')",
    )
    .unwrap();

    db
}

fn assert_modes_match(db: &mut SqlDatabase, sql: &str) {
    let row = db
        .execute_with_mode(sql, ExecutionMode::Row)
        .expect("row execution should succeed");
    let vectorized = db
        .execute_with_mode(sql, ExecutionMode::Vectorized)
        .expect("vectorized execution should succeed");
    assert_eq!(row, vectorized);
}

#[test]
fn vectorized_matches_row_for_simple_scan() {
    let mut db = setup_db();
    assert_modes_match(&mut db, "SELECT * FROM metrics");
}

#[test]
fn vectorized_handles_equality_predicates() {
    let mut db = setup_db();
    assert_modes_match(
        &mut db,
        "SELECT id, category FROM metrics WHERE category = 'A'",
    );
}

#[test]
fn vectorized_handles_between_predicates() {
    let mut db = setup_db();
    assert_modes_match(
        &mut db,
        "SELECT id FROM metrics WHERE amount BETWEEN 10 AND 30",
    );
}

#[test]
fn vectorized_handles_is_null_predicates() {
    let mut db = setup_db();
    assert_modes_match(&mut db, "SELECT id FROM metrics WHERE category IS NULL");
}

#[test]
fn vectorized_handles_in_table_predicates() {
    let mut db = setup_db();
    assert_modes_match(
        &mut db,
        "SELECT id FROM metrics WHERE category IN categories.name",
    );
}

#[test]
fn vectorized_handles_full_text_predicates() {
    let mut db = setup_db();
    assert_modes_match(
        &mut db,
        "SELECT id FROM documents WHERE body @@ 'database' LANGUAGE 'english'",
    );
}

#[test]
fn vectorized_mode_rejects_window_functions() {
    let mut db = setup_db();
    let sql = "SELECT id, ROW_NUMBER() OVER (ORDER BY id) FROM metrics";

    match db.execute_with_mode(sql, ExecutionMode::Vectorized) {
        Err(SqlDatabaseError::Unsupported) => {}
        other => panic!(
            "expected Unsupported error for vectorized mode, got {:?}",
            other
        ),
    }

    let row = db
        .execute_with_mode(sql, ExecutionMode::Row)
        .expect("row execution should succeed");
    let auto = db.execute(sql).expect("auto execution should succeed");
    assert_eq!(row, auto);
}

#[test]
fn auto_mode_matches_vectorized_when_available() {
    let mut db = setup_db();
    let sql = "SELECT id, amount FROM metrics WHERE amount BETWEEN 10 AND 40";

    let auto = db.execute(sql).expect("auto execution should succeed");
    let vectorized = db
        .execute_with_mode(sql, ExecutionMode::Vectorized)
        .expect("vectorized execution should succeed");
    assert_eq!(auto, vectorized);
}
