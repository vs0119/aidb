use std::sync::Arc;

use aidb_sql::{ColumnType, ParquetConnector, QueryResult, SqlDatabase, Value};

fn setup_database() -> (SqlDatabase, Arc<ParquetConnector>) {
    let mut db = SqlDatabase::new();
    db.execute("CREATE TABLE residents (id INTEGER, city TEXT)")
        .unwrap();
    db.execute("INSERT INTO residents (id, city) VALUES (1, 'Paris'), (2, 'Berlin'), (3, 'Oslo')")
        .unwrap();

    let connector = Arc::new(
        ParquetConnector::new(
            vec![
                ("city", ColumnType::Text),
                ("population", ColumnType::Integer),
            ],
            vec![
                vec![Value::Text("Paris".into()), Value::Integer(2_148_000)],
                vec![Value::Text("Madrid".into()), Value::Integer(3_823_000)],
                vec![Value::Text("Berlin".into()), Value::Integer(3_769_000)],
            ],
        )
        .expect("connector should be constructed"),
    );

    db.register_external_table("city_stats", connector.clone())
        .expect("external table should register");

    (db, connector)
}

#[test]
fn federated_filter_uses_external_data() {
    let (mut db, _connector) = setup_database();
    let result = db
        .execute("SELECT id, city FROM residents WHERE city IN city_stats.city")
        .expect("query should succeed");

    match result {
        QueryResult::Rows { columns, rows } => {
            assert_eq!(columns, vec!["id".to_string(), "city".to_string()]);
            assert_eq!(rows.len(), 2);
            assert!(rows.contains(&vec![Value::Integer(1), Value::Text("Paris".into())]));
            assert!(rows.contains(&vec![Value::Integer(2), Value::Text("Berlin".into())]));
        }
        other => panic!("unexpected result: {other:?}"),
    }
}

#[test]
fn external_predicates_are_pushed_down() {
    let (mut db, connector) = setup_database();
    let result = db
        .execute("SELECT city FROM city_stats WHERE population BETWEEN 2000000 AND 3500000")
        .expect("external query should succeed");

    match result {
        QueryResult::Rows { columns, rows } => {
            assert_eq!(columns, vec!["city".to_string()]);
            assert_eq!(rows, vec![vec![Value::Text("Paris".into())]]);
        }
        other => panic!("unexpected result: {other:?}"),
    }

    assert!(connector.pushdown_call_count() > 0);
}
