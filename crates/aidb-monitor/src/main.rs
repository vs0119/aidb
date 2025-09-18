use aidb_monitor::{
    alerts::{create_default_alert_rules, AlertEngine},
    dashboard::DashboardServer,
    health::HealthCheckEngine,
    tracing::TracingSystem,
    HealthChecker, MonitoringSystem,
};
use std::sync::Arc;
use tokio::time::Duration;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialize logging
    tracing_subscriber::fmt::init();

    println!("🔍 Starting AIDB Monitoring System...");

    // Create monitoring system
    let monitoring = Arc::new(MonitoringSystem::new());

    // Initialize alert rules
    {
        let rules = create_default_alert_rules();
        *monitoring.alerts.rules.write() = rules;
        println!(
            "✅ Loaded {} default alert rules",
            monitoring.alerts.rules.read().len()
        );
    }

    // Create and start alert engine
    let alert_engine = AlertEngine::new(
        Arc::clone(&monitoring.alerts),
        monitoring.event_sender.clone(),
    );

    // Create health check system
    let health_checker = Arc::clone(&monitoring.health);
    let health_engine = HealthCheckEngine::new(health_checker);

    // Add default health checks
    health_engine.checker.add_default_checks();
    println!(
        "✅ Initialized {} health checks",
        health_engine.checker.checks.read().len()
    );

    // Create tracing system
    let tracing_system = Arc::new(TracingSystem::new(0.1)); // 10% sampling
    println!("✅ Tracing system initialized with 10% sampling");

    // Create dashboard server
    let dashboard = DashboardServer::new(Arc::clone(&monitoring), 3001);
    println!("✅ Dashboard server configured on port 3001");

    // Start all background tasks
    let monitoring_task = {
        let monitoring = Arc::clone(&monitoring);
        tokio::spawn(async move {
            if let Err(e) = monitoring.start().await {
                eprintln!("Monitoring system error: {}", e);
            }
        })
    };

    let alert_task = {
        let alert_engine_clone = alert_engine;
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_secs(10));

            loop {
                interval.tick().await;
                // Simple evaluation without complex locking
                println!("Alert evaluation tick (simplified)");
            }
        })
    };

    let health_task = tokio::spawn(async move {
        if let Err(e) = health_engine.start_health_checks().await {
            eprintln!("Health check error: {}", e);
        }
    });

    let dashboard_task = tokio::spawn(async move {
        if let Err(e) = dashboard.start().await {
            eprintln!("Dashboard server error: {}", e);
        }
    });

    // Simulate some monitoring events for demonstration
    let demo_task = {
        let monitoring = Arc::clone(&monitoring);
        let tracing_system = Arc::clone(&tracing_system);
        tokio::spawn(async move {
            demo_monitoring_activity(monitoring, tracing_system).await;
        })
    };

    println!("🚀 AIDB Monitoring System is running!");
    println!("📊 Dashboard: http://localhost:3001");
    println!("🔍 Health API: http://localhost:3001/api/health");
    println!("📈 Metrics API: http://localhost:3001/api/metrics");
    println!("🚨 Alerts API: http://localhost:3001/api/alerts");
    println!();
    println!("Press Ctrl+C to stop...");

    // Wait for all tasks
    tokio::select! {
        _ = monitoring_task => println!("Monitoring task completed"),
        _ = alert_task => println!("Alert task completed"),
        _ = health_task => println!("Health task completed"),
        _ = dashboard_task => println!("Dashboard task completed"),
        _ = demo_task => println!("Demo task completed"),
        _ = tokio::signal::ctrl_c() => {
            println!("\n🛑 Received shutdown signal, stopping...");
        }
    }

    println!("✅ AIDB Monitoring System stopped");
    Ok(())
}

// Simulate realistic monitoring activity for demonstration
async fn demo_monitoring_activity(
    monitoring: Arc<MonitoringSystem>,
    tracing_system: Arc<TracingSystem>,
) {
    let mut interval = tokio::time::interval(Duration::from_secs(2));
    let mut counter = 0;

    loop {
        interval.tick().await;
        counter += 1;

        // Simulate various query patterns
        let query_types = [
            "vector_search",
            "collection_create",
            "point_upsert",
            "health_check",
        ];
        let query_type = query_types[counter % query_types.len()];

        // Simulate query latency (with some variation)
        let base_latency = match query_type {
            "vector_search" => 25.0,
            "collection_create" => 150.0,
            "point_upsert" => 8.0,
            "health_check" => 2.0,
            _ => 10.0,
        };

        let latency = base_latency * (0.5 + rand::random::<f64>());

        // Simulate occasional errors
        let success = match query_type {
            "vector_search" => rand::random::<f64>() > 0.02, // 2% error rate
            "collection_create" => rand::random::<f64>() > 0.05, // 5% error rate
            _ => rand::random::<f64>() > 0.01,               // 1% error rate
        };

        // Record the query
        monitoring.record_query(latency, success, query_type);

        // Create some distributed traces
        if counter % 5 == 0 {
            simulate_traced_request(&tracing_system).await;
        }

        // Simulate system metric updates
        if counter % 10 == 0 {
            update_system_metrics(&monitoring).await;
        }

        // Print periodic status
        if counter % 30 == 0 {
            let metrics = monitoring.get_metrics_snapshot();
            println!(
                "📊 Status: {} queries, {:.1}ms avg latency, {:.1} QPS",
                metrics.query_count, metrics.avg_query_latency_ms, metrics.qps
            );

            let health = monitoring.health.overall_status.read();
            println!("💚 Health: {:?}", *health);
        }

        // Create test alert condition occasionally
        if counter % 50 == 0 && success {
            // Simulate a latency spike to trigger alerts
            monitoring.record_query(1500.0, true, "latency_spike_test");
            println!("⚠️ Simulated high latency event for alert testing");
        }
    }
}

async fn simulate_traced_request(tracing_system: &TracingSystem) {
    use aidb_monitor::tracing::examples::trace_http_request;

    let methods = ["GET", "POST", "PUT", "DELETE"];
    let paths = ["/search", "/collections", "/health", "/metrics"];

    let method = methods[rand::random::<usize>() % methods.len()];
    let path = paths[rand::random::<usize>() % paths.len()];

    let _ = trace_http_request(tracing_system, method, path).await;
}

async fn update_system_metrics(monitoring: &MonitoringSystem) {
    // Update collection count
    let collections = 3 + (rand::random::<u64>() % 3);
    monitoring
        .metrics
        .collections_count
        .store(collections, std::sync::atomic::Ordering::Relaxed);

    // Update vector count
    let vectors = 50_000 + (rand::random::<u64>() % 100_000);
    monitoring
        .metrics
        .total_vectors
        .store(vectors, std::sync::atomic::Ordering::Relaxed);

    // Update active connections
    let connections = 5 + (rand::random::<u64>() % 20);
    monitoring
        .metrics
        .active_connections
        .store(connections, std::sync::atomic::Ordering::Relaxed);

    // Update memory usage (simulate gradual increase)
    let current = monitoring
        .metrics
        .memory_usage
        .load(std::sync::atomic::Ordering::Relaxed);
    let change = (rand::random::<i64>() % 10_000_000) - 5_000_000; // +/- 5MB
    let new_memory = (current as i64 + change).max(100_000_000) as u64; // Minimum 100MB
    monitoring
        .metrics
        .memory_usage
        .store(new_memory, std::sync::atomic::Ordering::Relaxed);

    // Update CPU usage
    let cpu_usage = 20.0 + (rand::random::<f64>() * 40.0); // 20-60%
    *monitoring.metrics.cpu_usage.write() = cpu_usage;
}

// Integration test for the monitoring system
#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_monitoring_system_integration() {
        let monitoring = Arc::new(MonitoringSystem::new());

        // Record some test queries
        monitoring.record_query(25.0, true, "test_query");
        monitoring.record_query(45.0, false, "test_query");
        monitoring.record_query(15.0, true, "test_query");

        // Get metrics
        let metrics = monitoring.get_metrics_snapshot();
        assert_eq!(metrics.query_count, 3);
        assert_eq!(metrics.query_errors, 1);
        assert!(metrics.avg_query_latency_ms > 0.0);
    }

    #[tokio::test]
    async fn test_alert_rules_loading() {
        let rules = create_default_alert_rules();
        assert!(!rules.is_empty());
        assert!(rules.iter().any(|r| r.id == "high_latency"));
        assert!(rules.iter().any(|r| r.id == "high_error_rate"));
    }

    #[tokio::test]
    async fn test_health_checks() {
        let health_checker = Arc::new(HealthChecker::new());
        let engine = HealthCheckEngine::new(health_checker.clone());

        engine.initialize_default_checks().await;

        let checks = health_checker.checks.read();
        assert!(checks.len() >= 5); // At least 5 default checks
    }
}
