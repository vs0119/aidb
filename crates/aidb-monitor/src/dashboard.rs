use axum::{
    extract::{Path, Query, State},
    http::StatusCode,
    response::{Html, IntoResponse},
    routing::{get, post},
    Json, Router,
};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use std::sync::Arc;
use tower_http::cors::CorsLayer;
use tower_http::services::ServeDir;

use crate::{HealthStatus, MetricsSnapshot, MonitoringEvent, MonitoringSystem};

// Dashboard web server
pub struct DashboardServer {
    monitoring: Arc<MonitoringSystem>,
    port: u16,
}

#[derive(Serialize, Deserialize)]
struct DashboardConfig {
    title: String,
    refresh_interval_seconds: u32,
    show_debug_events: bool,
    alert_sound_enabled: bool,
}

#[derive(Serialize)]
struct DashboardData {
    metrics: MetricsSnapshot,
    health_status: HealthStatus,
    recent_events: Vec<MonitoringEvent>,
    time_series: TimeSeriesData,
    active_alerts: Vec<crate::Alert>,
    config: DashboardConfig,
}

#[derive(Serialize)]
struct TimeSeriesData {
    timestamps: Vec<u64>,
    query_latency: Vec<f64>,
    qps: Vec<f64>,
    memory_usage: Vec<f64>,
    cpu_usage: Vec<f64>,
    error_rate: Vec<f64>,
}

impl DashboardServer {
    pub fn new(monitoring: Arc<MonitoringSystem>, port: u16) -> Self {
        Self { monitoring, port }
    }

    pub async fn start(self) -> anyhow::Result<()> {
        let app = Router::new()
            .route("/", get(dashboard_home))
            .route("/api/metrics", get(get_metrics))
            .route("/api/metrics/timeseries", get(get_timeseries))
            .route("/api/events", get(get_events))
            .route("/api/health", get(get_health))
            .route("/api/alerts", get(get_alerts))
            .route("/api/alerts/:id/acknowledge", post(acknowledge_alert))
            .route("/api/config", get(get_config).post(update_config))
            // AIDB Query Interface Endpoints
            .route(
                "/api/aidb/collections",
                get(aidb_list_collections).post(aidb_create_collection),
            )
            .route("/api/aidb/collections/:name", get(aidb_collection_info))
            .route(
                "/api/aidb/collections/:name/points",
                post(aidb_upsert_point),
            )
            .route(
                "/api/aidb/collections/:name/search",
                post(aidb_search_points),
            )
            .route(
                "/api/aidb/collections/:name/snapshot",
                post(aidb_snapshot_collection),
            )
            .route(
                "/api/aidb/collections/:name/compact",
                post(aidb_compact_collection),
            )
            .nest_service("/static", ServeDir::new("static"))
            .layer(CorsLayer::permissive())
            .with_state(Arc::clone(&self.monitoring));

        let addr = format!("0.0.0.0:{}", self.port);
        println!("🔍 Monitoring Dashboard starting on http://{}", addr);

        let listener = tokio::net::TcpListener::bind(&addr).await?;
        axum::serve(listener, app).await?;

        Ok(())
    }
}

// Dashboard HTML template
async fn dashboard_home() -> impl IntoResponse {
    let html = include_str!("../templates/dashboard.html");
    Html(html)
}

// API endpoints
async fn get_metrics(State(monitoring): State<Arc<MonitoringSystem>>) -> Json<MetricsSnapshot> {
    Json(monitoring.get_metrics_snapshot())
}

async fn get_timeseries(
    State(monitoring): State<Arc<MonitoringSystem>>,
    Query(params): Query<HashMap<String, String>>,
) -> Json<TimeSeriesData> {
    let time_series = monitoring.metrics.time_series.read();

    // Optional filtering by time range
    let limit = params
        .get("limit")
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(1000);

    let len = time_series.timestamps.len();
    let start_idx = if len > limit { len - limit } else { 0 };

    Json(TimeSeriesData {
        timestamps: time_series.timestamps[start_idx..].to_vec(),
        query_latency: time_series.query_latency[start_idx..].to_vec(),
        qps: time_series.qps[start_idx..].to_vec(),
        memory_usage: time_series.memory_usage[start_idx..].to_vec(),
        cpu_usage: time_series.cpu_usage[start_idx..].to_vec(),
        error_rate: time_series.error_rate[start_idx..].to_vec(),
    })
}

async fn get_events(
    State(monitoring): State<Arc<MonitoringSystem>>,
    Query(params): Query<HashMap<String, String>>,
) -> Json<Vec<MonitoringEvent>> {
    let mut events = Vec::new();
    let mut receiver = monitoring.event_sender.subscribe();

    let limit = params
        .get("limit")
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(50);

    // Get recent events (simplified - in real implementation would store event history)
    while events.len() < limit {
        match receiver.try_recv() {
            Ok(event) => events.push(event),
            Err(_) => break,
        }
    }

    Json(events)
}

async fn get_health(
    State(monitoring): State<Arc<MonitoringSystem>>,
) -> Json<Vec<crate::HealthCheck>> {
    let checks = monitoring.health.checks.read();
    Json(checks.clone())
}

async fn get_alerts(State(monitoring): State<Arc<MonitoringSystem>>) -> Json<Vec<crate::Alert>> {
    let alerts = monitoring.alerts.active_alerts.read();
    Json(alerts.clone())
}

async fn acknowledge_alert(
    State(monitoring): State<Arc<MonitoringSystem>>,
    Path(alert_id): Path<String>,
) -> Result<Json<String>, StatusCode> {
    let mut alerts = monitoring.alerts.active_alerts.write();

    if let Some(alert) = alerts.iter_mut().find(|a| a.id == alert_id) {
        alert.resolved_at = Some(
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
        );
        Ok(Json("Alert acknowledged".to_string()))
    } else {
        Err(StatusCode::NOT_FOUND)
    }
}

async fn get_config() -> Json<DashboardConfig> {
    Json(DashboardConfig {
        title: "AIDB Monitoring Dashboard".to_string(),
        refresh_interval_seconds: 5,
        show_debug_events: false,
        alert_sound_enabled: true,
    })
}

async fn update_config(Json(config): Json<DashboardConfig>) -> Json<String> {
    // In real implementation, would persist config
    Json("Configuration updated".to_string())
}

// Generate performance reports
pub fn generate_performance_report(monitoring: &MonitoringSystem) -> String {
    let metrics = monitoring.get_metrics_snapshot();
    let time_series = monitoring.metrics.time_series.read();

    let mut report = String::new();

    report.push_str("# AIDB Performance Report\n\n");
    report.push_str(&format!(
        "Generated at: {}\n\n",
        chrono::DateTime::from_timestamp(metrics.timestamp as i64, 0)
            .map(|dt| dt.format("%Y-%m-%d %H:%M:%S UTC").to_string())
            .unwrap_or_else(|| "Unknown".to_string())
    ));

    // Summary metrics
    report.push_str("## Summary\n\n");
    report.push_str(&format!("- **Total Queries**: {}\n", metrics.query_count));
    report.push_str(&format!("- **Query Errors**: {}\n", metrics.query_errors));
    report.push_str(&format!(
        "- **Average Latency**: {:.2}ms\n",
        metrics.avg_query_latency_ms
    ));
    report.push_str(&format!(
        "- **P95 Latency**: {:.2}ms\n",
        metrics.p95_query_latency_ms
    ));
    report.push_str(&format!("- **Queries per Second**: {:.1}\n", metrics.qps));
    report.push_str(&format!(
        "- **Memory Usage**: {:.1}MB\n",
        metrics.memory_usage_bytes as f64 / 1024.0 / 1024.0
    ));
    report.push_str(&format!(
        "- **CPU Usage**: {:.1}%\n",
        metrics.cpu_usage_percent
    ));
    report.push_str(&format!(
        "- **Collections**: {}\n",
        metrics.collections_count
    ));
    report.push_str(&format!("- **Total Vectors**: {}\n", metrics.total_vectors));

    let error_rate = if metrics.query_count > 0 {
        (metrics.query_errors as f64 / metrics.query_count as f64) * 100.0
    } else {
        0.0
    };
    report.push_str(&format!("- **Error Rate**: {:.2}%\n\n", error_rate));

    // Performance trends
    report.push_str("## Performance Trends\n\n");

    if !time_series.query_latency.is_empty() {
        let recent_latency: f64 =
            time_series.query_latency.iter().rev().take(10).sum::<f64>() / 10.0;
        let older_latency: f64 = time_series.query_latency.iter().take(10).sum::<f64>() / 10.0;
        let latency_change = ((recent_latency - older_latency) / older_latency) * 100.0;

        report.push_str(&format!(
            "- **Latency Trend**: {:.1}% over recent period\n",
            latency_change
        ));

        let recent_qps: f64 = time_series.qps.iter().rev().take(10).sum::<f64>() / 10.0;
        let older_qps: f64 = time_series.qps.iter().take(10).sum::<f64>() / 10.0;
        let qps_change = ((recent_qps - older_qps) / older_qps.max(1.0)) * 100.0;

        report.push_str(&format!(
            "- **QPS Trend**: {:.1}% over recent period\n",
            qps_change
        ));
    }

    // Health status
    report.push_str("\n## Health Status\n\n");
    let health_checks = monitoring.health.checks.read();
    let overall_status = monitoring.health.overall_status.read();

    report.push_str(&format!("- **Overall Status**: {:?}\n", *overall_status));
    report.push_str(&format!(
        "- **Health Checks**: {}/{} passing\n",
        health_checks
            .iter()
            .filter(|c| c.status == HealthStatus::Healthy)
            .count(),
        health_checks.len()
    ));

    // Active alerts
    let active_alerts = monitoring.alerts.active_alerts.read();
    report.push_str(&format!("\n## Active Alerts: {}\n\n", active_alerts.len()));

    for alert in active_alerts.iter() {
        report.push_str(&format!(
            "- **{}**: {} ({})\n",
            alert.rule_id,
            alert.message,
            format!("{:?}", alert.severity)
        ));
    }

    // Recommendations
    report.push_str("\n## Recommendations\n\n");

    if metrics.avg_query_latency_ms > 100.0 {
        report.push_str("- ⚠️ Consider optimizing queries or scaling resources (high latency)\n");
    }

    if error_rate > 1.0 {
        report.push_str("- ⚠️ Investigate error causes (high error rate)\n");
    }

    if metrics.cpu_usage_percent > 80.0 {
        report.push_str("- ⚠️ Consider adding more CPU resources\n");
    }

    if (metrics.memory_usage_bytes as f64 / 1024.0 / 1024.0 / 1024.0) > 4.0 {
        report.push_str("- ⚠️ High memory usage - consider enabling compression\n");
    }

    if metrics.qps < 10.0 && metrics.query_count > 100 {
        report.push_str("- 💡 Low throughput - investigate bottlenecks\n");
    }

    report
}

// AIDB Server Proxy Functions
const AIDB_SERVER_URL: &str = "http://localhost:8080";

async fn aidb_list_collections() -> Result<Json<Value>, (StatusCode, String)> {
    let client = Client::new();
    let response = client
        .get(&format!("{}/collections", AIDB_SERVER_URL))
        .send()
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    if response.status().is_success() {
        let json = response
            .json::<Value>()
            .await
            .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
        Ok(Json(json))
    } else {
        let status = response.status();
        let text = response.text().await.unwrap_or_default();
        Err((
            StatusCode::from_u16(status.as_u16()).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR),
            text,
        ))
    }
}

async fn aidb_create_collection(
    Json(payload): Json<Value>,
) -> Result<Json<Value>, (StatusCode, String)> {
    let client = Client::new();
    let response = client
        .post(&format!("{}/collections", AIDB_SERVER_URL))
        .json(&payload)
        .send()
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    if response.status().is_success() {
        let json = response
            .json::<Value>()
            .await
            .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
        Ok(Json(json))
    } else {
        let status = response.status();
        let text = response.text().await.unwrap_or_default();
        Err((
            StatusCode::from_u16(status.as_u16()).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR),
            text,
        ))
    }
}

async fn aidb_collection_info(
    Path(name): Path<String>,
) -> Result<Json<Value>, (StatusCode, String)> {
    let client = Client::new();
    let response = client
        .get(&format!("{}/collections/{}", AIDB_SERVER_URL, name))
        .send()
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    if response.status().is_success() {
        let json = response
            .json::<Value>()
            .await
            .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
        Ok(Json(json))
    } else {
        let status = response.status();
        let text = response.text().await.unwrap_or_default();
        Err((
            StatusCode::from_u16(status.as_u16()).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR),
            text,
        ))
    }
}

async fn aidb_upsert_point(
    Path(name): Path<String>,
    Json(payload): Json<Value>,
) -> Result<Json<Value>, (StatusCode, String)> {
    let client = Client::new();
    let response = client
        .post(&format!("{}/collections/{}/points", AIDB_SERVER_URL, name))
        .json(&payload)
        .send()
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    if response.status().is_success() {
        let json = response
            .json::<Value>()
            .await
            .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
        Ok(Json(json))
    } else {
        let status = response.status();
        let text = response.text().await.unwrap_or_default();
        Err((
            StatusCode::from_u16(status.as_u16()).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR),
            text,
        ))
    }
}

async fn aidb_search_points(
    Path(name): Path<String>,
    Json(payload): Json<Value>,
) -> Result<Json<Value>, (StatusCode, String)> {
    let client = Client::new();
    let response = client
        .post(&format!("{}/collections/{}/search", AIDB_SERVER_URL, name))
        .json(&payload)
        .send()
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    if response.status().is_success() {
        let json = response
            .json::<Value>()
            .await
            .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
        Ok(Json(json))
    } else {
        let status = response.status();
        let text = response.text().await.unwrap_or_default();
        Err((
            StatusCode::from_u16(status.as_u16()).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR),
            text,
        ))
    }
}

async fn aidb_snapshot_collection(
    Path(name): Path<String>,
    Json(payload): Json<Value>,
) -> Result<Json<Value>, (StatusCode, String)> {
    let client = Client::new();
    let response = client
        .post(&format!(
            "{}/collections/{}/snapshot",
            AIDB_SERVER_URL, name
        ))
        .json(&payload)
        .send()
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    if response.status().is_success() {
        let json = response
            .json::<Value>()
            .await
            .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
        Ok(Json(json))
    } else {
        let status = response.status();
        let text = response.text().await.unwrap_or_default();
        Err((
            StatusCode::from_u16(status.as_u16()).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR),
            text,
        ))
    }
}

async fn aidb_compact_collection(
    Path(name): Path<String>,
) -> Result<Json<Value>, (StatusCode, String)> {
    let client = Client::new();
    let response = client
        .post(&format!("{}/collections/{}/compact", AIDB_SERVER_URL, name))
        .send()
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    if response.status().is_success() {
        let json = response
            .json::<Value>()
            .await
            .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
        Ok(Json(json))
    } else {
        let status = response.status();
        let text = response.text().await.unwrap_or_default();
        Err((
            StatusCode::from_u16(status.as_u16()).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR),
            text,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dashboard_config_serialization() {
        let config = DashboardConfig {
            title: "Test Dashboard".to_string(),
            refresh_interval_seconds: 10,
            show_debug_events: true,
            alert_sound_enabled: false,
        };

        let json = serde_json::to_string(&config).unwrap();
        let deserialized: DashboardConfig = serde_json::from_str(&json).unwrap();

        assert_eq!(config.title, deserialized.title);
        assert_eq!(
            config.refresh_interval_seconds,
            deserialized.refresh_interval_seconds
        );
        assert_eq!(config.show_debug_events, deserialized.show_debug_events);
        assert_eq!(config.alert_sound_enabled, deserialized.alert_sound_enabled);
    }
}
