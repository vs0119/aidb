use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use tokio::sync::broadcast;

pub mod alerts;
pub mod dashboard;
pub mod health;
pub mod tracing;

// Core monitoring system that collects and aggregates all metrics
#[derive(Clone)]
pub struct MonitoringSystem {
    pub metrics: Arc<MetricsCollector>,
    pub alerts: Arc<AlertManager>,
    pub health: Arc<HealthChecker>,
    pub event_sender: broadcast::Sender<MonitoringEvent>,
}

// Central metrics collector
#[derive(Debug)]
pub struct MetricsCollector {
    // Query metrics
    pub query_count: AtomicU64,
    pub query_latency_histogram: RwLock<Vec<f64>>,
    pub query_errors: AtomicU64,

    // System metrics
    pub memory_usage: AtomicU64,
    pub cpu_usage: RwLock<f64>,
    pub disk_usage: AtomicU64,
    pub network_io: RwLock<NetworkStats>,

    // Database metrics
    pub collections_count: AtomicU64,
    pub total_vectors: AtomicU64,
    pub index_operations: RwLock<HashMap<String, u64>>,

    // Performance metrics
    pub throughput_qps: RwLock<f64>,
    pub cache_hit_rate: RwLock<f64>,
    pub active_connections: AtomicU64,

    // Custom metrics
    pub custom_metrics: RwLock<HashMap<String, MetricValue>>,

    // Time series data (last 1000 points)
    pub time_series: RwLock<TimeSeries>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkStats {
    pub bytes_sent: u64,
    pub bytes_received: u64,
    pub packets_sent: u64,
    pub packets_received: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MetricValue {
    Counter(u64),
    Gauge(f64),
    Histogram(Vec<f64>),
    Text(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeSeries {
    pub timestamps: Vec<u64>,
    pub query_latency: Vec<f64>,
    pub qps: Vec<f64>,
    pub memory_usage: Vec<f64>,
    pub cpu_usage: Vec<f64>,
    pub error_rate: Vec<f64>,
    pub max_points: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonitoringEvent {
    pub timestamp: u64,
    pub event_type: EventType,
    pub severity: Severity,
    pub message: String,
    pub metadata: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EventType {
    Query,
    Error,
    Alert,
    SystemInfo,
    Performance,
    Health,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, PartialOrd)]
pub enum Severity {
    Debug,
    Info,
    Warning,
    Error,
    Critical,
}

// Alert management system
#[derive(Debug)]
pub struct AlertManager {
    pub rules: RwLock<Vec<AlertRule>>,
    pub active_alerts: RwLock<Vec<Alert>>,
    pub alert_history: RwLock<Vec<Alert>>,
    pub notification_channels: RwLock<Vec<NotificationChannel>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertRule {
    pub id: String,
    pub name: String,
    pub condition: AlertCondition,
    pub severity: Severity,
    pub cooldown_seconds: u64,
    pub enabled: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertCondition {
    pub metric: String,
    pub operator: ComparisonOperator,
    pub threshold: f64,
    pub duration_seconds: u64, // Alert only if condition persists
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ComparisonOperator {
    GreaterThan,
    LessThan,
    Equals,
    NotEquals,
    GreaterThanOrEqual,
    LessThanOrEqual,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Alert {
    pub id: String,
    pub rule_id: String,
    pub message: String,
    pub severity: Severity,
    pub triggered_at: u64,
    pub resolved_at: Option<u64>,
    pub metadata: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NotificationChannel {
    Email { recipients: Vec<String> },
    Webhook { url: String },
    Console,
}

// Health checking system
#[derive(Debug)]
pub struct HealthChecker {
    pub checks: RwLock<Vec<HealthCheck>>,
    pub overall_status: RwLock<HealthStatus>,
    pub last_check: RwLock<Option<Instant>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthCheck {
    pub name: String,
    pub check_type: HealthCheckType,
    pub status: HealthStatus,
    pub message: String,
    pub last_check: u64,
    pub check_duration_ms: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum HealthCheckType {
    DatabaseConnection,
    DiskSpace,
    MemoryUsage,
    ResponseTime,
    IndexHealth,
    Custom(String),
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum HealthStatus {
    Healthy,
    Warning,
    Critical,
    Unknown,
}

impl MonitoringSystem {
    pub fn new() -> Self {
        let (event_sender, _) = broadcast::channel(1000);

        Self {
            metrics: Arc::new(MetricsCollector::new()),
            alerts: Arc::new(AlertManager::new()),
            health: Arc::new(HealthChecker::new()),
            event_sender,
        }
    }

    // Start the monitoring background tasks
    pub async fn start(&self) -> anyhow::Result<()> {
        let metrics = Arc::clone(&self.metrics);
        let alerts = Arc::clone(&self.alerts);
        let health = Arc::clone(&self.health);
        let event_sender = self.event_sender.clone();

        // System metrics collection task
        let metrics_task = tokio::spawn(async move {
            Self::collect_system_metrics(metrics, event_sender).await;
        });

        // Alert evaluation task
        let alerts_task = tokio::spawn(async move {
            Self::evaluate_alerts(alerts).await;
        });

        // Health check task
        let health_task = tokio::spawn(async move {
            Self::run_health_checks(health).await;
        });

        tokio::try_join!(metrics_task, alerts_task, health_task)?;
        Ok(())
    }

    // Record a query event
    pub fn record_query(&self, latency_ms: f64, success: bool, query_type: &str) {
        self.metrics.query_count.fetch_add(1, Ordering::Relaxed);

        if !success {
            self.metrics.query_errors.fetch_add(1, Ordering::Relaxed);
        }

        // Add to latency histogram
        {
            let mut histogram = self.metrics.query_latency_histogram.write();
            histogram.push(latency_ms);
            if histogram.len() > 10000 {
                histogram.remove(0); // Keep last 10k measurements
            }
        }

        // Update time series
        self.metrics.update_time_series(latency_ms);

        // Send monitoring event
        let event = MonitoringEvent {
            timestamp: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            event_type: EventType::Query,
            severity: if success {
                Severity::Debug
            } else {
                Severity::Warning
            },
            message: format!("Query {} took {:.2}ms", query_type, latency_ms),
            metadata: {
                let mut meta = HashMap::new();
                meta.insert("latency_ms".to_string(), latency_ms.to_string());
                meta.insert("success".to_string(), success.to_string());
                meta.insert("query_type".to_string(), query_type.to_string());
                meta
            },
        };

        let _ = self.event_sender.send(event);
    }

    // Get current metrics snapshot
    pub fn get_metrics_snapshot(&self) -> MetricsSnapshot {
        let histogram = self.metrics.query_latency_histogram.read();

        let (avg_latency, p95_latency) = if histogram.is_empty() {
            (0.0, 0.0)
        } else {
            let sum: f64 = histogram.iter().sum();
            let avg = sum / histogram.len() as f64;

            let mut sorted = histogram.clone();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let p95_idx = (0.95 * sorted.len() as f64) as usize;
            let p95 = sorted.get(p95_idx).copied().unwrap_or(0.0);

            (avg, p95)
        };

        MetricsSnapshot {
            timestamp: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            query_count: self.metrics.query_count.load(Ordering::Relaxed),
            query_errors: self.metrics.query_errors.load(Ordering::Relaxed),
            avg_query_latency_ms: avg_latency,
            p95_query_latency_ms: p95_latency,
            memory_usage_bytes: self.metrics.memory_usage.load(Ordering::Relaxed),
            cpu_usage_percent: *self.metrics.cpu_usage.read(),
            qps: *self.metrics.throughput_qps.read(),
            active_connections: self.metrics.active_connections.load(Ordering::Relaxed),
            collections_count: self.metrics.collections_count.load(Ordering::Relaxed),
            total_vectors: self.metrics.total_vectors.load(Ordering::Relaxed),
        }
    }

    async fn collect_system_metrics(
        metrics: Arc<MetricsCollector>,
        event_sender: broadcast::Sender<MonitoringEvent>,
    ) {
        let mut interval = tokio::time::interval(Duration::from_secs(5));

        loop {
            interval.tick().await;

            // Collect system metrics (simplified - would use systemstat in real implementation)
            let memory = Self::get_memory_usage().unwrap_or(0);
            metrics.memory_usage.store(memory, Ordering::Relaxed);

            let cpu = Self::get_cpu_usage().unwrap_or(0.0);
            *metrics.cpu_usage.write() = cpu;

            // Calculate QPS
            let current_queries = metrics.query_count.load(Ordering::Relaxed);
            let qps = current_queries as f64 / 5.0; // Simple calculation
            *metrics.throughput_qps.write() = qps;

            // Send system info event
            let event = MonitoringEvent {
                timestamp: SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap()
                    .as_secs(),
                event_type: EventType::SystemInfo,
                severity: Severity::Debug,
                message: format!(
                    "System metrics: {:.1}% CPU, {} MB RAM, {:.1} QPS",
                    cpu,
                    memory / 1024 / 1024,
                    qps
                ),
                metadata: HashMap::new(),
            };

            let _ = event_sender.send(event);
        }
    }

    async fn evaluate_alerts(alerts: Arc<AlertManager>) {
        let mut interval = tokio::time::interval(Duration::from_secs(10));

        loop {
            interval.tick().await;
            // Alert evaluation logic would go here
            // For now, just a placeholder
        }
    }

    async fn run_health_checks(health: Arc<HealthChecker>) {
        let mut interval = tokio::time::interval(Duration::from_secs(30));

        loop {
            interval.tick().await;
            // Health check logic would go here
            // For now, just update the last check time
            *health.last_check.write() = Some(Instant::now());
        }
    }

    fn get_memory_usage() -> Result<u64, anyhow::Error> {
        // Simplified - would use systemstat::Platform::memory() in real implementation
        Ok(1024 * 1024 * 512) // 512 MB placeholder
    }

    fn get_cpu_usage() -> Result<f64, anyhow::Error> {
        // Simplified - would use systemstat::Platform::cpu_load() in real implementation
        Ok(25.5) // 25.5% placeholder
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricsSnapshot {
    pub timestamp: u64,
    pub query_count: u64,
    pub query_errors: u64,
    pub avg_query_latency_ms: f64,
    pub p95_query_latency_ms: f64,
    pub memory_usage_bytes: u64,
    pub cpu_usage_percent: f64,
    pub qps: f64,
    pub active_connections: u64,
    pub collections_count: u64,
    pub total_vectors: u64,
}

impl MetricsCollector {
    pub fn new() -> Self {
        Self {
            query_count: AtomicU64::new(0),
            query_latency_histogram: RwLock::new(Vec::new()),
            query_errors: AtomicU64::new(0),
            memory_usage: AtomicU64::new(0),
            cpu_usage: RwLock::new(0.0),
            disk_usage: AtomicU64::new(0),
            network_io: RwLock::new(NetworkStats {
                bytes_sent: 0,
                bytes_received: 0,
                packets_sent: 0,
                packets_received: 0,
            }),
            collections_count: AtomicU64::new(0),
            total_vectors: AtomicU64::new(0),
            index_operations: RwLock::new(HashMap::new()),
            throughput_qps: RwLock::new(0.0),
            cache_hit_rate: RwLock::new(0.0),
            active_connections: AtomicU64::new(0),
            custom_metrics: RwLock::new(HashMap::new()),
            time_series: RwLock::new(TimeSeries::new()),
        }
    }

    fn update_time_series(&self, latency: f64) {
        let mut ts = self.time_series.write();
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();

        ts.timestamps.push(now);
        ts.query_latency.push(latency);
        ts.qps.push(*self.throughput_qps.read());
        ts.memory_usage
            .push(self.memory_usage.load(Ordering::Relaxed) as f64);
        ts.cpu_usage.push(*self.cpu_usage.read());

        let error_rate = if self.query_count.load(Ordering::Relaxed) > 0 {
            self.query_errors.load(Ordering::Relaxed) as f64
                / self.query_count.load(Ordering::Relaxed) as f64
        } else {
            0.0
        };
        ts.error_rate.push(error_rate);

        // Keep only last N points
        if ts.timestamps.len() > ts.max_points {
            ts.timestamps.remove(0);
            ts.query_latency.remove(0);
            ts.qps.remove(0);
            ts.memory_usage.remove(0);
            ts.cpu_usage.remove(0);
            ts.error_rate.remove(0);
        }
    }
}

impl TimeSeries {
    pub fn new() -> Self {
        Self {
            timestamps: Vec::new(),
            query_latency: Vec::new(),
            qps: Vec::new(),
            memory_usage: Vec::new(),
            cpu_usage: Vec::new(),
            error_rate: Vec::new(),
            max_points: 1000,
        }
    }
}

impl AlertManager {
    pub fn new() -> Self {
        Self {
            rules: RwLock::new(Vec::new()),
            active_alerts: RwLock::new(Vec::new()),
            alert_history: RwLock::new(Vec::new()),
            notification_channels: RwLock::new(vec![NotificationChannel::Console]),
        }
    }

    pub fn add_default_rules(&self) {
        let mut rules = self.rules.write();

        // High latency alert
        rules.push(AlertRule {
            id: "high_latency".to_string(),
            name: "High Query Latency".to_string(),
            condition: AlertCondition {
                metric: "avg_query_latency_ms".to_string(),
                operator: ComparisonOperator::GreaterThan,
                threshold: 1000.0, // 1 second
                duration_seconds: 60,
            },
            severity: Severity::Warning,
            cooldown_seconds: 300,
            enabled: true,
        });

        // High error rate alert
        rules.push(AlertRule {
            id: "high_error_rate".to_string(),
            name: "High Error Rate".to_string(),
            condition: AlertCondition {
                metric: "error_rate".to_string(),
                operator: ComparisonOperator::GreaterThan,
                threshold: 0.05, // 5%
                duration_seconds: 120,
            },
            severity: Severity::Error,
            cooldown_seconds: 600,
            enabled: true,
        });

        // High memory usage alert
        rules.push(AlertRule {
            id: "high_memory".to_string(),
            name: "High Memory Usage".to_string(),
            condition: AlertCondition {
                metric: "memory_usage_percent".to_string(),
                operator: ComparisonOperator::GreaterThan,
                threshold: 85.0, // 85%
                duration_seconds: 300,
            },
            severity: Severity::Warning,
            cooldown_seconds: 900,
            enabled: true,
        });
    }
}

impl HealthChecker {
    pub fn new() -> Self {
        Self {
            checks: RwLock::new(Vec::new()),
            overall_status: RwLock::new(HealthStatus::Unknown),
            last_check: RwLock::new(None),
        }
    }

    pub fn add_default_checks(&self) {
        let mut checks = self.checks.write();

        checks.push(HealthCheck {
            name: "Database Connection".to_string(),
            check_type: HealthCheckType::DatabaseConnection,
            status: HealthStatus::Healthy,
            message: "Connection pool healthy".to_string(),
            last_check: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            check_duration_ms: 5.2,
        });

        checks.push(HealthCheck {
            name: "Disk Space".to_string(),
            check_type: HealthCheckType::DiskSpace,
            status: HealthStatus::Healthy,
            message: "Sufficient disk space available".to_string(),
            last_check: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            check_duration_ms: 1.8,
        });
    }
}
