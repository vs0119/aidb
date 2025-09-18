use parking_lot::RwLock;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tokio::sync::broadcast;

use crate::{
    Alert, AlertCondition, AlertManager, AlertRule, ComparisonOperator, EventType, MetricsSnapshot,
    MonitoringEvent, NotificationChannel, Severity,
};

// Advanced alert engine with ML-based anomaly detection
pub struct AlertEngine {
    pub manager: Arc<AlertManager>,
    pub metrics_history: Arc<RwLock<Vec<MetricsSnapshot>>>,
    pub anomaly_detector: AnomalyDetector,
    pub notification_client: NotificationClient,
    event_sender: broadcast::Sender<MonitoringEvent>,
}

#[derive(Debug, Clone)]
pub struct AnomalyDetector {
    // Simple statistical anomaly detection
    baseline_window: usize,
    sensitivity: f64,
}

pub struct NotificationClient {
    http_client: Client,
    smtp_config: Option<SmtpConfig>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SmtpConfig {
    pub server: String,
    pub port: u16,
    pub username: String,
    pub password: String,
    pub from_email: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SlackWebhook {
    pub url: String,
    pub channel: String,
    pub username: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertNotification {
    pub alert_id: String,
    pub title: String,
    pub message: String,
    pub severity: Severity,
    pub timestamp: u64,
    pub metadata: HashMap<String, String>,
    pub actions: Vec<AlertAction>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertAction {
    pub name: String,
    pub url: String,
    pub method: String,
}

impl AlertEngine {
    pub fn new(
        manager: Arc<AlertManager>,
        event_sender: broadcast::Sender<MonitoringEvent>,
    ) -> Self {
        Self {
            manager,
            metrics_history: Arc::new(RwLock::new(Vec::new())),
            anomaly_detector: AnomalyDetector::new(50, 2.0), // 50 samples, 2 sigma
            notification_client: NotificationClient::new(),
            event_sender,
        }
    }

    // Main alert evaluation loop
    pub async fn run_evaluation_loop(&self) {
        let mut interval = tokio::time::interval(Duration::from_secs(10));

        loop {
            interval.tick().await;

            // Get current metrics (would come from monitoring system)
            let current_metrics = self.get_current_metrics().await;

            // Store in history
            {
                let mut history = self.metrics_history.write();
                history.push(current_metrics.clone());
                if history.len() > 1000 {
                    history.remove(0); // Keep last 1000 samples
                }
            }

            // Evaluate all rules
            self.evaluate_rules(&current_metrics).await;

            // Check for anomalies
            self.detect_anomalies(&current_metrics).await;

            // Clean up resolved alerts
            self.cleanup_resolved_alerts().await;
        }
    }

    async fn evaluate_rules(&self, metrics: &MetricsSnapshot) {
        let rules = {
            let rules_guard = self.manager.rules.read();
            rules_guard.clone()
        };

        for rule in rules.iter() {
            if !rule.enabled {
                continue;
            }

            let metric_value = self.extract_metric_value(metrics, &rule.condition.metric);
            let condition_met = self.evaluate_condition(&rule.condition, metric_value);

            // Check if alert already exists and handle atomically
            let mut active_alerts = self.manager.active_alerts.write();
            let existing_alert = active_alerts.iter().position(|a| a.rule_id == rule.id);

            if condition_met {
                if existing_alert.is_none() {
                    // Create new alert
                    let alert = Alert {
                        id: uuid::Uuid::new_v4().to_string(),
                        rule_id: rule.id.clone(),
                        message: format!(
                            "{}: {} {} {} (current: {:.2})",
                            rule.name,
                            rule.condition.metric,
                            self.operator_to_string(&rule.condition.operator),
                            rule.condition.threshold,
                            metric_value
                        ),
                        severity: rule.severity.clone(),
                        triggered_at: SystemTime::now()
                            .duration_since(UNIX_EPOCH)
                            .unwrap()
                            .as_secs(),
                        resolved_at: None,
                        metadata: {
                            let mut meta = HashMap::new();
                            meta.insert("metric".to_string(), rule.condition.metric.clone());
                            meta.insert(
                                "threshold".to_string(),
                                rule.condition.threshold.to_string(),
                            );
                            meta.insert("current_value".to_string(), metric_value.to_string());
                            meta
                        },
                    };

                    // Add to active alerts
                    active_alerts.push(alert.clone());
                    drop(active_alerts); // Release lock before async operations

                    // Send notification
                    self.send_alert_notification(&alert).await;

                    // Send monitoring event
                    let event = MonitoringEvent {
                        timestamp: SystemTime::now()
                            .duration_since(UNIX_EPOCH)
                            .unwrap()
                            .as_secs(),
                        event_type: EventType::Alert,
                        severity: rule.severity.clone(),
                        message: format!("Alert triggered: {}", alert.message),
                        metadata: alert.metadata.clone(),
                    };

                    let _ = self.event_sender.send(event);
                } else {
                    drop(active_alerts); // Release lock
                }
            } else if let Some(idx) = existing_alert {
                // Condition no longer met, resolve alert
                active_alerts[idx].resolved_at = Some(
                    SystemTime::now()
                        .duration_since(UNIX_EPOCH)
                        .unwrap()
                        .as_secs(),
                );

                let resolved_alert = active_alerts.remove(idx);
                drop(active_alerts); // Release lock before async operations

                // Move to history
                self.manager
                    .alert_history
                    .write()
                    .push(resolved_alert.clone());

                // Send resolution notification
                self.send_resolution_notification(&resolved_alert).await;
            } else {
                drop(active_alerts); // Release lock
            }
        }
    }

    async fn detect_anomalies(&self, current_metrics: &MetricsSnapshot) {
        let history_len = {
            let history = self.metrics_history.read();
            history.len()
        };

        if history_len < self.anomaly_detector.baseline_window {
            return; // Not enough data
        }

        let history = {
            let history_guard = self.metrics_history.read();
            history_guard.clone()
        };

        // Check query latency anomalies
        if let Some(anomaly) = self
            .anomaly_detector
            .detect_latency_anomaly(&history, current_metrics)
        {
            self.create_anomaly_alert("query_latency_anomaly", &anomaly)
                .await;
        }

        // Check QPS anomalies
        if let Some(anomaly) = self
            .anomaly_detector
            .detect_qps_anomaly(&history, current_metrics)
        {
            self.create_anomaly_alert("qps_anomaly", &anomaly).await;
        }

        // Check error rate spikes
        if let Some(anomaly) = self
            .anomaly_detector
            .detect_error_rate_anomaly(&history, current_metrics)
        {
            self.create_anomaly_alert("error_rate_spike", &anomaly)
                .await;
        }
    }

    async fn create_anomaly_alert(&self, anomaly_type: &str, anomaly_description: &str) {
        let alert = Alert {
            id: uuid::Uuid::new_v4().to_string(),
            rule_id: format!("anomaly_{}", anomaly_type),
            message: format!("Anomaly detected: {}", anomaly_description),
            severity: Severity::Warning,
            triggered_at: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            resolved_at: None,
            metadata: {
                let mut meta = HashMap::new();
                meta.insert("type".to_string(), "anomaly".to_string());
                meta.insert("anomaly_type".to_string(), anomaly_type.to_string());
                meta
            },
        };

        // Check if similar anomaly alert already exists
        let active_alerts = self.manager.active_alerts.read();
        let similar_exists = active_alerts.iter().any(|a| {
            a.rule_id == alert.rule_id
                && a.triggered_at
                    > SystemTime::now()
                        .duration_since(UNIX_EPOCH)
                        .unwrap()
                        .as_secs()
                        - 300
        });

        if !similar_exists {
            drop(active_alerts);
            self.manager.active_alerts.write().push(alert.clone());
            self.send_alert_notification(&alert).await;
        }
    }

    async fn send_alert_notification(&self, alert: &Alert) {
        let notification = AlertNotification {
            alert_id: alert.id.clone(),
            title: format!("🚨 AIDB Alert: {}", alert.rule_id),
            message: alert.message.clone(),
            severity: alert.severity.clone(),
            timestamp: alert.triggered_at,
            metadata: alert.metadata.clone(),
            actions: vec![
                AlertAction {
                    name: "View Dashboard".to_string(),
                    url: "http://localhost:3001".to_string(),
                    method: "GET".to_string(),
                },
                AlertAction {
                    name: "Acknowledge".to_string(),
                    url: format!("/api/alerts/{}/acknowledge", alert.id),
                    method: "POST".to_string(),
                },
            ],
        };

        let channels = self.manager.notification_channels.read();
        for channel in channels.iter() {
            match channel {
                NotificationChannel::Console => {
                    println!(
                        "🚨 ALERT: {} - {}",
                        notification.title, notification.message
                    );
                }
                NotificationChannel::Webhook { url } => {
                    if let Err(e) = self
                        .notification_client
                        .send_webhook(url, &notification)
                        .await
                    {
                        eprintln!("Failed to send webhook notification: {}", e);
                    }
                }
                NotificationChannel::Email { recipients } => {
                    for recipient in recipients {
                        if let Err(e) = self
                            .notification_client
                            .send_email(recipient, &notification)
                            .await
                        {
                            eprintln!("Failed to send email notification to {}: {}", recipient, e);
                        }
                    }
                }
            }
        }
    }

    async fn send_resolution_notification(&self, alert: &Alert) {
        let channels = self.manager.notification_channels.read();
        let resolved_message = format!("✅ RESOLVED: {} - Alert has been resolved", alert.rule_id);

        for channel in channels.iter() {
            match channel {
                NotificationChannel::Console => {
                    println!("{}", resolved_message);
                }
                NotificationChannel::Webhook { url } => {
                    let resolution_notification = AlertNotification {
                        alert_id: alert.id.clone(),
                        title: format!("✅ RESOLVED: {}", alert.rule_id),
                        message: "Alert condition no longer met".to_string(),
                        severity: Severity::Info,
                        timestamp: alert.resolved_at.unwrap_or(0),
                        metadata: alert.metadata.clone(),
                        actions: vec![],
                    };

                    let _ = self
                        .notification_client
                        .send_webhook(url, &resolution_notification)
                        .await;
                }
                _ => {} // Skip email for resolutions
            }
        }
    }

    async fn cleanup_resolved_alerts(&self) {
        let mut history = self.manager.alert_history.write();

        // Keep only last 1000 resolved alerts
        let history_len = history.len();
        if history_len > 1000 {
            history.drain(0..history_len - 1000);
        }
    }

    fn extract_metric_value(&self, metrics: &MetricsSnapshot, metric_name: &str) -> f64 {
        match metric_name {
            "avg_query_latency_ms" => metrics.avg_query_latency_ms,
            "p95_query_latency_ms" => metrics.p95_query_latency_ms,
            "qps" => metrics.qps,
            "error_rate" => {
                if metrics.query_count > 0 {
                    (metrics.query_errors as f64 / metrics.query_count as f64) * 100.0
                } else {
                    0.0
                }
            }
            "memory_usage_percent" => {
                // Simplified - would need actual system memory total
                (metrics.memory_usage_bytes as f64 / (8.0 * 1024.0 * 1024.0 * 1024.0)) * 100.0
            }
            "cpu_usage_percent" => metrics.cpu_usage_percent,
            _ => 0.0,
        }
    }

    fn evaluate_condition(&self, condition: &AlertCondition, value: f64) -> bool {
        match condition.operator {
            ComparisonOperator::GreaterThan => value > condition.threshold,
            ComparisonOperator::LessThan => value < condition.threshold,
            ComparisonOperator::Equals => (value - condition.threshold).abs() < f64::EPSILON,
            ComparisonOperator::NotEquals => (value - condition.threshold).abs() >= f64::EPSILON,
            ComparisonOperator::GreaterThanOrEqual => value >= condition.threshold,
            ComparisonOperator::LessThanOrEqual => value <= condition.threshold,
        }
    }

    fn operator_to_string(&self, op: &ComparisonOperator) -> &'static str {
        match op {
            ComparisonOperator::GreaterThan => ">",
            ComparisonOperator::LessThan => "<",
            ComparisonOperator::Equals => "=",
            ComparisonOperator::NotEquals => "!=",
            ComparisonOperator::GreaterThanOrEqual => ">=",
            ComparisonOperator::LessThanOrEqual => "<=",
        }
    }

    async fn get_current_metrics(&self) -> MetricsSnapshot {
        // In real implementation, this would get metrics from the monitoring system
        MetricsSnapshot {
            timestamp: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            query_count: 1000,
            query_errors: 5,
            avg_query_latency_ms: 25.5,
            p95_query_latency_ms: 45.2,
            memory_usage_bytes: 1024 * 1024 * 512, // 512MB
            cpu_usage_percent: 35.2,
            qps: 50.5,
            active_connections: 25,
            collections_count: 3,
            total_vectors: 50000,
        }
    }
}

impl AnomalyDetector {
    pub fn new(baseline_window: usize, sensitivity: f64) -> Self {
        Self {
            baseline_window,
            sensitivity,
        }
    }

    pub fn detect_latency_anomaly(
        &self,
        history: &[MetricsSnapshot],
        current: &MetricsSnapshot,
    ) -> Option<String> {
        let recent_latencies: Vec<f64> = history
            .iter()
            .rev()
            .take(self.baseline_window)
            .map(|m| m.avg_query_latency_ms)
            .collect();

        if recent_latencies.is_empty() {
            return None;
        }

        let mean = recent_latencies.iter().sum::<f64>() / recent_latencies.len() as f64;
        let variance = recent_latencies
            .iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f64>()
            / recent_latencies.len() as f64;
        let std_dev = variance.sqrt();

        let threshold = mean + (self.sensitivity * std_dev);

        if current.avg_query_latency_ms > threshold {
            Some(format!(
                "Query latency {:.2}ms is {:.1}x higher than baseline ({:.2}ms ± {:.2}ms)",
                current.avg_query_latency_ms,
                current.avg_query_latency_ms / mean,
                mean,
                std_dev
            ))
        } else {
            None
        }
    }

    pub fn detect_qps_anomaly(
        &self,
        history: &[MetricsSnapshot],
        current: &MetricsSnapshot,
    ) -> Option<String> {
        let recent_qps: Vec<f64> = history
            .iter()
            .rev()
            .take(self.baseline_window)
            .map(|m| m.qps)
            .collect();

        if recent_qps.is_empty() {
            return None;
        }

        let mean = recent_qps.iter().sum::<f64>() / recent_qps.len() as f64;
        let variance =
            recent_qps.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / recent_qps.len() as f64;
        let std_dev = variance.sqrt();

        // Check for both high and low QPS anomalies
        let upper_threshold = mean + (self.sensitivity * std_dev);
        let lower_threshold = mean - (self.sensitivity * std_dev);

        if current.qps > upper_threshold {
            Some(format!(
                "QPS spike: {:.1} QPS is {:.1}x higher than baseline ({:.1} ± {:.1})",
                current.qps,
                current.qps / mean,
                mean,
                std_dev
            ))
        } else if current.qps < lower_threshold && current.qps > 0.0 {
            Some(format!(
                "QPS drop: {:.1} QPS is {:.1}x lower than baseline ({:.1} ± {:.1})",
                current.qps,
                mean / current.qps,
                mean,
                std_dev
            ))
        } else {
            None
        }
    }

    pub fn detect_error_rate_anomaly(
        &self,
        history: &[MetricsSnapshot],
        current: &MetricsSnapshot,
    ) -> Option<String> {
        let current_error_rate = if current.query_count > 0 {
            (current.query_errors as f64 / current.query_count as f64) * 100.0
        } else {
            0.0
        };

        let recent_error_rates: Vec<f64> = history
            .iter()
            .rev()
            .take(self.baseline_window)
            .map(|m| {
                if m.query_count > 0 {
                    (m.query_errors as f64 / m.query_count as f64) * 100.0
                } else {
                    0.0
                }
            })
            .collect();

        if recent_error_rates.is_empty() {
            return None;
        }

        let mean = recent_error_rates.iter().sum::<f64>() / recent_error_rates.len() as f64;
        let variance = recent_error_rates
            .iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f64>()
            / recent_error_rates.len() as f64;
        let std_dev = variance.sqrt();

        let threshold = mean + (self.sensitivity * std_dev);

        if current_error_rate > threshold && current_error_rate > 1.0 {
            Some(format!(
                "Error rate spike: {:.2}% is {:.1}x higher than baseline ({:.2}% ± {:.2}%)",
                current_error_rate,
                if mean > 0.0 {
                    current_error_rate / mean
                } else {
                    1.0
                },
                mean,
                std_dev
            ))
        } else {
            None
        }
    }
}

impl NotificationClient {
    pub fn new() -> Self {
        Self {
            http_client: Client::new(),
            smtp_config: None,
        }
    }

    pub async fn send_webhook(
        &self,
        url: &str,
        notification: &AlertNotification,
    ) -> anyhow::Result<()> {
        let payload = serde_json::json!({
            "text": format!("{}\n{}", notification.title, notification.message),
            "attachments": [{
                "color": match notification.severity {
                    Severity::Critical => "danger",
                    Severity::Error => "danger",
                    Severity::Warning => "warning",
                    _ => "good"
                },
                "fields": [
                    {
                        "title": "Alert ID",
                        "value": &notification.alert_id,
                        "short": true
                    },
                    {
                        "title": "Timestamp",
                        "value": chrono::DateTime::from_timestamp(notification.timestamp as i64, 0)
                            .map(|dt| dt.format("%Y-%m-%d %H:%M:%S UTC").to_string())
                            .unwrap_or_else(|| "Unknown".to_string()),
                        "short": true
                    }
                ],
                "actions": notification.actions.iter().map(|action| {
                    serde_json::json!({
                        "type": "button",
                        "text": &action.name,
                        "url": &action.url
                    })
                }).collect::<Vec<_>>()
            }]
        });

        self.http_client
            .post(url)
            .json(&payload)
            .send()
            .await?
            .error_for_status()?;

        Ok(())
    }

    pub async fn send_email(
        &self,
        _recipient: &str,
        _notification: &AlertNotification,
    ) -> anyhow::Result<()> {
        // Email implementation would go here using lettre crate
        // For now, just log
        println!("Would send email notification to {}", _recipient);
        Ok(())
    }
}

// Predefined alert rule templates
pub fn create_default_alert_rules() -> Vec<AlertRule> {
    vec![
        AlertRule {
            id: "high_latency".to_string(),
            name: "High Query Latency".to_string(),
            condition: AlertCondition {
                metric: "avg_query_latency_ms".to_string(),
                operator: ComparisonOperator::GreaterThan,
                threshold: 500.0, // 500ms
                duration_seconds: 60,
            },
            severity: Severity::Warning,
            cooldown_seconds: 300,
            enabled: true,
        },
        AlertRule {
            id: "critical_latency".to_string(),
            name: "Critical Query Latency".to_string(),
            condition: AlertCondition {
                metric: "avg_query_latency_ms".to_string(),
                operator: ComparisonOperator::GreaterThan,
                threshold: 2000.0, // 2 seconds
                duration_seconds: 30,
            },
            severity: Severity::Critical,
            cooldown_seconds: 180,
            enabled: true,
        },
        AlertRule {
            id: "high_error_rate".to_string(),
            name: "High Error Rate".to_string(),
            condition: AlertCondition {
                metric: "error_rate".to_string(),
                operator: ComparisonOperator::GreaterThan,
                threshold: 5.0, // 5%
                duration_seconds: 120,
            },
            severity: Severity::Error,
            cooldown_seconds: 600,
            enabled: true,
        },
        AlertRule {
            id: "low_qps".to_string(),
            name: "Low Query Throughput".to_string(),
            condition: AlertCondition {
                metric: "qps".to_string(),
                operator: ComparisonOperator::LessThan,
                threshold: 1.0, // 1 QPS
                duration_seconds: 300,
            },
            severity: Severity::Warning,
            cooldown_seconds: 900,
            enabled: true,
        },
        AlertRule {
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
        },
    ]
}
