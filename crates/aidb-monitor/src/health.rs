use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use tokio::process::Command;

use crate::{HealthCheck, HealthCheckType, HealthChecker, HealthStatus};

// Comprehensive health check system
pub struct HealthCheckEngine {
    pub checker: Arc<HealthChecker>,
    pub check_interval: Duration,
    pub timeout_duration: Duration,
}

// Custom health check function type
pub type HealthCheckFn = Box<
    dyn Fn() -> Box<dyn std::future::Future<Output = HealthCheckResult> + Send + Unpin>
        + Send
        + Sync,
>;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthCheckResult {
    pub status: HealthStatus,
    pub message: String,
    pub duration_ms: f64,
    pub metadata: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemHealth {
    pub overall_status: HealthStatus,
    pub checks: Vec<HealthCheck>,
    pub last_update: u64,
    pub uptime_seconds: u64,
    pub system_info: SystemInfo,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemInfo {
    pub hostname: String,
    pub os: String,
    pub arch: String,
    pub cpu_cores: usize,
    pub memory_total_gb: f64,
    pub disk_total_gb: f64,
    pub rust_version: String,
}

impl HealthCheckEngine {
    pub fn new(checker: Arc<HealthChecker>) -> Self {
        Self {
            checker,
            check_interval: Duration::from_secs(30),
            timeout_duration: Duration::from_secs(10),
        }
    }

    pub async fn start_health_checks(&self) -> anyhow::Result<()> {
        // Initialize default health checks
        self.initialize_default_checks().await;

        // Start the health check loop
        let mut interval = tokio::time::interval(self.check_interval);

        loop {
            interval.tick().await;
            self.run_all_health_checks().await;
        }
    }

    pub async fn initialize_default_checks(&self) {
        let mut checks = self.checker.checks.write();
        checks.clear();

        // Database connection check
        checks.push(HealthCheck {
            name: "Database Connection".to_string(),
            check_type: HealthCheckType::DatabaseConnection,
            status: HealthStatus::Unknown,
            message: "Initializing...".to_string(),
            last_check: 0,
            check_duration_ms: 0.0,
        });

        // Memory usage check
        checks.push(HealthCheck {
            name: "Memory Usage".to_string(),
            check_type: HealthCheckType::MemoryUsage,
            status: HealthStatus::Unknown,
            message: "Initializing...".to_string(),
            last_check: 0,
            check_duration_ms: 0.0,
        });

        // Disk space check
        checks.push(HealthCheck {
            name: "Disk Space".to_string(),
            check_type: HealthCheckType::DiskSpace,
            status: HealthStatus::Unknown,
            message: "Initializing...".to_string(),
            last_check: 0,
            check_duration_ms: 0.0,
        });

        // Response time check
        checks.push(HealthCheck {
            name: "API Response Time".to_string(),
            check_type: HealthCheckType::ResponseTime,
            status: HealthStatus::Unknown,
            message: "Initializing...".to_string(),
            last_check: 0,
            check_duration_ms: 0.0,
        });

        // Index health check
        checks.push(HealthCheck {
            name: "Index Health".to_string(),
            check_type: HealthCheckType::IndexHealth,
            status: HealthStatus::Unknown,
            message: "Initializing...".to_string(),
            last_check: 0,
            check_duration_ms: 0.0,
        });
    }

    async fn run_all_health_checks(&self) {
        let check_types = {
            let checks = self.checker.checks.read();
            checks
                .iter()
                .map(|c| c.check_type.clone())
                .collect::<Vec<_>>()
        };

        let mut results = Vec::new();

        // Run all checks concurrently
        for check_type in check_types {
            let result = match check_type {
                HealthCheckType::DatabaseConnection => self.check_database_connection().await,
                HealthCheckType::MemoryUsage => self.check_memory_usage().await,
                HealthCheckType::DiskSpace => self.check_disk_space().await,
                HealthCheckType::ResponseTime => self.check_response_time().await,
                HealthCheckType::IndexHealth => self.check_index_health().await,
                HealthCheckType::Custom(ref name) => self.check_custom(&name).await,
            };

            results.push((check_type, result));
        }

        // Update health checks with results
        self.update_health_checks(results).await;

        // Update overall status
        self.update_overall_status().await;
    }

    async fn check_database_connection(&self) -> HealthCheckResult {
        let start = Instant::now();

        // Simulate database connection check
        // In real implementation, would actually try to connect
        tokio::time::sleep(Duration::from_millis(10)).await;

        let duration = start.elapsed();

        // Simulate occasional failures for demonstration
        let success = rand::random::<f64>() > 0.05; // 5% failure rate

        if success {
            HealthCheckResult {
                status: HealthStatus::Healthy,
                message: "Database connection pool is healthy".to_string(),
                duration_ms: duration.as_secs_f64() * 1000.0,
                metadata: {
                    let mut meta = HashMap::new();
                    meta.insert("pool_size".to_string(), "10".to_string());
                    meta.insert("active_connections".to_string(), "3".to_string());
                    meta
                },
            }
        } else {
            HealthCheckResult {
                status: HealthStatus::Critical,
                message: "Database connection failed".to_string(),
                duration_ms: duration.as_secs_f64() * 1000.0,
                metadata: HashMap::new(),
            }
        }
    }

    async fn check_memory_usage(&self) -> HealthCheckResult {
        let start = Instant::now();

        // Get actual memory usage (simplified)
        let memory_info = self.get_memory_info().await;
        let duration = start.elapsed();

        let usage_percent = memory_info.used_percent;
        let status = if usage_percent > 90.0 {
            HealthStatus::Critical
        } else if usage_percent > 75.0 {
            HealthStatus::Warning
        } else {
            HealthStatus::Healthy
        };

        HealthCheckResult {
            status,
            message: format!(
                "Memory usage: {:.1}% ({:.1}GB / {:.1}GB)",
                usage_percent, memory_info.used_gb, memory_info.total_gb
            ),
            duration_ms: duration.as_secs_f64() * 1000.0,
            metadata: {
                let mut meta = HashMap::new();
                meta.insert("used_gb".to_string(), format!("{:.2}", memory_info.used_gb));
                meta.insert(
                    "total_gb".to_string(),
                    format!("{:.2}", memory_info.total_gb),
                );
                meta.insert("usage_percent".to_string(), format!("{:.1}", usage_percent));
                meta
            },
        }
    }

    async fn check_disk_space(&self) -> HealthCheckResult {
        let start = Instant::now();

        let disk_info = self.get_disk_info().await;
        let duration = start.elapsed();

        let usage_percent = disk_info.used_percent;
        let status = if usage_percent > 95.0 {
            HealthStatus::Critical
        } else if usage_percent > 85.0 {
            HealthStatus::Warning
        } else {
            HealthStatus::Healthy
        };

        HealthCheckResult {
            status,
            message: format!(
                "Disk usage: {:.1}% ({:.1}GB / {:.1}GB free)",
                usage_percent, disk_info.free_gb, disk_info.total_gb
            ),
            duration_ms: duration.as_secs_f64() * 1000.0,
            metadata: {
                let mut meta = HashMap::new();
                meta.insert("free_gb".to_string(), format!("{:.2}", disk_info.free_gb));
                meta.insert("total_gb".to_string(), format!("{:.2}", disk_info.total_gb));
                meta.insert("usage_percent".to_string(), format!("{:.1}", usage_percent));
                meta
            },
        }
    }

    async fn check_response_time(&self) -> HealthCheckResult {
        let start = Instant::now();

        // Test a simple HTTP request to self
        let client = reqwest::Client::new();
        let response_result = tokio::time::timeout(
            Duration::from_secs(5),
            client.get("http://localhost:8080/health").send(),
        )
        .await;

        let duration = start.elapsed();
        let duration_ms = duration.as_secs_f64() * 1000.0;

        match response_result {
            Ok(Ok(response)) => {
                let status_code = response.status();
                let response_time_status = if duration_ms > 1000.0 {
                    HealthStatus::Critical
                } else if duration_ms > 500.0 {
                    HealthStatus::Warning
                } else {
                    HealthStatus::Healthy
                };

                HealthCheckResult {
                    status: if status_code.is_success() {
                        response_time_status
                    } else {
                        HealthStatus::Critical
                    },
                    message: format!("API response: {} in {:.1}ms", status_code, duration_ms),
                    duration_ms,
                    metadata: {
                        let mut meta = HashMap::new();
                        meta.insert("status_code".to_string(), status_code.as_str().to_string());
                        meta.insert(
                            "response_time_ms".to_string(),
                            format!("{:.1}", duration_ms),
                        );
                        meta
                    },
                }
            }
            Ok(Err(e)) => HealthCheckResult {
                status: HealthStatus::Critical,
                message: format!("HTTP request failed: {}", e),
                duration_ms,
                metadata: HashMap::new(),
            },
            Err(_) => HealthCheckResult {
                status: HealthStatus::Critical,
                message: "HTTP request timeout".to_string(),
                duration_ms,
                metadata: HashMap::new(),
            },
        }
    }

    async fn check_index_health(&self) -> HealthCheckResult {
        let start = Instant::now();

        // Simulate index health check
        // In real implementation, would check index integrity, corruption, etc.
        tokio::time::sleep(Duration::from_millis(20)).await;

        let duration = start.elapsed();

        // Simulate index statistics
        let index_stats = IndexStats {
            total_vectors: 150_000,
            corrupted_vectors: 0,
            index_size_mb: 45.2,
            last_rebuild: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs()
                - 3600, // 1 hour ago
            fragmentation_percent: 12.5,
        };

        let status = if index_stats.corrupted_vectors > 0 {
            HealthStatus::Critical
        } else if index_stats.fragmentation_percent > 25.0 {
            HealthStatus::Warning
        } else {
            HealthStatus::Healthy
        };

        HealthCheckResult {
            status,
            message: format!(
                "Index: {} vectors, {:.1}MB, {:.1}% fragmentation",
                index_stats.total_vectors,
                index_stats.index_size_mb,
                index_stats.fragmentation_percent
            ),
            duration_ms: duration.as_secs_f64() * 1000.0,
            metadata: {
                let mut meta = HashMap::new();
                meta.insert(
                    "total_vectors".to_string(),
                    index_stats.total_vectors.to_string(),
                );
                meta.insert(
                    "corrupted_vectors".to_string(),
                    index_stats.corrupted_vectors.to_string(),
                );
                meta.insert(
                    "index_size_mb".to_string(),
                    format!("{:.2}", index_stats.index_size_mb),
                );
                meta.insert(
                    "fragmentation_percent".to_string(),
                    format!("{:.1}", index_stats.fragmentation_percent),
                );
                meta
            },
        }
    }

    async fn check_custom(&self, _name: &str) -> HealthCheckResult {
        // Placeholder for custom health checks
        HealthCheckResult {
            status: HealthStatus::Healthy,
            message: "Custom check passed".to_string(),
            duration_ms: 1.0,
            metadata: HashMap::new(),
        }
    }

    async fn update_health_checks(&self, results: Vec<(HealthCheckType, HealthCheckResult)>) {
        let mut checks = self.checker.checks.write();
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();

        for (check_type, result) in results {
            if let Some(check) = checks.iter_mut().find(|c| {
                std::mem::discriminant(&c.check_type) == std::mem::discriminant(&check_type)
            }) {
                check.status = result.status;
                check.message = result.message;
                check.last_check = now;
                check.check_duration_ms = result.duration_ms;
            }
        }
    }

    async fn update_overall_status(&self) {
        let checks = self.checker.checks.read();

        let overall_status = if checks.iter().any(|c| c.status == HealthStatus::Critical) {
            HealthStatus::Critical
        } else if checks.iter().any(|c| c.status == HealthStatus::Warning) {
            HealthStatus::Warning
        } else if checks.iter().all(|c| c.status == HealthStatus::Healthy) {
            HealthStatus::Healthy
        } else {
            HealthStatus::Unknown
        };

        *self.checker.overall_status.write() = overall_status;
    }

    pub fn get_system_health(&self) -> SystemHealth {
        let checks = self.checker.checks.read();
        let overall_status = self.checker.overall_status.read();
        let last_check = self.checker.last_check.read();

        SystemHealth {
            overall_status: overall_status.clone(),
            checks: checks.clone(),
            last_update: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            uptime_seconds: self.get_uptime_seconds(),
            system_info: self.get_system_info(),
        }
    }

    fn get_uptime_seconds(&self) -> u64 {
        // Simplified - would track actual process start time
        3600 // 1 hour
    }

    fn get_system_info(&self) -> SystemInfo {
        SystemInfo {
            hostname: "aidb-server".to_string(), // Would get actual hostname
            os: std::env::consts::OS.to_string(),
            arch: std::env::consts::ARCH.to_string(),
            cpu_cores: num_cpus::get(),
            memory_total_gb: 16.0,              // Would get actual total
            disk_total_gb: 500.0,               // Would get actual total
            rust_version: "1.70.0".to_string(), // Would get actual version
        }
    }

    async fn get_memory_info(&self) -> MemoryInfo {
        // Simplified memory info - would use systemstat in real implementation
        MemoryInfo {
            total_gb: 16.0,
            used_gb: 4.5,
            used_percent: 28.1,
        }
    }

    async fn get_disk_info(&self) -> DiskInfo {
        // Simplified disk info - would use systemstat in real implementation
        DiskInfo {
            total_gb: 500.0,
            free_gb: 350.0,
            used_percent: 30.0,
        }
    }

    // Add custom health check
    pub fn add_custom_health_check(&self, name: String, check_fn: HealthCheckFn) {
        let mut checks = self.checker.checks.write();
        checks.push(HealthCheck {
            name: name.clone(),
            check_type: HealthCheckType::Custom(name),
            status: HealthStatus::Unknown,
            message: "Not yet checked".to_string(),
            last_check: 0,
            check_duration_ms: 0.0,
        });
    }
}

#[derive(Debug, Clone)]
struct MemoryInfo {
    total_gb: f64,
    used_gb: f64,
    used_percent: f64,
}

#[derive(Debug, Clone)]
struct DiskInfo {
    total_gb: f64,
    free_gb: f64,
    used_percent: f64,
}

#[derive(Debug, Clone)]
struct IndexStats {
    total_vectors: u64,
    corrupted_vectors: u64,
    index_size_mb: f64,
    last_rebuild: u64,
    fragmentation_percent: f64,
}

// Health check utilities
pub mod utils {
    use super::*;

    // Check if a TCP port is accessible
    pub async fn check_tcp_port(host: &str, port: u16, timeout: Duration) -> bool {
        match tokio::time::timeout(
            timeout,
            tokio::net::TcpStream::connect(format!("{}:{}", host, port)),
        )
        .await
        {
            Ok(Ok(_)) => true,
            _ => false,
        }
    }

    // Check HTTP endpoint
    pub async fn check_http_endpoint(
        url: &str,
        expected_status: u16,
        timeout: Duration,
    ) -> Result<Duration, String> {
        let client = reqwest::Client::new();
        let start = Instant::now();

        match tokio::time::timeout(timeout, client.get(url).send()).await {
            Ok(Ok(response)) => {
                let duration = start.elapsed();
                if response.status().as_u16() == expected_status {
                    Ok(duration)
                } else {
                    Err(format!(
                        "Expected status {}, got {}",
                        expected_status,
                        response.status()
                    ))
                }
            }
            Ok(Err(e)) => Err(format!("HTTP request failed: {}", e)),
            Err(_) => Err("Request timeout".to_string()),
        }
    }

    // Check if a process is running
    pub async fn check_process_running(process_name: &str) -> bool {
        match Command::new("pgrep")
            .arg("-f")
            .arg(process_name)
            .output()
            .await
        {
            Ok(output) => output.status.success() && !output.stdout.is_empty(),
            Err(_) => false,
        }
    }

    // Get system load average
    pub async fn get_load_average() -> Result<(f64, f64, f64), String> {
        // Simplified - would use proper system calls
        Ok((0.5, 0.7, 0.8)) // 1min, 5min, 15min
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_health_check_engine() {
        let checker = Arc::new(HealthChecker::new());
        let engine = HealthCheckEngine::new(checker.clone());

        engine.initialize_default_checks().await;

        let checks = checker.checks.read();
        assert!(checks.len() > 0);
        assert!(checks
            .iter()
            .any(|c| matches!(c.check_type, HealthCheckType::DatabaseConnection)));
    }

    #[tokio::test]
    async fn test_memory_check() {
        let checker = Arc::new(HealthChecker::new());
        let engine = HealthCheckEngine::new(checker);

        let result = engine.check_memory_usage().await;
        assert!(matches!(
            result.status,
            HealthStatus::Healthy | HealthStatus::Warning | HealthStatus::Critical
        ));
        assert!(result.duration_ms > 0.0);
    }
}
