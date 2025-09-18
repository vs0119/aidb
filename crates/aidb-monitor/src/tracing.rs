use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use uuid::Uuid;

// Distributed tracing system for tracking requests across services
#[derive(Debug, Clone)]
pub struct TracingSystem {
    pub active_traces: Arc<RwLock<HashMap<String, Trace>>>,
    pub completed_traces: Arc<RwLock<Vec<Trace>>>,
    pub sampling_rate: f64, // 0.0 to 1.0
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Trace {
    pub trace_id: String,
    pub parent_span_id: Option<String>,
    pub spans: Vec<Span>,
    pub start_time: u64,
    pub end_time: Option<u64>,
    pub duration_ms: Option<f64>,
    pub status: TraceStatus,
    pub tags: HashMap<String, String>,
    pub error: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Span {
    pub span_id: String,
    pub parent_span_id: Option<String>,
    pub operation_name: String,
    pub start_time: u64,
    pub end_time: Option<u64>,
    pub duration_ms: Option<f64>,
    pub tags: HashMap<String, String>,
    pub logs: Vec<SpanLog>,
    pub status: SpanStatus,
    pub component: String, // e.g., "database", "index", "http"
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpanLog {
    pub timestamp: u64,
    pub level: LogLevel,
    pub message: String,
    pub fields: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum TraceStatus {
    InProgress,
    Completed,
    Error,
    Timeout,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum SpanStatus {
    InProgress,
    Completed,
    Error,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LogLevel {
    Debug,
    Info,
    Warning,
    Error,
}

// Span builder for easy span creation
pub struct SpanBuilder {
    span: Span,
    start_time: Instant,
}

// Context for propagating trace information
#[derive(Debug, Clone)]
pub struct TraceContext {
    pub trace_id: String,
    pub span_id: String,
    pub parent_span_id: Option<String>,
    pub baggage: HashMap<String, String>,
}

impl TracingSystem {
    pub fn new(sampling_rate: f64) -> Self {
        Self {
            active_traces: Arc::new(RwLock::new(HashMap::new())),
            completed_traces: Arc::new(RwLock::new(Vec::new())),
            sampling_rate: sampling_rate.clamp(0.0, 1.0),
        }
    }

    // Start a new trace
    pub fn start_trace(&self, operation_name: &str) -> Option<TraceContext> {
        // Apply sampling
        if rand::random::<f64>() > self.sampling_rate {
            return None;
        }

        let trace_id = Uuid::new_v4().to_string();
        let span_id = Uuid::new_v4().to_string();

        let trace = Trace {
            trace_id: trace_id.clone(),
            parent_span_id: None,
            spans: Vec::new(),
            start_time: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_millis() as u64,
            end_time: None,
            duration_ms: None,
            status: TraceStatus::InProgress,
            tags: HashMap::new(),
            error: None,
        };

        self.active_traces.write().insert(trace_id.clone(), trace);

        Some(TraceContext {
            trace_id,
            span_id,
            parent_span_id: None,
            baggage: HashMap::new(),
        })
    }

    // Start a child span
    pub fn start_span(
        &self,
        context: &TraceContext,
        operation_name: &str,
        component: &str,
    ) -> SpanBuilder {
        let span_id = Uuid::new_v4().to_string();

        let span = Span {
            span_id: span_id.clone(),
            parent_span_id: Some(context.span_id.clone()),
            operation_name: operation_name.to_string(),
            start_time: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_millis() as u64,
            end_time: None,
            duration_ms: None,
            tags: HashMap::new(),
            logs: Vec::new(),
            status: SpanStatus::InProgress,
            component: component.to_string(),
        };

        SpanBuilder {
            span,
            start_time: Instant::now(),
        }
    }

    // Finish a span and add it to the trace
    pub fn finish_span(&self, trace_id: &str, span: Span) {
        let mut active_traces = self.active_traces.write();
        if let Some(trace) = active_traces.get_mut(trace_id) {
            trace.spans.push(span);
        }
    }

    // Finish a trace
    pub fn finish_trace(&self, trace_id: &str, status: TraceStatus) {
        let mut active_traces = self.active_traces.write();
        if let Some(mut trace) = active_traces.remove(trace_id) {
            let end_time = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_millis() as u64;
            trace.end_time = Some(end_time);
            trace.duration_ms = Some((end_time - trace.start_time) as f64);
            trace.status = status;

            // Move to completed traces
            let mut completed = self.completed_traces.write();
            completed.push(trace);

            // Keep only last 1000 completed traces
            if completed.len() > 1000 {
                completed.remove(0);
            }
        }
    }

    // Get trace by ID
    pub fn get_trace(&self, trace_id: &str) -> Option<Trace> {
        // Check active traces first
        if let Some(trace) = self.active_traces.read().get(trace_id) {
            return Some(trace.clone());
        }

        // Check completed traces
        self.completed_traces
            .read()
            .iter()
            .find(|t| t.trace_id == trace_id)
            .cloned()
    }

    // Get traces by time range
    pub fn get_traces_by_time_range(&self, start_time: u64, end_time: u64) -> Vec<Trace> {
        self.completed_traces
            .read()
            .iter()
            .filter(|trace| trace.start_time >= start_time && trace.start_time <= end_time)
            .cloned()
            .collect()
    }

    // Get slow traces (above threshold)
    pub fn get_slow_traces(&self, threshold_ms: f64) -> Vec<Trace> {
        self.completed_traces
            .read()
            .iter()
            .filter(|trace| trace.duration_ms.unwrap_or(0.0) > threshold_ms)
            .cloned()
            .collect()
    }

    // Get error traces
    pub fn get_error_traces(&self) -> Vec<Trace> {
        self.completed_traces
            .read()
            .iter()
            .filter(|trace| trace.status == TraceStatus::Error)
            .cloned()
            .collect()
    }

    // Generate trace summary
    pub fn generate_trace_summary(&self) -> TraceSummary {
        let completed = self.completed_traces.read();
        let total_traces = completed.len();

        if total_traces == 0 {
            return TraceSummary::default();
        }

        let total_duration: f64 = completed.iter().filter_map(|t| t.duration_ms).sum();

        let error_count = completed
            .iter()
            .filter(|t| t.status == TraceStatus::Error)
            .count();

        let mut durations: Vec<f64> = completed.iter().filter_map(|t| t.duration_ms).collect();
        durations.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let avg_duration = total_duration / total_traces as f64;
        let p95_duration = if !durations.is_empty() {
            let idx = (0.95 * durations.len() as f64) as usize;
            durations.get(idx).copied().unwrap_or(0.0)
        } else {
            0.0
        };

        let p99_duration = if !durations.is_empty() {
            let idx = (0.99 * durations.len() as f64) as usize;
            durations.get(idx).copied().unwrap_or(0.0)
        } else {
            0.0
        };

        TraceSummary {
            total_traces,
            error_rate: (error_count as f64 / total_traces as f64) * 100.0,
            avg_duration_ms: avg_duration,
            p95_duration_ms: p95_duration,
            p99_duration_ms: p99_duration,
            slowest_traces: completed
                .iter()
                .filter_map(|t| t.duration_ms.map(|d| (t.trace_id.clone(), d)))
                .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
                .map(|(id, duration)| vec![(id, duration)])
                .unwrap_or_default(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct TraceSummary {
    pub total_traces: usize,
    pub error_rate: f64,
    pub avg_duration_ms: f64,
    pub p95_duration_ms: f64,
    pub p99_duration_ms: f64,
    pub slowest_traces: Vec<(String, f64)>, // (trace_id, duration)
}

impl SpanBuilder {
    pub fn with_tag(mut self, key: &str, value: &str) -> Self {
        self.span.tags.insert(key.to_string(), value.to_string());
        self
    }

    pub fn with_tags(mut self, tags: HashMap<String, String>) -> Self {
        self.span.tags.extend(tags);
        self
    }

    pub fn log(&mut self, level: LogLevel, message: &str) {
        self.log_with_fields(level, message, HashMap::new());
    }

    pub fn log_with_fields(
        &mut self,
        level: LogLevel,
        message: &str,
        fields: HashMap<String, String>,
    ) {
        let log_entry = SpanLog {
            timestamp: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_millis() as u64,
            level,
            message: message.to_string(),
            fields,
        };
        self.span.logs.push(log_entry);
    }

    pub fn finish(mut self, tracing_system: &TracingSystem, trace_id: &str) -> Span {
        let end_time = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64;
        let duration = self.start_time.elapsed();

        self.span.end_time = Some(end_time);
        self.span.duration_ms = Some(duration.as_secs_f64() * 1000.0);
        self.span.status = SpanStatus::Completed;

        tracing_system.finish_span(trace_id, self.span.clone());
        self.span
    }

    pub fn finish_with_error(
        mut self,
        tracing_system: &TracingSystem,
        trace_id: &str,
        error_msg: &str,
    ) -> Span {
        self.log(LogLevel::Error, error_msg);
        self.span.status = SpanStatus::Error;

        let end_time = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64;
        let duration = self.start_time.elapsed();

        self.span.end_time = Some(end_time);
        self.span.duration_ms = Some(duration.as_secs_f64() * 1000.0);

        tracing_system.finish_span(trace_id, self.span.clone());
        self.span
    }
}

impl TraceContext {
    pub fn child_context(&self) -> Self {
        Self {
            trace_id: self.trace_id.clone(),
            span_id: Uuid::new_v4().to_string(),
            parent_span_id: Some(self.span_id.clone()),
            baggage: self.baggage.clone(),
        }
    }

    pub fn with_baggage(mut self, key: &str, value: &str) -> Self {
        self.baggage.insert(key.to_string(), value.to_string());
        self
    }
}

// Macro for easy tracing
#[macro_export]
macro_rules! trace_span {
    ($tracing:expr, $context:expr, $operation:expr, $component:expr, $block:block) => {{
        let mut span_builder = $tracing.start_span($context, $operation, $component);
        let result = $block;
        span_builder.finish($tracing, &$context.trace_id);
        result
    }};
}

// Example usage patterns
pub mod examples {
    use super::*;

    pub async fn trace_database_query(
        tracing: &TracingSystem,
        context: &TraceContext,
        query: &str,
    ) -> Result<Vec<String>, String> {
        let mut span = tracing
            .start_span(context, "database_query", "database")
            .with_tag("query.type", "vector_search")
            .with_tag("db.statement", query);

        span.log(LogLevel::Info, "Starting database query");

        // Simulate database work
        tokio::time::sleep(Duration::from_millis(50)).await;

        let result = if query.contains("error") {
            span.log(LogLevel::Error, "Query failed");
            span.finish_with_error(tracing, &context.trace_id, "Database connection failed");
            Err("Database error".to_string())
        } else {
            span.log(LogLevel::Info, "Query completed successfully");
            span.finish(tracing, &context.trace_id);
            Ok(vec!["result1".to_string(), "result2".to_string()])
        };

        result
    }

    pub async fn trace_http_request(
        tracing: &TracingSystem,
        method: &str,
        path: &str,
    ) -> Result<String, String> {
        // Start a new trace for HTTP request
        let context = match tracing.start_trace("http_request") {
            Some(ctx) => ctx,
            None => return Ok("Request not traced".to_string()), // Sampling
        };

        let mut span = tracing
            .start_span(&context, "http_handler", "http")
            .with_tag("http.method", method)
            .with_tag("http.url", path)
            .with_tag("component", "web_server");

        span.log(LogLevel::Info, "Processing HTTP request");

        // Simulate request processing with child spans
        let db_result = trace_database_query(tracing, &context, "SELECT * FROM vectors").await;

        let result = match db_result {
            Ok(_) => {
                span.log(LogLevel::Info, "Request processed successfully");
                span.finish(tracing, &context.trace_id);
                tracing.finish_trace(&context.trace_id, TraceStatus::Completed);
                Ok("Success".to_string())
            }
            Err(e) => {
                span.log(LogLevel::Error, &format!("Request failed: {}", e));
                span.finish_with_error(tracing, &context.trace_id, &e);
                tracing.finish_trace(&context.trace_id, TraceStatus::Error);
                Err(e)
            }
        };

        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_basic_tracing() {
        let tracing = TracingSystem::new(1.0); // 100% sampling for test

        let context = tracing.start_trace("test_operation").unwrap();

        let span = tracing
            .start_span(&context, "test_span", "test")
            .with_tag("test.key", "test.value")
            .finish(&tracing, &context.trace_id);

        assert_eq!(span.operation_name, "test_span");
        assert_eq!(span.component, "test");
        assert!(span.tags.contains_key("test.key"));

        tracing.finish_trace(&context.trace_id, TraceStatus::Completed);

        let trace = tracing.get_trace(&context.trace_id).unwrap();
        assert_eq!(trace.status, TraceStatus::Completed);
        assert_eq!(trace.spans.len(), 1);
    }

    #[tokio::test]
    async fn test_trace_summary() {
        let tracing = TracingSystem::new(1.0);

        // Create some test traces
        for i in 0..10 {
            let context = tracing.start_trace(&format!("test_{}", i)).unwrap();
            let status = if i < 8 {
                TraceStatus::Completed
            } else {
                TraceStatus::Error
            };
            tracing.finish_trace(&context.trace_id, status);
        }

        let summary = tracing.generate_trace_summary();
        assert_eq!(summary.total_traces, 10);
        assert_eq!(summary.error_rate, 20.0); // 2 out of 10
    }
}
