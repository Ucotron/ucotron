//! Prometheus metrics instrumentation for the Ucotron server.
//!
//! Exposes metrics in the Prometheus text exposition format (0.0.4) at `GET /metrics`.
//! Tracks request latency histograms, counters for operations and errors,
//! and gauges for system state.

use std::sync::Arc;
use std::time::Instant;

use axum::extract::State;
use axum::http::{Request, StatusCode};
use axum::middleware::Next;
use axum::response::{IntoResponse, Response};
use prometheus_client::encoding::text::encode;
use prometheus_client::encoding::EncodeLabelSet;
use prometheus_client::metrics::counter::Counter;
use prometheus_client::metrics::family::Family;
use prometheus_client::metrics::gauge::Gauge;
use prometheus_client::metrics::histogram::{exponential_buckets, Histogram};
use prometheus_client::registry::Registry;

/// Label set for request-level metrics (method + path + status).
#[derive(Clone, Debug, Hash, PartialEq, Eq, EncodeLabelSet)]
pub struct RequestLabels {
    pub method: String,
    pub path: String,
    pub status: String,
}

/// Label set for error-type metrics.
#[derive(Clone, Debug, Hash, PartialEq, Eq, EncodeLabelSet)]
pub struct ErrorLabels {
    pub error_type: String,
}

/// All Prometheus metrics for the Ucotron server.
pub struct PrometheusMetrics {
    /// Prometheus registry holding all metrics.
    pub registry: Registry,

    // -- Counters --
    /// Total HTTP requests processed.
    pub http_requests_total: Family<RequestLabels, Counter>,
    /// Total ingestion operations.
    pub ingestions_total: Counter,
    /// Total search operations.
    pub searches_total: Counter,
    /// Total errors by type.
    pub errors_total: Family<ErrorLabels, Counter>,

    // -- Histograms --
    /// HTTP request duration in seconds.
    pub http_request_duration_seconds: Family<RequestLabels, Histogram>,
    /// Ingestion latency in seconds.
    pub ingestion_duration_seconds: Histogram,
    /// Search latency in seconds.
    pub search_duration_seconds: Histogram,

    // -- Gauges --
    /// Server uptime in seconds (updated on each scrape).
    pub uptime_seconds: Gauge,
    /// Total node count in the graph.
    pub graph_nodes_total: Gauge,
    /// Total edge count in the graph.
    pub graph_edges_total: Gauge,
    /// Process RSS memory in bytes.
    pub process_rss_bytes: Gauge,

    // -- New metrics (US-21.8) --
    /// Model inference latency in seconds (NER, embedding, etc.).
    pub model_inference_duration_seconds: Family<ModelLabels, Histogram>,
    /// LMDB memory map usage in bytes.
    pub lmdb_map_usage_bytes: Gauge,
}

/// Label set for model-level metrics.
#[derive(Clone, Debug, Hash, PartialEq, Eq, EncodeLabelSet)]
pub struct ModelLabels {
    pub model_name: String,
}

/// Create latency histogram buckets.
/// Covers: 1ms, 2.5ms, 5ms, 10ms, 25ms, 50ms, 100ms, 250ms, 500ms, 1s, 2.5s, 5s, 10s
fn make_histogram() -> Histogram {
    Histogram::new(exponential_buckets(0.001, 2.5, 13))
}

impl Default for PrometheusMetrics {
    fn default() -> Self {
        Self::new()
    }
}

impl PrometheusMetrics {
    /// Create a new PrometheusMetrics with all metrics registered.
    pub fn new() -> Self {
        let mut registry = Registry::default();

        // Counters
        let http_requests_total = Family::<RequestLabels, Counter>::default();
        registry.register(
            "ucotron_http_requests_total",
            "Total number of HTTP requests processed",
            http_requests_total.clone(),
        );

        let ingestions_total = Counter::default();
        registry.register(
            "ucotron_ingestions_total",
            "Total number of memory ingestion operations",
            ingestions_total.clone(),
        );

        let searches_total = Counter::default();
        registry.register(
            "ucotron_searches_total",
            "Total number of search operations",
            searches_total.clone(),
        );

        let errors_total = Family::<ErrorLabels, Counter>::default();
        registry.register(
            "ucotron_errors_total",
            "Total number of errors by type",
            errors_total.clone(),
        );

        // Histograms
        let http_request_duration_seconds =
            Family::<RequestLabels, Histogram>::new_with_constructor(make_histogram);
        registry.register(
            "ucotron_http_request_duration_seconds",
            "HTTP request duration in seconds",
            http_request_duration_seconds.clone(),
        );

        let ingestion_duration_seconds = make_histogram();
        registry.register(
            "ucotron_ingestion_duration_seconds",
            "Memory ingestion latency in seconds",
            ingestion_duration_seconds.clone(),
        );

        let search_duration_seconds = make_histogram();
        registry.register(
            "ucotron_search_duration_seconds",
            "Search operation latency in seconds",
            search_duration_seconds.clone(),
        );

        // Gauges
        let uptime_seconds = Gauge::default();
        registry.register(
            "ucotron_uptime_seconds",
            "Server uptime in seconds",
            uptime_seconds.clone(),
        );

        let graph_nodes_total = Gauge::default();
        registry.register(
            "ucotron_graph_nodes_total",
            "Total number of nodes in the knowledge graph",
            graph_nodes_total.clone(),
        );

        let graph_edges_total = Gauge::default();
        registry.register(
            "ucotron_graph_edges_total",
            "Total number of edges in the knowledge graph",
            graph_edges_total.clone(),
        );

        let process_rss_bytes = Gauge::default();
        registry.register(
            "ucotron_process_rss_bytes",
            "Resident set size of the server process in bytes",
            process_rss_bytes.clone(),
        );

        // New metrics (US-21.8)
        let model_inference_duration_seconds =
            Family::<ModelLabels, Histogram>::new_with_constructor(make_histogram);
        registry.register(
            "ucotron_model_inference_duration_seconds",
            "Model inference latency in seconds (NER, embedding, etc.)",
            model_inference_duration_seconds.clone(),
        );

        let lmdb_map_usage_bytes = Gauge::default();
        registry.register(
            "ucotron_lmdb_map_usage_bytes",
            "LMDB memory map usage in bytes",
            lmdb_map_usage_bytes.clone(),
        );

        Self {
            registry,
            http_requests_total,
            ingestions_total,
            searches_total,
            errors_total,
            http_request_duration_seconds,
            ingestion_duration_seconds,
            search_duration_seconds,
            uptime_seconds,
            graph_nodes_total,
            graph_edges_total,
            process_rss_bytes,
            model_inference_duration_seconds,
            lmdb_map_usage_bytes,
        }
    }
}

/// Normalize a request path for metric labels.
/// Replaces path parameters with placeholders to avoid high cardinality.
fn normalize_path(path: &str) -> String {
    let parts: Vec<&str> = path.split('/').collect();
    let mut normalized = Vec::with_capacity(parts.len());
    for (i, part) in parts.iter().enumerate() {
        if part.is_empty() {
            normalized.push(*part);
            continue;
        }
        if i > 0 {
            let prev = parts[i - 1];
            if (prev == "memories" || prev == "entities") && part.parse::<u64>().is_ok() {
                normalized.push(":id");
                continue;
            }
            if prev == "namespaces" && *part != "namespaces" {
                normalized.push(":name");
                continue;
            }
        }
        normalized.push(part);
    }
    normalized.join("/")
}

/// Axum middleware that records request duration and count in Prometheus metrics.
pub async fn prometheus_middleware(
    State(state): State<Arc<crate::state::AppState>>,
    request: Request<axum::body::Body>,
    next: Next,
) -> Response {
    let method = request.method().to_string();
    let path = normalize_path(request.uri().path());
    let start = Instant::now();

    let response = next.run(request).await;

    let duration = start.elapsed().as_secs_f64();
    let status = response.status().as_u16().to_string();

    let status_code = response.status().as_u16();

    if let Some(prom) = &state.prometheus {
        let labels = RequestLabels {
            method: method.clone(),
            path: path.clone(),
            status: status.clone(),
        };
        prom.http_requests_total.get_or_create(&labels).inc();
        prom.http_request_duration_seconds
            .get_or_create(&labels)
            .observe(duration);

        // Track errors
        if status_code >= 400 {
            let error_type = if status_code >= 500 {
                "server_error"
            } else {
                "client_error"
            };
            prom.errors_total
                .get_or_create(&ErrorLabels {
                    error_type: error_type.to_string(),
                })
                .inc();
        }
    }

    // Also record in OTLP metrics if enabled.
    if let Some(otel) = &state.otel_metrics {
        otel.record_http_request(&method, &path, status_code, duration);
    }

    response
}

/// Handler for `GET /metrics` — returns Prometheus text exposition format.
pub async fn prometheus_metrics_handler(
    State(state): State<Arc<crate::state::AppState>>,
) -> impl IntoResponse {
    let Some(prom) = &state.prometheus else {
        return (
            StatusCode::NOT_FOUND,
            "Prometheus metrics not enabled".to_string(),
        );
    };

    // Update dynamic gauges before encoding
    prom.uptime_seconds
        .set(state.start_time.elapsed().as_secs() as i64);

    // Update RSS memory gauge
    #[cfg(target_os = "macos")]
    {
        use std::mem;
        #[allow(deprecated)]
        let task_self = unsafe { libc::mach_task_self() };
        let mut info: libc::mach_task_basic_info = unsafe { mem::zeroed() };
        let mut count = libc::MACH_TASK_BASIC_INFO_COUNT;
        let result = unsafe {
            libc::task_info(
                task_self,
                libc::MACH_TASK_BASIC_INFO,
                &mut info as *mut _ as *mut _,
                &mut count,
            )
        };
        if result == libc::KERN_SUCCESS {
            prom.process_rss_bytes.set(info.resident_size as i64);
        }
    }

    #[cfg(target_os = "linux")]
    {
        if let Ok(status) = std::fs::read_to_string("/proc/self/status") {
            for line in status.lines() {
                if let Some(rest) = line.strip_prefix("VmRSS:") {
                    let kb: i64 = rest
                        .trim()
                        .trim_end_matches(" kB")
                        .trim()
                        .parse()
                        .unwrap_or(0);
                    prom.process_rss_bytes.set(kb * 1024);
                    break;
                }
            }
        }
    }

    // Update graph size gauges (best-effort)
    let node_count = state
        .registry
        .graph()
        .get_all_nodes()
        .map(|n| n.len() as i64)
        .unwrap_or(0);
    let edge_count = state
        .registry
        .graph()
        .get_all_edges()
        .map(|e| e.len() as i64)
        .unwrap_or(0);
    prom.graph_nodes_total.set(node_count);
    prom.graph_edges_total.set(edge_count);

    // Push gauge values to OTLP metrics if enabled.
    if let Some(otel) = &state.otel_metrics {
        let rss = prom.process_rss_bytes.get() as u64;
        otel.record_gauges(
            state.start_time.elapsed().as_secs(),
            node_count as u64,
            edge_count as u64,
            rss,
        );
    }

    // Encode all metrics to Prometheus text format
    let mut buf = String::new();
    if encode(&mut buf, &prom.registry).is_err() {
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            "Failed to encode metrics".to_string(),
        );
    }

    (StatusCode::OK, buf)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normalize_path_basic() {
        assert_eq!(normalize_path("/api/v1/health"), "/api/v1/health");
        assert_eq!(normalize_path("/api/v1/metrics"), "/api/v1/metrics");
        assert_eq!(normalize_path("/api/v1/memories"), "/api/v1/memories");
    }

    #[test]
    fn test_normalize_path_with_id() {
        assert_eq!(
            normalize_path("/api/v1/memories/123"),
            "/api/v1/memories/:id"
        );
        assert_eq!(
            normalize_path("/api/v1/entities/456"),
            "/api/v1/entities/:id"
        );
    }

    #[test]
    fn test_normalize_path_with_namespace() {
        assert_eq!(
            normalize_path("/api/v1/admin/namespaces/production"),
            "/api/v1/admin/namespaces/:name"
        );
    }

    #[test]
    fn test_normalize_path_non_numeric_subpath() {
        assert_eq!(
            normalize_path("/api/v1/memories/search"),
            "/api/v1/memories/search"
        );
    }

    #[test]
    fn test_prometheus_metrics_creation() {
        let metrics = PrometheusMetrics::new();

        // Verify counters can be incremented
        metrics.ingestions_total.inc();
        metrics.searches_total.inc();

        let labels = RequestLabels {
            method: "GET".to_string(),
            path: "/api/v1/health".to_string(),
            status: "200".to_string(),
        };
        metrics.http_requests_total.get_or_create(&labels).inc();
        metrics
            .http_request_duration_seconds
            .get_or_create(&labels)
            .observe(0.005);

        // Verify gauges
        metrics.uptime_seconds.set(42);
        metrics.graph_nodes_total.set(100);
        metrics.graph_edges_total.set(500);
        metrics.process_rss_bytes.set(1024 * 1024 * 50);
    }

    #[test]
    fn test_prometheus_metrics_encode() {
        let metrics = PrometheusMetrics::new();

        metrics.ingestions_total.inc();
        metrics.searches_total.inc();
        metrics.uptime_seconds.set(42);

        let labels = RequestLabels {
            method: "POST".to_string(),
            path: "/api/v1/memories".to_string(),
            status: "201".to_string(),
        };
        metrics.http_requests_total.get_or_create(&labels).inc();

        let mut buf = String::new();
        encode(&mut buf, &metrics.registry).expect("encoding should succeed");

        assert!(buf.contains("ucotron_http_requests_total"));
        assert!(buf.contains("ucotron_ingestions_total"));
        assert!(buf.contains("ucotron_searches_total"));
        assert!(buf.contains("ucotron_uptime_seconds"));
    }

    #[test]
    fn test_error_labels_encode() {
        let metrics = PrometheusMetrics::new();

        metrics
            .errors_total
            .get_or_create(&ErrorLabels {
                error_type: "server_error".to_string(),
            })
            .inc();

        let mut buf = String::new();
        encode(&mut buf, &metrics.registry).expect("encoding should succeed");
        assert!(buf.contains("ucotron_errors_total"));
        assert!(buf.contains("server_error"));
    }

    #[test]
    fn test_histogram_records_values() {
        let metrics = PrometheusMetrics::new();

        metrics.ingestion_duration_seconds.observe(0.015);
        metrics.search_duration_seconds.observe(0.008);

        let labels = RequestLabels {
            method: "POST".to_string(),
            path: "/api/v1/memories".to_string(),
            status: "201".to_string(),
        };
        metrics
            .http_request_duration_seconds
            .get_or_create(&labels)
            .observe(0.025);

        let mut buf = String::new();
        encode(&mut buf, &metrics.registry).expect("encoding should succeed");
        assert!(buf.contains("ucotron_ingestion_duration_seconds"));
        assert!(buf.contains("ucotron_search_duration_seconds"));
        assert!(buf.contains("ucotron_http_request_duration_seconds"));
    }

    #[test]
    fn test_model_inference_metric() {
        let metrics = PrometheusMetrics::new();

        let labels = ModelLabels {
            model_name: "ner".to_string(),
        };
        metrics
            .model_inference_duration_seconds
            .get_or_create(&labels)
            .observe(0.035);

        let embed_labels = ModelLabels {
            model_name: "embedding".to_string(),
        };
        metrics
            .model_inference_duration_seconds
            .get_or_create(&embed_labels)
            .observe(0.012);

        let mut buf = String::new();
        encode(&mut buf, &metrics.registry).expect("encoding should succeed");
        assert!(buf.contains("ucotron_model_inference_duration_seconds"));
        assert!(buf.contains("ner"));
        assert!(buf.contains("embedding"));
    }

    #[test]
    fn test_lmdb_map_usage_metric() {
        let metrics = PrometheusMetrics::new();
        metrics.lmdb_map_usage_bytes.set(1024 * 1024 * 512); // 512 MB

        let mut buf = String::new();
        encode(&mut buf, &metrics.registry).expect("encoding should succeed");
        assert!(buf.contains("ucotron_lmdb_map_usage_bytes"));
    }
}
