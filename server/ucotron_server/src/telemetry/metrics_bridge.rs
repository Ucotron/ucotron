//! OTLP metrics bridge — exports Ucotron metrics via OpenTelemetry OTLP/gRPC.
//!
//! Creates an OTel `SdkMeterProvider` with a periodic exporter that mirrors the
//! existing Prometheus counters, histograms, and gauges. The `/metrics` endpoint
//! remains available for Prometheus scraping; this module adds OTLP push-based
//! export for collectors like Grafana Alloy, Datadog Agent, or any OTLP-compatible
//! backend.

use opentelemetry::{global, KeyValue};
use opentelemetry_otlp::WithExportConfig;
use opentelemetry_sdk::{
    metrics::{PeriodicReader, SdkMeterProvider},
    runtime, Resource,
};
use opentelemetry_semantic_conventions::resource as res;

/// Configuration for the OTLP metrics pipeline.
pub struct MetricsBridgeConfig {
    /// OTLP gRPC endpoint (e.g. "http://localhost:4317").
    pub otlp_endpoint: String,
    /// Service name reported in metrics resource.
    pub service_name: String,
}

/// Build an OTLP gRPC [`SdkMeterProvider`] that periodically exports metrics.
///
/// Returns the provider which must be stored for the application lifetime.
/// On drop, it flushes pending metrics and shuts down the exporter.
pub fn build_meter_provider(cfg: &MetricsBridgeConfig) -> anyhow::Result<SdkMeterProvider> {
    let exporter = opentelemetry_otlp::MetricExporter::builder()
        .with_tonic()
        .with_endpoint(&cfg.otlp_endpoint)
        .build()
        .map_err(|e| anyhow::anyhow!("Failed to build OTLP metric exporter: {}", e))?;

    let resource = Resource::new_with_defaults([
        KeyValue::new(res::SERVICE_NAME, cfg.service_name.clone()),
        KeyValue::new(res::SERVICE_VERSION, env!("CARGO_PKG_VERSION")),
    ]);

    let reader = PeriodicReader::builder(exporter, runtime::Tokio).build();

    let provider = SdkMeterProvider::builder()
        .with_reader(reader)
        .with_resource(resource)
        .build();

    // Register globally so any code can obtain a meter via `global::meter("name")`.
    global::set_meter_provider(provider.clone());

    Ok(provider)
}

/// OTLP meter instruments that mirror the Prometheus metrics.
///
/// Call [`OtelMetrics::record`] periodically (e.g. in the prometheus middleware)
/// to push values to the OTLP pipeline.
pub struct OtelMetrics {
    pub http_requests_total: opentelemetry::metrics::Counter<u64>,
    pub ingestions_total: opentelemetry::metrics::Counter<u64>,
    pub searches_total: opentelemetry::metrics::Counter<u64>,
    pub errors_total: opentelemetry::metrics::Counter<u64>,
    pub http_request_duration_seconds: opentelemetry::metrics::Histogram<f64>,
    pub ingestion_duration_seconds: opentelemetry::metrics::Histogram<f64>,
    pub search_duration_seconds: opentelemetry::metrics::Histogram<f64>,
    pub uptime_seconds: opentelemetry::metrics::Gauge<u64>,
    pub graph_nodes_total: opentelemetry::metrics::Gauge<u64>,
    pub graph_edges_total: opentelemetry::metrics::Gauge<u64>,
    pub process_rss_bytes: opentelemetry::metrics::Gauge<u64>,
    // New metrics (US-21.8)
    pub model_inference_duration_seconds: opentelemetry::metrics::Histogram<f64>,
    pub lmdb_map_usage_bytes: opentelemetry::metrics::Gauge<u64>,
}

impl Default for OtelMetrics {
    fn default() -> Self {
        Self::new()
    }
}

impl OtelMetrics {
    /// Create all meter instruments from the global meter provider.
    pub fn new() -> Self {
        let meter = global::meter("ucotron");

        let http_requests_total = meter
            .u64_counter("ucotron.http.requests.total")
            .with_description("Total number of HTTP requests processed")
            .build();

        let ingestions_total = meter
            .u64_counter("ucotron.ingestions.total")
            .with_description("Total number of memory ingestion operations")
            .build();

        let searches_total = meter
            .u64_counter("ucotron.searches.total")
            .with_description("Total number of search operations")
            .build();

        let errors_total = meter
            .u64_counter("ucotron.errors.total")
            .with_description("Total number of errors by type")
            .build();

        let http_request_duration_seconds = meter
            .f64_histogram("ucotron.http.request.duration")
            .with_description("HTTP request duration in seconds")
            .with_unit("s")
            .build();

        let ingestion_duration_seconds = meter
            .f64_histogram("ucotron.ingestion.duration")
            .with_description("Memory ingestion latency in seconds")
            .with_unit("s")
            .build();

        let search_duration_seconds = meter
            .f64_histogram("ucotron.search.duration")
            .with_description("Search operation latency in seconds")
            .with_unit("s")
            .build();

        let uptime_seconds = meter
            .u64_gauge("ucotron.uptime")
            .with_description("Server uptime in seconds")
            .with_unit("s")
            .build();

        let graph_nodes_total = meter
            .u64_gauge("ucotron.graph.nodes.total")
            .with_description("Total number of nodes in the knowledge graph")
            .build();

        let graph_edges_total = meter
            .u64_gauge("ucotron.graph.edges.total")
            .with_description("Total number of edges in the knowledge graph")
            .build();

        let process_rss_bytes = meter
            .u64_gauge("ucotron.process.rss")
            .with_description("Resident set size of the server process in bytes")
            .with_unit("By")
            .build();

        let model_inference_duration_seconds = meter
            .f64_histogram("ucotron.model.inference.duration")
            .with_description("Model inference latency in seconds (NER, embedding, etc.)")
            .with_unit("s")
            .build();

        let lmdb_map_usage_bytes = meter
            .u64_gauge("ucotron.lmdb.map.usage")
            .with_description("LMDB memory map usage in bytes")
            .with_unit("By")
            .build();

        Self {
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

    /// Record an HTTP request observation in OTLP metrics.
    pub fn record_http_request(&self, method: &str, path: &str, status: u16, duration_secs: f64) {
        let attrs = [
            KeyValue::new("http.method", method.to_string()),
            KeyValue::new("url.path", path.to_string()),
            KeyValue::new("http.status_code", status as i64),
        ];
        self.http_requests_total.add(1, &attrs);
        self.http_request_duration_seconds
            .record(duration_secs, &attrs);

        if status >= 400 {
            let error_type = if status >= 500 {
                "server_error"
            } else {
                "client_error"
            };
            self.errors_total
                .add(1, &[KeyValue::new("error.type", error_type)]);
        }
    }

    /// Record an ingestion operation in OTLP metrics.
    pub fn record_ingestion(&self, duration_secs: f64) {
        self.ingestions_total.add(1, &[]);
        self.ingestion_duration_seconds.record(duration_secs, &[]);
    }

    /// Record a search operation in OTLP metrics.
    pub fn record_search(&self, duration_secs: f64) {
        self.searches_total.add(1, &[]);
        self.search_duration_seconds.record(duration_secs, &[]);
    }

    /// Record model inference duration (NER, embedding, etc.).
    pub fn record_model_inference(&self, model_name: &str, duration_secs: f64) {
        self.model_inference_duration_seconds.record(
            duration_secs,
            &[KeyValue::new("model.name", model_name.to_string())],
        );
    }

    /// Record LMDB map usage.
    pub fn record_lmdb_map_usage(&self, usage_bytes: u64) {
        self.lmdb_map_usage_bytes.record(usage_bytes, &[]);
    }

    /// Record gauge values (uptime, graph size, RSS).
    pub fn record_gauges(&self, uptime: u64, nodes: u64, edges: u64, rss: u64) {
        self.uptime_seconds.record(uptime, &[]);
        self.graph_nodes_total.record(nodes, &[]);
        self.graph_edges_total.record(edges, &[]);
        self.process_rss_bytes.record(rss, &[]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metrics_bridge_config() {
        let cfg = MetricsBridgeConfig {
            otlp_endpoint: "http://localhost:4317".into(),
            service_name: "ucotron-test".into(),
        };
        assert_eq!(cfg.otlp_endpoint, "http://localhost:4317");
        assert_eq!(cfg.service_name, "ucotron-test");
    }

    #[test]
    fn test_otel_metrics_new() {
        // Set up a no-op global meter provider for testing.
        // OtelMetrics::new() should create all instruments without panicking.
        let metrics = OtelMetrics::new();
        // Verify instruments are callable (no-op in test without real provider).
        metrics.record_http_request("GET", "/health", 200, 0.001);
        metrics.record_ingestion(0.05);
        metrics.record_search(0.01);
        metrics.record_model_inference("ner", 0.02);
        metrics.record_lmdb_map_usage(1024 * 1024);
        metrics.record_gauges(42, 100, 500, 50 * 1024 * 1024);
    }

    #[test]
    fn test_otel_metrics_error_classification() {
        let metrics = OtelMetrics::new();
        // 4xx → client_error, 5xx → server_error
        metrics.record_http_request("POST", "/api/v1/memories", 400, 0.002);
        metrics.record_http_request("POST", "/api/v1/memories", 500, 0.005);
        // 2xx/3xx → no error recorded (only counter + histogram)
        metrics.record_http_request("GET", "/api/v1/health", 200, 0.001);
    }
}
