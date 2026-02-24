//! OTLP gRPC exporter and TracerProvider setup.
//!
//! Configures an OpenTelemetry [`TracerProvider`] that exports spans via OTLP/gRPC
//! to a collector (Jaeger, Grafana Tempo, Datadog Agent, etc.).

use opentelemetry_otlp::{SpanExporter, WithExportConfig};
use opentelemetry_sdk::{
    runtime,
    trace::{RandomIdGenerator, Sampler, TracerProvider},
    Resource,
};
use opentelemetry_semantic_conventions::resource as res;

/// Configuration for the OTLP tracer.
pub struct TracerConfig {
    /// OTLP gRPC endpoint (e.g. "http://localhost:4317").
    pub otlp_endpoint: String,
    /// Service name reported in spans.
    pub service_name: String,
    /// Sampling ratio (0.0 = none, 1.0 = all).
    pub sample_rate: f64,
}

/// Build an OTLP gRPC [`TracerProvider`] with the given configuration.
///
/// The returned provider manages span batching, export, and shutdown.
/// Call `provider.shutdown()` on graceful termination to flush pending spans.
pub fn build_tracer_provider(cfg: &TracerConfig) -> anyhow::Result<TracerProvider> {
    // Build the OTLP span exporter pointing at the configured endpoint.
    let exporter = SpanExporter::builder()
        .with_tonic()
        .with_endpoint(&cfg.otlp_endpoint)
        .build()
        .map_err(|e| anyhow::anyhow!("Failed to build OTLP span exporter: {}", e))?;

    // Build resource attributes describing this service.
    let resource = Resource::new_with_defaults([
        opentelemetry::KeyValue::new(res::SERVICE_NAME, cfg.service_name.clone()),
        opentelemetry::KeyValue::new(res::SERVICE_VERSION, env!("CARGO_PKG_VERSION")),
    ]);

    // Build the tracer provider with batch exporter.
    let sampler = if cfg.sample_rate >= 1.0 {
        Sampler::AlwaysOn
    } else if cfg.sample_rate <= 0.0 {
        Sampler::AlwaysOff
    } else {
        Sampler::TraceIdRatioBased(cfg.sample_rate)
    };

    let provider = TracerProvider::builder()
        .with_batch_exporter(exporter, runtime::Tokio)
        .with_resource(resource)
        .with_sampler(sampler)
        .with_id_generator(RandomIdGenerator::default())
        .build();

    // Register the provider as the global tracer provider so `tracing` spans
    // flow through OpenTelemetry automatically.
    let _prev = opentelemetry::global::set_tracer_provider(provider.clone());

    Ok(provider)
}

/// Create a named tracer from the global provider.
///
/// Use this when you need a standalone `Tracer` instance for manual span
/// creation outside of the `tracing` crate integration.
pub fn tracer(name: &'static str) -> opentelemetry::global::BoxedTracer {
    opentelemetry::global::tracer(name)
}
