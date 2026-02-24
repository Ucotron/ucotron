//! # Telemetry Module
//!
//! Initializes OpenTelemetry tracing and metrics with OTLP gRPC export and
//! integrates them with the `tracing` subscriber so that all `tracing::info!`,
//! `#[instrument]`, and manual spans flow through both the console logger and
//! the OTLP collector. Metrics are exported via a periodic OTLP push exporter
//! alongside the existing Prometheus `/metrics` endpoint.
//!
//! ## Usage
//!
//! ```no_run
//! use ucotron_server::telemetry;
//!
//! // With OpenTelemetry enabled:
//! let guard = telemetry::init_telemetry(telemetry::TelemetryInit {
//!     enabled: true,
//!     otlp_endpoint: "http://localhost:4317".into(),
//!     service_name: "ucotron".into(),
//!     sample_rate: 1.0,
//!     log_level: "info".into(),
//!     export_traces: true,
//!     export_metrics: true,
//!     log_format: "json".into(), // "json" for structured logs with trace_id/span_id
//! }).unwrap();
//!
//! // On shutdown:
//! drop(guard);
//! ```

pub mod http_layer;
pub mod metrics_bridge;
pub mod tracer;

use opentelemetry::trace::TracerProvider as _;
use opentelemetry_sdk::metrics::SdkMeterProvider;
use opentelemetry_sdk::trace::TracerProvider;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;
use tracing_subscriber::EnvFilter;

use self::metrics_bridge::{MetricsBridgeConfig, OtelMetrics};
use self::tracer::TracerConfig;

mod trace_id_layer;

/// Parameters for telemetry initialization.
pub struct TelemetryInit {
    /// Whether OpenTelemetry export is enabled.
    /// When `false`, only the console fmt subscriber is installed.
    pub enabled: bool,
    /// OTLP gRPC collector endpoint (e.g. "http://localhost:4317").
    pub otlp_endpoint: String,
    /// Service name to report in traces.
    pub service_name: String,
    /// Trace sampling ratio (0.0–1.0).
    pub sample_rate: f64,
    /// Log level filter string (e.g. "info", "ucotron_server=debug,info").
    pub log_level: String,
    /// Whether to export traces via OTLP.
    pub export_traces: bool,
    /// Whether to export metrics via OTLP.
    pub export_metrics: bool,
    /// Log format: "json" for structured JSON with trace_id/span_id, "text" for human-readable.
    pub log_format: String,
}

/// Guard that shuts down the OpenTelemetry tracer and meter providers on drop.
///
/// Keep this alive for the duration of the application. When dropped, it
/// flushes all pending spans/metrics and gracefully shuts down the exporters.
pub struct TelemetryGuard {
    trace_provider: Option<TracerProvider>,
    meter_provider: Option<SdkMeterProvider>,
}

impl TelemetryGuard {
    /// Get the OTLP metrics instruments, if metrics export was enabled.
    /// Returns `None` if metrics export is disabled.
    pub fn otel_metrics(&self) -> Option<OtelMetrics> {
        if self.meter_provider.is_some() {
            Some(OtelMetrics::new())
        } else {
            None
        }
    }
}

impl Drop for TelemetryGuard {
    fn drop(&mut self) {
        if let Some(provider) = self.meter_provider.take() {
            if let Err(e) = provider.shutdown() {
                eprintln!("OpenTelemetry metrics shutdown error: {e}");
            }
        }
        if let Some(provider) = self.trace_provider.take() {
            if let Err(e) = provider.shutdown() {
                eprintln!("OpenTelemetry trace shutdown error: {e}");
            }
        }
        opentelemetry::global::shutdown_tracer_provider();
    }
}

/// Initialize the tracing subscriber with optional OpenTelemetry export.
///
/// When `init.enabled` is `true`, traces and/or metrics are exported via OTLP
/// gRPC to the configured collector endpoint **in addition** to the console fmt
/// layer and the Prometheus `/metrics` endpoint.
///
/// Returns a [`TelemetryGuard`] that must be held for the application lifetime.
/// Dropping the guard flushes and shuts down the exporters.
pub fn init_telemetry(init: TelemetryInit) -> anyhow::Result<TelemetryGuard> {
    let env_filter =
        EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new(&init.log_level));

    let use_json = init.log_format == "json";

    let mut trace_provider = None;
    let mut meter_provider = None;

    // Build optional OTEL tracing layer.
    let otel_layer = if init.enabled && init.export_traces {
        let provider = tracer::build_tracer_provider(&TracerConfig {
            otlp_endpoint: init.otlp_endpoint.clone(),
            service_name: init.service_name.clone(),
            sample_rate: init.sample_rate,
        })?;
        let otel_tracer = provider.tracer("ucotron");
        trace_provider = Some(provider);
        Some(tracing_opentelemetry::layer().with_tracer(otel_tracer))
    } else {
        None
    };

    // Build fmt layers — exactly one of json or text is Some.
    let (json_layer, text_layer) = if use_json {
        // JSON structured logging with trace_id/span_id fields from OpenTelemetry context.
        let json =
            tracing_subscriber::fmt::layer().event_format(trace_id_layer::TraceIdJsonFormat::new());
        (Some(json), None)
    } else {
        let text = tracing_subscriber::fmt::layer()
            .with_target(true)
            .with_thread_ids(false)
            .with_file(false);
        (None, Some(text))
    };

    // Compose the subscriber. `Option<Layer>` is itself a Layer (no-op when None).
    tracing_subscriber::registry()
        .with(env_filter)
        .with(json_layer)
        .with(text_layer)
        .with(otel_layer)
        .init();

    if init.enabled && init.export_metrics {
        // Build the OTLP metrics pipeline.
        let provider = metrics_bridge::build_meter_provider(&MetricsBridgeConfig {
            otlp_endpoint: init.otlp_endpoint.clone(),
            service_name: init.service_name.clone(),
        })?;

        tracing::info!("OTLP metrics export initialized");
        meter_provider = Some(provider);
    }

    tracing::info!(
        otel = init.enabled,
        traces = init.enabled && init.export_traces,
        metrics = init.enabled && init.export_metrics,
        log_format = init.log_format.as_str(),
        "Telemetry initialized"
    );

    Ok(TelemetryGuard {
        trace_provider,
        meter_provider,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn telemetry_init_disabled() {
        // Verify that guard is constructed correctly with no providers.
        let guard = TelemetryGuard {
            trace_provider: None,
            meter_provider: None,
        };
        assert!(guard.trace_provider.is_none());
        assert!(guard.meter_provider.is_none());
        assert!(guard.otel_metrics().is_none());
        drop(guard);
    }

    #[test]
    fn telemetry_init_config_struct() {
        let init = TelemetryInit {
            enabled: false,
            otlp_endpoint: "http://localhost:4317".into(),
            service_name: "ucotron-test".into(),
            sample_rate: 0.5,
            log_level: "info".into(),
            export_traces: true,
            export_metrics: true,
            log_format: "text".into(),
        };
        assert!(!init.enabled);
        assert_eq!(init.otlp_endpoint, "http://localhost:4317");
        assert_eq!(init.service_name, "ucotron-test");
        assert!((init.sample_rate - 0.5).abs() < f64::EPSILON);
        assert!(init.export_traces);
        assert!(init.export_metrics);
        assert_eq!(init.log_format, "text");
    }

    #[test]
    fn telemetry_init_selective_export() {
        // Verify that export flags can be set independently.
        let init = TelemetryInit {
            enabled: true,
            otlp_endpoint: "http://localhost:4317".into(),
            service_name: "ucotron".into(),
            sample_rate: 1.0,
            log_level: "info".into(),
            export_traces: false,
            export_metrics: true,
            log_format: "json".into(),
        };
        assert!(init.enabled);
        assert!(!init.export_traces);
        assert!(init.export_metrics);
        assert_eq!(init.log_format, "json");
    }

    #[test]
    fn telemetry_init_json_format() {
        let init = TelemetryInit {
            enabled: false,
            otlp_endpoint: "http://localhost:4317".into(),
            service_name: "ucotron".into(),
            sample_rate: 1.0,
            log_level: "info".into(),
            export_traces: false,
            export_metrics: false,
            log_format: "json".into(),
        };
        assert_eq!(init.log_format, "json");
    }

    #[test]
    fn tracer_config_sampling() {
        let cfg_full = TracerConfig {
            otlp_endpoint: "http://localhost:4317".into(),
            service_name: "test".into(),
            sample_rate: 1.0,
        };
        assert!((cfg_full.sample_rate - 1.0).abs() < f64::EPSILON);

        let cfg_none = TracerConfig {
            otlp_endpoint: "http://localhost:4317".into(),
            service_name: "test".into(),
            sample_rate: 0.0,
        };
        assert!((cfg_none.sample_rate - 0.0).abs() < f64::EPSILON);

        let cfg_partial = TracerConfig {
            otlp_endpoint: "http://localhost:4317".into(),
            service_name: "test".into(),
            sample_rate: 0.25,
        };
        assert!((cfg_partial.sample_rate - 0.25).abs() < f64::EPSILON);
    }
}
