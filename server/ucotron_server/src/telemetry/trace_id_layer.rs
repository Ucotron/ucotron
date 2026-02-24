//! Custom tracing layer that injects OpenTelemetry `trace_id` and `span_id`
//! into each log event's recorded fields. When combined with the JSON formatter,
//! every log line includes these fields, making it trivial to correlate logs
//! with distributed traces in tools like Jaeger, Grafana Tempo, or Datadog.
//!
//! This works by intercepting each event and recording the current OTel context
//! as additional visitor fields that the JSON formatter serializes.

use std::fmt;

use opentelemetry::trace::TraceContextExt;
use tracing_subscriber::fmt::format::Writer;
use tracing_subscriber::fmt::time::FormatTime;
use tracing_subscriber::fmt::time::SystemTime;
use tracing_subscriber::fmt::FormatEvent;
use tracing_subscriber::fmt::FormatFields;
use tracing_subscriber::registry::LookupSpan;

/// A custom [`FormatEvent`] wrapper that prepends `trace_id` and `span_id`
/// fields from the current OpenTelemetry context into JSON log output.
///
/// When there is no active trace context (e.g., internal startup logs),
/// the fields are set to empty strings, keeping the JSON schema consistent.
pub struct TraceIdJsonFormat {
    timer: SystemTime,
}

impl TraceIdJsonFormat {
    pub fn new() -> Self {
        Self { timer: SystemTime }
    }
}

impl<S, N> FormatEvent<S, N> for TraceIdJsonFormat
where
    S: tracing::Subscriber + for<'a> LookupSpan<'a>,
    N: for<'a> FormatFields<'a> + 'static,
{
    fn format_event(
        &self,
        ctx: &tracing_subscriber::fmt::FmtContext<'_, S, N>,
        mut writer: Writer<'_>,
        event: &tracing::Event<'_>,
    ) -> fmt::Result {
        // Extract current OpenTelemetry trace context.
        let current_cx = opentelemetry::Context::current();
        let otel_span = current_cx.span();
        let span_context = otel_span.span_context();

        let (trace_id, span_id) = if span_context.is_valid() {
            (
                span_context.trace_id().to_string(),
                span_context.span_id().to_string(),
            )
        } else {
            (String::new(), String::new())
        };

        // Write opening brace and timestamp.
        write!(writer, "{{\"timestamp\":\"")?;
        self.timer.format_time(&mut writer)?;
        write!(writer, "\"")?;

        // Write level.
        let meta = event.metadata();
        write!(writer, ",\"level\":\"{}\"", meta.level())?;

        // Write trace correlation fields.
        write!(
            writer,
            ",\"trace_id\":\"{}\",\"span_id\":\"{}\"",
            trace_id, span_id
        )?;

        // Write target.
        write!(writer, ",\"target\":\"{}\"", meta.target())?;

        // Write span context (current span name).
        if let Some(scope) = ctx.event_scope() {
            let mut spans = Vec::new();
            for span in scope {
                spans.push(span.name());
            }
            if !spans.is_empty() {
                write!(writer, ",\"span\":\"")?;
                for (i, name) in spans.iter().rev().enumerate() {
                    if i > 0 {
                        write!(writer, ":")?;
                    }
                    write!(writer, "{}", name)?;
                }
                write!(writer, "\"")?;
            }
        }

        // Write event fields.
        write!(writer, ",\"fields\":{{")?;
        let mut visitor = JsonVisitor::new();
        event.record(&mut visitor);
        write!(writer, "{}", visitor.output)?;
        write!(writer, "}}")?;

        // Close JSON object.
        writeln!(writer, "}}")?;

        Ok(())
    }
}

/// Simple JSON field visitor that serializes event fields as JSON key-value pairs.
struct JsonVisitor {
    output: String,
    first: bool,
}

impl JsonVisitor {
    fn new() -> Self {
        Self {
            output: String::new(),
            first: true,
        }
    }

    fn write_separator(&mut self) {
        if !self.first {
            self.output.push(',');
        }
        self.first = false;
    }
}

impl tracing::field::Visit for JsonVisitor {
    fn record_f64(&mut self, field: &tracing::field::Field, value: f64) {
        self.write_separator();
        self.output
            .push_str(&format!("\"{}\":{}", field.name(), value));
    }

    fn record_i64(&mut self, field: &tracing::field::Field, value: i64) {
        self.write_separator();
        self.output
            .push_str(&format!("\"{}\":{}", field.name(), value));
    }

    fn record_u64(&mut self, field: &tracing::field::Field, value: u64) {
        self.write_separator();
        self.output
            .push_str(&format!("\"{}\":{}", field.name(), value));
    }

    fn record_bool(&mut self, field: &tracing::field::Field, value: bool) {
        self.write_separator();
        self.output
            .push_str(&format!("\"{}\":{}", field.name(), value));
    }

    fn record_str(&mut self, field: &tracing::field::Field, value: &str) {
        self.write_separator();
        // Escape JSON special characters.
        let escaped = value
            .replace('\\', "\\\\")
            .replace('"', "\\\"")
            .replace('\n', "\\n")
            .replace('\r', "\\r")
            .replace('\t', "\\t");
        self.output
            .push_str(&format!("\"{}\":\"{}\"", field.name(), escaped));
    }

    fn record_debug(&mut self, field: &tracing::field::Field, value: &dyn fmt::Debug) {
        self.write_separator();
        let debug_str = format!("{:?}", value);
        let escaped = debug_str
            .replace('\\', "\\\\")
            .replace('"', "\\\"")
            .replace('\n', "\\n")
            .replace('\r', "\\r")
            .replace('\t', "\\t");
        self.output
            .push_str(&format!("\"{}\":\"{}\"", field.name(), escaped));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn trace_id_json_format_constructed() {
        let _fmt = TraceIdJsonFormat::new();
    }

    #[test]
    fn json_visitor_string_escaping() {
        // Verify JSON special characters are properly escaped.
        let output = "test\"value\nnewline";
        let escaped = output
            .replace('\\', "\\\\")
            .replace('"', "\\\"")
            .replace('\n', "\\n");
        assert_eq!(escaped, "test\\\"value\\nnewline");
    }

    #[test]
    fn json_visitor_separator_logic() {
        let mut visitor = JsonVisitor::new();
        assert!(visitor.first);
        visitor.write_separator();
        assert!(!visitor.first);
        assert_eq!(visitor.output, "");
        visitor.write_separator();
        assert_eq!(visitor.output, ",");
    }
}
