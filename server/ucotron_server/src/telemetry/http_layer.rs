//! OpenTelemetry HTTP instrumentation layer.
//!
//! Adds W3C Trace Context propagation and HTTP semantic convention attributes
//! to every request span. When OTLP export is enabled, these spans appear in
//! Jaeger, Grafana Tempo, or any OTLP-compatible collector with proper HTTP
//! attributes (method, path, status_code, duration_ms).
//!
//! Implemented as a tower `Layer`/`Service` pair so it can be applied at any
//! position in the middleware stack regardless of Axum state type.
//!
//! ## W3C Trace Context
//!
//! Inbound `traceparent` and `tracestate` headers are extracted to link
//! distributed traces across services. Outbound response headers propagate
//! the trace context back to callers.

use std::future::Future;
use std::pin::Pin;
use std::task::{Context as TaskContext, Poll};
use std::time::Instant;

use axum::body::Body;
use axum::http::Request;
use axum::response::Response;
use opentelemetry::propagation::TextMapPropagator;
use opentelemetry::trace::TraceContextExt;
use opentelemetry::Context;
use opentelemetry_sdk::propagation::TraceContextPropagator;
use tower::{Layer, Service};

/// HTTP header extractor for W3C Trace Context propagation.
struct HeaderExtractor<'a>(&'a axum::http::HeaderMap);

impl opentelemetry::propagation::Extractor for HeaderExtractor<'_> {
    fn get(&self, key: &str) -> Option<&str> {
        self.0.get(key).and_then(|v| v.to_str().ok())
    }

    fn keys(&self) -> Vec<&str> {
        self.0.keys().map(|k| k.as_str()).collect()
    }
}

/// HTTP header injector for W3C Trace Context propagation.
struct HeaderInjector<'a>(&'a mut axum::http::HeaderMap);

impl opentelemetry::propagation::Injector for HeaderInjector<'_> {
    fn set(&mut self, key: &str, value: String) {
        if let Ok(name) = axum::http::header::HeaderName::from_bytes(key.as_bytes()) {
            if let Ok(val) = axum::http::header::HeaderValue::from_str(&value) {
                self.0.insert(name, val);
            }
        }
    }
}

/// Tower layer that wraps services with OpenTelemetry HTTP instrumentation.
///
/// Apply this to an Axum router via `.layer(OtelHttpLayer)`.
#[derive(Clone)]
pub struct OtelHttpLayer;

impl<S> Layer<S> for OtelHttpLayer {
    type Service = OtelHttpService<S>;

    fn layer(&self, inner: S) -> Self::Service {
        OtelHttpService { inner }
    }
}

/// Tower service that instruments HTTP requests with OpenTelemetry spans.
///
/// For each request, this service:
/// 1. Extracts W3C Trace Context from inbound headers (`traceparent`, `tracestate`)
/// 2. Creates a tracing span with HTTP semantic convention attributes
/// 3. Calls the inner service
/// 4. Records response status code and duration
/// 5. Injects trace context into response headers
///
/// Span attributes follow [OpenTelemetry HTTP semantic conventions]:
/// - `http.request.method` — HTTP method (GET, POST, etc.)
/// - `url.path` — Request path
/// - `http.response.status_code` — Response status code
/// - `http.request.duration_ms` — Request duration in milliseconds
/// - `otel.kind` — Always `server` for inbound HTTP requests
///
/// [OpenTelemetry HTTP semantic conventions]: https://opentelemetry.io/docs/specs/semconv/http/http-spans/
#[derive(Clone)]
pub struct OtelHttpService<S> {
    inner: S,
}

impl<S> Service<Request<Body>> for OtelHttpService<S>
where
    S: Service<Request<Body>, Response = Response> + Clone + Send + 'static,
    S::Future: Send + 'static,
{
    type Response = Response;
    type Error = S::Error;
    type Future = Pin<Box<dyn Future<Output = Result<Self::Response, Self::Error>> + Send>>;

    fn poll_ready(&mut self, cx: &mut TaskContext<'_>) -> Poll<Result<(), Self::Error>> {
        self.inner.poll_ready(cx)
    }

    fn call(&mut self, request: Request<Body>) -> Self::Future {
        let mut inner = self.inner.clone();
        // Swap cloned service with the ready one (standard tower pattern)
        std::mem::swap(&mut self.inner, &mut inner);

        // Extract trace context and build span synchronously (before async boundary)
        // to avoid sending non-Send ContextGuard across await points.
        let start = Instant::now();
        let propagator = TraceContextPropagator::new();
        let parent_context = propagator.extract(&HeaderExtractor(request.headers()));

        let method = request.method().to_string();
        let path = request.uri().path().to_string();

        // Create span while parent context is attached (synchronous scope).
        let span = {
            let _guard = if parent_context.span().span_context().is_valid() {
                Some(parent_context.attach())
            } else {
                None
            };

            tracing::info_span!(
                "HTTP request",
                otel.kind = "server",
                http.request.method = %method,
                url.path = %path,
                http.response.status_code = tracing::field::Empty,
                http.request.duration_ms = tracing::field::Empty,
            )
            // _guard dropped here — parent context detached, span already created
        };

        Box::pin(async move {
            // Run the inner service within the span.
            let response = {
                let _enter = span.enter();
                inner.call(request).await?
            };

            // Record response attributes on the span.
            let status = response.status().as_u16();
            let duration_ms = start.elapsed().as_secs_f64() * 1000.0;
            span.record("http.response.status_code", status);
            span.record(
                "http.request.duration_ms",
                format!("{:.2}", duration_ms).as_str(),
            );

            // Log a warning for error responses.
            if status >= 400 {
                tracing::warn!(
                    parent: &span,
                    status = status,
                    "HTTP error response"
                );
            }

            // Inject trace context into response headers.
            let mut response = response;
            let propagator = TraceContextPropagator::new();
            let current_context = Context::current();
            propagator.inject_context(
                &current_context,
                &mut HeaderInjector(response.headers_mut()),
            );

            Ok(response)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::http::StatusCode;
    use axum::response::IntoResponse;
    use axum::routing::get;
    use axum::Router;
    use tower::ServiceExt;

    async fn ok_handler() -> impl IntoResponse {
        StatusCode::OK
    }

    async fn not_found_handler() -> impl IntoResponse {
        StatusCode::NOT_FOUND
    }

    async fn error_handler() -> impl IntoResponse {
        StatusCode::INTERNAL_SERVER_ERROR
    }

    fn build_app(handler: axum::routing::MethodRouter) -> Router {
        Router::new()
            .route("/test", handler)
            .layer(OtelHttpLayer)
    }

    #[tokio::test]
    async fn test_otel_layer_passes_through() {
        let app = build_app(get(ok_handler));
        let request = Request::builder()
            .uri("/test")
            .body(Body::empty())
            .unwrap();

        let response = app.oneshot(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_otel_layer_with_traceparent() {
        let app = build_app(get(ok_handler));
        let request = Request::builder()
            .uri("/test")
            .header(
                "traceparent",
                "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01",
            )
            .body(Body::empty())
            .unwrap();

        let response = app.oneshot(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_otel_layer_with_tracestate() {
        let app = build_app(get(ok_handler));
        let request = Request::builder()
            .uri("/test")
            .header(
                "traceparent",
                "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01",
            )
            .header("tracestate", "congo=t61rcWkgMzE")
            .body(Body::empty())
            .unwrap();

        let response = app.oneshot(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_otel_layer_error_status() {
        let app = build_app(get(not_found_handler));
        let request = Request::builder()
            .uri("/test")
            .body(Body::empty())
            .unwrap();

        let response = app.oneshot(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn test_otel_layer_server_error() {
        let app = build_app(get(error_handler));
        let request = Request::builder()
            .uri("/test")
            .body(Body::empty())
            .unwrap();

        let response = app.oneshot(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::INTERNAL_SERVER_ERROR);
    }

    #[tokio::test]
    async fn test_otel_layer_post_method() {
        let app = Router::new()
            .route("/test", axum::routing::post(ok_handler))
            .layer(OtelHttpLayer);
        let request = Request::builder()
            .method("POST")
            .uri("/test")
            .body(Body::empty())
            .unwrap();

        let response = app.oneshot(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_otel_layer_invalid_traceparent() {
        // Invalid traceparent should not cause errors — middleware should
        // gracefully handle it and proceed without parent context.
        let app = build_app(get(ok_handler));
        let request = Request::builder()
            .uri("/test")
            .header("traceparent", "invalid-traceparent-header")
            .body(Body::empty())
            .unwrap();

        let response = app.oneshot(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::OK);
    }
}
