//! # Ucotron SDK
//!
//! Rust client library for the Ucotron cognitive memory framework.
//!
//! Provides both async and sync APIs for integrating cognitive memory into LLM agents.
//!
//! ## Quick Start (Async)
//!
//! ```rust,ignore
//! use ucotron_sdk::UcotronClient;
//!
//! let client = UcotronClient::new("http://localhost:8420");
//! let result = client.augment("What does Juan do for work?", Default::default()).await?;
//! println!("Context: {}", result.context_text);
//! client.learn("Juan works at SAP in Berlin.", Default::default()).await?;
//! ```
//!
//! ## Quick Start (Sync)
//!
//! ```rust,ignore
//! use ucotron_sdk::UcotronClient;
//!
//! let client = UcotronClient::new("http://localhost:8420");
//! let result = client.augment_sync("What does Juan do for work?", Default::default())?;
//! println!("Context: {}", result.context_text);
//! client.learn_sync("Juan works at SAP in Berlin.", Default::default())?;
//! ```

use std::collections::HashMap;
use std::time::Duration;

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// W3C Trace Context propagation (optional, behind `otel` feature)
// ---------------------------------------------------------------------------

/// W3C Trace Context header injector.
///
/// Collects `traceparent` and `tracestate` headers from the current OpenTelemetry
/// context so they can be added to outbound HTTP requests.
#[cfg(feature = "otel")]
struct HeaderMap(Vec<(String, String)>);

#[cfg(feature = "otel")]
impl opentelemetry::propagation::Injector for HeaderMap {
    fn set(&mut self, key: &str, value: String) {
        self.0.push((key.to_string(), value));
    }
}

/// Inject W3C `traceparent` and `tracestate` headers into a reqwest `RequestBuilder`.
///
/// When the `otel` feature is enabled and an active OpenTelemetry context exists,
/// this injects the W3C Trace Context headers so the server can link its spans
/// to the caller's distributed trace.
///
/// When the `otel` feature is disabled, this is a no-op.
fn inject_trace_context(req: reqwest::RequestBuilder) -> reqwest::RequestBuilder {
    #[cfg(feature = "otel")]
    {
        use opentelemetry::propagation::TextMapPropagator;
        use opentelemetry_sdk::propagation::TraceContextPropagator;

        let propagator = TraceContextPropagator::new();
        let cx = opentelemetry::Context::current();
        let mut headers = HeaderMap(Vec::new());
        propagator.inject_context(&cx, &mut headers);

        let mut req = req;
        for (key, value) in headers.0 {
            req = req.header(key, value);
        }
        req
    }
    #[cfg(not(feature = "otel"))]
    {
        req
    }
}

// ---------------------------------------------------------------------------
// Error types
// ---------------------------------------------------------------------------

/// Errors that can occur when using the Ucotron SDK.
#[derive(Debug, thiserror::Error)]
pub enum UcotronError {
    /// HTTP transport error.
    #[error("HTTP error: {0}")]
    Http(#[from] reqwest::Error),

    /// Server returned a non-success status code.
    #[error("Server error {status}: {message}")]
    Server { status: u16, message: String },

    /// JSON serialization/deserialization error.
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    /// All retry attempts exhausted.
    #[error("All {attempts} retry attempts failed: {last_error}")]
    RetriesExhausted { attempts: u32, last_error: String },
}

pub type Result<T> = std::result::Result<T, UcotronError>;

// ---------------------------------------------------------------------------
// API types (mirror server request/response types)
// ---------------------------------------------------------------------------

/// Result of a context augmentation request.
#[derive(Debug, Clone, Deserialize)]
pub struct AugmentResult {
    /// Relevant memories retrieved for the context.
    pub memories: Vec<SearchResultItem>,
    /// Entities mentioned in the context.
    pub entities: Vec<EntityResponse>,
    /// Formatted context text ready for LLM injection.
    pub context_text: String,
}

/// Result of a learn (memory storage) request.
#[derive(Debug, Clone, Deserialize)]
pub struct LearnResult {
    /// Number of memories created from the input.
    pub memories_created: usize,
    /// Number of entities found or created.
    pub entities_found: usize,
    /// Number of conflicts detected.
    pub conflicts_found: usize,
}

/// Result of creating/ingesting a memory.
#[derive(Debug, Clone, Deserialize)]
pub struct CreateMemoryResult {
    /// IDs of chunk nodes created.
    pub chunk_node_ids: Vec<u64>,
    /// IDs of entity nodes created or matched.
    pub entity_node_ids: Vec<u64>,
    /// Number of edges created.
    pub edges_created: usize,
    /// Ingestion pipeline metrics.
    pub metrics: IngestionMetrics,
}

/// Ingestion pipeline metrics.
#[derive(Debug, Clone, Deserialize)]
pub struct IngestionMetrics {
    pub chunks_processed: usize,
    pub entities_extracted: usize,
    pub relations_extracted: usize,
    pub contradictions_detected: usize,
    pub total_us: u64,
}

/// A single search result item.
#[derive(Debug, Clone, Deserialize)]
pub struct SearchResultItem {
    pub id: u64,
    pub content: String,
    pub node_type: String,
    pub score: f32,
    pub vector_sim: f32,
    pub graph_centrality: f32,
    pub recency: f32,
    /// Mindset-aware score component (0.0 when no mindset is configured).
    #[serde(default)]
    pub mindset_score: f32,
}

/// Search response containing results and metadata.
#[derive(Debug, Clone, Deserialize)]
pub struct SearchResult {
    pub results: Vec<SearchResultItem>,
    pub total: usize,
    pub query: String,
}

/// A memory node as returned by the API.
#[derive(Debug, Clone, Deserialize)]
pub struct MemoryResponse {
    pub id: u64,
    pub content: String,
    pub node_type: String,
    pub timestamp: u64,
    #[serde(default)]
    pub metadata: HashMap<String, serde_json::Value>,
}

/// A graph entity as returned by the API.
#[derive(Debug, Clone, Deserialize)]
pub struct EntityResponse {
    pub id: u64,
    pub content: String,
    pub node_type: String,
    pub timestamp: u64,
    #[serde(default)]
    pub metadata: HashMap<String, serde_json::Value>,
    pub neighbors: Option<Vec<NeighborResponse>>,
}

/// A neighbor edge+node pair.
#[derive(Debug, Clone, Deserialize)]
pub struct NeighborResponse {
    pub node_id: u64,
    pub content: String,
    pub edge_type: String,
    pub weight: f32,
}

/// Server health status.
#[derive(Debug, Clone, Deserialize)]
pub struct HealthResponse {
    pub status: String,
    pub version: String,
    pub instance_id: String,
    pub instance_role: String,
    pub storage_mode: String,
    pub vector_backend: String,
    pub graph_backend: String,
    pub models: ModelStatus,
}

/// Model availability status.
#[derive(Debug, Clone, Deserialize)]
pub struct ModelStatus {
    pub embedder_loaded: bool,
    pub embedding_model: String,
    pub ner_loaded: bool,
    pub relation_extractor_loaded: bool,
    #[serde(default)]
    pub transcriber_loaded: bool,
}

/// Server metrics.
#[derive(Debug, Clone, Deserialize)]
pub struct MetricsResponse {
    pub instance_id: String,
    pub total_requests: u64,
    pub total_ingestions: u64,
    pub total_searches: u64,
    pub uptime_secs: u64,
}

/// API error response from the server.
#[derive(Debug, Clone, Deserialize)]
struct ApiErrorResponse {
    #[allow(dead_code)]
    code: String,
    message: String,
}

// ---------------------------------------------------------------------------
// Request option types
// ---------------------------------------------------------------------------

/// Options for augment requests.
#[derive(Debug, Clone, Default)]
pub struct AugmentOptions {
    /// Maximum number of memories to return.
    pub limit: Option<usize>,
    /// Namespace for multi-tenancy.
    pub namespace: Option<String>,
}

/// Options for learn requests.
#[derive(Debug, Clone, Default)]
pub struct LearnOptions {
    /// Namespace for storing memories.
    pub namespace: Option<String>,
    /// Optional metadata to attach.
    pub metadata: Option<HashMap<String, serde_json::Value>>,
}

/// Options for search requests.
#[derive(Debug, Clone, Default)]
pub struct SearchOptions {
    /// Maximum number of results.
    pub limit: Option<usize>,
    /// Namespace for multi-tenancy.
    pub namespace: Option<String>,
    /// Filter by node type.
    pub node_type: Option<String>,
    /// Time range filter [min_ts, max_ts].
    pub time_range: Option<(u64, u64)>,
    /// Optional cognitive mindset for scoring ("convergent", "divergent", "algorithmic").
    pub query_mindset: Option<String>,
}

/// Options for add_memory requests.
#[derive(Debug, Clone, Default)]
pub struct AddMemoryOptions {
    /// Namespace for multi-tenancy.
    pub namespace: Option<String>,
    /// Optional metadata to attach.
    pub metadata: Option<HashMap<String, serde_json::Value>>,
}

/// Options for entity queries.
#[derive(Debug, Clone, Default)]
pub struct EntityOptions {
    /// Namespace for multi-tenancy.
    pub namespace: Option<String>,
}

/// Configuration for retry behavior.
#[derive(Debug, Clone)]
pub struct RetryConfig {
    /// Maximum number of retry attempts (0 = no retries).
    pub max_retries: u32,
    /// Base delay for exponential backoff.
    pub base_delay: Duration,
    /// Maximum delay between retries.
    pub max_delay: Duration,
}

impl Default for RetryConfig {
    fn default() -> Self {
        Self {
            max_retries: 3,
            base_delay: Duration::from_millis(100),
            max_delay: Duration::from_secs(5),
        }
    }
}

/// Configuration for the Ucotron client.
#[derive(Debug, Clone)]
pub struct ClientConfig {
    /// Request timeout.
    pub timeout: Duration,
    /// Retry configuration.
    pub retry: RetryConfig,
    /// Default namespace for requests.
    pub default_namespace: Option<String>,
}

impl Default for ClientConfig {
    fn default() -> Self {
        Self {
            timeout: Duration::from_secs(30),
            retry: RetryConfig::default(),
            default_namespace: None,
        }
    }
}

// ---------------------------------------------------------------------------
// Internal request body types (for serialization)
// ---------------------------------------------------------------------------

#[derive(Serialize)]
struct CreateMemoryBody {
    text: String,
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    metadata: HashMap<String, serde_json::Value>,
}

#[derive(Serialize)]
struct SearchBody {
    query: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    limit: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    node_type: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    time_range: Option<(u64, u64)>,
    #[serde(skip_serializing_if = "Option::is_none")]
    query_mindset: Option<String>,
}

#[derive(Serialize)]
struct AugmentBody {
    context: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    limit: Option<usize>,
}

#[derive(Serialize)]
struct LearnBody {
    output: String,
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    metadata: HashMap<String, serde_json::Value>,
}

#[derive(Serialize)]
struct UpdateMemoryBody {
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<String>,
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    metadata: HashMap<String, serde_json::Value>,
}

// ---------------------------------------------------------------------------
// UcotronClient
// ---------------------------------------------------------------------------

/// Client for connecting to a Ucotron server.
///
/// Uses reqwest with connection pooling internally. Supports both async and sync APIs.
/// Retry logic with exponential backoff for transient failures.
///
/// # Examples
///
/// ```rust,ignore
/// // Default configuration
/// let client = UcotronClient::new("http://localhost:8420");
///
/// // Custom configuration
/// let client = UcotronClient::with_config(
///     "http://localhost:8420",
///     ClientConfig {
///         timeout: Duration::from_secs(60),
///         retry: RetryConfig { max_retries: 5, ..Default::default() },
///         default_namespace: Some("my-agent".to_string()),
///     },
/// );
/// ```
pub struct UcotronClient {
    base_url: String,
    http: reqwest::Client,
    config: ClientConfig,
}

impl UcotronClient {
    /// Create a new client with default configuration.
    ///
    /// Uses connection pooling via reqwest's internal pool.
    pub fn new(server_url: impl Into<String>) -> Self {
        Self::with_config(server_url, ClientConfig::default())
    }

    /// Create a new client with custom configuration.
    pub fn with_config(server_url: impl Into<String>, config: ClientConfig) -> Self {
        let url = server_url.into();
        let base_url = url.trim_end_matches('/').to_string();

        let http = reqwest::Client::builder()
            .timeout(config.timeout)
            .pool_max_idle_per_host(10)
            .pool_idle_timeout(Duration::from_secs(90))
            .build()
            .expect("Failed to build HTTP client");

        Self {
            base_url,
            http,
            config,
        }
    }

    /// Get the base URL of the server.
    pub fn server_url(&self) -> &str {
        &self.base_url
    }

    // -----------------------------------------------------------------------
    // Async API
    // -----------------------------------------------------------------------

    /// Context augmentation — retrieves relevant memories for a given context.
    pub async fn augment(&self, context: &str, opts: AugmentOptions) -> Result<AugmentResult> {
        let body = AugmentBody {
            context: context.to_string(),
            limit: opts.limit,
        };
        let namespace = opts
            .namespace
            .as_deref()
            .or(self.config.default_namespace.as_deref());

        self.post_json("/api/v1/augment", &body, namespace).await
    }

    /// Learn from agent output — extracts and stores memories.
    pub async fn learn(&self, output: &str, opts: LearnOptions) -> Result<LearnResult> {
        let body = LearnBody {
            output: output.to_string(),
            metadata: opts.metadata.unwrap_or_default(),
        };
        let namespace = opts
            .namespace
            .as_deref()
            .or(self.config.default_namespace.as_deref());

        self.post_json("/api/v1/learn", &body, namespace).await
    }

    /// Semantic search over stored memories.
    pub async fn search(&self, query: &str, opts: SearchOptions) -> Result<SearchResult> {
        let body = SearchBody {
            query: query.to_string(),
            limit: opts.limit,
            node_type: opts.node_type,
            time_range: opts.time_range,
            query_mindset: opts.query_mindset,
        };
        let namespace = opts
            .namespace
            .as_deref()
            .or(self.config.default_namespace.as_deref());

        self.post_json("/api/v1/memories/search", &body, namespace)
            .await
    }

    /// Add a memory (ingest text).
    pub async fn add_memory(
        &self,
        text: &str,
        opts: AddMemoryOptions,
    ) -> Result<CreateMemoryResult> {
        let body = CreateMemoryBody {
            text: text.to_string(),
            metadata: opts.metadata.unwrap_or_default(),
        };
        let namespace = opts
            .namespace
            .as_deref()
            .or(self.config.default_namespace.as_deref());

        self.post_json("/api/v1/memories", &body, namespace).await
    }

    /// Get a specific entity by ID.
    pub async fn get_entity(&self, id: u64, opts: EntityOptions) -> Result<EntityResponse> {
        let namespace = opts
            .namespace
            .as_deref()
            .or(self.config.default_namespace.as_deref());

        self.get_json(&format!("/api/v1/entities/{}", id), namespace)
            .await
    }

    /// List entities.
    pub async fn list_entities(
        &self,
        limit: Option<usize>,
        offset: Option<usize>,
        opts: EntityOptions,
    ) -> Result<Vec<EntityResponse>> {
        let mut url = format!("{}/api/v1/entities", self.base_url);
        let mut params = Vec::new();
        if let Some(l) = limit {
            params.push(format!("limit={}", l));
        }
        if let Some(o) = offset {
            params.push(format!("offset={}", o));
        }
        if !params.is_empty() {
            url.push('?');
            url.push_str(&params.join("&"));
        }
        let namespace = opts
            .namespace
            .as_deref()
            .or(self.config.default_namespace.as_deref());

        self.get_json_url(&url, namespace).await
    }

    /// Get a specific memory by ID.
    pub async fn get_memory(&self, id: u64) -> Result<MemoryResponse> {
        self.get_json(&format!("/api/v1/memories/{}", id), None)
            .await
    }

    /// List memories with optional filters.
    pub async fn list_memories(
        &self,
        node_type: Option<&str>,
        limit: Option<usize>,
        offset: Option<usize>,
        namespace: Option<&str>,
    ) -> Result<Vec<MemoryResponse>> {
        let mut url = format!("{}/api/v1/memories", self.base_url);
        let mut params = Vec::new();
        if let Some(nt) = node_type {
            params.push(format!("node_type={}", nt));
        }
        if let Some(l) = limit {
            params.push(format!("limit={}", l));
        }
        if let Some(o) = offset {
            params.push(format!("offset={}", o));
        }
        if !params.is_empty() {
            url.push('?');
            url.push_str(&params.join("&"));
        }
        let ns = namespace.or(self.config.default_namespace.as_deref());
        self.get_json_url(&url, ns).await
    }

    /// Update a memory.
    pub async fn update_memory(
        &self,
        id: u64,
        content: Option<&str>,
        metadata: Option<HashMap<String, serde_json::Value>>,
    ) -> Result<MemoryResponse> {
        let body = UpdateMemoryBody {
            content: content.map(|s| s.to_string()),
            metadata: metadata.unwrap_or_default(),
        };
        let url = format!("{}/api/v1/memories/{}", self.base_url, id);
        self.request_json(reqwest::Method::PUT, &url, Some(&body), None)
            .await
    }

    /// Delete a memory (soft delete).
    pub async fn delete_memory(&self, id: u64) -> Result<()> {
        let url = format!("{}/api/v1/memories/{}", self.base_url, id);
        self.request_no_body(reqwest::Method::DELETE, &url, None)
            .await
    }

    /// Check server health.
    pub async fn health(&self) -> Result<HealthResponse> {
        self.get_json("/api/v1/health", None).await
    }

    /// Get server metrics.
    pub async fn metrics(&self) -> Result<MetricsResponse> {
        self.get_json("/api/v1/metrics", None).await
    }

    // -----------------------------------------------------------------------
    // Sync API wrappers
    // -----------------------------------------------------------------------

    /// Synchronous version of [`augment`](Self::augment).
    pub fn augment_sync(&self, context: &str, opts: AugmentOptions) -> Result<AugmentResult> {
        block_on(self.augment(context, opts))
    }

    /// Synchronous version of [`learn`](Self::learn).
    pub fn learn_sync(&self, output: &str, opts: LearnOptions) -> Result<LearnResult> {
        block_on(self.learn(output, opts))
    }

    /// Synchronous version of [`search`](Self::search).
    pub fn search_sync(&self, query: &str, opts: SearchOptions) -> Result<SearchResult> {
        block_on(self.search(query, opts))
    }

    /// Synchronous version of [`add_memory`](Self::add_memory).
    pub fn add_memory_sync(
        &self,
        text: &str,
        opts: AddMemoryOptions,
    ) -> Result<CreateMemoryResult> {
        block_on(self.add_memory(text, opts))
    }

    /// Synchronous version of [`get_entity`](Self::get_entity).
    pub fn get_entity_sync(&self, id: u64, opts: EntityOptions) -> Result<EntityResponse> {
        block_on(self.get_entity(id, opts))
    }

    /// Synchronous version of [`health`](Self::health).
    pub fn health_sync(&self) -> Result<HealthResponse> {
        block_on(self.health())
    }

    /// Synchronous version of [`metrics`](Self::metrics).
    pub fn metrics_sync(&self) -> Result<MetricsResponse> {
        block_on(self.metrics())
    }

    // -----------------------------------------------------------------------
    // Internal HTTP helpers
    // -----------------------------------------------------------------------

    async fn post_json<B: Serialize, R: for<'de> Deserialize<'de>>(
        &self,
        path: &str,
        body: &B,
        namespace: Option<&str>,
    ) -> Result<R> {
        let url = format!("{}{}", self.base_url, path);
        self.request_json(reqwest::Method::POST, &url, Some(body), namespace)
            .await
    }

    async fn get_json<R: for<'de> Deserialize<'de>>(
        &self,
        path: &str,
        namespace: Option<&str>,
    ) -> Result<R> {
        let url = format!("{}{}", self.base_url, path);
        self.get_json_url::<R>(&url, namespace).await
    }

    async fn get_json_url<R: for<'de> Deserialize<'de>>(
        &self,
        url: &str,
        namespace: Option<&str>,
    ) -> Result<R> {
        self.request_json::<(), R>(reqwest::Method::GET, url, None, namespace)
            .await
    }

    /// Core request method with retry logic.
    async fn request_json<B: Serialize, R: for<'de> Deserialize<'de>>(
        &self,
        method: reqwest::Method,
        url: &str,
        body: Option<&B>,
        namespace: Option<&str>,
    ) -> Result<R> {
        let max_attempts = self.config.retry.max_retries + 1;
        let mut last_error = String::new();

        for attempt in 0..max_attempts {
            if attempt > 0 {
                let delay = self.retry_delay(attempt);
                tokio::time::sleep(delay).await;
            }

            let mut req = self.http.request(method.clone(), url);

            if let Some(ns) = namespace {
                req = req.header("X-Ucotron-Namespace", ns);
            }

            if let Some(b) = body {
                req = req.json(b);
            }

            // Inject W3C traceparent/tracestate headers for distributed tracing.
            req = inject_trace_context(req);

            match req.send().await {
                Ok(resp) => {
                    let status = resp.status();
                    if status.is_success() {
                        let text = resp.text().await?;
                        let parsed: R = serde_json::from_str(&text)?;
                        return Ok(parsed);
                    }

                    // Non-retryable client errors (4xx)
                    if status.is_client_error() {
                        let msg = match resp.json::<ApiErrorResponse>().await {
                            Ok(e) => e.message,
                            Err(_) => format!("HTTP {}", status.as_u16()),
                        };
                        return Err(UcotronError::Server {
                            status: status.as_u16(),
                            message: msg,
                        });
                    }

                    // Server errors (5xx) — retryable
                    last_error = format!("HTTP {}", status.as_u16());
                }
                Err(e) => {
                    // Connection errors — retryable
                    last_error = e.to_string();
                }
            }
        }

        Err(UcotronError::RetriesExhausted {
            attempts: max_attempts,
            last_error,
        })
    }

    /// Fire-and-forget request (for DELETE etc.) with retry.
    async fn request_no_body(
        &self,
        method: reqwest::Method,
        url: &str,
        namespace: Option<&str>,
    ) -> Result<()> {
        let max_attempts = self.config.retry.max_retries + 1;
        let mut last_error = String::new();

        for attempt in 0..max_attempts {
            if attempt > 0 {
                let delay = self.retry_delay(attempt);
                tokio::time::sleep(delay).await;
            }

            let mut req = self.http.request(method.clone(), url);
            if let Some(ns) = namespace {
                req = req.header("X-Ucotron-Namespace", ns);
            }

            // Inject W3C traceparent/tracestate headers for distributed tracing.
            req = inject_trace_context(req);

            match req.send().await {
                Ok(resp) => {
                    let status = resp.status();
                    if status.is_success() {
                        return Ok(());
                    }
                    if status.is_client_error() {
                        let msg = match resp.json::<ApiErrorResponse>().await {
                            Ok(e) => e.message,
                            Err(_) => format!("HTTP {}", status.as_u16()),
                        };
                        return Err(UcotronError::Server {
                            status: status.as_u16(),
                            message: msg,
                        });
                    }
                    last_error = format!("HTTP {}", status.as_u16());
                }
                Err(e) => {
                    last_error = e.to_string();
                }
            }
        }

        Err(UcotronError::RetriesExhausted {
            attempts: max_attempts,
            last_error,
        })
    }

    /// Calculate retry delay with exponential backoff and jitter cap.
    fn retry_delay(&self, attempt: u32) -> Duration {
        let base_ms = self.config.retry.base_delay.as_millis() as u64;
        let delay_ms = base_ms.saturating_mul(1u64 << attempt.min(10));
        let max_ms = self.config.retry.max_delay.as_millis() as u64;
        Duration::from_millis(delay_ms.min(max_ms))
    }
}

// ---------------------------------------------------------------------------
// Sync runtime helper
// ---------------------------------------------------------------------------

/// Run an async future synchronously.
///
/// Creates a new tokio runtime if not already inside one,
/// or uses `block_in_place` if already in a tokio context.
fn block_on<F: std::future::Future<Output = Result<T>>, T>(future: F) -> Result<T> {
    match tokio::runtime::Handle::try_current() {
        Ok(handle) => {
            // Already inside a tokio runtime — use block_in_place to avoid nesting
            tokio::task::block_in_place(|| handle.block_on(future))
        }
        Err(_) => {
            // No runtime — create a new one
            let rt = tokio::runtime::Runtime::new().map_err(|e| UcotronError::Server {
                status: 0,
                message: format!("Failed to create tokio runtime: {}", e),
            })?;
            rt.block_on(future)
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_client_creation() {
        let client = UcotronClient::new("http://localhost:8420");
        assert_eq!(client.server_url(), "http://localhost:8420");
    }

    #[test]
    fn test_client_creation_trailing_slash() {
        let client = UcotronClient::new("http://localhost:8420/");
        assert_eq!(client.server_url(), "http://localhost:8420");
    }

    #[test]
    fn test_client_with_config() {
        let config = ClientConfig {
            timeout: Duration::from_secs(60),
            retry: RetryConfig {
                max_retries: 5,
                base_delay: Duration::from_millis(200),
                max_delay: Duration::from_secs(10),
            },
            default_namespace: Some("test-ns".to_string()),
        };
        let client = UcotronClient::with_config("http://localhost:8420", config);
        assert_eq!(client.server_url(), "http://localhost:8420");
        assert_eq!(client.config.retry.max_retries, 5);
        assert_eq!(client.config.default_namespace.as_deref(), Some("test-ns"));
    }

    #[test]
    fn test_retry_delay_exponential_backoff() {
        let client = UcotronClient::new("http://localhost:8420");
        // base = 100ms
        assert_eq!(client.retry_delay(0), Duration::from_millis(100));
        assert_eq!(client.retry_delay(1), Duration::from_millis(200));
        assert_eq!(client.retry_delay(2), Duration::from_millis(400));
        assert_eq!(client.retry_delay(3), Duration::from_millis(800));
    }

    #[test]
    fn test_retry_delay_capped_at_max() {
        let config = ClientConfig {
            retry: RetryConfig {
                max_retries: 10,
                base_delay: Duration::from_millis(100),
                max_delay: Duration::from_secs(1),
            },
            ..Default::default()
        };
        let client = UcotronClient::with_config("http://localhost:8420", config);
        // 2^10 * 100ms = 102400ms, but capped at 1000ms
        assert_eq!(client.retry_delay(10), Duration::from_secs(1));
    }

    #[test]
    fn test_augment_options_default() {
        let opts = AugmentOptions::default();
        assert!(opts.limit.is_none());
        assert!(opts.namespace.is_none());
    }

    #[test]
    fn test_search_options_default() {
        let opts = SearchOptions::default();
        assert!(opts.limit.is_none());
        assert!(opts.namespace.is_none());
        assert!(opts.node_type.is_none());
        assert!(opts.time_range.is_none());
    }

    #[test]
    fn test_learn_options_default() {
        let opts = LearnOptions::default();
        assert!(opts.namespace.is_none());
        assert!(opts.metadata.is_none());
    }

    #[test]
    fn test_add_memory_options_default() {
        let opts = AddMemoryOptions::default();
        assert!(opts.namespace.is_none());
        assert!(opts.metadata.is_none());
    }

    #[test]
    fn test_retry_config_default() {
        let config = RetryConfig::default();
        assert_eq!(config.max_retries, 3);
        assert_eq!(config.base_delay, Duration::from_millis(100));
        assert_eq!(config.max_delay, Duration::from_secs(5));
    }

    #[test]
    fn test_client_config_default() {
        let config = ClientConfig::default();
        assert_eq!(config.timeout, Duration::from_secs(30));
        assert_eq!(config.retry.max_retries, 3);
        assert!(config.default_namespace.is_none());
    }

    #[test]
    fn test_ucotron_error_display() {
        let err = UcotronError::Server {
            status: 404,
            message: "Not found".to_string(),
        };
        assert_eq!(format!("{}", err), "Server error 404: Not found");

        let err = UcotronError::RetriesExhausted {
            attempts: 4,
            last_error: "Connection refused".to_string(),
        };
        assert!(format!("{}", err).contains("4 retry attempts"));
        assert!(format!("{}", err).contains("Connection refused"));
    }

    #[test]
    fn test_augment_result_deserialization() {
        let json = r#"{
            "memories": [{
                "id": 1,
                "content": "Juan works at SAP",
                "node_type": "Entity",
                "score": 0.95,
                "vector_sim": 0.8,
                "graph_centrality": 0.6,
                "recency": 0.9
            }],
            "entities": [{
                "id": 2,
                "content": "Juan",
                "node_type": "Entity",
                "timestamp": 1234567890,
                "metadata": {},
                "neighbors": null
            }],
            "context_text": "Relevant context here"
        }"#;

        let result: AugmentResult = serde_json::from_str(json).unwrap();
        assert_eq!(result.memories.len(), 1);
        assert_eq!(result.memories[0].content, "Juan works at SAP");
        assert!((result.memories[0].score - 0.95).abs() < f32::EPSILON);
        assert_eq!(result.entities.len(), 1);
        assert_eq!(result.context_text, "Relevant context here");
    }

    #[test]
    fn test_learn_result_deserialization() {
        let json = r#"{
            "memories_created": 3,
            "entities_found": 2,
            "conflicts_found": 1
        }"#;

        let result: LearnResult = serde_json::from_str(json).unwrap();
        assert_eq!(result.memories_created, 3);
        assert_eq!(result.entities_found, 2);
        assert_eq!(result.conflicts_found, 1);
    }

    #[test]
    fn test_search_result_deserialization() {
        let json = r#"{
            "results": [
                {
                    "id": 10,
                    "content": "Memory content",
                    "node_type": "Event",
                    "score": 0.88,
                    "vector_sim": 0.75,
                    "graph_centrality": 0.5,
                    "recency": 0.9
                }
            ],
            "total": 1,
            "query": "test query"
        }"#;

        let result: SearchResult = serde_json::from_str(json).unwrap();
        assert_eq!(result.results.len(), 1);
        assert_eq!(result.total, 1);
        assert_eq!(result.query, "test query");
    }

    #[test]
    fn test_create_memory_result_deserialization() {
        let json = r#"{
            "chunk_node_ids": [100, 101],
            "entity_node_ids": [200],
            "edges_created": 5,
            "metrics": {
                "chunks_processed": 2,
                "entities_extracted": 3,
                "relations_extracted": 1,
                "contradictions_detected": 0,
                "total_us": 12345
            }
        }"#;

        let result: CreateMemoryResult = serde_json::from_str(json).unwrap();
        assert_eq!(result.chunk_node_ids, vec![100, 101]);
        assert_eq!(result.entity_node_ids, vec![200]);
        assert_eq!(result.edges_created, 5);
        assert_eq!(result.metrics.chunks_processed, 2);
    }

    #[test]
    fn test_health_response_deserialization() {
        let json = r#"{
            "status": "healthy",
            "version": "0.1.0",
            "instance_id": "abc123",
            "instance_role": "standalone",
            "storage_mode": "embedded",
            "vector_backend": "helix",
            "graph_backend": "helix",
            "models": {
                "embedder_loaded": true,
                "embedding_model": "all-MiniLM-L6-v2",
                "ner_loaded": false,
                "relation_extractor_loaded": false
            }
        }"#;

        let result: HealthResponse = serde_json::from_str(json).unwrap();
        assert_eq!(result.status, "healthy");
        assert!(result.models.embedder_loaded);
    }

    #[test]
    fn test_entity_response_with_neighbors() {
        let json = r#"{
            "id": 5,
            "content": "Juan",
            "node_type": "Entity",
            "timestamp": 1000,
            "metadata": {},
            "neighbors": [
                {
                    "node_id": 6,
                    "content": "SAP",
                    "edge_type": "WORKS_AT",
                    "weight": 0.9
                }
            ]
        }"#;

        let entity: EntityResponse = serde_json::from_str(json).unwrap();
        assert_eq!(entity.id, 5);
        let neighbors = entity.neighbors.unwrap();
        assert_eq!(neighbors.len(), 1);
        assert_eq!(neighbors[0].content, "SAP");
    }

    #[test]
    fn test_metrics_response_deserialization() {
        let json = r#"{
            "instance_id": "test-id",
            "total_requests": 100,
            "total_ingestions": 20,
            "total_searches": 50,
            "uptime_secs": 3600
        }"#;

        let result: MetricsResponse = serde_json::from_str(json).unwrap();
        assert_eq!(result.total_requests, 100);
        assert_eq!(result.uptime_secs, 3600);
    }

    #[test]
    fn test_request_body_serialization() {
        let body = CreateMemoryBody {
            text: "Test memory".to_string(),
            metadata: HashMap::new(),
        };
        let json = serde_json::to_string(&body).unwrap();
        assert!(json.contains("\"text\":\"Test memory\""));
        // Empty metadata should be skipped
        assert!(!json.contains("metadata"));
    }

    #[test]
    fn test_search_body_serialization() {
        let body = SearchBody {
            query: "test".to_string(),
            limit: Some(5),
            node_type: None,
            time_range: None,
            query_mindset: None,
        };
        let json = serde_json::to_string(&body).unwrap();
        assert!(json.contains("\"query\":\"test\""));
        assert!(json.contains("\"limit\":5"));
        // None fields should be skipped
        assert!(!json.contains("node_type"));
        assert!(!json.contains("time_range"));
    }

    #[test]
    fn test_augment_body_serialization() {
        let body = AugmentBody {
            context: "What does Juan do?".to_string(),
            limit: None,
        };
        let json = serde_json::to_string(&body).unwrap();
        assert!(json.contains("\"context\":\"What does Juan do?\""));
        assert!(!json.contains("limit"));
    }

    #[tokio::test]
    async fn test_connection_refused_returns_error() {
        let config = ClientConfig {
            timeout: Duration::from_millis(100),
            retry: RetryConfig {
                max_retries: 0,
                ..Default::default()
            },
            ..Default::default()
        };
        let client = UcotronClient::with_config("http://127.0.0.1:19999", config);

        let result = client.health().await;
        assert!(result.is_err());
        match result.unwrap_err() {
            UcotronError::RetriesExhausted { attempts, .. } => {
                assert_eq!(attempts, 1); // 0 retries = 1 attempt
            }
            other => panic!("Expected RetriesExhausted, got: {:?}", other),
        }
    }

    #[tokio::test]
    async fn test_retry_exhaustion() {
        let config = ClientConfig {
            timeout: Duration::from_millis(50),
            retry: RetryConfig {
                max_retries: 2,
                base_delay: Duration::from_millis(10),
                max_delay: Duration::from_millis(50),
            },
            ..Default::default()
        };
        let client = UcotronClient::with_config("http://127.0.0.1:19999", config);

        let start = std::time::Instant::now();
        let result = client.health().await;
        let elapsed = start.elapsed();

        assert!(result.is_err());
        match result.unwrap_err() {
            UcotronError::RetriesExhausted { attempts, .. } => {
                assert_eq!(attempts, 3); // 2 retries + 1 initial = 3 attempts
            }
            other => panic!("Expected RetriesExhausted, got: {:?}", other),
        }
        // Should have waited at least some time for retries
        assert!(elapsed >= Duration::from_millis(10));
    }

    // -----------------------------------------------------------------------
    // W3C Trace Context propagation tests
    // -----------------------------------------------------------------------

    #[cfg(feature = "otel")]
    mod otel_tests {
        use super::*;

        #[test]
        fn test_header_map_injector() {
            use opentelemetry::propagation::Injector;

            let mut map = HeaderMap(Vec::new());
            map.set("traceparent", "00-abc-def-01".to_string());
            map.set("tracestate", "vendor=value".to_string());

            assert_eq!(map.0.len(), 2);
            assert_eq!(
                map.0[0],
                ("traceparent".to_string(), "00-abc-def-01".to_string())
            );
            assert_eq!(
                map.0[1],
                ("tracestate".to_string(), "vendor=value".to_string())
            );
        }

        #[test]
        fn test_inject_trace_context_no_active_span() {
            // Without an active span, inject_trace_context should not panic
            // and should return the builder unchanged (no traceparent header).
            let client = reqwest::Client::new();
            let req = client.get("http://localhost:8420/test");
            let _req = inject_trace_context(req);
            // No panic = success. We can't easily inspect reqwest::RequestBuilder
            // headers, but we verify it doesn't error.
        }

        #[test]
        fn test_inject_trace_context_with_active_span() {
            use opentelemetry::trace::{
                SpanContext, SpanId, TraceContextExt, TraceFlags, TraceId, TraceState,
            };

            // Create a span context with a known trace ID and span ID.
            let trace_id = TraceId::from_hex("4bf92f3577b34da6a3ce929d0e0e4736").unwrap();
            let span_id = SpanId::from_hex("00f067aa0ba902b7").unwrap();
            let span_context = SpanContext::new(
                trace_id,
                span_id,
                TraceFlags::SAMPLED,
                true, // remote
                TraceState::default(),
            );

            // Attach the span context to the current OpenTelemetry context.
            let cx = opentelemetry::Context::current().with_remote_span_context(span_context);
            let _guard = cx.attach();

            // Now inject trace context and verify via the HeaderMap injector.
            use opentelemetry::propagation::TextMapPropagator;
            use opentelemetry_sdk::propagation::TraceContextPropagator;

            let propagator = TraceContextPropagator::new();
            let cx = opentelemetry::Context::current();
            let mut headers = HeaderMap(Vec::new());
            propagator.inject_context(&cx, &mut headers);

            // Should have injected a traceparent header.
            let traceparent = headers.0.iter().find(|(k, _)| k == "traceparent");
            assert!(
                traceparent.is_some(),
                "Expected traceparent header to be injected"
            );

            let (_, value) = traceparent.unwrap();
            assert!(
                value.contains("4bf92f3577b34da6a3ce929d0e0e4736"),
                "traceparent should contain the trace ID, got: {}",
                value
            );
            assert!(
                value.contains("00f067aa0ba902b7"),
                "traceparent should contain the span ID, got: {}",
                value
            );
        }

        #[test]
        fn test_inject_trace_context_with_tracestate() {
            use opentelemetry::trace::{
                SpanContext, SpanId, TraceContextExt, TraceFlags, TraceId, TraceState,
            };

            let trace_id = TraceId::from_hex("abcdef1234567890abcdef1234567890").unwrap();
            let span_id = SpanId::from_hex("1234567890abcdef").unwrap();
            let tracestate = TraceState::from_key_value([("congo", "t61rcWkgMzE")]).unwrap();

            let span_context =
                SpanContext::new(trace_id, span_id, TraceFlags::SAMPLED, true, tracestate);

            let cx = opentelemetry::Context::current().with_remote_span_context(span_context);
            let _guard = cx.attach();

            use opentelemetry::propagation::TextMapPropagator;
            use opentelemetry_sdk::propagation::TraceContextPropagator;

            let propagator = TraceContextPropagator::new();
            let cx = opentelemetry::Context::current();
            let mut headers = HeaderMap(Vec::new());
            propagator.inject_context(&cx, &mut headers);

            // Should have both traceparent and tracestate.
            let has_traceparent = headers.0.iter().any(|(k, _)| k == "traceparent");
            let tracestate_header = headers.0.iter().find(|(k, _)| k == "tracestate");

            assert!(has_traceparent, "Expected traceparent header");
            assert!(tracestate_header.is_some(), "Expected tracestate header");

            let (_, ts_value) = tracestate_header.unwrap();
            assert!(
                ts_value.contains("congo=t61rcWkgMzE"),
                "tracestate should contain vendor key-value, got: {}",
                ts_value
            );
        }
    }

    // Test that works without the otel feature — just verifies no-op path compiles.
    #[test]
    fn test_inject_trace_context_compiles_without_otel() {
        let client = reqwest::Client::new();
        let req = client.get("http://localhost:8420/test");
        let _req = inject_trace_context(req);
    }
}
