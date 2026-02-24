//! # Ucotron Config
//!
//! Configuration system for the Ucotron cognitive memory framework.
//!
//! Provides TOML-based configuration parsing and validation for the server,
//! storage backends, model pipelines, consolidation settings, namespaces, and auth.
//!
//! # Configuration Schema
//!
//! The configuration file (`ucotron.toml`) supports the following sections:
//! - `[server]` — HTTP server settings (host, port, workers, log_level)
//! - `[storage]` — Storage backend selection and settings
//! - `[models]` — Embedding, NER, and LLM model configuration
//! - `[consolidation]` — Background consolidation worker settings
//! - `[namespaces]` — Multi-tenancy namespace configuration
//! - `[auth]` — Authentication settings (API key, JWT)
//! - `[mcp]` — MCP (Model Context Protocol) server settings
//! - `[telemetry]` — OpenTelemetry tracing and metrics export
//!
//! # Environment Variable Overrides
//!
//! Every config field can be overridden via environment variables using the
//! `UCOTRON_` prefix and `_` as section separator:
//! - `UCOTRON_SERVER_HOST` → `server.host`
//! - `UCOTRON_SERVER_PORT` → `server.port`
//! - `UCOTRON_SERVER_WORKERS` → `server.workers`
//! - `UCOTRON_SERVER_LOG_LEVEL` → `server.log_level`
//! - `UCOTRON_SERVER_LOG_FORMAT` → `server.log_format`
//! - `UCOTRON_STORAGE_MODE` → `storage.mode`
//! - `UCOTRON_MODELS_DIR` → `models.models_dir`
//! - `UCOTRON_CONSOLIDATION_TRIGGER_INTERVAL` → `consolidation.trigger_interval`
//! - `UCOTRON_AUTH_API_KEY` → `auth.api_key`
//! - etc.

use serde::{Deserialize, Serialize};

/// Top-level Ucotron configuration.
///
/// Parsed from `ucotron.toml` or constructed programmatically.
/// Environment variables with the `UCOTRON_` prefix override TOML values.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct UcotronConfig {
    /// HTTP server settings.
    #[serde(default)]
    pub server: ServerConfig,
    /// Storage backend configuration.
    #[serde(default)]
    pub storage: StorageConfig,
    /// ML model configuration.
    #[serde(default)]
    pub models: ModelsConfig,
    /// Background consolidation settings.
    #[serde(default)]
    pub consolidation: ConsolidationConfig,
    /// MCP server settings.
    #[serde(default)]
    pub mcp: McpConfig,
    /// Multi-tenancy namespace configuration.
    #[serde(default)]
    pub namespaces: NamespacesConfig,
    /// Authentication settings.
    #[serde(default)]
    pub auth: AuthConfig,
    /// Multi-instance configuration.
    #[serde(default)]
    pub instance: InstanceConfig,
    /// GDPR compliance configuration.
    #[serde(default)]
    pub gdpr: GdprConfig,
    /// Audit logging configuration.
    #[serde(default)]
    pub audit: AuditConfig,
    /// OpenTelemetry configuration.
    #[serde(default)]
    pub telemetry: TelemetryConfig,
    /// Mindset auto-detection configuration.
    #[serde(default)]
    pub mindset: MindsetDetectorConfig,
    /// Connector scheduling configuration.
    #[serde(default)]
    pub connectors: ConnectorsConfig,
}

/// HTTP server configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerConfig {
    /// Bind address (default: "0.0.0.0").
    #[serde(default = "default_host")]
    pub host: String,
    /// HTTP port (default: 8420).
    #[serde(default = "default_port")]
    pub port: u16,
    /// Number of worker threads (default: 4).
    #[serde(default = "default_workers")]
    pub workers: usize,
    /// Log level (default: "info").
    #[serde(default = "default_log_level")]
    pub log_level: String,
    /// Log format: "text" (default) or "json" for structured JSON logging with trace IDs.
    #[serde(default = "default_log_format")]
    pub log_format: String,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            host: default_host(),
            port: default_port(),
            workers: default_workers(),
            log_level: default_log_level(),
            log_format: default_log_format(),
        }
    }
}

fn default_host() -> String {
    "0.0.0.0".to_string()
}
fn default_port() -> u16 {
    8420
}
fn default_workers() -> usize {
    4
}
fn default_log_level() -> String {
    "info".to_string()
}
fn default_log_format() -> String {
    "text".to_string()
}

/// Storage backend configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageConfig {
    /// Storage mode: "embedded" (default) or "external".
    #[serde(default = "default_storage_mode")]
    pub mode: String,
    /// Shared data directory for multi-instance mode.
    /// When `mode = "shared"`, all instances must point to the same directory.
    /// Both vector and graph backends will use sub-directories under this path.
    /// If set, overrides `vector.data_dir` and `graph.data_dir`.
    #[serde(default)]
    pub shared_data_dir: Option<String>,
    /// Directory for persisting uploaded media files (images, audio, video).
    /// Defaults to "data/media". Files are stored as `{node_id}.{ext}`.
    #[serde(default = "default_media_dir")]
    pub media_dir: String,
    /// Vector backend configuration.
    #[serde(default)]
    pub vector: VectorBackendConfig,
    /// Graph backend configuration.
    #[serde(default)]
    pub graph: GraphBackendConfig,
}

impl Default for StorageConfig {
    fn default() -> Self {
        Self {
            mode: default_storage_mode(),
            shared_data_dir: None,
            media_dir: default_media_dir(),
            vector: VectorBackendConfig::default(),
            graph: GraphBackendConfig::default(),
        }
    }
}

impl StorageConfig {
    /// Returns the effective data directory for vector storage.
    /// In shared mode with `shared_data_dir` set, returns the shared path.
    /// Otherwise returns the backend's own `data_dir`.
    pub fn effective_vector_data_dir(&self) -> &str {
        if self.mode == "shared" {
            if let Some(ref dir) = self.shared_data_dir {
                return dir;
            }
        }
        &self.vector.data_dir
    }

    /// Returns the effective data directory for graph storage.
    /// In shared mode with `shared_data_dir` set, returns the shared path.
    /// Otherwise returns the backend's own `data_dir`.
    pub fn effective_graph_data_dir(&self) -> &str {
        if self.mode == "shared" {
            if let Some(ref dir) = self.shared_data_dir {
                return dir;
            }
        }
        &self.graph.data_dir
    }

    /// Returns the media storage directory path.
    pub fn effective_media_dir(&self) -> &str {
        &self.media_dir
    }
}

fn default_storage_mode() -> String {
    "embedded".to_string()
}

/// Vector backend settings.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorBackendConfig {
    /// Backend type: "helix" (default), "qdrant", "custom".
    #[serde(default = "default_backend")]
    pub backend: String,
    /// Data directory for embedded backends.
    #[serde(default = "default_data_dir")]
    pub data_dir: String,
    /// Maximum database size in bytes (for LMDB map_size).
    #[serde(default = "default_max_db_size")]
    pub max_db_size: u64,
    /// External service URL (for qdrant, etc.).
    pub url: Option<String>,
    /// HNSW configuration (only used when vector index is HNSW).
    #[serde(default)]
    pub hnsw: HnswConfig,
}

/// HNSW vector index parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HnswConfig {
    /// Number of bi-directional links per node (default: 24).
    /// Higher values improve recall at the cost of memory and build time.
    #[serde(default = "default_hnsw_ef_construction")]
    pub ef_construction: usize,
    /// Search parameter: number of candidates to evaluate during search (default: 200).
    #[serde(default = "default_hnsw_ef_search")]
    pub ef_search: usize,
    /// Enable HNSW index (default: true). When false, falls back to brute-force SIMD.
    #[serde(default = "default_hnsw_enabled")]
    pub enabled: bool,
}

impl Default for HnswConfig {
    fn default() -> Self {
        Self {
            ef_construction: default_hnsw_ef_construction(),
            ef_search: default_hnsw_ef_search(),
            enabled: default_hnsw_enabled(),
        }
    }
}

fn default_hnsw_ef_construction() -> usize {
    200
}
fn default_hnsw_ef_search() -> usize {
    200
}
fn default_hnsw_enabled() -> bool {
    true
}

impl Default for VectorBackendConfig {
    fn default() -> Self {
        Self {
            backend: default_backend(),
            data_dir: default_data_dir(),
            max_db_size: default_max_db_size(),
            url: None,
            hnsw: HnswConfig::default(),
        }
    }
}

/// Graph backend settings.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphBackendConfig {
    /// Backend type: "helix" (default), "falkordb", "custom".
    #[serde(default = "default_backend")]
    pub backend: String,
    /// Data directory for embedded backends.
    #[serde(default = "default_data_dir")]
    pub data_dir: String,
    /// Maximum database size in bytes.
    #[serde(default = "default_max_db_size")]
    pub max_db_size: u64,
    /// Batch size for bulk operations.
    #[serde(default = "default_batch_size")]
    pub batch_size: usize,
    /// External service URL (for falkordb, etc.).
    pub url: Option<String>,
}

impl Default for GraphBackendConfig {
    fn default() -> Self {
        Self {
            backend: default_backend(),
            data_dir: default_data_dir(),
            max_db_size: default_max_db_size(),
            batch_size: default_batch_size(),
            url: None,
        }
    }
}

fn default_backend() -> String {
    "helix".to_string()
}
fn default_data_dir() -> String {
    "data".to_string()
}
fn default_media_dir() -> String {
    "data/media".to_string()
}
fn default_max_db_size() -> u64 {
    10 * 1024 * 1024 * 1024 // 10GB
}
fn default_batch_size() -> usize {
    10_000
}

/// ML model configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelsConfig {
    /// Embedding model name (default: "all-MiniLM-L6-v2").
    #[serde(default = "default_embedding_model")]
    pub embedding_model: String,
    /// NER model name (default: "gliner-multi-v2.1").
    #[serde(default = "default_ner_model")]
    pub ner_model: String,
    /// LLM model for relation extraction (default: "Qwen3-4B-GGUF").
    #[serde(default = "default_llm_model")]
    pub llm_model: String,
    /// LLM backend: "candle" or "llama_cpp" (default: "candle").
    #[serde(default = "default_llm_backend")]
    pub llm_backend: String,
    /// CLIP model name for image embedding (default: "clip-vit-base-patch32").
    #[serde(default = "default_clip_model")]
    pub clip_model: String,
    /// Directory for storing model files.
    #[serde(default = "default_models_dir")]
    pub models_dir: String,
    /// Enable document OCR pipeline (default: true).
    #[serde(default = "default_enable_ocr")]
    pub enable_ocr: bool,
    /// Language for Tesseract OCR (default: "eng").
    #[serde(default = "default_ocr_language")]
    pub ocr_language: String,
    /// Path to the tesseract binary (default: "tesseract", relies on PATH).
    #[serde(default = "default_tesseract_path")]
    pub tesseract_path: String,
    /// Fine-tuned relation extraction model name on Fireworks (e.g., "accounts/ucotron/models/re-qwen2-5-7b").
    /// When set and non-empty, the extraction pipeline will use this model via Fireworks API
    /// instead of co-occurrence. Falls back to co-occurrence on API errors.
    #[serde(default)]
    pub fine_tuned_re_model: String,
    /// Fireworks inference API endpoint (default: "https://api.fireworks.ai/inference/v1").
    #[serde(default = "default_fine_tuned_re_endpoint")]
    pub fine_tuned_re_endpoint: String,
    /// Name of the environment variable holding the Fireworks API key (default: "FIREWORKS_API_KEY").
    /// The actual key is read from this env var at runtime — never stored in config files.
    #[serde(default = "default_fine_tuned_re_api_key_env")]
    pub fine_tuned_re_api_key_env: String,
}

impl Default for ModelsConfig {
    fn default() -> Self {
        Self {
            embedding_model: default_embedding_model(),
            ner_model: default_ner_model(),
            llm_model: default_llm_model(),
            llm_backend: default_llm_backend(),
            clip_model: default_clip_model(),
            models_dir: default_models_dir(),
            enable_ocr: default_enable_ocr(),
            ocr_language: default_ocr_language(),
            tesseract_path: default_tesseract_path(),
            fine_tuned_re_model: String::new(),
            fine_tuned_re_endpoint: default_fine_tuned_re_endpoint(),
            fine_tuned_re_api_key_env: default_fine_tuned_re_api_key_env(),
        }
    }
}

fn default_embedding_model() -> String {
    "all-MiniLM-L6-v2".to_string()
}
fn default_ner_model() -> String {
    "gliner-multi-v2.1".to_string()
}
fn default_llm_model() -> String {
    "Qwen3-4B-GGUF".to_string()
}
fn default_llm_backend() -> String {
    "candle".to_string()
}
fn default_clip_model() -> String {
    "clip-vit-base-patch32".to_string()
}
fn default_models_dir() -> String {
    "models".to_string()
}
fn default_enable_ocr() -> bool {
    true
}
fn default_ocr_language() -> String {
    "eng".to_string()
}
fn default_tesseract_path() -> String {
    "tesseract".to_string()
}
fn default_fine_tuned_re_endpoint() -> String {
    "https://api.fireworks.ai/inference/v1".to_string()
}
fn default_fine_tuned_re_api_key_env() -> String {
    "FIREWORKS_API_KEY".to_string()
}

/// Background consolidation worker configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsolidationConfig {
    /// Number of messages between consolidation runs (default: 100).
    #[serde(default = "default_trigger_interval")]
    pub trigger_interval: usize,
    /// Enable memory decay for old nodes (default: true).
    #[serde(default = "default_enable_decay")]
    pub enable_decay: bool,
    /// Decay half-life in seconds (default: 30 days).
    #[serde(default = "default_decay_halflife")]
    pub decay_halflife_secs: u64,
}

impl Default for ConsolidationConfig {
    fn default() -> Self {
        Self {
            trigger_interval: default_trigger_interval(),
            enable_decay: default_enable_decay(),
            decay_halflife_secs: default_decay_halflife(),
        }
    }
}

fn default_trigger_interval() -> usize {
    100
}
fn default_enable_decay() -> bool {
    true
}
fn default_decay_halflife() -> u64 {
    30 * 24 * 3600 // 30 days
}

/// MCP (Model Context Protocol) server configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpConfig {
    /// Enable MCP server (default: true).
    #[serde(default = "default_mcp_enabled")]
    pub enabled: bool,
    /// Transport mode: "stdio" or "sse" (default: "stdio").
    #[serde(default = "default_mcp_transport")]
    pub transport: String,
    /// SSE port (only used when transport = "sse", default: 8421).
    #[serde(default = "default_mcp_port")]
    pub port: u16,
}

impl Default for McpConfig {
    fn default() -> Self {
        Self {
            enabled: default_mcp_enabled(),
            transport: default_mcp_transport(),
            port: default_mcp_port(),
        }
    }
}

fn default_mcp_enabled() -> bool {
    true
}
fn default_mcp_transport() -> String {
    "stdio".to_string()
}
fn default_mcp_port() -> u16 {
    8421
}

/// Multi-tenancy namespace configuration.
///
/// Namespaces isolate memory data between different tenants,
/// projects, users, agents, or threads.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NamespacesConfig {
    /// Default namespace when no `X-Ucotron-Namespace` header is provided.
    #[serde(default = "default_namespace")]
    pub default_namespace: String,
    /// If non-empty, only these namespaces are allowed.
    /// Empty means any namespace is allowed.
    #[serde(default)]
    pub allowed_namespaces: Vec<String>,
    /// Maximum number of namespaces allowed (0 = unlimited).
    #[serde(default)]
    pub max_namespaces: usize,
}

impl Default for NamespacesConfig {
    fn default() -> Self {
        Self {
            default_namespace: default_namespace(),
            allowed_namespaces: Vec::new(),
            max_namespaces: 0,
        }
    }
}

fn default_namespace() -> String {
    "default".to_string()
}

/// Authentication configuration.
///
/// Optional authentication for the REST API and MCP server.
/// When `enabled` is false (default), all requests are accepted.
///
/// RBAC roles (ordered by privilege):
/// - `admin`: full access including API key management and admin endpoints
/// - `writer`: read + write (ingest, learn, update, delete, GDPR)
/// - `reader`: read-only (search, augment, get, list, export)
/// - `viewer`: health + metrics only
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AuthConfig {
    /// Enable authentication (default: false).
    #[serde(default)]
    pub enabled: bool,
    /// Legacy single API key for simple auth (checked via `Authorization: Bearer <key>` header).
    /// Set via TOML or `UCOTRON_AUTH_API_KEY` env var.
    /// When used alone (without `api_keys`), grants `admin` role.
    #[serde(default)]
    pub api_key: Option<String>,
    /// JWT secret for token-based auth (future use).
    #[serde(default)]
    pub jwt_secret: Option<String>,
    /// JWT issuer (future use).
    #[serde(default)]
    pub jwt_issuer: Option<String>,
    /// Named API keys with role-based access control.
    /// Each key has a role and optional namespace scope.
    #[serde(default)]
    pub api_keys: Vec<ApiKeyEntry>,
}

/// A named API key with role and optional namespace scope.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiKeyEntry {
    /// Human-readable name for this key (e.g., "backend-service", "analytics-reader").
    pub name: String,
    /// The secret key value (checked via `Authorization: Bearer <key>` header).
    pub key: String,
    /// Role assigned to this key: "admin", "writer", "reader", or "viewer".
    #[serde(default = "default_api_key_role")]
    pub role: String,
    /// Optional namespace scope. If set, this key can only access the specified namespace.
    /// If empty/unset, the key can access all namespaces.
    #[serde(default)]
    pub namespace: Option<String>,
    /// Whether this key is active. Set to false to revoke without deleting.
    #[serde(default = "default_true")]
    pub active: bool,
}

fn default_api_key_role() -> String {
    "reader".to_string()
}

fn default_true() -> bool {
    true
}

/// RBAC role with ordered privilege levels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum AuthRole {
    Viewer = 0,
    Reader = 1,
    Writer = 2,
    Admin = 3,
}

impl AuthRole {
    /// Parse a role string into an AuthRole.
    pub fn parse_role(s: &str) -> Option<Self> {
        match s {
            "admin" => Some(AuthRole::Admin),
            "writer" => Some(AuthRole::Writer),
            "reader" => Some(AuthRole::Reader),
            "viewer" => Some(AuthRole::Viewer),
            _ => None,
        }
    }

    /// Whether this role has at least the given privilege level.
    pub fn has_privilege(&self, required: AuthRole) -> bool {
        (*self as u8) >= (required as u8)
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            AuthRole::Admin => "admin",
            AuthRole::Writer => "writer",
            AuthRole::Reader => "reader",
            AuthRole::Viewer => "viewer",
        }
    }
}

impl AuthConfig {
    /// Look up an API key and return its role and namespace scope.
    /// Checks named `api_keys` first, then falls back to legacy `api_key` (admin role).
    pub fn authenticate(&self, bearer_token: &str) -> Option<(AuthRole, Option<String>)> {
        // Check named API keys first.
        for entry in &self.api_keys {
            if entry.active && entry.key == bearer_token {
                if let Some(role) = AuthRole::parse_role(&entry.role) {
                    return Some((role, entry.namespace.clone()));
                }
            }
        }
        // Fall back to legacy single API key (grants admin).
        if let Some(ref legacy_key) = self.api_key {
            if legacy_key == bearer_token {
                return Some((AuthRole::Admin, None));
            }
        }
        None
    }
}

/// Multi-instance configuration.
///
/// Controls how this server instance participates in a multi-instance deployment.
/// In single-instance mode (default), all settings can be left at defaults.
///
/// For multi-instance deployments:
/// - Each instance needs a unique `instance_id`
/// - `role` determines whether this instance can write (`writer`), only read (`reader`), or auto-detect
/// - `id_range_start` and `id_range_size` partition the node ID space to avoid collisions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InstanceConfig {
    /// Unique identifier for this server instance.
    /// Auto-generated from hostname + PID if not set.
    #[serde(default = "default_instance_id")]
    pub instance_id: String,
    /// Instance role: "auto" (default), "writer", or "reader".
    /// - "auto": single-instance mode, acts as both reader and writer
    /// - "writer": can perform writes (ingestion, learn, update, delete)
    /// - "reader": read-only (search, augment, get operations only)
    #[serde(default = "default_instance_role")]
    pub role: String,
    /// Starting node ID for this instance's ID allocation range.
    /// Each instance should have a non-overlapping range to avoid ID collisions.
    /// Default: 1_000_000 (same as single-instance).
    #[serde(default = "default_id_range_start")]
    pub id_range_start: u64,
    /// Size of this instance's node ID allocation range.
    /// Default: 1_000_000_000 (1 billion IDs per instance).
    #[serde(default = "default_id_range_size")]
    pub id_range_size: u64,
}

impl Default for InstanceConfig {
    fn default() -> Self {
        Self {
            instance_id: default_instance_id(),
            role: default_instance_role(),
            id_range_start: default_id_range_start(),
            id_range_size: default_id_range_size(),
        }
    }
}

fn default_instance_id() -> String {
    "auto".to_string()
}
fn default_instance_role() -> String {
    "auto".to_string()
}
fn default_id_range_start() -> u64 {
    1_000_000
}
fn default_id_range_size() -> u64 {
    1_000_000_000
}

impl InstanceConfig {
    /// Resolve the instance_id. If set to "auto", generate from hostname + PID.
    pub fn resolved_instance_id(&self) -> String {
        if self.instance_id == "auto" {
            let hostname = hostname::get()
                .ok()
                .and_then(|h| h.into_string().ok())
                .unwrap_or_else(|| "unknown".to_string());
            let pid = std::process::id();
            format!("{}-{}", hostname, pid)
        } else {
            self.instance_id.clone()
        }
    }

    /// Whether this instance can perform write operations.
    pub fn can_write(&self) -> bool {
        matches!(self.role.as_str(), "auto" | "writer")
    }

    /// Whether this instance is a dedicated reader (no writes).
    pub fn is_reader_only(&self) -> bool {
        self.role == "reader"
    }
}

/// GDPR compliance configuration.
///
/// Controls data retention policies and right-to-be-forgotten behavior.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GdprConfig {
    /// Enable GDPR endpoints (default: true).
    #[serde(default = "default_gdpr_enabled")]
    pub enabled: bool,
    /// Default data retention TTL in seconds (0 = no automatic expiry).
    #[serde(default)]
    pub default_retention_ttl_secs: u64,
    /// Per-namespace retention policies: list of {namespace, ttl_secs}.
    #[serde(default)]
    pub retention_policies: Vec<GdprRetentionPolicyConfig>,
}

/// A single retention policy entry in configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GdprRetentionPolicyConfig {
    /// Namespace this policy applies to ("*" = all namespaces).
    pub namespace: String,
    /// Time-to-live in seconds (0 = no expiry).
    pub ttl_secs: u64,
}

impl Default for GdprConfig {
    fn default() -> Self {
        Self {
            enabled: default_gdpr_enabled(),
            default_retention_ttl_secs: 0,
            retention_policies: Vec::new(),
        }
    }
}

fn default_gdpr_enabled() -> bool {
    true
}

/// Audit logging configuration.
///
/// Controls the immutable audit trail that records all API operations.
/// Audit entries are stored in an append-only in-memory log and persisted
/// as special graph nodes for durability.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditConfig {
    /// Enable audit logging (default: true).
    #[serde(default = "default_audit_enabled")]
    pub enabled: bool,
    /// Retention period for audit entries in seconds.
    /// Entries older than this are eligible for pruning.
    /// 0 = keep forever. Default: 7776000 (90 days).
    #[serde(default = "default_audit_retention_secs")]
    pub retention_secs: u64,
    /// Maximum number of audit entries kept in memory.
    /// Oldest entries are evicted when this limit is exceeded.
    /// Default: 100000.
    #[serde(default = "default_audit_max_entries")]
    pub max_entries: usize,
}

impl Default for AuditConfig {
    fn default() -> Self {
        Self {
            enabled: default_audit_enabled(),
            retention_secs: default_audit_retention_secs(),
            max_entries: default_audit_max_entries(),
        }
    }
}

fn default_audit_enabled() -> bool {
    true
}

fn default_audit_retention_secs() -> u64 {
    7_776_000 // 90 days
}

fn default_audit_max_entries() -> usize {
    100_000
}

/// OpenTelemetry configuration.
///
/// Controls OTLP trace/metric/log export to an OpenTelemetry collector.
/// Disabled by default — enable and point to a collector (e.g., Jaeger, Grafana Tempo).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TelemetryConfig {
    /// Enable OTLP telemetry export (default: false).
    #[serde(default)]
    pub enabled: bool,
    /// OTLP gRPC collector endpoint (default: "http://localhost:4317").
    #[serde(default = "default_telemetry_otlp_endpoint")]
    pub otlp_endpoint: String,
    /// Service name reported in OTLP traces (default: "ucotron").
    #[serde(default = "default_telemetry_service_name")]
    pub service_name: String,
    /// Trace sampling ratio, 0.0 to 1.0 (default: 1.0 = sample everything).
    #[serde(default = "default_telemetry_sample_rate")]
    pub sample_rate: f64,
    /// Export traces via OTLP (default: true).
    #[serde(default = "default_true")]
    pub export_traces: bool,
    /// Export metrics via OTLP (default: true).
    #[serde(default = "default_true")]
    pub export_metrics: bool,
    /// Export logs via OTLP (default: false).
    #[serde(default)]
    pub export_logs: bool,
}

impl Default for TelemetryConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            otlp_endpoint: default_telemetry_otlp_endpoint(),
            service_name: default_telemetry_service_name(),
            sample_rate: default_telemetry_sample_rate(),
            export_traces: true,
            export_metrics: true,
            export_logs: false,
        }
    }
}

fn default_telemetry_otlp_endpoint() -> String {
    "http://localhost:4317".to_string()
}
fn default_telemetry_service_name() -> String {
    "ucotron".to_string()
}
fn default_telemetry_sample_rate() -> f64 {
    1.0
}

/// Mindset auto-detection configuration.
///
/// Controls automatic detection of cognitive mindset (Convergent, Divergent,
/// Algorithmic) from query keywords. When enabled and no explicit mindset is
/// provided in the search request, the system scans for keyword patterns.
///
/// ```toml
/// [mindset]
/// enabled = true
/// algorithmic_keywords = ["verify", "confirm", "check", "validate", "prove", "correct"]
/// divergent_keywords = ["what if", "explore", "brainstorm", "alternative", "imagine", "creative"]
/// convergent_keywords = ["summarize", "consensus", "agree", "common", "overview", "conclude"]
/// spatial_keywords = ["connected", "path", "route", "bridge", "relationship", "link", "network", "graph"]
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MindsetDetectorConfig {
    /// Enable automatic mindset detection from query keywords (default: true).
    #[serde(default = "default_true")]
    pub enabled: bool,
    /// Keywords that trigger Algorithmic mindset (verification, logical checking).
    #[serde(default = "default_algorithmic_keywords")]
    pub algorithmic_keywords: Vec<String>,
    /// Keywords that trigger Divergent mindset (exploration, brainstorming).
    #[serde(default = "default_divergent_keywords")]
    pub divergent_keywords: Vec<String>,
    /// Keywords that trigger Convergent mindset (synthesis, consensus).
    #[serde(default = "default_convergent_keywords")]
    pub convergent_keywords: Vec<String>,
    /// Keywords that trigger Spatial mindset (graph traversal, path-based reasoning).
    #[serde(default = "default_spatial_keywords")]
    pub spatial_keywords: Vec<String>,
}

impl Default for MindsetDetectorConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            algorithmic_keywords: default_algorithmic_keywords(),
            divergent_keywords: default_divergent_keywords(),
            convergent_keywords: default_convergent_keywords(),
            spatial_keywords: default_spatial_keywords(),
        }
    }
}

fn default_algorithmic_keywords() -> Vec<String> {
    vec![
        "verify".into(),
        "confirm".into(),
        "check".into(),
        "validate".into(),
        "prove".into(),
        "correct".into(),
    ]
}

fn default_divergent_keywords() -> Vec<String> {
    vec![
        "what if".into(),
        "explore".into(),
        "brainstorm".into(),
        "alternative".into(),
        "imagine".into(),
        "creative".into(),
    ]
}

fn default_convergent_keywords() -> Vec<String> {
    vec![
        "summarize".into(),
        "consensus".into(),
        "agree".into(),
        "common".into(),
        "overview".into(),
        "conclude".into(),
    ]
}

fn default_spatial_keywords() -> Vec<String> {
    vec![
        "connected".into(),
        "path".into(),
        "route".into(),
        "bridge".into(),
        "relationship".into(),
        "link".into(),
        "network".into(),
        "graph".into(),
    ]
}

/// Connector scheduling configuration.
///
/// Controls cron-based periodic sync for external data source connectors
/// (Slack, GitHub, Notion, etc.). Individual connector schedules are
/// configured as entries in the `[[connectors.schedules]]` array.
///
/// ```toml
/// [connectors]
/// enabled = true
/// check_interval_secs = 60
///
/// [[connectors.schedules]]
/// connector_id = "my-slack"
/// cron_expression = "0 */6 * * * *"
/// timeout_secs = 300
/// max_retries = 3
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConnectorsConfig {
    /// Enable the connector scheduler (default: false).
    #[serde(default)]
    pub enabled: bool,
    /// How often the scheduler checks for due cron jobs (seconds, default: 60).
    #[serde(default = "default_connector_check_interval")]
    pub check_interval_secs: u64,
    /// Connector schedule entries.
    #[serde(default)]
    pub schedules: Vec<ConnectorScheduleEntry>,
}

impl Default for ConnectorsConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            check_interval_secs: default_connector_check_interval(),
            schedules: Vec::new(),
        }
    }
}

/// A single connector schedule entry in the configuration file.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConnectorScheduleEntry {
    /// Connector instance ID (must match a registered connector).
    pub connector_id: String,
    /// Cron expression for periodic sync (e.g., "0 */6 * * * *").
    /// Uses 6-field format: sec min hour day month weekday.
    pub cron_expression: Option<String>,
    /// Whether this schedule is active (default: true).
    #[serde(default = "default_true")]
    pub enabled: bool,
    /// Timeout for a single sync operation in seconds (default: 300).
    #[serde(default = "default_connector_timeout")]
    pub timeout_secs: u64,
    /// Number of retries on sync failure (default: 3).
    #[serde(default = "default_connector_retries")]
    pub max_retries: u32,
}

fn default_connector_check_interval() -> u64 {
    60
}

fn default_connector_timeout() -> u64 {
    300
}

fn default_connector_retries() -> u32 {
    3
}

impl UcotronConfig {
    /// Load configuration from a TOML file, then apply environment variable overrides.
    pub fn from_file(path: &str) -> anyhow::Result<Self> {
        let contents = std::fs::read_to_string(path)
            .map_err(|e| anyhow::anyhow!("Failed to read config file '{}': {}", path, e))?;
        Self::parse_toml(&contents)
    }

    /// Parse configuration from a TOML string, apply env overrides, then validate.
    pub fn parse_toml(toml_str: &str) -> anyhow::Result<Self> {
        let mut config: UcotronConfig = toml::from_str(toml_str)
            .map_err(|e| anyhow::anyhow!("Failed to parse TOML config: {}", e))?;
        config.apply_env_overrides();
        config.validate()?;
        Ok(config)
    }

    /// Apply environment variable overrides to the configuration.
    ///
    /// Variables use the `UCOTRON_` prefix with `_` as section separator:
    /// - `UCOTRON_SERVER_HOST` → `server.host`
    /// - `UCOTRON_SERVER_PORT` → `server.port`
    /// - `UCOTRON_SERVER_WORKERS` → `server.workers`
    /// - `UCOTRON_SERVER_LOG_LEVEL` → `server.log_level`
    /// - `UCOTRON_SERVER_LOG_FORMAT` → `server.log_format`
    /// - `UCOTRON_STORAGE_MODE` → `storage.mode`
    /// - `UCOTRON_STORAGE_VECTOR_BACKEND` → `storage.vector.backend`
    /// - `UCOTRON_STORAGE_VECTOR_DATA_DIR` → `storage.vector.data_dir`
    /// - `UCOTRON_STORAGE_GRAPH_BACKEND` → `storage.graph.backend`
    /// - `UCOTRON_STORAGE_GRAPH_DATA_DIR` → `storage.graph.data_dir`
    /// - `UCOTRON_STORAGE_GRAPH_BATCH_SIZE` → `storage.graph.batch_size`
    /// - `UCOTRON_MODELS_EMBEDDING_MODEL` → `models.embedding_model`
    /// - `UCOTRON_MODELS_NER_MODEL` → `models.ner_model`
    /// - `UCOTRON_MODELS_LLM_MODEL` → `models.llm_model`
    /// - `UCOTRON_MODELS_LLM_BACKEND` → `models.llm_backend`
    /// - `UCOTRON_MODELS_DIR` → `models.models_dir`
    /// - `UCOTRON_CONSOLIDATION_TRIGGER_INTERVAL` → `consolidation.trigger_interval`
    /// - `UCOTRON_CONSOLIDATION_ENABLE_DECAY` → `consolidation.enable_decay`
    /// - `UCOTRON_CONSOLIDATION_DECAY_HALFLIFE_SECS` → `consolidation.decay_halflife_secs`
    /// - `UCOTRON_MCP_ENABLED` → `mcp.enabled`
    /// - `UCOTRON_MCP_TRANSPORT` → `mcp.transport`
    /// - `UCOTRON_MCP_PORT` → `mcp.port`
    /// - `UCOTRON_NAMESPACES_DEFAULT` → `namespaces.default_namespace`
    /// - `UCOTRON_AUTH_ENABLED` → `auth.enabled`
    /// - `UCOTRON_AUTH_API_KEY` → `auth.api_key`
    /// - `UCOTRON_AUTH_JWT_SECRET` → `auth.jwt_secret`
    /// - `UCOTRON_TELEMETRY_ENABLED` → `telemetry.enabled`
    /// - `UCOTRON_TELEMETRY_OTLP_ENDPOINT` → `telemetry.otlp_endpoint`
    /// - `UCOTRON_TELEMETRY_SERVICE_NAME` → `telemetry.service_name`
    /// - `UCOTRON_TELEMETRY_SAMPLE_RATE` → `telemetry.sample_rate`
    pub fn apply_env_overrides(&mut self) {
        // Server overrides
        if let Ok(v) = std::env::var("UCOTRON_SERVER_HOST") {
            self.server.host = v;
        }
        if let Ok(v) = std::env::var("UCOTRON_SERVER_PORT") {
            if let Ok(port) = v.parse::<u16>() {
                self.server.port = port;
            }
        }
        if let Ok(v) = std::env::var("UCOTRON_SERVER_WORKERS") {
            if let Ok(w) = v.parse::<usize>() {
                self.server.workers = w;
            }
        }
        if let Ok(v) = std::env::var("UCOTRON_SERVER_LOG_LEVEL") {
            self.server.log_level = v;
        }
        if let Ok(v) = std::env::var("UCOTRON_SERVER_LOG_FORMAT") {
            self.server.log_format = v;
        }

        // Storage overrides
        if let Ok(v) = std::env::var("UCOTRON_STORAGE_MODE") {
            self.storage.mode = v;
        }
        if let Ok(v) = std::env::var("UCOTRON_STORAGE_SHARED_DATA_DIR") {
            self.storage.shared_data_dir = Some(v);
        }
        if let Ok(v) = std::env::var("UCOTRON_STORAGE_MEDIA_DIR") {
            self.storage.media_dir = v;
        }
        if let Ok(v) = std::env::var("UCOTRON_STORAGE_VECTOR_BACKEND") {
            self.storage.vector.backend = v;
        }
        if let Ok(v) = std::env::var("UCOTRON_STORAGE_VECTOR_DATA_DIR") {
            self.storage.vector.data_dir = v;
        }
        if let Ok(v) = std::env::var("UCOTRON_STORAGE_GRAPH_BACKEND") {
            self.storage.graph.backend = v;
        }
        if let Ok(v) = std::env::var("UCOTRON_STORAGE_GRAPH_DATA_DIR") {
            self.storage.graph.data_dir = v;
        }
        if let Ok(v) = std::env::var("UCOTRON_STORAGE_GRAPH_BATCH_SIZE") {
            if let Ok(bs) = v.parse::<usize>() {
                self.storage.graph.batch_size = bs;
            }
        }

        // Models overrides
        if let Ok(v) = std::env::var("UCOTRON_MODELS_EMBEDDING_MODEL") {
            self.models.embedding_model = v;
        }
        if let Ok(v) = std::env::var("UCOTRON_MODELS_NER_MODEL") {
            self.models.ner_model = v;
        }
        if let Ok(v) = std::env::var("UCOTRON_MODELS_LLM_MODEL") {
            self.models.llm_model = v;
        }
        if let Ok(v) = std::env::var("UCOTRON_MODELS_LLM_BACKEND") {
            self.models.llm_backend = v;
        }
        if let Ok(v) = std::env::var("UCOTRON_MODELS_DIR") {
            self.models.models_dir = v;
        }
        if let Ok(v) = std::env::var("UCOTRON_MODELS_ENABLE_OCR") {
            if let Ok(b) = v.parse::<bool>() {
                self.models.enable_ocr = b;
            }
        }
        if let Ok(v) = std::env::var("UCOTRON_MODELS_OCR_LANGUAGE") {
            self.models.ocr_language = v;
        }
        if let Ok(v) = std::env::var("UCOTRON_MODELS_TESSERACT_PATH") {
            self.models.tesseract_path = v;
        }
        if let Ok(v) = std::env::var("UCOTRON_MODELS_FINE_TUNED_RE_MODEL") {
            self.models.fine_tuned_re_model = v;
        }
        if let Ok(v) = std::env::var("UCOTRON_MODELS_FINE_TUNED_RE_ENDPOINT") {
            self.models.fine_tuned_re_endpoint = v;
        }
        if let Ok(v) = std::env::var("UCOTRON_MODELS_FINE_TUNED_RE_API_KEY_ENV") {
            self.models.fine_tuned_re_api_key_env = v;
        }

        // Consolidation overrides
        if let Ok(v) = std::env::var("UCOTRON_CONSOLIDATION_TRIGGER_INTERVAL") {
            if let Ok(ti) = v.parse::<usize>() {
                self.consolidation.trigger_interval = ti;
            }
        }
        if let Ok(v) = std::env::var("UCOTRON_CONSOLIDATION_ENABLE_DECAY") {
            if let Ok(b) = v.parse::<bool>() {
                self.consolidation.enable_decay = b;
            }
        }
        if let Ok(v) = std::env::var("UCOTRON_CONSOLIDATION_DECAY_HALFLIFE_SECS") {
            if let Ok(s) = v.parse::<u64>() {
                self.consolidation.decay_halflife_secs = s;
            }
        }

        // MCP overrides
        if let Ok(v) = std::env::var("UCOTRON_MCP_ENABLED") {
            if let Ok(b) = v.parse::<bool>() {
                self.mcp.enabled = b;
            }
        }
        if let Ok(v) = std::env::var("UCOTRON_MCP_TRANSPORT") {
            self.mcp.transport = v;
        }
        if let Ok(v) = std::env::var("UCOTRON_MCP_PORT") {
            if let Ok(port) = v.parse::<u16>() {
                self.mcp.port = port;
            }
        }

        // Namespaces overrides
        if let Ok(v) = std::env::var("UCOTRON_NAMESPACES_DEFAULT") {
            self.namespaces.default_namespace = v;
        }

        // Auth overrides
        if let Ok(v) = std::env::var("UCOTRON_AUTH_ENABLED") {
            if let Ok(b) = v.parse::<bool>() {
                self.auth.enabled = b;
            }
        }
        if let Ok(v) = std::env::var("UCOTRON_AUTH_API_KEY") {
            self.auth.api_key = Some(v);
        }
        if let Ok(v) = std::env::var("UCOTRON_AUTH_JWT_SECRET") {
            self.auth.jwt_secret = Some(v);
        }

        // GDPR overrides
        if let Ok(v) = std::env::var("UCOTRON_GDPR_ENABLED") {
            if let Ok(b) = v.parse::<bool>() {
                self.gdpr.enabled = b;
            }
        }
        if let Ok(v) = std::env::var("UCOTRON_GDPR_DEFAULT_RETENTION_TTL_SECS") {
            if let Ok(s) = v.parse::<u64>() {
                self.gdpr.default_retention_ttl_secs = s;
            }
        }

        // Audit overrides
        if let Ok(v) = std::env::var("UCOTRON_AUDIT_ENABLED") {
            if let Ok(b) = v.parse::<bool>() {
                self.audit.enabled = b;
            }
        }
        if let Ok(v) = std::env::var("UCOTRON_AUDIT_RETENTION_SECS") {
            if let Ok(s) = v.parse::<u64>() {
                self.audit.retention_secs = s;
            }
        }
        if let Ok(v) = std::env::var("UCOTRON_AUDIT_MAX_ENTRIES") {
            if let Ok(n) = v.parse::<usize>() {
                self.audit.max_entries = n;
            }
        }

        // Instance overrides
        if let Ok(v) = std::env::var("UCOTRON_INSTANCE_ID") {
            self.instance.instance_id = v;
        }
        if let Ok(v) = std::env::var("UCOTRON_INSTANCE_ROLE") {
            self.instance.role = v;
        }
        if let Ok(v) = std::env::var("UCOTRON_INSTANCE_ID_RANGE_START") {
            if let Ok(n) = v.parse::<u64>() {
                self.instance.id_range_start = n;
            }
        }
        if let Ok(v) = std::env::var("UCOTRON_INSTANCE_ID_RANGE_SIZE") {
            if let Ok(n) = v.parse::<u64>() {
                self.instance.id_range_size = n;
            }
        }

        // Telemetry overrides
        if let Ok(v) = std::env::var("UCOTRON_TELEMETRY_ENABLED") {
            if let Ok(b) = v.parse::<bool>() {
                self.telemetry.enabled = b;
            }
        }
        if let Ok(v) = std::env::var("UCOTRON_TELEMETRY_OTLP_ENDPOINT") {
            self.telemetry.otlp_endpoint = v;
        }
        if let Ok(v) = std::env::var("UCOTRON_TELEMETRY_SERVICE_NAME") {
            self.telemetry.service_name = v;
        }
        if let Ok(v) = std::env::var("UCOTRON_TELEMETRY_SAMPLE_RATE") {
            if let Ok(r) = v.parse::<f64>() {
                self.telemetry.sample_rate = r;
            }
        }
        if let Ok(v) = std::env::var("UCOTRON_TELEMETRY_EXPORT_TRACES") {
            if let Ok(b) = v.parse::<bool>() {
                self.telemetry.export_traces = b;
            }
        }
        if let Ok(v) = std::env::var("UCOTRON_TELEMETRY_EXPORT_METRICS") {
            if let Ok(b) = v.parse::<bool>() {
                self.telemetry.export_metrics = b;
            }
        }
        if let Ok(v) = std::env::var("UCOTRON_TELEMETRY_EXPORT_LOGS") {
            if let Ok(b) = v.parse::<bool>() {
                self.telemetry.export_logs = b;
            }
        }

        // Connectors overrides
        if let Ok(v) = std::env::var("UCOTRON_CONNECTORS_ENABLED") {
            if let Ok(b) = v.parse::<bool>() {
                self.connectors.enabled = b;
            }
        }
        if let Ok(v) = std::env::var("UCOTRON_CONNECTORS_CHECK_INTERVAL_SECS") {
            if let Ok(n) = v.parse::<u64>() {
                self.connectors.check_interval_secs = n;
            }
        }
    }

    // --- Telemetry accessors ---

    /// Whether OTLP telemetry export is enabled.
    pub fn telemetry_enabled(&self) -> bool {
        self.telemetry.enabled
    }

    /// OTLP gRPC collector endpoint.
    pub fn telemetry_otlp_endpoint(&self) -> String {
        self.telemetry.otlp_endpoint.clone()
    }

    /// Service name reported in OTLP traces.
    pub fn telemetry_service_name(&self) -> String {
        self.telemetry.service_name.clone()
    }

    /// Trace sampling ratio (0.0–1.0).
    pub fn telemetry_sample_rate(&self) -> f64 {
        self.telemetry.sample_rate
    }

    /// Validate configuration values with detailed error messages.
    pub fn validate(&self) -> anyhow::Result<()> {
        // --- Server validation ---
        if self.server.port == 0 {
            anyhow::bail!(
                "server.port must be > 0 (got 0). Set a valid port in ucotron.toml or via UCOTRON_SERVER_PORT env var."
            );
        }
        if self.server.workers == 0 {
            anyhow::bail!(
                "server.workers must be > 0 (got 0). Set the number of worker threads in ucotron.toml or via UCOTRON_SERVER_WORKERS env var."
            );
        }
        let valid_log_levels = ["trace", "debug", "info", "warn", "error"];
        if !valid_log_levels.contains(&self.server.log_level.as_str()) {
            anyhow::bail!(
                "server.log_level must be one of: {} (got '{}').",
                valid_log_levels.join(", "),
                self.server.log_level
            );
        }
        let valid_log_formats = ["text", "json"];
        if !valid_log_formats.contains(&self.server.log_format.as_str()) {
            anyhow::bail!(
                "server.log_format must be one of: {} (got '{}').",
                valid_log_formats.join(", "),
                self.server.log_format
            );
        }

        // --- Storage validation ---
        let valid_modes = ["embedded", "external", "shared"];
        if !valid_modes.contains(&self.storage.mode.as_str()) {
            anyhow::bail!(
                "storage.mode must be one of: {} (got '{}').",
                valid_modes.join(", "),
                self.storage.mode
            );
        }
        let valid_vector_backends = ["helix", "qdrant", "custom"];
        if !valid_vector_backends.contains(&self.storage.vector.backend.as_str()) {
            anyhow::bail!(
                "storage.vector.backend must be one of: {} (got '{}').",
                valid_vector_backends.join(", "),
                self.storage.vector.backend
            );
        }
        let valid_graph_backends = ["helix", "falkordb", "custom"];
        if !valid_graph_backends.contains(&self.storage.graph.backend.as_str()) {
            anyhow::bail!(
                "storage.graph.backend must be one of: {} (got '{}').",
                valid_graph_backends.join(", "),
                self.storage.graph.backend
            );
        }
        // Shared mode requires shared_data_dir when using helix backends
        if self.storage.mode == "shared"
            && self.storage.vector.backend == "helix"
            && self.storage.graph.backend == "helix"
            && self.storage.shared_data_dir.is_none()
        {
            anyhow::bail!(
                "storage.shared_data_dir is required when storage.mode is 'shared' with helix backends. \
                 All instances must point to the same directory. \
                 Set via ucotron.toml or UCOTRON_STORAGE_SHARED_DATA_DIR env var."
            );
        }

        // External backends require a URL
        if self.storage.mode == "external" || self.storage.mode == "shared" {
            if self.storage.vector.backend != "helix" && self.storage.vector.url.is_none() {
                anyhow::bail!(
                    "storage.vector.url is required when using external vector backend '{}' in '{}' mode.",
                    self.storage.vector.backend,
                    self.storage.mode
                );
            }
            if self.storage.graph.backend != "helix" && self.storage.graph.url.is_none() {
                anyhow::bail!(
                    "storage.graph.url is required when using external graph backend '{}' in '{}' mode.",
                    self.storage.graph.backend,
                    self.storage.mode
                );
            }
        }

        // --- HNSW validation ---
        if self.storage.vector.hnsw.ef_construction == 0 {
            anyhow::bail!("storage.vector.hnsw.ef_construction must be > 0.");
        }
        if self.storage.vector.hnsw.ef_search == 0 {
            anyhow::bail!("storage.vector.hnsw.ef_search must be > 0.");
        }

        // --- Models validation ---
        let valid_llm_backends = ["candle", "llama_cpp"];
        if !valid_llm_backends.contains(&self.models.llm_backend.as_str()) {
            anyhow::bail!(
                "models.llm_backend must be one of: {} (got '{}').",
                valid_llm_backends.join(", "),
                self.models.llm_backend
            );
        }
        if self.models.embedding_model.is_empty() {
            anyhow::bail!("models.embedding_model must not be empty.");
        }

        // --- MCP validation ---
        let valid_transports = ["stdio", "sse"];
        if !valid_transports.contains(&self.mcp.transport.as_str()) {
            anyhow::bail!(
                "mcp.transport must be one of: {} (got '{}').",
                valid_transports.join(", "),
                self.mcp.transport
            );
        }
        if self.mcp.transport == "sse" && self.mcp.port == 0 {
            anyhow::bail!("mcp.port must be > 0 when mcp.transport is 'sse'.");
        }
        // MCP and server ports must not collide
        if self.mcp.enabled && self.mcp.transport == "sse" && self.mcp.port == self.server.port {
            anyhow::bail!(
                "mcp.port ({}) must differ from server.port ({}) to avoid port collision.",
                self.mcp.port,
                self.server.port
            );
        }

        // --- Auth validation ---
        if self.auth.enabled
            && self.auth.api_key.is_none()
            && self.auth.jwt_secret.is_none()
            && self.auth.api_keys.is_empty()
        {
            anyhow::bail!(
                "auth.enabled is true but no authentication method is configured. \
                 Provide auth.api_key, auth.jwt_secret, or at least one [[auth.api_keys]] entry."
            );
        }
        let valid_auth_roles = ["admin", "writer", "reader", "viewer"];
        for (i, entry) in self.auth.api_keys.iter().enumerate() {
            if entry.name.is_empty() {
                anyhow::bail!("auth.api_keys[{}].name must not be empty.", i);
            }
            if entry.key.is_empty() {
                anyhow::bail!(
                    "auth.api_keys[{}].key must not be empty (name='{}').",
                    i,
                    entry.name
                );
            }
            if !valid_auth_roles.contains(&entry.role.as_str()) {
                anyhow::bail!(
                    "auth.api_keys[{}].role must be one of: {} (got '{}', name='{}').",
                    i,
                    valid_auth_roles.join(", "),
                    entry.role,
                    entry.name
                );
            }
        }

        // --- Telemetry validation ---
        if self.telemetry.sample_rate < 0.0 || self.telemetry.sample_rate > 1.0 {
            anyhow::bail!(
                "telemetry.sample_rate must be between 0.0 and 1.0 (got {}).",
                self.telemetry.sample_rate
            );
        }
        if self.telemetry.enabled && self.telemetry.otlp_endpoint.is_empty() {
            anyhow::bail!("telemetry.otlp_endpoint must not be empty when telemetry is enabled.");
        }
        if self.telemetry.enabled && self.telemetry.service_name.is_empty() {
            anyhow::bail!("telemetry.service_name must not be empty when telemetry is enabled.");
        }

        // --- Namespaces validation ---
        if self.namespaces.default_namespace.is_empty() {
            anyhow::bail!("namespaces.default_namespace must not be empty.");
        }

        // --- Instance validation ---
        let valid_roles = ["auto", "writer", "reader"];
        if !valid_roles.contains(&self.instance.role.as_str()) {
            anyhow::bail!(
                "instance.role must be one of: {} (got '{}').",
                valid_roles.join(", "),
                self.instance.role
            );
        }
        if self.instance.id_range_size == 0 {
            anyhow::bail!("instance.id_range_size must be > 0.");
        }
        // Shared mode requires either writer or reader role
        if self.storage.mode == "shared" && self.instance.role == "auto" {
            anyhow::bail!(
                "instance.role must be 'writer' or 'reader' when storage.mode is 'shared'. \
                 Set via ucotron.toml or UCOTRON_INSTANCE_ROLE env var."
            );
        }

        // --- Connectors validation ---
        if self.connectors.check_interval_secs == 0 {
            anyhow::bail!("connectors.check_interval_secs must be > 0.");
        }
        for (i, entry) in self.connectors.schedules.iter().enumerate() {
            if entry.connector_id.is_empty() {
                anyhow::bail!(
                    "connectors.schedules[{}].connector_id must not be empty.",
                    i
                );
            }
            if entry.timeout_secs == 0 {
                anyhow::bail!(
                    "connectors.schedules[{}].timeout_secs must be > 0 (connector_id='{}').",
                    i,
                    entry.connector_id
                );
            }
        }

        Ok(())
    }

    /// Generate an example configuration as a TOML string (plain, no comments).
    pub fn example_toml() -> String {
        let config = UcotronConfig::default();
        toml::to_string_pretty(&config)
            .unwrap_or_else(|_| "# Failed to generate example".to_string())
    }

    /// Generate a fully commented example configuration file.
    ///
    /// This is suitable for `ucotron_server --init-config` output.
    pub fn example_toml_commented() -> String {
        format!(
            r#"# =============================================================================
# Ucotron Configuration File
# =============================================================================
# This file configures the Ucotron cognitive memory server.
# All values shown below are defaults — uncomment and modify as needed.
#
# Environment variables override TOML values. Use the UCOTRON_ prefix:
#   UCOTRON_SERVER_PORT=9000 ucotron_server
#
# For full documentation, see: https://ucotron.com/docs/server/configuration

# -----------------------------------------------------------------------------
# [server] — HTTP server settings
# -----------------------------------------------------------------------------
[server]
# Bind address for the REST API.
host = "0.0.0.0"
# HTTP port for the REST API.
port = 8420
# Number of worker threads for request handling.
workers = 4
# Log level: trace, debug, info, warn, error
log_level = "info"
# Log format: "text" (human-readable) or "json" (structured with trace IDs)
log_format = "text"

# -----------------------------------------------------------------------------
# [storage] — Backend storage configuration
# -----------------------------------------------------------------------------
[storage]
# Storage mode:
#   "embedded" — All data stored locally in LMDB (default, single-instance)
#   "external" — Use external backends (Qdrant, FalkorDB, etc.)
#   "shared"   — Multiple server instances sharing the same storage directory
mode = "embedded"
# Shared data directory for multi-instance mode (required when mode = "shared").
# All instances must point to the same directory (e.g., NFS mount, shared volume).
# shared_data_dir = "/data/ucotron-shared"

# Vector backend configuration.
[storage.vector]
# Backend type: "helix" (embedded LMDB+HNSW), "qdrant" (external), "custom"
backend = "helix"
# Data directory for embedded backends (relative to working dir).
data_dir = "data"
# Maximum database size in bytes (10 GB default, for LMDB map_size).
max_db_size = {max_db_size}
# URL for external vector backend (required when backend != "helix").
# url = "http://localhost:6333"

# HNSW vector index parameters.
[storage.vector.hnsw]
# Number of bi-directional links per node during index construction.
# Higher = better recall, more memory, slower build.
ef_construction = 200
# Number of candidates evaluated during search.
# Higher = better recall, slower search.
ef_search = 200
# Enable HNSW index. When false, falls back to brute-force SIMD search.
enabled = true

# Graph backend configuration.
[storage.graph]
# Backend type: "helix" (embedded LMDB), "falkordb" (external), "custom"
backend = "helix"
# Data directory for embedded backends.
data_dir = "data"
# Maximum database size in bytes (10 GB default).
max_db_size = {max_db_size}
# Batch size for bulk operations (node/edge inserts).
batch_size = 10000
# URL for external graph backend (required when backend != "helix").
# url = "redis://localhost:6379"

# -----------------------------------------------------------------------------
# [models] — ML model configuration
# -----------------------------------------------------------------------------
[models]
# Sentence embedding model (ONNX format, 384-dim output).
embedding_model = "all-MiniLM-L6-v2"
# Named Entity Recognition model (GLiNER, ONNX format).
ner_model = "gliner-multi-v2.1"
# LLM model for relation extraction (GGUF quantized).
# Set to "none" or "" to use co-occurrence fallback (no LLM).
llm_model = "Qwen3-4B-GGUF"
# LLM backend: "candle" (Rust native) or "llama_cpp" (C++ bindings).
llm_backend = "candle"
# Directory containing model files (downloaded via scripts/download_models.sh).
models_dir = "models"
# Enable document OCR pipeline (PDF text extraction + Tesseract image OCR).
enable_ocr = true
# Language for Tesseract OCR (e.g., "eng", "spa", "deu", "eng+spa").
ocr_language = "eng"
# Path to the tesseract binary. Set to full path if not on PATH.
tesseract_path = "tesseract"

# -----------------------------------------------------------------------------
# [consolidation] — Background "dreaming" worker
# -----------------------------------------------------------------------------
[consolidation]
# Number of ingested messages between consolidation runs.
trigger_interval = 100
# Enable temporal memory decay for old, unaccessed nodes.
enable_decay = true
# Decay half-life in seconds (default: 30 days = 2592000).
decay_halflife_secs = 2592000

# -----------------------------------------------------------------------------
# [telemetry] — OpenTelemetry observability
# -----------------------------------------------------------------------------
[telemetry]
# Enable OTLP telemetry export (traces, metrics, logs).
# Requires a running OpenTelemetry collector (e.g., Jaeger, Grafana Tempo).
enabled = false
# OTLP gRPC collector endpoint.
otlp_endpoint = "http://localhost:4317"
# Service name reported in traces and metrics.
service_name = "ucotron"
# Trace sampling ratio: 0.0 (no traces) to 1.0 (all traces).
sample_rate = 1.0
# Export traces via OTLP.
export_traces = true
# Export metrics via OTLP.
export_metrics = true
# Export logs via OTLP (may be verbose, disabled by default).
export_logs = false

# -----------------------------------------------------------------------------
# [mcp] — Model Context Protocol server
# -----------------------------------------------------------------------------
[mcp]
# Enable MCP server for Claude Desktop, Cursor, etc.
enabled = true
# Transport mode: "stdio" (default, for CLI tools) or "sse" (HTTP streaming).
transport = "stdio"
# Port for SSE transport (only used when transport = "sse").
port = 8421

# -----------------------------------------------------------------------------
# [namespaces] — Multi-tenancy configuration
# -----------------------------------------------------------------------------
[namespaces]
# Default namespace when no X-Ucotron-Namespace header is provided.
default_namespace = "default"
# Restrict to specific namespaces (empty = allow any).
# allowed_namespaces = ["org1", "org2"]
# Maximum number of namespaces (0 = unlimited).
max_namespaces = 0

# -----------------------------------------------------------------------------
# [auth] — Authentication (optional)
# -----------------------------------------------------------------------------
[auth]
# Enable authentication. When false, all requests are accepted.
enabled = false
# API key for Bearer token auth. Set via UCOTRON_AUTH_API_KEY env var.
# api_key = "your-secret-api-key"
# JWT secret for token-based auth (future use).
# jwt_secret = "your-jwt-secret"
# JWT issuer (future use).
# jwt_issuer = "ucotron"

# -----------------------------------------------------------------------------
# [audit] — Immutable audit logging
# -----------------------------------------------------------------------------
[audit]
# Enable audit logging for all API operations.
enabled = true
# Retention period in seconds (0 = keep forever). Default: 90 days.
retention_secs = 7776000
# Maximum entries kept in memory. Oldest entries evicted when exceeded.
max_entries = 100000

# -----------------------------------------------------------------------------
# [instance] — Multi-instance configuration
# -----------------------------------------------------------------------------
[instance]
# Unique identifier for this server instance.
# Set to "auto" to generate from hostname + PID.
instance_id = "auto"
# Instance role:
#   "auto"   — Single-instance mode (default), acts as both reader and writer
#   "writer" — Can perform writes (ingestion, learn, update, delete)
#   "reader" — Read-only (search, augment, get operations only)
role = "auto"
# Starting node ID for this instance's ID allocation range.
# Each instance in a multi-instance deployment needs a non-overlapping range.
id_range_start = 1000000
# Size of this instance's node ID allocation range (default: 1 billion).
id_range_size = 1000000000
"#,
            max_db_size = 10u64 * 1024 * 1024 * 1024
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = UcotronConfig::default();
        assert_eq!(config.server.host, "0.0.0.0");
        assert_eq!(config.server.port, 8420);
        assert_eq!(config.server.workers, 4);
        assert_eq!(config.server.log_level, "info");
        assert_eq!(config.server.log_format, "text");
        assert_eq!(config.storage.mode, "embedded");
        assert_eq!(config.storage.vector.backend, "helix");
        assert_eq!(config.storage.graph.backend, "helix");
        assert_eq!(config.models.embedding_model, "all-MiniLM-L6-v2");
        assert_eq!(config.namespaces.default_namespace, "default");
        assert!(config.namespaces.allowed_namespaces.is_empty());
        assert!(!config.auth.enabled);
        assert!(config.auth.api_key.is_none());
    }

    #[test]
    fn test_parse_minimal_toml() {
        let toml = "";
        let config = UcotronConfig::parse_toml(toml).unwrap();
        assert_eq!(config.server.port, 8420);
        assert_eq!(config.namespaces.default_namespace, "default");
    }

    #[test]
    fn test_parse_custom_toml() {
        let toml = r#"
[server]
host = "127.0.0.1"
port = 9000
workers = 8

[storage]
mode = "embedded"

[storage.vector]
backend = "helix"
data_dir = "/tmp/ucotron"

[storage.graph]
backend = "helix"
batch_size = 5000

[namespaces]
default_namespace = "my-project"
allowed_namespaces = ["my-project", "staging"]

[auth]
enabled = true
api_key = "test-key-123"
"#;
        let config = UcotronConfig::parse_toml(toml).unwrap();
        assert_eq!(config.server.host, "127.0.0.1");
        assert_eq!(config.server.port, 9000);
        assert_eq!(config.server.workers, 8);
        assert_eq!(config.storage.vector.data_dir, "/tmp/ucotron");
        assert_eq!(config.storage.graph.batch_size, 5000);
        assert_eq!(config.namespaces.default_namespace, "my-project");
        assert_eq!(
            config.namespaces.allowed_namespaces,
            vec!["my-project", "staging"]
        );
        assert!(config.auth.enabled);
        assert_eq!(config.auth.api_key.as_deref(), Some("test-key-123"));
    }

    #[test]
    fn test_invalid_storage_mode() {
        let toml = r#"
[storage]
mode = "invalid"
"#;
        let result = UcotronConfig::parse_toml(toml);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("storage.mode"));
        assert!(err.contains("invalid"));
    }

    #[test]
    fn test_invalid_port() {
        let toml = r#"
[server]
port = 0
"#;
        let result = UcotronConfig::parse_toml(toml);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("server.port"));
    }

    #[test]
    fn test_invalid_log_level() {
        let toml = r#"
[server]
log_level = "verbose"
"#;
        let result = UcotronConfig::parse_toml(toml);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("server.log_level"));
        assert!(err.contains("verbose"));
    }

    #[test]
    fn test_invalid_log_format() {
        let toml = r#"
[server]
log_format = "xml"
"#;
        let result = UcotronConfig::parse_toml(toml);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("server.log_format"));
        assert!(err.contains("xml"));
    }

    #[test]
    fn test_log_format_json_valid() {
        let toml = r#"
[server]
log_format = "json"
"#;
        let config = UcotronConfig::parse_toml(toml).unwrap();
        assert_eq!(config.server.log_format, "json");
    }

    #[test]
    fn test_env_override_log_format() {
        std::env::set_var("UCOTRON_SERVER_LOG_FORMAT", "json");
        let mut config = UcotronConfig::default();
        config.apply_env_overrides();
        assert_eq!(config.server.log_format, "json");
        std::env::remove_var("UCOTRON_SERVER_LOG_FORMAT");
    }

    #[test]
    fn test_example_toml_generation() {
        let example = UcotronConfig::example_toml();
        assert!(example.contains("port"));
        assert!(example.contains("8420"));
        assert!(example.contains("helix"));
        // Verify it round-trips
        let _config = UcotronConfig::parse_toml(&example).unwrap();
    }

    #[test]
    fn test_example_toml_commented() {
        let commented = UcotronConfig::example_toml_commented();
        // Should contain section headers
        assert!(commented.contains("[server]"));
        assert!(commented.contains("[storage]"));
        assert!(commented.contains("[models]"));
        assert!(commented.contains("[consolidation]"));
        assert!(commented.contains("[mcp]"));
        assert!(commented.contains("[namespaces]"));
        assert!(commented.contains("[auth]"));
        // Should contain inline comments
        assert!(commented.contains("# Bind address"));
        assert!(commented.contains("UCOTRON_"));
    }

    #[test]
    fn test_serialization_roundtrip() {
        let config = UcotronConfig::default();
        let toml_str = toml::to_string_pretty(&config).unwrap();
        let parsed: UcotronConfig = toml::from_str(&toml_str).unwrap();
        assert_eq!(parsed.server.port, config.server.port);
        assert_eq!(parsed.storage.mode, config.storage.mode);
        assert_eq!(
            parsed.namespaces.default_namespace,
            config.namespaces.default_namespace
        );
    }

    #[test]
    fn test_env_override_server_port() {
        let mut config = UcotronConfig::default();
        // Simulate env override
        std::env::set_var("UCOTRON_SERVER_PORT", "9999");
        config.apply_env_overrides();
        assert_eq!(config.server.port, 9999);
        std::env::remove_var("UCOTRON_SERVER_PORT");
    }

    #[test]
    fn test_env_override_server_host() {
        let mut config = UcotronConfig::default();
        std::env::set_var("UCOTRON_SERVER_HOST", "127.0.0.1");
        config.apply_env_overrides();
        assert_eq!(config.server.host, "127.0.0.1");
        std::env::remove_var("UCOTRON_SERVER_HOST");
    }

    #[test]
    fn test_env_override_storage_mode() {
        let mut config = UcotronConfig::default();
        std::env::set_var("UCOTRON_STORAGE_MODE", "external");
        config.apply_env_overrides();
        assert_eq!(config.storage.mode, "external");
        std::env::remove_var("UCOTRON_STORAGE_MODE");
    }

    #[test]
    fn test_env_override_auth_api_key() {
        let mut config = UcotronConfig::default();
        std::env::set_var("UCOTRON_AUTH_API_KEY", "secret-from-env");
        config.apply_env_overrides();
        assert_eq!(config.auth.api_key.as_deref(), Some("secret-from-env"));
        std::env::remove_var("UCOTRON_AUTH_API_KEY");
    }

    #[test]
    fn test_env_override_models_dir() {
        let mut config = UcotronConfig::default();
        std::env::set_var("UCOTRON_MODELS_DIR", "/opt/models");
        config.apply_env_overrides();
        assert_eq!(config.models.models_dir, "/opt/models");
        std::env::remove_var("UCOTRON_MODELS_DIR");
    }

    #[test]
    fn test_env_override_invalid_port_ignored() {
        let mut config = UcotronConfig::default();
        std::env::set_var("UCOTRON_SERVER_PORT", "not-a-number");
        config.apply_env_overrides();
        // Should keep the default since parse fails
        assert_eq!(config.server.port, 8420);
        std::env::remove_var("UCOTRON_SERVER_PORT");
    }

    #[test]
    fn test_auth_enabled_without_credentials() {
        let toml = r#"
[auth]
enabled = true
"#;
        let result = UcotronConfig::parse_toml(toml);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("auth.enabled"));
        assert!(err.contains("api_key"));
    }

    #[test]
    fn test_auth_enabled_with_api_key() {
        let toml = r#"
[auth]
enabled = true
api_key = "my-secret"
"#;
        let config = UcotronConfig::parse_toml(toml).unwrap();
        assert!(config.auth.enabled);
        assert_eq!(config.auth.api_key.as_deref(), Some("my-secret"));
    }

    #[test]
    fn test_external_mode_requires_url() {
        let toml = r#"
[storage]
mode = "external"

[storage.vector]
backend = "qdrant"
"#;
        let result = UcotronConfig::parse_toml(toml);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("storage.vector.url"));
        assert!(err.contains("qdrant"));
    }

    #[test]
    fn test_external_mode_helix_no_url_needed() {
        // Helix in external mode doesn't need URL (it's embedded)
        let toml = r#"
[storage]
mode = "external"

[storage.vector]
backend = "helix"

[storage.graph]
backend = "helix"
"#;
        let config = UcotronConfig::parse_toml(toml).unwrap();
        assert_eq!(config.storage.mode, "external");
    }

    #[test]
    fn test_mcp_sse_port_collision() {
        let toml = r#"
[server]
port = 8420

[mcp]
enabled = true
transport = "sse"
port = 8420
"#;
        let result = UcotronConfig::parse_toml(toml);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("port collision"));
    }

    #[test]
    fn test_mcp_transport_validation() {
        let toml = r#"
[mcp]
transport = "grpc"
"#;
        let result = UcotronConfig::parse_toml(toml);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("mcp.transport"));
    }

    #[test]
    fn test_empty_namespace_rejected() {
        let toml = r#"
[namespaces]
default_namespace = ""
"#;
        let result = UcotronConfig::parse_toml(toml);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("namespaces.default_namespace"));
    }

    #[test]
    fn test_hnsw_zero_ef_construction() {
        let toml = r#"
[storage.vector.hnsw]
ef_construction = 0
"#;
        let result = UcotronConfig::parse_toml(toml);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("ef_construction"));
    }

    #[test]
    fn test_empty_embedding_model_rejected() {
        let toml = r#"
[models]
embedding_model = ""
"#;
        let result = UcotronConfig::parse_toml(toml);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("models.embedding_model"));
    }

    // --- Instance config tests ---

    #[test]
    fn test_instance_config_defaults() {
        let config = UcotronConfig::default();
        assert_eq!(config.instance.instance_id, "auto");
        assert_eq!(config.instance.role, "auto");
        assert_eq!(config.instance.id_range_start, 1_000_000);
        assert_eq!(config.instance.id_range_size, 1_000_000_000);
        assert!(config.instance.can_write());
        assert!(!config.instance.is_reader_only());
    }

    #[test]
    fn test_instance_config_writer_role() {
        let toml = r#"
[instance]
instance_id = "writer-1"
role = "writer"
id_range_start = 0
id_range_size = 500000000
"#;
        let config = UcotronConfig::parse_toml(toml).unwrap();
        assert_eq!(config.instance.instance_id, "writer-1");
        assert_eq!(config.instance.role, "writer");
        assert_eq!(config.instance.id_range_start, 0);
        assert_eq!(config.instance.id_range_size, 500_000_000);
        assert!(config.instance.can_write());
        assert!(!config.instance.is_reader_only());
    }

    #[test]
    fn test_instance_config_reader_role() {
        let toml = r#"
[instance]
instance_id = "reader-1"
role = "reader"
"#;
        let config = UcotronConfig::parse_toml(toml).unwrap();
        assert_eq!(config.instance.role, "reader");
        assert!(!config.instance.can_write());
        assert!(config.instance.is_reader_only());
    }

    #[test]
    fn test_instance_invalid_role() {
        let toml = r#"
[instance]
role = "master"
"#;
        let result = UcotronConfig::parse_toml(toml);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("instance.role"));
        assert!(err.contains("master"));
    }

    #[test]
    fn test_instance_zero_range_size_rejected() {
        let toml = r#"
[instance]
id_range_size = 0
"#;
        let result = UcotronConfig::parse_toml(toml);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("id_range_size"));
    }

    #[test]
    fn test_shared_mode_requires_explicit_role() {
        let toml = r#"
[storage]
mode = "shared"
shared_data_dir = "/data/shared"

[instance]
role = "auto"
"#;
        let result = UcotronConfig::parse_toml(toml);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("instance.role"));
        assert!(err.contains("shared"));
    }

    #[test]
    fn test_shared_mode_writer_role_ok() {
        let toml = r#"
[storage]
mode = "shared"
shared_data_dir = "/data/shared"

[instance]
instance_id = "w1"
role = "writer"
"#;
        let config = UcotronConfig::parse_toml(toml).unwrap();
        assert_eq!(config.storage.mode, "shared");
        assert_eq!(config.instance.role, "writer");
        assert_eq!(
            config.storage.shared_data_dir.as_deref(),
            Some("/data/shared")
        );
    }

    #[test]
    fn test_shared_mode_reader_role_ok() {
        let toml = r#"
[storage]
mode = "shared"
shared_data_dir = "/data/shared"

[instance]
instance_id = "r1"
role = "reader"
"#;
        let config = UcotronConfig::parse_toml(toml).unwrap();
        assert_eq!(config.storage.mode, "shared");
        assert_eq!(config.instance.role, "reader");
    }

    #[test]
    fn test_shared_mode_requires_shared_data_dir() {
        let toml = r#"
[storage]
mode = "shared"

[instance]
instance_id = "w1"
role = "writer"
"#;
        let result = UcotronConfig::parse_toml(toml);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("shared_data_dir"));
    }

    #[test]
    fn test_effective_data_dirs_embedded() {
        let config = UcotronConfig::default();
        assert_eq!(config.storage.effective_vector_data_dir(), "data");
        assert_eq!(config.storage.effective_graph_data_dir(), "data");
    }

    #[test]
    fn test_effective_data_dirs_shared() {
        let toml = r#"
[storage]
mode = "shared"
shared_data_dir = "/mnt/shared"

[instance]
role = "writer"
"#;
        let config = UcotronConfig::parse_toml(toml).unwrap();
        assert_eq!(config.storage.effective_vector_data_dir(), "/mnt/shared");
        assert_eq!(config.storage.effective_graph_data_dir(), "/mnt/shared");
    }

    #[test]
    fn test_instance_resolved_id_auto() {
        let config = UcotronConfig::default();
        let resolved = config.instance.resolved_instance_id();
        // Should contain hostname and PID
        assert!(resolved.contains('-'));
        assert!(!resolved.is_empty());
        assert_ne!(resolved, "auto");
    }

    #[test]
    fn test_instance_resolved_id_explicit() {
        let toml = r#"
[instance]
instance_id = "my-server-1"
"#;
        let config = UcotronConfig::parse_toml(toml).unwrap();
        assert_eq!(config.instance.resolved_instance_id(), "my-server-1");
    }

    #[test]
    fn test_env_override_instance_role() {
        let mut config = UcotronConfig::default();
        std::env::set_var("UCOTRON_INSTANCE_ROLE", "reader");
        config.apply_env_overrides();
        assert_eq!(config.instance.role, "reader");
        assert!(config.instance.is_reader_only());
        std::env::remove_var("UCOTRON_INSTANCE_ROLE");
    }

    #[test]
    fn test_env_override_instance_id() {
        let mut config = UcotronConfig::default();
        std::env::set_var("UCOTRON_INSTANCE_ID", "env-instance-42");
        config.apply_env_overrides();
        assert_eq!(config.instance.instance_id, "env-instance-42");
        assert_eq!(config.instance.resolved_instance_id(), "env-instance-42");
        std::env::remove_var("UCOTRON_INSTANCE_ID");
    }

    #[test]
    fn test_env_override_id_range() {
        let mut config = UcotronConfig::default();
        std::env::set_var("UCOTRON_INSTANCE_ID_RANGE_START", "2000000000");
        std::env::set_var("UCOTRON_INSTANCE_ID_RANGE_SIZE", "500000000");
        config.apply_env_overrides();
        assert_eq!(config.instance.id_range_start, 2_000_000_000);
        assert_eq!(config.instance.id_range_size, 500_000_000);
        std::env::remove_var("UCOTRON_INSTANCE_ID_RANGE_START");
        std::env::remove_var("UCOTRON_INSTANCE_ID_RANGE_SIZE");
    }

    #[test]
    fn test_example_toml_contains_instance_section() {
        let commented = UcotronConfig::example_toml_commented();
        assert!(commented.contains("[instance]"));
        assert!(commented.contains("instance_id"));
        assert!(commented.contains("role"));
        assert!(commented.contains("id_range_start"));
        assert!(commented.contains("id_range_size"));
    }

    #[test]
    fn test_full_config_all_sections() {
        let toml = r#"
[server]
host = "10.0.0.1"
port = 3000
workers = 16
log_level = "debug"

[storage]
mode = "embedded"

[storage.vector]
backend = "helix"
data_dir = "/data/vectors"

[storage.vector.hnsw]
ef_construction = 100
ef_search = 100
enabled = true

[storage.graph]
backend = "helix"
data_dir = "/data/graph"
batch_size = 20000

[models]
embedding_model = "custom-model"
ner_model = "custom-ner"
llm_model = "none"
llm_backend = "candle"
models_dir = "/opt/models"

[consolidation]
trigger_interval = 50
enable_decay = false
decay_halflife_secs = 86400

[mcp]
enabled = false
transport = "stdio"
port = 9000

[namespaces]
default_namespace = "prod"
allowed_namespaces = ["prod", "staging", "dev"]
max_namespaces = 10

[auth]
enabled = true
api_key = "super-secret-key"
"#;
        let config = UcotronConfig::parse_toml(toml).unwrap();
        assert_eq!(config.server.host, "10.0.0.1");
        assert_eq!(config.server.port, 3000);
        assert_eq!(config.server.workers, 16);
        assert_eq!(config.server.log_level, "debug");
        assert_eq!(config.storage.vector.data_dir, "/data/vectors");
        assert_eq!(config.storage.vector.hnsw.ef_construction, 100);
        assert_eq!(config.storage.graph.data_dir, "/data/graph");
        assert_eq!(config.storage.graph.batch_size, 20000);
        assert_eq!(config.models.llm_model, "none");
        assert_eq!(config.consolidation.trigger_interval, 50);
        assert!(!config.consolidation.enable_decay);
        assert!(!config.mcp.enabled);
        assert_eq!(config.namespaces.default_namespace, "prod");
        assert_eq!(config.namespaces.allowed_namespaces.len(), 3);
        assert_eq!(config.namespaces.max_namespaces, 10);
        assert!(config.auth.enabled);
    }

    // --- Telemetry config tests ---

    #[test]
    fn test_telemetry_config_defaults() {
        let config = UcotronConfig::default();
        assert!(!config.telemetry.enabled);
        assert_eq!(config.telemetry.otlp_endpoint, "http://localhost:4317");
        assert_eq!(config.telemetry.service_name, "ucotron");
        assert_eq!(config.telemetry.sample_rate, 1.0);
        assert!(config.telemetry.export_traces);
        assert!(config.telemetry.export_metrics);
        assert!(!config.telemetry.export_logs);
    }

    #[test]
    fn test_telemetry_config_parse_toml() {
        let toml = r#"
[telemetry]
enabled = true
otlp_endpoint = "http://otel-collector:4317"
service_name = "ucotron-prod"
sample_rate = 0.5
export_traces = true
export_metrics = false
export_logs = true
"#;
        let config = UcotronConfig::parse_toml(toml).unwrap();
        assert!(config.telemetry.enabled);
        assert_eq!(config.telemetry.otlp_endpoint, "http://otel-collector:4317");
        assert_eq!(config.telemetry.service_name, "ucotron-prod");
        assert_eq!(config.telemetry.sample_rate, 0.5);
        assert!(config.telemetry.export_traces);
        assert!(!config.telemetry.export_metrics);
        assert!(config.telemetry.export_logs);
    }

    #[test]
    fn test_telemetry_accessors() {
        let toml = r#"
[telemetry]
enabled = true
otlp_endpoint = "http://collector:4317"
service_name = "test-svc"
sample_rate = 0.75
"#;
        let config = UcotronConfig::parse_toml(toml).unwrap();
        assert!(config.telemetry_enabled());
        assert_eq!(config.telemetry_otlp_endpoint(), "http://collector:4317");
        assert_eq!(config.telemetry_service_name(), "test-svc");
        assert_eq!(config.telemetry_sample_rate(), 0.75);
    }

    #[test]
    fn test_telemetry_sample_rate_out_of_range() {
        let toml = r#"
[telemetry]
sample_rate = 1.5
"#;
        let result = UcotronConfig::parse_toml(toml);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("telemetry.sample_rate"));
        assert!(err.contains("1.5"));
    }

    #[test]
    fn test_telemetry_sample_rate_negative() {
        let toml = r#"
[telemetry]
sample_rate = -0.1
"#;
        let result = UcotronConfig::parse_toml(toml);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("telemetry.sample_rate"));
    }

    #[test]
    fn test_telemetry_enabled_empty_endpoint_rejected() {
        let toml = r#"
[telemetry]
enabled = true
otlp_endpoint = ""
"#;
        let result = UcotronConfig::parse_toml(toml);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("telemetry.otlp_endpoint"));
    }

    #[test]
    fn test_telemetry_enabled_empty_service_name_rejected() {
        let toml = r#"
[telemetry]
enabled = true
service_name = ""
"#;
        let result = UcotronConfig::parse_toml(toml);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("telemetry.service_name"));
    }

    #[test]
    fn test_env_override_telemetry_enabled() {
        let mut config = UcotronConfig::default();
        std::env::set_var("UCOTRON_TELEMETRY_ENABLED", "true");
        config.apply_env_overrides();
        assert!(config.telemetry.enabled);
        std::env::remove_var("UCOTRON_TELEMETRY_ENABLED");
    }

    #[test]
    fn test_env_override_telemetry_endpoint() {
        let mut config = UcotronConfig::default();
        std::env::set_var("UCOTRON_TELEMETRY_OTLP_ENDPOINT", "http://custom:4317");
        config.apply_env_overrides();
        assert_eq!(config.telemetry.otlp_endpoint, "http://custom:4317");
        std::env::remove_var("UCOTRON_TELEMETRY_OTLP_ENDPOINT");
    }

    #[test]
    fn test_env_override_telemetry_service_name() {
        let mut config = UcotronConfig::default();
        std::env::set_var("UCOTRON_TELEMETRY_SERVICE_NAME", "my-svc");
        config.apply_env_overrides();
        assert_eq!(config.telemetry.service_name, "my-svc");
        std::env::remove_var("UCOTRON_TELEMETRY_SERVICE_NAME");
    }

    #[test]
    fn test_env_override_telemetry_sample_rate() {
        let mut config = UcotronConfig::default();
        std::env::set_var("UCOTRON_TELEMETRY_SAMPLE_RATE", "0.25");
        config.apply_env_overrides();
        assert_eq!(config.telemetry.sample_rate, 0.25);
        std::env::remove_var("UCOTRON_TELEMETRY_SAMPLE_RATE");
    }

    #[test]
    fn test_example_toml_contains_telemetry_section() {
        let commented = UcotronConfig::example_toml_commented();
        assert!(commented.contains("[telemetry]"));
        assert!(commented.contains("otlp_endpoint"));
        assert!(commented.contains("service_name"));
        assert!(commented.contains("sample_rate"));
        assert!(commented.contains("export_traces"));
        assert!(commented.contains("export_metrics"));
        assert!(commented.contains("export_logs"));
    }

    // --- Mindset config tests ---

    #[test]
    fn test_mindset_config_defaults() {
        let config = UcotronConfig::default();
        assert!(config.mindset.enabled);
        assert!(!config.mindset.algorithmic_keywords.is_empty());
        assert!(!config.mindset.divergent_keywords.is_empty());
        assert!(!config.mindset.convergent_keywords.is_empty());
        assert!(config
            .mindset
            .algorithmic_keywords
            .contains(&"verify".to_string()));
        assert!(config
            .mindset
            .divergent_keywords
            .contains(&"explore".to_string()));
        assert!(config
            .mindset
            .convergent_keywords
            .contains(&"summarize".to_string()));
    }

    #[test]
    fn test_mindset_config_parse_toml() {
        let toml = r#"
[mindset]
enabled = false
algorithmic_keywords = ["audit", "fact-check"]
divergent_keywords = ["hypothesize"]
convergent_keywords = ["wrap up"]
"#;
        let config = UcotronConfig::parse_toml(toml).unwrap();
        assert!(!config.mindset.enabled);
        assert_eq!(
            config.mindset.algorithmic_keywords,
            vec!["audit", "fact-check"]
        );
        assert_eq!(config.mindset.divergent_keywords, vec!["hypothesize"]);
        assert_eq!(config.mindset.convergent_keywords, vec!["wrap up"]);
    }

    #[test]
    fn test_mindset_config_empty_uses_defaults() {
        let toml = "";
        let config = UcotronConfig::parse_toml(toml).unwrap();
        assert!(config.mindset.enabled);
        assert_eq!(config.mindset.algorithmic_keywords.len(), 6);
        assert_eq!(config.mindset.divergent_keywords.len(), 6);
        assert_eq!(config.mindset.convergent_keywords.len(), 6);
    }
}
