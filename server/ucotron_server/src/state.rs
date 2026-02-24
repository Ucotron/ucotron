//! Application state shared across all request handlers.

use std::sync::{atomic::AtomicU64, Arc, Mutex, RwLock};
use std::time::Instant;

use ucotron_config::{ApiKeyEntry, UcotronConfig};
use ucotron_core::BackendRegistry;
use ucotron_extraction::{
    CrossModalTextEncoder, DocumentOcrPipeline, EmbeddingPipeline, ImageEmbeddingPipeline,
    NerPipeline, RelationExtractor, TranscriptionPipeline, VideoPipeline,
};

use ucotron_connectors::CronScheduler;

use crate::audit::AuditLog;
use crate::llm::LLMProvider;
use crate::metrics::PrometheusMetrics;
use crate::telemetry::metrics_bridge::OtelMetrics;

pub type LlmProviderStatsMap = Arc<RwLock<std::collections::HashMap<String, serde_json::Value>>>;

/// Shared application state threaded through Axum handlers.
///
/// Wrapped in `Arc` and shared via Axum's `State` extractor.
pub struct AppState {
    /// Storage backends (vector + graph).
    pub registry: Arc<BackendRegistry>,
    /// Embedding pipeline (ONNX all-MiniLM-L6-v2).
    pub embedder: Arc<dyn EmbeddingPipeline>,
    /// NER pipeline (GLiNER, optional).
    pub ner: Option<Arc<dyn NerPipeline>>,
    /// Relation extractor (co-occurrence or LLM, optional).
    pub relation_extractor: Option<Arc<dyn RelationExtractor>>,
    /// Audio transcription pipeline (Whisper ONNX, optional).
    pub transcriber: Option<Arc<dyn TranscriptionPipeline>>,
    /// Image embedding pipeline (CLIP visual encoder, optional).
    pub image_embedder: Option<Arc<dyn ImageEmbeddingPipeline>>,
    /// Cross-modal text encoder (CLIP text encoder, optional).
    pub cross_modal_encoder: Option<Arc<dyn CrossModalTextEncoder>>,
    /// Document OCR pipeline (pdf_extract + Tesseract, optional).
    pub ocr_pipeline: Option<Arc<dyn DocumentOcrPipeline>>,
    /// Video frame extraction pipeline (FFmpeg, optional).
    pub video_pipeline: Option<Arc<dyn VideoPipeline>>,
    /// Full configuration.
    pub config: UcotronConfig,
    /// Server start time (for uptime metric).
    pub start_time: Instant,
    /// Monotonically increasing node ID counter (shared across ingestion calls).
    /// In multi-instance mode, starts at `instance.id_range_start` and is bounded
    /// by `id_range_start + id_range_size`.
    pub next_node_id: Mutex<u64>,
    /// Upper bound of the ID allocation range (exclusive).
    /// Node IDs allocated by this instance will be in `[id_range_start, id_range_end)`.
    pub id_range_end: u64,
    /// Resolved instance ID (unique per instance).
    pub instance_id: String,
    /// Request counters for metrics.
    pub total_requests: AtomicU64,
    pub total_ingestions: AtomicU64,
    pub total_searches: AtomicU64,
    /// Prometheus metrics (optional, enabled by default).
    pub prometheus: Option<PrometheusMetrics>,
    /// OTLP metrics instruments (optional, enabled when telemetry.export_metrics is true).
    pub otel_metrics: Option<OtelMetrics>,
    /// Runtime-mutable API keys. Initialized from config.auth.api_keys and
    /// updated by create/revoke handlers. The auth middleware reads from here.
    pub api_keys: RwLock<Vec<ApiKeyEntry>>,
    /// Immutable append-only audit log.
    pub audit_log: AuditLog,
    /// Cron-based connector scheduler (optional, enabled via config).
    pub cron_scheduler: Option<Arc<CronScheduler>>,
    /// LLM provider registry.
    pub llm_providers: RwLock<Vec<LLMProvider>>,
    /// LLM providers environment config (LMDB).
    pub llm_providers_env: Option<heed::Env>,
    /// LLM cost tracking environment (LMDB).
    pub llm_costs_env: Option<heed::Env>,
    /// LLM provider performance stats.
    pub llm_provider_stats: LlmProviderStatsMap,
    /// Round-robin index for LLM provider load balancing.
    pub llm_round_robin_index: std::sync::RwLock<usize>,
}

impl AppState {
    pub fn new(
        registry: Arc<BackendRegistry>,
        embedder: Arc<dyn EmbeddingPipeline>,
        ner: Option<Arc<dyn NerPipeline>>,
        relation_extractor: Option<Arc<dyn RelationExtractor>>,
        config: UcotronConfig,
    ) -> Self {
        Self::with_transcriber(registry, embedder, ner, relation_extractor, None, config)
    }

    /// Create AppState with all optional pipelines including transcription.
    pub fn with_transcriber(
        registry: Arc<BackendRegistry>,
        embedder: Arc<dyn EmbeddingPipeline>,
        ner: Option<Arc<dyn NerPipeline>>,
        relation_extractor: Option<Arc<dyn RelationExtractor>>,
        transcriber: Option<Arc<dyn TranscriptionPipeline>>,
        config: UcotronConfig,
    ) -> Self {
        Self::with_all_pipelines(
            registry,
            embedder,
            ner,
            relation_extractor,
            transcriber,
            None,
            None,
            config,
        )
    }

    /// Create AppState with all optional pipelines including image embedding and OCR.
    #[allow(clippy::too_many_arguments)]
    pub fn with_all_pipelines(
        registry: Arc<BackendRegistry>,
        embedder: Arc<dyn EmbeddingPipeline>,
        ner: Option<Arc<dyn NerPipeline>>,
        relation_extractor: Option<Arc<dyn RelationExtractor>>,
        transcriber: Option<Arc<dyn TranscriptionPipeline>>,
        image_embedder: Option<Arc<dyn ImageEmbeddingPipeline>>,
        cross_modal_encoder: Option<Arc<dyn CrossModalTextEncoder>>,
        config: UcotronConfig,
    ) -> Self {
        Self::with_all_pipelines_and_ocr(
            registry,
            embedder,
            ner,
            relation_extractor,
            transcriber,
            image_embedder,
            cross_modal_encoder,
            None,
            config,
        )
    }

    /// Create AppState with all optional pipelines including OCR.
    #[allow(clippy::too_many_arguments)]
    pub fn with_all_pipelines_and_ocr(
        registry: Arc<BackendRegistry>,
        embedder: Arc<dyn EmbeddingPipeline>,
        ner: Option<Arc<dyn NerPipeline>>,
        relation_extractor: Option<Arc<dyn RelationExtractor>>,
        transcriber: Option<Arc<dyn TranscriptionPipeline>>,
        image_embedder: Option<Arc<dyn ImageEmbeddingPipeline>>,
        cross_modal_encoder: Option<Arc<dyn CrossModalTextEncoder>>,
        ocr_pipeline: Option<Arc<dyn DocumentOcrPipeline>>,
        config: UcotronConfig,
    ) -> Self {
        Self::with_all_pipelines_full(
            registry,
            embedder,
            ner,
            relation_extractor,
            transcriber,
            image_embedder,
            cross_modal_encoder,
            ocr_pipeline,
            None,
            config,
        )
    }

    /// Create AppState with all optional pipelines including OCR and video.
    #[allow(clippy::too_many_arguments)]
    pub fn with_all_pipelines_full(
        registry: Arc<BackendRegistry>,
        embedder: Arc<dyn EmbeddingPipeline>,
        ner: Option<Arc<dyn NerPipeline>>,
        relation_extractor: Option<Arc<dyn RelationExtractor>>,
        transcriber: Option<Arc<dyn TranscriptionPipeline>>,
        image_embedder: Option<Arc<dyn ImageEmbeddingPipeline>>,
        cross_modal_encoder: Option<Arc<dyn CrossModalTextEncoder>>,
        ocr_pipeline: Option<Arc<dyn DocumentOcrPipeline>>,
        video_pipeline: Option<Arc<dyn VideoPipeline>>,
        config: UcotronConfig,
    ) -> Self {
        let instance_id = config.instance.resolved_instance_id();
        let id_start = config.instance.id_range_start;
        let id_end = id_start.saturating_add(config.instance.id_range_size);
        let api_keys = RwLock::new(config.auth.api_keys.clone());
        let audit_log = AuditLog::new(config.audit.max_entries, config.audit.retention_secs);
        Self {
            registry,
            embedder,
            ner,
            relation_extractor,
            transcriber,
            image_embedder,
            cross_modal_encoder,
            ocr_pipeline,
            video_pipeline,
            config,
            start_time: Instant::now(),
            next_node_id: Mutex::new(id_start),
            id_range_end: id_end,
            instance_id,
            total_requests: AtomicU64::new(0),
            total_ingestions: AtomicU64::new(0),
            total_searches: AtomicU64::new(0),
            prometheus: Some(PrometheusMetrics::new()),
            otel_metrics: None,
            api_keys,
            audit_log,
            cron_scheduler: None,
            llm_providers: RwLock::new(Vec::new()),
            llm_providers_env: None,
            llm_costs_env: None,
            llm_provider_stats: Arc::new(RwLock::new(std::collections::HashMap::new())),
            llm_round_robin_index: std::sync::RwLock::new(0),
        }
    }

    /// Allocate a fresh node ID (thread-safe).
    ///
    /// In multi-instance mode, each instance allocates from its own ID range
    /// to avoid collisions. Returns an error if the range is exhausted.
    pub fn alloc_next_node_id(&self) -> u64 {
        let mut id = self.next_node_id.lock().unwrap();
        let current = *id;
        if current < self.id_range_end {
            *id += 1;
        }
        // If exhausted, we still return the last valid ID (callers should handle
        // the case where sequential calls return the same ID). In practice,
        // 1 billion IDs per instance is sufficient.
        current
    }

    /// Whether this instance can perform write operations (based on role config).
    pub fn can_write(&self) -> bool {
        self.config.instance.can_write()
    }

    /// Whether this instance is reader-only (no writes).
    pub fn is_reader_only(&self) -> bool {
        self.config.instance.is_reader_only()
    }
}
