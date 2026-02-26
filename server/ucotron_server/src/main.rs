//! # Ucotron Server
//!
//! The Ucotron cognitive trust server ("El Hipocampo").
//!
//! Provides:
//! - REST API (Axum) for memory CRUD, search, augment, and learn operations
//! - Multi-tenancy via `X-Ucotron-Namespace` header
//! - Middleware: logging, request timing, CORS
//! - Background consolidation worker ("dreaming")
//!
//! # Configuration
//!
//! Set `UCOTRON_CONFIG` env var to a TOML config file path, or use defaults.
//! The server binds to the configured `host:port` (default `0.0.0.0:8420`).
//!
//! # CLI Usage
//!
//! ```bash
//! # Start server with default config
//! ucotron_server
//!
//! # Start server with custom config file
//! ucotron_server --config ucotron.toml
//!
//! # Generate example config file with inline documentation
//! ucotron_server --init-config
//!
//! # Override specific settings via env vars
//! UCOTRON_SERVER_PORT=9000 ucotron_server
//! ```

use axum::extract::{DefaultBodyLimit, State};
use axum::http::Request;
use axum::middleware::{self, Next};
use axum::response::Response;
use axum::routing::{delete, get, post, put};
use axum::Router;
use clap::{Parser, Subcommand};
use std::sync::atomic::Ordering;
use std::sync::Arc;
use tower_http::cors::{Any, CorsLayer};
use tower_http::trace::TraceLayer;

use utoipa::OpenApi;
use utoipa_swagger_ui::SwaggerUi;

use ucotron_config::UcotronConfig;
use ucotron_server::audit;
use ucotron_server::auth;
use ucotron_server::handlers;
use ucotron_server::metrics;
use ucotron_server::openapi::ApiDoc;
use ucotron_server::state::AppState;
use ucotron_server::telemetry;
use ucotron_server::writer_lock::WriterLock;

/// Ucotron cognitive trust server.
#[derive(Parser, Debug)]
#[command(name = "ucotron_server")]
#[command(about = "Ucotron cognitive trust server — REST API + MCP for LLM memory management")]
#[command(version)]
struct Cli {
    /// Path to ucotron.toml config file.
    /// Can also be set via UCOTRON_CONFIG env var.
    #[arg(short, long, env = "UCOTRON_CONFIG", global = true)]
    config: Option<String>,

    /// Generate an example ucotron.toml config file with documentation and exit.
    #[arg(long)]
    init_config: bool,

    /// Subcommand to run.
    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Import memories from a third-party memory system.
    Migrate {
        /// Source system to migrate from.
        #[arg(long, value_parser = ["mem0", "zep"])]
        from: String,

        /// Path to a JSON file containing the exported memories.
        /// For Mem0: the output of `GET /v1/memories/` or a file export.
        #[arg(long)]
        file: String,

        /// Target namespace for imported memories (default: "{source}_import").
        #[arg(long)]
        namespace: Option<String>,

        /// Whether to link memories from the same user with edges.
        #[arg(long, default_value = "true")]
        link_same_user: bool,

        /// Whether to link memories from the same agent with edges.
        #[arg(long, default_value = "false")]
        link_same_agent: bool,
    },
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    // Handle --init-config: print example config and exit.
    if cli.init_config {
        print!("{}", UcotronConfig::example_toml_commented());
        return Ok(());
    }

    // Handle subcommands.
    if let Some(command) = &cli.command {
        return handle_command(command, &cli).await;
    }

    // Load configuration from file or defaults, then apply env var overrides.
    // Config must be loaded before telemetry so we can read OTLP settings.
    let config = if let Some(path) = &cli.config {
        UcotronConfig::from_file(path)?
    } else {
        let mut cfg = UcotronConfig::default();
        cfg.apply_env_overrides();
        cfg.validate()?;
        cfg
    };

    // Initialize telemetry (tracing + optional OTLP export for traces and metrics).
    // The guard must live for the entire server lifetime to flush spans/metrics on shutdown.
    let _telemetry_guard = telemetry::init_telemetry(telemetry::TelemetryInit {
        enabled: config.telemetry_enabled(),
        otlp_endpoint: config.telemetry_otlp_endpoint(),
        service_name: config.telemetry_service_name(),
        sample_rate: config.telemetry_sample_rate(),
        log_level: config.server.log_level.clone(),
        export_traces: config.telemetry.export_traces,
        export_metrics: config.telemetry.export_metrics,
        log_format: config.server.log_format.clone(),
    })?;

    tracing::info!(
        "Ucotron Server starting on {}:{}",
        config.server.host,
        config.server.port
    );
    tracing::info!(
        "Storage mode: {}, vector backend: {}, graph backend: {}",
        config.storage.mode,
        config.storage.vector.backend,
        config.storage.graph.backend
    );
    tracing::info!(
        "Instance: id={}, role={}, id_range=[{}..{})",
        config.instance.resolved_instance_id(),
        config.instance.role,
        config.instance.id_range_start,
        config
            .instance
            .id_range_start
            .saturating_add(config.instance.id_range_size),
    );

    // In shared mode, override data dirs to the shared path and acquire writer lock.
    let _writer_lock: Option<WriterLock>;
    let mut vector_config = config.storage.vector.clone();
    let mut graph_config = config.storage.graph.clone();

    if config.storage.mode == "shared" {
        let eff_vec_dir = config.storage.effective_vector_data_dir().to_string();
        let eff_graph_dir = config.storage.effective_graph_data_dir().to_string();
        tracing::info!(
            "Shared mode: vector_dir={}, graph_dir={}",
            eff_vec_dir,
            eff_graph_dir,
        );
        vector_config.data_dir = eff_vec_dir;
        graph_config.data_dir = eff_graph_dir;

        // Acquire writer lock if this instance has writer role.
        if config.instance.can_write() {
            let shared_dir = config
                .storage
                .shared_data_dir
                .as_deref()
                .unwrap_or(&config.storage.vector.data_dir);
            let instance_id = config.instance.resolved_instance_id();
            _writer_lock = Some(WriterLock::acquire(shared_dir, &instance_id)?);
        } else {
            _writer_lock = None;
        }
    } else {
        _writer_lock = None;
    }

    // Initialize storage backends.
    // If CLIP models are present, also create the visual vector backend (512-dim).
    let clip_model_dir = format!("{}/{}", config.models.models_dir, config.models.clip_model);
    let clip_available =
        std::path::Path::new(&format!("{}/visual_model.onnx", clip_model_dir)).exists();

    let registry = if clip_available {
        match ucotron_helix::create_helix_backends_with_visual(&vector_config, &graph_config) {
            Ok((vector_backend, graph_backend, visual_backend)) => {
                tracing::info!("Visual vector backend initialized for CLIP search");
                Arc::new(ucotron_core::BackendRegistry::with_visual(
                    vector_backend,
                    graph_backend,
                    visual_backend,
                ))
            }
            Err(e) => {
                tracing::warn!(
                    "Failed to create visual backend: {}. Falling back to text-only.",
                    e
                );
                let (vector_backend, graph_backend) =
                    ucotron_helix::create_helix_backends(&vector_config, &graph_config)?;
                Arc::new(ucotron_core::BackendRegistry::new(
                    vector_backend,
                    graph_backend,
                ))
            }
        }
    } else {
        let (vector_backend, graph_backend) =
            ucotron_helix::create_helix_backends(&vector_config, &graph_config)?;
        Arc::new(ucotron_core::BackendRegistry::new(
            vector_backend,
            graph_backend,
        ))
    };

    // Initialize embedding pipeline.
    // Try to load ONNX model; fall back to a stub if model files are not present.
    let embedder: Arc<dyn ucotron_extraction::EmbeddingPipeline> = match try_init_embedder(&config)
    {
        Ok(e) => e,
        Err(e) => {
            tracing::warn!(
                "Failed to load embedding model: {}. Using stub embedder.",
                e
            );
            Arc::new(StubEmbedder)
        }
    };

    // Initialize CLIP image/text pipelines (optional).
    let (image_embedder, cross_modal_encoder) = try_init_clip(&config);

    // Initialize Whisper transcription pipeline (optional).
    let transcriber = try_init_whisper(&config);

    // Initialize FFmpeg video pipeline (optional — requires ffmpeg libs).
    let video_pipeline = try_init_video();

    // Build application state with all optional pipelines.
    let mut app_state = AppState::with_all_pipelines_full(
        registry,
        embedder,
        None, // NER pipeline loaded separately if model present
        None, // Relation extractor loaded separately if model present
        transcriber,
        image_embedder,
        cross_modal_encoder,
        None, // OCR pipeline loaded separately if model present
        video_pipeline,
        config.clone(),
    );
    // Attach OTLP metrics instruments if metrics export is enabled.
    app_state.otel_metrics = _telemetry_guard.otel_metrics();

    let state = Arc::new(app_state);

    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods(Any)
        .allow_headers(Any);

    // Build the Axum router.
    let app = Router::new()
        // Health & Metrics
        .route("/api/v1/health", get(handlers::health_handler))
        .route("/api/v1/metrics", get(handlers::metrics_handler))
        // Memories CRUD
        .route("/api/v1/memories", post(handlers::create_memory_handler))
        .route("/api/v1/memories", get(handlers::list_memories_handler))
        .route("/api/v1/memories/{id}", get(handlers::get_memory_handler))
        .route(
            "/api/v1/memories/{id}",
            put(handlers::update_memory_handler),
        )
        .route(
            "/api/v1/memories/{id}",
            delete(handlers::delete_memory_handler),
        )
        // Multimodal text ingestion
        .route(
            "/api/v1/memories/text",
            post(handlers::create_text_memory_handler),
        )
        // Multimodal audio ingestion
        .route(
            "/api/v1/memories/audio",
            post(handlers::create_audio_memory_handler),
        )
        // Multimodal image ingestion
        .route(
            "/api/v1/memories/image",
            post(handlers::create_image_memory_handler),
        )
        // Multimodal video ingestion
        .route(
            "/api/v1/memories/video",
            post(handlers::create_video_memory_handler),
        )
        // Search
        .route("/api/v1/memories/search", post(handlers::search_handler))
        // Entities
        .route("/api/v1/entities", get(handlers::list_entities_handler))
        .route("/api/v1/entities/{id}", get(handlers::get_entity_handler))
        // Graph visualization
        .route("/api/v1/graph", get(handlers::graph_handler))
        // Augment & Learn
        .route("/api/v1/augment", post(handlers::augment_handler))
        .route("/api/v1/learn", post(handlers::learn_handler))
        // Export & Import
        .route("/api/v1/export", get(handlers::export_handler))
        .route("/api/v1/import", post(handlers::import_handler))
        .route("/api/v1/import/mem0", post(handlers::mem0_import_handler))
        .route("/api/v1/import/zep", post(handlers::zep_import_handler))
        // Audio transcription
        .route("/api/v1/transcribe", post(handlers::transcribe_handler))
        // Image embedding (CLIP)
        .route("/api/v1/images", post(handlers::index_image_handler))
        .route(
            "/api/v1/images/search",
            post(handlers::image_search_handler),
        )
        // Multimodal cross-modal search
        .route(
            "/api/v1/search/multimodal",
            post(handlers::multimodal_search_handler),
        )
        // Media file serving (image, audio, video)
        .route("/api/v1/media/{id}", get(handlers::get_media_handler))
        // Video segment navigation
        .route(
            "/api/v1/videos/{parent_id}/segments",
            get(handlers::get_video_segments_handler),
        )
        // Document OCR (pdf_extract + Tesseract)
        .route("/api/v1/ocr", post(handlers::ocr_handler))
        // Prometheus metrics (text exposition format)
        .route("/metrics", get(metrics::prometheus_metrics_handler))
        // Admin: namespace management, config, and system info
        .route(
            "/api/v1/admin/namespaces",
            get(handlers::list_namespaces_handler),
        )
        .route(
            "/api/v1/admin/namespaces",
            post(handlers::create_namespace_handler),
        )
        .route(
            "/api/v1/admin/namespaces/{name}",
            get(handlers::get_namespace_handler),
        )
        .route(
            "/api/v1/admin/namespaces/{name}",
            delete(handlers::delete_namespace_handler),
        )
        .route("/api/v1/admin/config", get(handlers::admin_config_handler))
        .route("/api/v1/admin/system", get(handlers::admin_system_handler))
        // Auth / RBAC
        .route("/api/v1/auth/whoami", get(handlers::whoami_handler))
        .route("/api/v1/auth/keys", get(handlers::list_api_keys_handler))
        .route("/api/v1/auth/keys", post(handlers::create_api_key_handler))
        .route(
            "/api/v1/auth/keys/{name}",
            delete(handlers::revoke_api_key_handler),
        )
        // Audit
        .route("/api/v1/audit", get(handlers::audit_query_handler))
        .route("/api/v1/audit/export", get(handlers::audit_export_handler))
        // Fine-tuning
        .route(
            "/api/v1/finetune/generate-dataset",
            post(handlers::generate_dataset_handler),
        )
        // Agents
        .route("/api/v1/agents", post(handlers::create_agent_handler))
        .route("/api/v1/agents", get(handlers::list_agents_handler))
        .route("/api/v1/agents/{id}", get(handlers::get_agent_handler))
        .route(
            "/api/v1/agents/{id}",
            delete(handlers::delete_agent_handler),
        )
        .route(
            "/api/v1/agents/{id}/clone",
            post(handlers::clone_agent_handler),
        )
        .route(
            "/api/v1/agents/{id}/merge",
            post(handlers::merge_agent_handler),
        )
        .route(
            "/api/v1/agents/{id}/share",
            post(handlers::create_share_handler).get(handlers::list_shares_handler),
        )
        .route(
            "/api/v1/agents/{id}/share/{target}",
            delete(handlers::delete_share_handler),
        )
        // Conversations
        .route(
            "/api/v1/conversations",
            get(handlers::list_conversations_handler),
        )
        .route(
            "/api/v1/conversations/{id}/messages",
            get(handlers::get_conversation_messages_handler),
        )
        // Connector sync
        .route(
            "/api/v1/connectors/{id}/sync",
            post(handlers::trigger_connector_sync_handler),
        )
        .route(
            "/api/v1/connectors/schedules",
            get(handlers::list_connector_schedules_handler),
        )
        .route(
            "/api/v1/connectors/{id}/history",
            get(handlers::connector_sync_history_handler),
        )
        // MCP Streamable HTTP (SSE) transport — all methods on /mcp
        .nest_service("/mcp", {
            use rmcp::transport::streamable_http_server::{
                session::local::LocalSessionManager,
                tower::{StreamableHttpServerConfig, StreamableHttpService},
            };
            use ucotron_server::mcp::UcotronMcpServer;
            let mcp_state = state.clone();
            let session_mgr = Arc::new(LocalSessionManager::default());
            let mcp_config = StreamableHttpServerConfig::default();
            StreamableHttpService::new(
                move || Ok(UcotronMcpServer::new(mcp_state.clone())),
                session_mgr,
                mcp_config,
            )
        })
        // Swagger UI for interactive API exploration
        // SwaggerUi serves the OpenAPI JSON at the URL passed to .url()
        .merge(SwaggerUi::new("/swagger-ui").url("/api/v1/openapi.json", ApiDoc::openapi()))
        // Middleware (order matters: first added = outermost)
        .layer(middleware::from_fn_with_state(
            state.clone(),
            metrics::prometheus_middleware,
        ))
        .layer(middleware::from_fn_with_state(
            state.clone(),
            audit::audit_middleware,
        ))
        .layer(middleware::from_fn_with_state(
            state.clone(),
            request_counter_middleware,
        ))
        .layer(middleware::from_fn_with_state(
            state.clone(),
            auth::auth_middleware,
        ))
        .layer(TraceLayer::new_for_http())
        .layer(telemetry::http_layer::OtelHttpLayer)
        .layer(DefaultBodyLimit::max(50 * 1024 * 1024)) // 50 MB max request body (for file uploads)
        .layer(cors)
        .with_state(state);

    // Bind and serve.
    let addr = format!("{}:{}", config.server.host, config.server.port);
    let listener = tokio::net::TcpListener::bind(&addr).await?;
    tracing::info!("Listening on {}", addr);

    axum::serve(listener, app).await?;

    Ok(())
}

/// Middleware that increments the global request counter.
async fn request_counter_middleware(
    State(state): State<Arc<AppState>>,
    request: Request<axum::body::Body>,
    next: Next,
) -> Response {
    state.total_requests.fetch_add(1, Ordering::Relaxed);
    next.run(request).await
}

/// Handle CLI subcommands (migrate, etc.).
async fn handle_command(command: &Commands, cli: &Cli) -> anyhow::Result<()> {
    match command {
        Commands::Migrate {
            from,
            file,
            namespace,
            link_same_user,
            link_same_agent,
        } => {
            handle_migrate(
                from,
                file,
                namespace.as_deref(),
                *link_same_user,
                *link_same_agent,
                cli,
            )
            .await
        }
    }
}

/// Handle the `migrate` subcommand: import memories from a third-party system.
async fn handle_migrate(
    from: &str,
    file_path: &str,
    namespace: Option<&str>,
    link_same_user: bool,
    link_same_agent: bool,
    cli: &Cli,
) -> anyhow::Result<()> {
    // Initialize tracing.
    tracing_subscriber::fmt::init();

    // Load config.
    let config = if let Some(path) = &cli.config {
        UcotronConfig::from_file(path)?
    } else {
        let mut cfg = UcotronConfig::default();
        cfg.apply_env_overrides();
        cfg.validate()?;
        cfg
    };

    // Read the input file.
    let json_data = std::fs::read_to_string(file_path)
        .map_err(|e| anyhow::anyhow!("Failed to read file '{}': {}", file_path, e))?;

    match from {
        "mem0" => {
            let default_ns = "mem0_import".to_string();
            let target_ns = namespace.unwrap_or(&default_ns);

            // Parse Mem0 data.
            let memories = ucotron_core::mem0_adapter::parse_mem0_json(&json_data)?;
            println!(
                "Parsed {} Mem0 memories from '{}'",
                memories.len(),
                file_path
            );

            if memories.is_empty() {
                println!("No memories to import.");
                return Ok(());
            }

            // Convert to Ucotron format.
            let options = ucotron_core::mem0_adapter::Mem0ImportOptions {
                namespace: target_ns.to_string(),
                link_same_user,
                link_same_agent,
            };
            let parse_result = ucotron_core::mem0_adapter::mem0_to_ucotron(&memories, &options);
            println!(
                "Converted: {} nodes, {} edges inferred",
                parse_result.memories_parsed, parse_result.edges_inferred
            );

            // Initialize storage backends.
            let (vector_backend, graph_backend) = ucotron_helix::create_helix_backends(
                &config.storage.vector,
                &config.storage.graph,
            )?;
            let registry = ucotron_core::BackendRegistry::new(vector_backend, graph_backend);

            // Import using the standard pipeline.
            // Find the next available node ID by scanning existing nodes.
            let existing_nodes = registry.graph().get_all_nodes().unwrap_or_default();
            let next_id = existing_nodes.iter().map(|n| n.id).max().unwrap_or(0) + 1;

            let import_result = ucotron_core::jsonld_export::import_graph(
                &parse_result.export,
                next_id,
                target_ns,
            )?;

            // Insert into backends.
            if !import_result.nodes.is_empty() {
                registry.graph().upsert_nodes(&import_result.nodes)?;
                let embeddings: Vec<(u64, Vec<f32>)> = import_result
                    .nodes
                    .iter()
                    .map(|n| (n.id, n.embedding.clone()))
                    .collect();
                registry.vector().upsert_embeddings(&embeddings)?;
            }
            if !import_result.edges.is_empty() {
                registry.graph().upsert_edges(&import_result.edges)?;
            }

            println!("Migration complete:");
            println!("  Nodes imported: {}", import_result.nodes_imported);
            println!("  Edges imported: {}", import_result.edges_imported);
            println!("  Target namespace: {}", target_ns);
        }
        "zep" => {
            let default_ns = "zep_import".to_string();
            let target_ns = namespace.unwrap_or(&default_ns);

            // Parse Zep/Graphiti data.
            let zep_data = ucotron_core::zep_adapter::parse_zep_json(&json_data)?;

            // Count items for user feedback.
            let entity_count = zep_data.entities.as_ref().map_or(0, |e| e.len());
            let episode_count = zep_data.episodes.as_ref().map_or(0, |e| e.len());
            let edge_count = zep_data.edges.as_ref().map_or(0, |e| e.len());
            let session_count = zep_data.sessions.as_ref().map_or(0, |s| s.len());
            let fact_count = zep_data.facts.as_ref().map_or(0, |f| f.len());

            println!(
                "Parsed from '{}': {} entities, {} episodes, {} edges, {} sessions, {} facts",
                file_path, entity_count, episode_count, edge_count, session_count, fact_count
            );

            if entity_count + episode_count + session_count + fact_count == 0 {
                println!("No data to import.");
                return Ok(());
            }

            // Convert to Ucotron format.
            let options = ucotron_core::zep_adapter::ZepImportOptions {
                namespace: target_ns.to_string(),
                link_same_user,
                link_same_group: false,
                preserve_expired: true,
            };
            let parse_result = ucotron_core::zep_adapter::zep_to_ucotron(&zep_data, &options);
            println!(
                "Converted: {} nodes, {} edges",
                parse_result.memories_parsed, parse_result.edges_inferred
            );

            // Initialize storage backends.
            let (vector_backend, graph_backend) = ucotron_helix::create_helix_backends(
                &config.storage.vector,
                &config.storage.graph,
            )?;
            let registry = ucotron_core::BackendRegistry::new(vector_backend, graph_backend);

            // Import using the standard pipeline.
            let existing_nodes = registry.graph().get_all_nodes().unwrap_or_default();
            let next_id = existing_nodes.iter().map(|n| n.id).max().unwrap_or(0) + 1;

            let import_result = ucotron_core::jsonld_export::import_graph(
                &parse_result.export,
                next_id,
                target_ns,
            )?;

            // Insert into backends.
            if !import_result.nodes.is_empty() {
                registry.graph().upsert_nodes(&import_result.nodes)?;
                let embeddings: Vec<(u64, Vec<f32>)> = import_result
                    .nodes
                    .iter()
                    .map(|n| (n.id, n.embedding.clone()))
                    .collect();
                registry.vector().upsert_embeddings(&embeddings)?;
            }
            if !import_result.edges.is_empty() {
                registry.graph().upsert_edges(&import_result.edges)?;
            }

            println!("Migration complete:");
            println!("  Nodes imported: {}", import_result.nodes_imported);
            println!("  Edges imported: {}", import_result.edges_imported);
            println!("  Target namespace: {}", target_ns);
        }
        other => {
            anyhow::bail!("Unknown source system: '{}'. Supported: mem0, zep", other);
        }
    }

    Ok(())
}

/// Try to initialize the real ONNX embedding pipeline.
fn try_init_embedder(
    config: &UcotronConfig,
) -> anyhow::Result<Arc<dyn ucotron_extraction::EmbeddingPipeline>> {
    use ucotron_extraction::embeddings::OnnxEmbeddingPipeline;

    let models_dir = &config.models.models_dir;
    let model_name = &config.models.embedding_model;
    let model_dir = format!("{}/{}", models_dir, model_name);

    let model_path = format!("{}/onnx/model.onnx", model_dir);
    let tokenizer_path = format!("{}/tokenizer.json", model_dir);

    let pipeline = OnnxEmbeddingPipeline::new(&model_path, &tokenizer_path, 4)?;
    Ok(Arc::new(pipeline))
}

/// Try to initialize Whisper transcription pipeline for audio support.
/// Returns None if Whisper model files are not present.
fn try_init_whisper(
    config: &UcotronConfig,
) -> Option<Arc<dyn ucotron_extraction::TranscriptionPipeline>> {
    use ucotron_extraction::audio::{WhisperConfig, WhisperOnnxPipeline};

    let models_dir = &config.models.models_dir;
    let whisper_dir = format!("{}/whisper-tiny", models_dir);

    let encoder_path = format!("{}/encoder.onnx", whisper_dir);
    let decoder_path = format!("{}/decoder.onnx", whisper_dir);
    let tokens_path = format!("{}/tokens.txt", whisper_dir);

    match WhisperOnnxPipeline::new(
        &encoder_path,
        &decoder_path,
        &tokens_path,
        WhisperConfig::default(),
    ) {
        Ok(pipeline) => {
            tracing::info!("Whisper transcription pipeline loaded from {}", whisper_dir);
            Some(Arc::new(pipeline))
        }
        Err(e) => {
            tracing::info!("Whisper transcription pipeline not available: {}", e);
            None
        }
    }
}

/// Try to initialize CLIP image and text pipelines for multimodal support.
/// Returns (None, None) if CLIP model files are not present.
#[allow(clippy::type_complexity)]
fn try_init_clip(
    config: &UcotronConfig,
) -> (
    Option<Arc<dyn ucotron_extraction::ImageEmbeddingPipeline>>,
    Option<Arc<dyn ucotron_extraction::CrossModalTextEncoder>>,
) {
    use ucotron_extraction::image::{ClipConfig, ClipImagePipeline, ClipTextPipeline};

    let models_dir = &config.models.models_dir;
    let clip_model = &config.models.clip_model;
    let clip_dir = format!("{}/{}", models_dir, clip_model);

    let visual_path = format!("{}/visual_model.onnx", clip_dir);
    let text_path = format!("{}/text_model.onnx", clip_dir);
    let tokenizer_path = format!("{}/tokenizer.json", clip_dir);

    // Try loading the visual encoder
    let image_embedder: Option<Arc<dyn ucotron_extraction::ImageEmbeddingPipeline>> =
        match ClipImagePipeline::new(&visual_path, ClipConfig::default()) {
            Ok(pipeline) => {
                tracing::info!("CLIP visual encoder loaded from {}", visual_path);
                Some(Arc::new(pipeline))
            }
            Err(e) => {
                tracing::info!("CLIP visual encoder not available: {}", e);
                None
            }
        };

    // Try loading the text encoder (for cross-modal search)
    let cross_modal_encoder: Option<Arc<dyn ucotron_extraction::CrossModalTextEncoder>> =
        match ClipTextPipeline::new(&text_path, &tokenizer_path, 4) {
            Ok(pipeline) => {
                tracing::info!("CLIP text encoder loaded from {}", text_path);
                Some(Arc::new(pipeline))
            }
            Err(e) => {
                tracing::info!("CLIP text encoder not available: {}", e);
                None
            }
        };

    (image_embedder, cross_modal_encoder)
}

/// Try to initialize FFmpeg video pipeline for video ingestion support.
/// Returns None if FFmpeg initialization fails.
fn try_init_video() -> Option<Arc<dyn ucotron_extraction::VideoPipeline>> {
    use ucotron_extraction::video::{FfmpegVideoPipeline, VideoConfig};

    match std::panic::catch_unwind(|| FfmpegVideoPipeline::new(VideoConfig::default())) {
        Ok(pipeline) => {
            tracing::info!("FFmpeg video pipeline initialized");
            Some(Arc::new(pipeline))
        }
        Err(_) => {
            tracing::info!("FFmpeg video pipeline not available");
            None
        }
    }
}

/// Stub embedding pipeline for when ONNX models are not available.
/// Produces zero vectors — suitable for testing and development.
struct StubEmbedder;

impl ucotron_extraction::EmbeddingPipeline for StubEmbedder {
    fn embed_text(&self, _text: &str) -> anyhow::Result<Vec<f32>> {
        Ok(vec![0.0f32; 384])
    }

    fn embed_batch(&self, texts: &[&str]) -> anyhow::Result<Vec<Vec<f32>>> {
        Ok(texts.iter().map(|_| vec![0.0f32; 384]).collect())
    }
}
