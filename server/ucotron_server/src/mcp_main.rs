//! # Ucotron MCP Server (stdio)
//!
//! Standalone MCP server binary for Ucotron, using stdio transport.
//!
//! This binary is designed for integration with Claude Desktop, Cursor,
//! and other MCP-compatible clients.
//!
//! # Usage
//!
//! ```bash
//! ucotron_mcp
//! ```
//!
//! # Configuration
//!
//! Set `UCOTRON_CONFIG` env var to a TOML config file path, or use defaults.
//! Logs are written to stderr (stdout is reserved for MCP JSON-RPC protocol).

use std::sync::Arc;

use rmcp::ServiceExt;

use ucotron_config::UcotronConfig;
use ucotron_server::mcp::UcotronMcpServer;
use ucotron_server::state::AppState;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialize tracing to stderr (stdout is for MCP protocol).
    tracing_subscriber::fmt()
        .with_writer(std::io::stderr)
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    // Load configuration.
    let config = if let Ok(path) = std::env::var("UCOTRON_CONFIG") {
        tracing::info!("Loading config from {}", path);
        UcotronConfig::from_file(&path)?
    } else {
        tracing::info!("No UCOTRON_CONFIG set, using default configuration");
        UcotronConfig::default()
    };

    tracing::info!("Ucotron MCP Server starting (stdio transport)");

    // Initialize storage backends.
    let (vector_backend, graph_backend) =
        ucotron_helix::create_helix_backends(&config.storage.vector, &config.storage.graph)?;
    let registry = Arc::new(ucotron_core::BackendRegistry::new(
        vector_backend,
        graph_backend,
    ));

    // Initialize embedding pipeline.
    let embedder: Arc<dyn ucotron_extraction::EmbeddingPipeline> =
        match try_init_embedder(&config) {
            Ok(e) => e,
            Err(e) => {
                tracing::warn!("Failed to load embedding model: {}. Using stub embedder.", e);
                Arc::new(StubEmbedder)
            }
        };

    // Build application state.
    let state = Arc::new(AppState::new(
        registry,
        embedder,
        None,
        None,
        config,
        None,
    ));

    // Create MCP server and serve via stdio.
    let server = UcotronMcpServer::new(state);
    let transport = rmcp::transport::stdio();
    let service = server.serve(transport).await?;

    tracing::info!("MCP server running on stdio");
    service.waiting().await?;

    tracing::info!("MCP server shut down");
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

/// Stub embedding pipeline for when ONNX models are not available.
struct StubEmbedder;

impl ucotron_extraction::EmbeddingPipeline for StubEmbedder {
    fn embed_text(&self, _text: &str) -> anyhow::Result<Vec<f32>> {
        Ok(vec![0.0f32; 384])
    }
    fn embed_batch(&self, texts: &[&str]) -> anyhow::Result<Vec<Vec<f32>>> {
        Ok(texts.iter().map(|_| vec![0.0f32; 384]).collect())
    }
}
