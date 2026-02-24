//! OpenAPI 3.1 specification generation for the Ucotron REST API.
//!
//! Uses utoipa to generate the spec from annotated handlers and types.
//! The spec is served at `/api/v1/openapi.json` and Swagger UI at `/swagger-ui`.

use utoipa::OpenApi;

use crate::handlers;
use crate::types::*;

/// OpenAPI specification for the Ucotron REST API.
#[derive(OpenApi)]
#[openapi(
    info(
        title = "Ucotron API",
        description = "Cognitive memory framework for LLMs — REST API for memory management, \
                       semantic search, context augmentation, and knowledge graph operations.",
        version = "0.1.0",
        contact(name = "Ucotron Team"),
        license(name = "MIT OR Apache-2.0")
    ),
    servers(
        (url = "http://localhost:8420", description = "Local development server")
    ),
    tags(
        (name = "Health", description = "Server health and metrics endpoints"),
        (name = "Memories", description = "Memory CRUD operations — create, read, update, delete memory nodes"),
        (name = "Search", description = "Semantic search using vector similarity and graph-based re-ranking"),
        (name = "Entities", description = "Knowledge graph entity operations"),
        (name = "Graph", description = "Graph visualization — nodes and edges for force-directed rendering"),
        (name = "Augment & Learn", description = "Core agent integration — augment context with memories and learn from agent output"),
        (name = "Export & Import", description = "JSON-LD graph export and import for data portability and migration"),
        (name = "Import", description = "Import adapters for third-party memory systems (Mem0, Zep)"),
        (name = "Audio", description = "Audio transcription via Whisper ONNX — voice-to-memory ingestion"),
        (name = "Images", description = "Image embedding via CLIP ONNX — cross-modal visual search"),
        (name = "Documents", description = "Document OCR — PDF text extraction and scanned document OCR via Tesseract"),
        (name = "Multimodal", description = "Cross-modal search and media management — unified text/image/audio/video search, \
                                              video segments, and media file serving"),
        (name = "Media", description = "Media file serving — stream images, audio, and video files by node ID"),
        (name = "Admin", description = "Administration — namespace management, configuration view, and system monitoring"),
        (name = "Auth", description = "Authentication and RBAC — API key management, role-based access control"),
        (name = "Agents", description = "Agent management — create, list, clone, merge, and share agent graphs"),
        (name = "Fine-Tuning", description = "Training dataset generation and model fine-tuning pipeline")
    ),
    paths(
        handlers::health_handler,
        handlers::metrics_handler,
        handlers::create_memory_handler,
        handlers::list_memories_handler,
        handlers::get_memory_handler,
        handlers::update_memory_handler,
        handlers::delete_memory_handler,
        handlers::search_handler,
        handlers::list_entities_handler,
        handlers::get_entity_handler,
        handlers::graph_handler,
        handlers::augment_handler,
        handlers::learn_handler,
        handlers::export_handler,
        handlers::import_handler,
        handlers::mem0_import_handler,
        handlers::zep_import_handler,
        handlers::transcribe_handler,
        handlers::index_image_handler,
        handlers::image_search_handler,
        handlers::ocr_handler,
        handlers::list_namespaces_handler,
        handlers::create_namespace_handler,
        handlers::get_namespace_handler,
        handlers::delete_namespace_handler,
        handlers::admin_config_handler,
        handlers::admin_system_handler,
        handlers::gdpr_forget_handler,
        handlers::gdpr_export_handler,
        handlers::gdpr_retention_status_handler,
        handlers::gdpr_retention_sweep_handler,
        handlers::whoami_handler,
        handlers::list_api_keys_handler,
        handlers::create_api_key_handler,
        handlers::revoke_api_key_handler,
        handlers::generate_dataset_handler,
        handlers::create_text_memory_handler,
        handlers::create_audio_memory_handler,
        handlers::create_image_memory_handler,
        handlers::create_agent_handler,
        handlers::list_agents_handler,
        handlers::get_agent_handler,
        handlers::delete_agent_handler,
        handlers::clone_agent_handler,
        handlers::merge_agent_handler,
        handlers::create_share_handler,
        handlers::list_shares_handler,
        handlers::delete_share_handler,
        handlers::create_video_memory_handler,
        handlers::multimodal_search_handler,
        handlers::get_video_segments_handler,
        handlers::get_media_handler,
        handlers::storage::upload_handler,
        handlers::storage::download_handler,
        handlers::storage::delete_handler,
        handlers::storage::presign_handler,
    ),
    components(schemas(
        CreateMemoryRequest,
        CreateMemoryResponse,
        IngestionMetricsResponse,
        MemoryResponse,
        UpdateMemoryRequest,
        SearchRequest,
        SearchResultItem,
        SearchResponse,
        EntityResponse,
        NeighborResponse,
        GraphNode,
        GraphEdge,
        GraphResponse,
        AugmentRequest,
        AugmentResponse,
        ExplainabilityInfo,
        ExplainSourceNode,
        RetrievalPath,
        ContextContribution,
        LearnRequest,
        LearnResponse,
        HealthResponse,
        ModelStatus,
        MetricsResponse,
        ApiErrorResponse,
        ExportResponse,
        ExportNodeResponse,
        ExportEdgeResponse,
        ExportStatsResponse,
        ImportRequest,
        ImportResponse,
        Mem0ImportRequest,
        Mem0ImportResponse,
        ZepImportRequest,
        ZepImportResponse,
        TranscribeResponse,
        TranscribeChunk,
        AudioMetadataResponse,
        ImageIndexResponse,
        ImageSearchRequest,
        ImageSearchResultItem,
        ImageSearchResponse,
        OcrResponse,
        OcrPageResponse,
        OcrDocumentMetadata,
        NamespaceListResponse,
        NamespaceInfo,
        CreateNamespaceRequest,
        CreateNamespaceResponse,
        DeleteNamespaceResponse,
        ConfigSummaryResponse,
        ConfigServerSection,
        ConfigStorageSection,
        ConfigModelsSection,
        ConfigInstanceSection,
        ConfigNamespacesSection,
        SystemInfoResponse,
        GdprForgetResponse,
        GdprExportResponse,
        GdprExportEdge,
        GdprExportStats,
        RetentionStatusResponse,
        RetentionPolicy,
        SetRetentionRequest,
        SetRetentionResponse,
        RetentionSweepResponse,
        GdprAuditEntry,
        CreateApiKeyRequest,
        CreateApiKeyResponse,
        ListApiKeysResponse,
        ApiKeyInfo,
        RevokeApiKeyResponse,
        WhoamiResponse,
        GenerateDatasetRequest,
        GenerateDatasetResponse,
        CreateTextMemoryRequest,
        CreateTextMemoryResponse,
        CreateAudioMemoryResponse,
        CreateImageMemoryResponse,
        CreateAgentRequest,
        CreateAgentResponse,
        ListAgentsResponse,
        AgentResponse,
        CloneAgentRequest,
        CloneAgentResponse,
        MergeAgentRequest,
        MergeAgentResponse,
        CreateShareRequest,
        CreateShareResponse,
        ListSharesResponse,
        ShareResponse,
        CreateVideoMemoryResponse,
        VideoSegmentInfo,
        VideoSegmentDetail,
        VideoSegmentsResponse,
        MultimodalSearchRequest,
        MultimodalSearchResultItem,
        MultimodalSearchResponse,
        MultimodalSearchMetrics,
        handlers::storage::UploadResponse,
        handlers::storage::DownloadResponse,
        handlers::storage::PresignedUrlResponse,
        handlers::storage::DeleteResponse,
        handlers::storage::StorageApiErrorResponse,
        FrameEmbedKeyResponse,
    ))
)]
pub struct ApiDoc;

#[cfg(test)]
mod tests {
    use super::*;
    use utoipa::OpenApi;

    #[test]
    fn test_openapi_spec_generates() {
        let spec = ApiDoc::openapi();
        let json = spec.to_json().expect("Failed to serialize OpenAPI spec");
        assert!(json.contains("Ucotron API"));
        assert!(json.contains("/api/v1/health"));
        assert!(json.contains("/api/v1/memories"));
        assert!(json.contains("/api/v1/memories/search"));
        assert!(json.contains("/api/v1/entities"));
        assert!(json.contains("/api/v1/augment"));
        assert!(json.contains("/api/v1/learn"));
    }

    #[test]
    fn test_openapi_spec_has_all_endpoints() {
        let spec = ApiDoc::openapi();
        let json = spec.to_json().expect("Failed to serialize OpenAPI spec");

        // Core + agent endpoints should be present
        let paths = [
            "/api/v1/health",
            "/api/v1/metrics",
            "/api/v1/memories",
            "/api/v1/memories/{id}",
            "/api/v1/memories/search",
            "/api/v1/entities",
            "/api/v1/entities/{id}",
            "/api/v1/augment",
            "/api/v1/learn",
            "/api/v1/agents",
            "/api/v1/agents/{id}",
            "/api/v1/agents/{id}/clone",
            "/api/v1/agents/{id}/merge",
            "/api/v1/agents/{id}/share",
            "/api/v1/agents/{id}/share/{target}",
        ];
        for path in paths {
            assert!(json.contains(path), "Missing path: {}", path);
        }
    }

    #[test]
    fn test_openapi_spec_has_schemas() {
        let spec = ApiDoc::openapi();
        let json = spec.to_json().expect("Failed to serialize OpenAPI spec");

        let schemas = [
            "CreateMemoryRequest",
            "CreateMemoryResponse",
            "MemoryResponse",
            "SearchRequest",
            "SearchResponse",
            "EntityResponse",
            "AugmentRequest",
            "AugmentResponse",
            "LearnRequest",
            "LearnResponse",
            "HealthResponse",
            "MetricsResponse",
            "CreateAgentRequest",
            "CreateAgentResponse",
            "AgentResponse",
            "ListAgentsResponse",
            "CloneAgentRequest",
            "CloneAgentResponse",
            "MergeAgentRequest",
            "MergeAgentResponse",
            "CreateShareRequest",
            "CreateShareResponse",
            "ListSharesResponse",
            "ShareResponse",
        ];
        for schema in schemas {
            assert!(json.contains(schema), "Missing schema: {}", schema);
        }
    }

    #[test]
    fn test_openapi_spec_has_tags() {
        let spec = ApiDoc::openapi();
        let json = spec.to_json().expect("Failed to serialize OpenAPI spec");

        assert!(json.contains("Health"));
        assert!(json.contains("Memories"));
        assert!(json.contains("Search"));
        assert!(json.contains("Entities"));
        assert!(json.contains("Augment & Learn"));
        assert!(json.contains("Agents"));
    }

    #[test]
    fn test_openapi_spec_has_agent_examples() {
        let spec = ApiDoc::openapi();
        let json_str = spec.to_json().expect("Failed to serialize OpenAPI spec");
        let parsed: serde_json::Value =
            serde_json::from_str(&json_str).expect("OpenAPI spec is not valid JSON");

        // Check that agent schemas include example values
        let schemas = parsed["components"]["schemas"].as_object().unwrap();
        assert!(
            schemas
                .get("CreateAgentRequest")
                .unwrap()
                .get("example")
                .is_some(),
            "CreateAgentRequest should have an example"
        );
        assert!(
            schemas
                .get("CreateShareRequest")
                .unwrap()
                .get("example")
                .is_some(),
            "CreateShareRequest should have an example"
        );
        assert!(
            schemas
                .get("ShareResponse")
                .unwrap()
                .get("example")
                .is_some(),
            "ShareResponse should have an example"
        );
    }

    #[test]
    fn test_openapi_spec_valid_json() {
        let spec = ApiDoc::openapi();
        let json_str = spec.to_json().expect("Failed to serialize OpenAPI spec");
        let parsed: serde_json::Value =
            serde_json::from_str(&json_str).expect("OpenAPI spec is not valid JSON");

        // Check top-level fields
        assert!(parsed.get("openapi").is_some());
        assert!(parsed.get("info").is_some());
        assert!(parsed.get("paths").is_some());
        assert!(parsed.get("components").is_some());
    }

    #[test]
    fn test_openapi_spec_has_multimodal_tag() {
        let spec = ApiDoc::openapi();
        let json_str = spec.to_json().expect("Failed to serialize OpenAPI spec");
        let parsed: serde_json::Value =
            serde_json::from_str(&json_str).expect("OpenAPI spec is not valid JSON");

        let tags = parsed["tags"].as_array().expect("tags should be an array");
        let tag_names: Vec<&str> = tags.iter().filter_map(|t| t["name"].as_str()).collect();

        assert!(tag_names.contains(&"Multimodal"), "Missing Multimodal tag");
        assert!(tag_names.contains(&"Media"), "Missing Media tag");
    }

    #[test]
    fn test_openapi_spec_has_multimodal_endpoints() {
        let spec = ApiDoc::openapi();
        let json_str = spec.to_json().expect("Failed to serialize OpenAPI spec");

        // Multimodal search, video memory, video segments, media serving
        let multimodal_paths = [
            "/api/v1/search/multimodal",
            "/api/v1/memories/video",
            "/api/v1/videos/{parent_id}/segments",
            "/api/v1/media/{id}",
        ];
        for path in multimodal_paths {
            assert!(json_str.contains(path), "Missing multimodal path: {}", path);
        }
    }

    #[test]
    fn test_openapi_spec_has_multimodal_schemas() {
        let spec = ApiDoc::openapi();
        let json_str = spec.to_json().expect("Failed to serialize OpenAPI spec");
        let parsed: serde_json::Value =
            serde_json::from_str(&json_str).expect("OpenAPI spec is not valid JSON");

        let schemas = parsed["components"]["schemas"]
            .as_object()
            .expect("schemas should be an object");

        let multimodal_schemas = [
            "MultimodalSearchRequest",
            "MultimodalSearchResponse",
            "MultimodalSearchResultItem",
            "MultimodalSearchMetrics",
            "CreateVideoMemoryResponse",
            "VideoSegmentInfo",
            "VideoSegmentDetail",
            "VideoSegmentsResponse",
        ];
        for schema in multimodal_schemas {
            assert!(
                schemas.contains_key(schema),
                "Missing multimodal schema: {}",
                schema
            );
        }
    }

    #[test]
    fn test_openapi_multimodal_search_has_multipart_docs() {
        let spec = ApiDoc::openapi();
        let json_str = spec.to_json().expect("Failed to serialize OpenAPI spec");
        let parsed: serde_json::Value =
            serde_json::from_str(&json_str).expect("OpenAPI spec is not valid JSON");

        // Video memory ingestion should document multipart/form-data
        let video_path = &parsed["paths"]["/api/v1/memories/video"]["post"];
        assert!(video_path.is_object(), "Video memory path should exist");
        let video_body = video_path["requestBody"]["content"]
            .as_object()
            .expect("Video handler should have request body content");
        assert!(
            video_body.contains_key("multipart/form-data"),
            "Video memory handler should document multipart/form-data"
        );
    }
}
