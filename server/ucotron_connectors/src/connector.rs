//! Core connector trait and types for data source integrations.
//!
//! Each connector implements the [`Connector`] trait to fetch content from
//! an external data source (Slack, GitHub, Notion, etc.) and produce
//! [`ContentItem`]s ready for ingestion into the Ucotron memory graph.

use std::collections::HashMap;

use anyhow::Result;
use serde::{Deserialize, Serialize};

/// Unique identifier for a connector instance.
pub type ConnectorId = String;

/// Authentication credentials for a connector.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum AuthConfig {
    /// OAuth2 flow (Slack, Google Docs, Google Drive).
    OAuth2 {
        client_id: String,
        client_secret: String,
        access_token: Option<String>,
        refresh_token: Option<String>,
    },
    /// Personal Access Token (GitHub, GitLab, Bitbucket).
    Token { token: String },
    /// Bot token (Discord, Telegram).
    BotToken { token: String },
    /// API key (Notion).
    ApiKey { key: String },
    /// Connection string (Postgres, MongoDB).
    ConnectionString { uri: String },
    /// No auth required (Obsidian local vault).
    None,
}

/// Configuration for a specific connector instance.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConnectorConfig {
    /// Unique identifier for this connector instance.
    pub id: ConnectorId,
    /// Human-readable name (e.g., "My Slack Workspace").
    pub name: String,
    /// The connector type (e.g., "slack", "github", "notion").
    pub connector_type: String,
    /// Authentication credentials.
    pub auth: AuthConfig,
    /// Namespace to ingest content into.
    pub namespace: String,
    /// Connector-specific settings (e.g., channels, repos, pages).
    pub settings: HashMap<String, serde_json::Value>,
    /// Whether this connector is enabled.
    pub enabled: bool,
}

/// Metadata about a content item's source.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SourceMetadata {
    /// Connector type that produced this item (e.g., "slack").
    pub connector_type: String,
    /// Connector instance ID.
    pub connector_id: ConnectorId,
    /// Source-specific identifier (e.g., message ID, issue number).
    pub source_id: String,
    /// URL to the original content (if available).
    pub source_url: Option<String>,
    /// Author/creator name.
    pub author: Option<String>,
    /// Original creation timestamp (Unix seconds).
    pub created_at: Option<u64>,
    /// Additional source-specific metadata.
    pub extra: HashMap<String, serde_json::Value>,
}

/// A content item ready for ingestion into Ucotron.
///
/// Connectors fetch raw data from external sources and map them
/// into `ContentItem`s that the ingestion pipeline can process.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContentItem {
    /// The text content to be ingested.
    pub content: String,
    /// Metadata about the source of this content.
    pub source: SourceMetadata,
    /// Optional media data (images, audio, video).
    pub media: Option<MediaAttachment>,
}

/// Media attachment for multimodal content items.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MediaAttachment {
    /// MIME type (e.g., "image/png", "audio/wav").
    pub mime_type: String,
    /// Raw bytes of the media file.
    #[serde(with = "base64_bytes")]
    pub data: Vec<u8>,
    /// Original filename (if available).
    pub filename: Option<String>,
}

/// Serde helper for base64-encoding bytes.
mod base64_bytes {
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    pub fn serialize<S: Serializer>(bytes: &[u8], s: S) -> Result<S::Ok, S::Error> {
        // Simple hex encoding (avoids base64 crate dependency)
        let hex: String = bytes.iter().map(|b| format!("{:02x}", b)).collect();
        hex.serialize(s)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<Vec<u8>, D::Error> {
        let hex = String::deserialize(d)?;
        (0..hex.len())
            .step_by(2)
            .map(|i| {
                u8::from_str_radix(&hex[i..i + 2], 16).map_err(serde::de::Error::custom)
            })
            .collect()
    }
}

/// Cursor for incremental sync — tracks where the last sync left off.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SyncCursor {
    /// Opaque cursor value (e.g., Slack cursor, GitHub since timestamp).
    pub value: Option<String>,
    /// Last sync timestamp (Unix seconds).
    pub last_sync: Option<u64>,
}

/// Result of a sync operation.
#[derive(Debug, Clone)]
pub struct SyncResult {
    /// Content items fetched in this sync.
    pub items: Vec<ContentItem>,
    /// Updated cursor for next incremental sync.
    pub cursor: SyncCursor,
    /// Number of items skipped (e.g., due to filters).
    pub skipped: usize,
}

/// Webhook payload from an external service.
#[derive(Debug, Clone)]
pub struct WebhookPayload {
    /// Raw request body.
    pub body: Vec<u8>,
    /// HTTP headers.
    pub headers: HashMap<String, String>,
    /// Content type.
    pub content_type: Option<String>,
}

/// Trait for data source connectors.
///
/// Each connector integrates with one external service (Slack, GitHub, etc.)
/// and provides methods to fetch, sync, and receive webhooks.
#[allow(async_fn_in_trait)]
pub trait Connector: Send + Sync {
    /// Returns the unique type identifier (e.g., "slack", "github").
    fn id(&self) -> &str;

    /// Returns the human-readable connector name (e.g., "Slack").
    fn name(&self) -> &str;

    /// Returns a JSON schema describing the connector's configuration.
    fn config_schema(&self) -> serde_json::Value;

    /// Validates the provided configuration.
    fn validate_config(&self, config: &ConnectorConfig) -> Result<()>;

    /// Fetches all content from the source (full sync).
    async fn fetch(&self, config: &ConnectorConfig) -> Result<Vec<ContentItem>>;

    /// Fetches content incrementally from the last cursor position.
    async fn sync_incremental(
        &self,
        config: &ConnectorConfig,
        cursor: &SyncCursor,
    ) -> Result<SyncResult>;

    /// Handles an incoming webhook from the external service.
    async fn handle_webhook(
        &self,
        config: &ConnectorConfig,
        payload: WebhookPayload,
    ) -> Result<Vec<ContentItem>>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_auth_config_serialization() {
        let auth = AuthConfig::Token {
            token: "ghp_test123".to_string(),
        };
        let json = serde_json::to_string(&auth).unwrap();
        assert!(json.contains("Token"));
        assert!(json.contains("ghp_test123"));

        let deserialized: AuthConfig = serde_json::from_str(&json).unwrap();
        match deserialized {
            AuthConfig::Token { token } => assert_eq!(token, "ghp_test123"),
            _ => panic!("expected Token variant"),
        }
    }

    #[test]
    fn test_connector_config_creation() {
        let config = ConnectorConfig {
            id: "conn-1".to_string(),
            name: "My GitHub".to_string(),
            connector_type: "github".to_string(),
            auth: AuthConfig::Token {
                token: "ghp_abc".to_string(),
            },
            namespace: "default".to_string(),
            settings: HashMap::new(),
            enabled: true,
        };
        assert_eq!(config.id, "conn-1");
        assert!(config.enabled);
    }

    #[test]
    fn test_content_item_serialization() {
        let item = ContentItem {
            content: "Hello from Slack".to_string(),
            source: SourceMetadata {
                connector_type: "slack".to_string(),
                connector_id: "conn-1".to_string(),
                source_id: "msg-123".to_string(),
                source_url: Some("https://slack.com/archives/C01/p123".to_string()),
                author: Some("Alice".to_string()),
                created_at: Some(1700000000),
                extra: HashMap::new(),
            },
            media: None,
        };
        let json = serde_json::to_string(&item).unwrap();
        let deserialized: ContentItem = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.content, "Hello from Slack");
        assert_eq!(deserialized.source.source_id, "msg-123");
    }

    #[test]
    fn test_sync_cursor_default() {
        let cursor = SyncCursor::default();
        assert!(cursor.value.is_none());
        assert!(cursor.last_sync.is_none());
    }

    #[test]
    fn test_media_attachment_serialization() {
        let attachment = MediaAttachment {
            mime_type: "image/png".to_string(),
            data: vec![0x89, 0x50, 0x4E, 0x47],
            filename: Some("test.png".to_string()),
        };
        let json = serde_json::to_string(&attachment).unwrap();
        let deserialized: MediaAttachment = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.data, vec![0x89, 0x50, 0x4E, 0x47]);
        assert_eq!(deserialized.mime_type, "image/png");
    }
}
