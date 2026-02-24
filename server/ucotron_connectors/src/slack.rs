//! Slack connector — fetches messages from Slack channels via the Web API.
//!
//! Uses OAuth2 access tokens to authenticate with the Slack API.
//! Supports full sync (all messages from configured channels) and
//! incremental sync via cursor-based pagination.

use std::collections::HashMap;

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};

use crate::connector::{
    AuthConfig, Connector, ConnectorConfig, ContentItem, SourceMetadata, SyncCursor, SyncResult,
    WebhookPayload,
};

const SLACK_API_BASE: &str = "https://slack.com/api";

/// Slack connector for fetching messages from Slack channels.
///
/// Requires an OAuth2 access token with the following scopes:
/// - `channels:history` — read messages from public channels
/// - `channels:read` — list public channels
/// - `users:read` — resolve user IDs to display names
pub struct SlackConnector {
    client: reqwest::Client,
}

impl SlackConnector {
    /// Creates a new SlackConnector with a default HTTP client.
    pub fn new() -> Self {
        Self {
            client: reqwest::Client::new(),
        }
    }

    /// Creates a new SlackConnector with a custom HTTP client.
    ///
    /// Useful for testing with mock servers or custom timeouts.
    pub fn with_client(client: reqwest::Client) -> Self {
        Self { client }
    }

    /// Extracts the access token from the connector config.
    fn get_token(config: &ConnectorConfig) -> Result<&str> {
        match &config.auth {
            AuthConfig::OAuth2 {
                access_token: Some(token),
                ..
            } => Ok(token.as_str()),
            AuthConfig::OAuth2 {
                access_token: None, ..
            } => bail!("Slack connector requires an access_token in OAuth2 config"),
            _ => bail!("Slack connector requires OAuth2 authentication"),
        }
    }

    /// Extracts configured channel IDs from settings, or returns empty vec for "all channels".
    fn get_channels(config: &ConnectorConfig) -> Vec<String> {
        config
            .settings
            .get("channels")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str().map(String::from))
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Lists public channels in the workspace.
    async fn list_channels(&self, token: &str) -> Result<Vec<SlackChannel>> {
        let mut channels = Vec::new();
        let mut cursor = String::new();

        loop {
            let mut params = vec![("limit", "200".to_string()), ("types", "public_channel".to_string())];
            if !cursor.is_empty() {
                params.push(("cursor", cursor.clone()));
            }

            let resp: ConversationsListResponse = self
                .client
                .get(format!("{}/conversations.list", SLACK_API_BASE))
                .bearer_auth(token)
                .query(&params)
                .send()
                .await
                .context("Failed to call conversations.list")?
                .json()
                .await
                .context("Failed to parse conversations.list response")?;

            if !resp.ok {
                bail!(
                    "Slack API error in conversations.list: {}",
                    resp.error.unwrap_or_default()
                );
            }

            channels.extend(resp.channels);

            match resp.response_metadata.and_then(|m| m.next_cursor) {
                Some(c) if !c.is_empty() => cursor = c,
                _ => break,
            }
        }

        Ok(channels)
    }

    /// Fetches message history from a single channel.
    async fn fetch_channel_history(
        &self,
        token: &str,
        channel_id: &str,
        oldest: Option<&str>,
        cursor: Option<&str>,
    ) -> Result<(Vec<SlackMessage>, Option<String>)> {
        let mut messages = Vec::new();
        let mut page_cursor = cursor.map(String::from);

        loop {
            let mut params = vec![
                ("channel", channel_id.to_string()),
                ("limit", "200".to_string()),
            ];
            if let Some(ts) = oldest {
                params.push(("oldest", ts.to_string()));
            }
            if let Some(ref c) = page_cursor {
                params.push(("cursor", c.clone()));
            }

            let resp: ConversationsHistoryResponse = self
                .client
                .get(format!("{}/conversations.history", SLACK_API_BASE))
                .bearer_auth(token)
                .query(&params)
                .send()
                .await
                .context("Failed to call conversations.history")?
                .json()
                .await
                .context("Failed to parse conversations.history response")?;

            if !resp.ok {
                bail!(
                    "Slack API error in conversations.history: {}",
                    resp.error.unwrap_or_default()
                );
            }

            messages.extend(resp.messages);

            match resp.response_metadata.and_then(|m| m.next_cursor) {
                Some(c) if !c.is_empty() => page_cursor = Some(c),
                _ => break,
            }
        }

        // Return the latest message timestamp as the new cursor
        let latest_ts = messages.first().map(|m| m.ts.clone());
        Ok((messages, latest_ts))
    }

    /// Resolves a Slack user ID to a display name.
    async fn resolve_user(&self, token: &str, user_id: &str) -> Result<String> {
        let resp: UsersInfoResponse = self
            .client
            .get(format!("{}/users.info", SLACK_API_BASE))
            .bearer_auth(token)
            .query(&[("user", user_id)])
            .send()
            .await
            .context("Failed to call users.info")?
            .json()
            .await
            .context("Failed to parse users.info response")?;

        if !resp.ok {
            return Ok(user_id.to_string());
        }

        Ok(resp
            .user
            .and_then(|u| {
                u.profile
                    .and_then(|p| p.display_name)
                    .or(u.real_name)
                    .or(Some(u.name))
            })
            .unwrap_or_else(|| user_id.to_string()))
    }

    /// Converts a Slack message to a ContentItem.
    fn message_to_content_item(
        &self,
        msg: &SlackMessage,
        channel_id: &str,
        channel_name: &str,
        connector_id: &str,
        author_name: Option<&str>,
    ) -> Option<ContentItem> {
        // Skip messages with no text (e.g., bot join/leave messages)
        let text = msg.text.as_deref().unwrap_or("").trim();
        if text.is_empty() {
            return None;
        }

        // Parse Slack timestamp (e.g., "1234567890.123456") to Unix seconds
        let created_at = msg
            .ts
            .split('.')
            .next()
            .and_then(|s| s.parse::<u64>().ok());

        let mut extra = HashMap::new();
        extra.insert(
            "channel_id".to_string(),
            serde_json::Value::String(channel_id.to_string()),
        );
        extra.insert(
            "channel_name".to_string(),
            serde_json::Value::String(channel_name.to_string()),
        );
        if let Some(ref subtype) = msg.subtype {
            extra.insert(
                "subtype".to_string(),
                serde_json::Value::String(subtype.clone()),
            );
        }
        if let Some(ref thread_ts) = msg.thread_ts {
            extra.insert(
                "thread_ts".to_string(),
                serde_json::Value::String(thread_ts.clone()),
            );
        }

        // Build content with channel context
        let content = format!("[#{}] {}", channel_name, text);

        Some(ContentItem {
            content,
            source: SourceMetadata {
                connector_type: "slack".to_string(),
                connector_id: connector_id.to_string(),
                source_id: msg.ts.clone(),
                source_url: Some(format!(
                    "https://slack.com/archives/{}/p{}",
                    channel_id,
                    msg.ts.replace('.', "")
                )),
                author: author_name.map(String::from).or(msg.user.clone()),
                created_at,
                extra,
            },
            media: None,
        })
    }
}

impl Default for SlackConnector {
    fn default() -> Self {
        Self::new()
    }
}

impl Connector for SlackConnector {
    fn id(&self) -> &str {
        "slack"
    }

    fn name(&self) -> &str {
        "Slack"
    }

    fn config_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "auth": {
                    "type": "object",
                    "description": "OAuth2 credentials",
                    "properties": {
                        "client_id": { "type": "string" },
                        "client_secret": { "type": "string" },
                        "access_token": { "type": "string", "description": "Bot or user OAuth token" },
                        "refresh_token": { "type": "string" }
                    },
                    "required": ["access_token"]
                },
                "settings": {
                    "type": "object",
                    "properties": {
                        "channels": {
                            "type": "array",
                            "items": { "type": "string" },
                            "description": "Channel IDs to sync. If empty, syncs all public channels."
                        }
                    }
                }
            },
            "required": ["auth"]
        })
    }

    fn validate_config(&self, config: &ConnectorConfig) -> Result<()> {
        if config.connector_type != "slack" {
            bail!(
                "Invalid connector type '{}', expected 'slack'",
                config.connector_type
            );
        }
        Self::get_token(config)?;
        Ok(())
    }

    async fn fetch(&self, config: &ConnectorConfig) -> Result<Vec<ContentItem>> {
        let token = Self::get_token(config)?;
        let configured_channels = Self::get_channels(config);

        // Get channels to fetch from
        let channels = if configured_channels.is_empty() {
            self.list_channels(token).await?
        } else {
            configured_channels
                .into_iter()
                .map(|id| SlackChannel {
                    id: id.clone(),
                    name: id,
                    is_member: true,
                })
                .collect()
        };

        let mut items = Vec::new();
        // Cache user name lookups
        let mut user_cache: HashMap<String, String> = HashMap::new();

        for channel in &channels {
            let (messages, _) = self
                .fetch_channel_history(token, &channel.id, None, None)
                .await?;

            for msg in &messages {
                // Resolve user name
                let author = if let Some(ref user_id) = msg.user {
                    if let Some(cached) = user_cache.get(user_id) {
                        Some(cached.as_str())
                    } else {
                        match self.resolve_user(token, user_id).await {
                            Ok(name) => {
                                user_cache.insert(user_id.clone(), name);
                                user_cache.get(user_id).map(|s| s.as_str())
                            }
                            Err(_) => Some(user_id.as_str()),
                        }
                    }
                } else {
                    None
                };

                if let Some(item) = self.message_to_content_item(
                    msg,
                    &channel.id,
                    &channel.name,
                    &config.id,
                    author,
                ) {
                    items.push(item);
                }
            }
        }

        Ok(items)
    }

    async fn sync_incremental(
        &self,
        config: &ConnectorConfig,
        cursor: &SyncCursor,
    ) -> Result<SyncResult> {
        let token = Self::get_token(config)?;
        let configured_channels = Self::get_channels(config);

        let channels = if configured_channels.is_empty() {
            self.list_channels(token).await?
        } else {
            configured_channels
                .into_iter()
                .map(|id| SlackChannel {
                    id: id.clone(),
                    name: id,
                    is_member: true,
                })
                .collect()
        };

        let mut items = Vec::new();
        let mut latest_ts: Option<String> = cursor.value.clone();
        let mut user_cache: HashMap<String, String> = HashMap::new();

        for channel in &channels {
            let oldest = cursor.value.as_deref();
            let (messages, channel_latest) = self
                .fetch_channel_history(token, &channel.id, oldest, None)
                .await?;

            // Track the most recent timestamp across all channels
            if let Some(ref ts) = channel_latest {
                if latest_ts.as_ref().map_or(true, |current| ts > current) {
                    latest_ts = Some(ts.clone());
                }
            }

            for msg in &messages {
                let author = if let Some(ref user_id) = msg.user {
                    if let Some(cached) = user_cache.get(user_id) {
                        Some(cached.as_str())
                    } else {
                        match self.resolve_user(token, user_id).await {
                            Ok(name) => {
                                user_cache.insert(user_id.clone(), name);
                                user_cache.get(user_id).map(|s| s.as_str())
                            }
                            Err(_) => Some(user_id.as_str()),
                        }
                    }
                } else {
                    None
                };

                if let Some(item) = self.message_to_content_item(
                    msg,
                    &channel.id,
                    &channel.name,
                    &config.id,
                    author,
                ) {
                    items.push(item);
                }
            }
        }

        let now_secs = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        Ok(SyncResult {
            items,
            cursor: SyncCursor {
                value: latest_ts,
                last_sync: Some(now_secs),
            },
            skipped: 0,
        })
    }

    async fn handle_webhook(
        &self,
        config: &ConnectorConfig,
        payload: WebhookPayload,
    ) -> Result<Vec<ContentItem>> {
        // Parse the webhook body as JSON
        let body: serde_json::Value =
            serde_json::from_slice(&payload.body).context("Invalid webhook JSON body")?;

        // Handle Slack URL verification challenge
        if let Some(challenge) = body.get("challenge").and_then(|v| v.as_str()) {
            // For URL verification, return the challenge as-is (caller handles HTTP response)
            // We return an empty vec since there's no content to ingest
            let _ = challenge;
            return Ok(Vec::new());
        }

        // Handle event callbacks
        let event = body
            .get("event")
            .context("Webhook body missing 'event' field")?;

        let event_type = event
            .get("type")
            .and_then(|v| v.as_str())
            .unwrap_or_default();

        if event_type != "message" {
            return Ok(Vec::new());
        }

        // Skip message subtypes that aren't regular messages (e.g., message_changed, message_deleted)
        if event.get("subtype").is_some() {
            return Ok(Vec::new());
        }

        let text = event
            .get("text")
            .and_then(|v| v.as_str())
            .unwrap_or_default()
            .trim();
        if text.is_empty() {
            return Ok(Vec::new());
        }

        let channel_id = event
            .get("channel")
            .and_then(|v| v.as_str())
            .unwrap_or("unknown");
        let user_id = event
            .get("user")
            .and_then(|v| v.as_str())
            .map(String::from);
        let ts = event
            .get("ts")
            .and_then(|v| v.as_str())
            .unwrap_or("0")
            .to_string();
        let thread_ts = event
            .get("thread_ts")
            .and_then(|v| v.as_str())
            .map(String::from);

        let created_at = ts.split('.').next().and_then(|s| s.parse::<u64>().ok());

        let mut extra = HashMap::new();
        extra.insert(
            "channel_id".to_string(),
            serde_json::Value::String(channel_id.to_string()),
        );
        if let Some(ref tts) = thread_ts {
            extra.insert(
                "thread_ts".to_string(),
                serde_json::Value::String(tts.clone()),
            );
        }

        let content = format!("[#{}] {}", channel_id, text);

        Ok(vec![ContentItem {
            content,
            source: SourceMetadata {
                connector_type: "slack".to_string(),
                connector_id: config.id.clone(),
                source_id: ts.clone(),
                source_url: Some(format!(
                    "https://slack.com/archives/{}/p{}",
                    channel_id,
                    ts.replace('.', "")
                )),
                author: user_id,
                created_at,
                extra,
            },
            media: None,
        }])
    }
}

// --- Slack API response types ---

#[derive(Debug, Deserialize)]
struct ConversationsListResponse {
    ok: bool,
    #[serde(default)]
    channels: Vec<SlackChannel>,
    error: Option<String>,
    response_metadata: Option<ResponseMetadata>,
}

#[derive(Debug, Deserialize, Clone)]
struct SlackChannel {
    id: String,
    name: String,
    #[serde(default)]
    #[allow(dead_code)]
    is_member: bool,
}

#[derive(Debug, Deserialize)]
struct ConversationsHistoryResponse {
    ok: bool,
    #[serde(default)]
    messages: Vec<SlackMessage>,
    error: Option<String>,
    response_metadata: Option<ResponseMetadata>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
struct SlackMessage {
    /// Message timestamp (unique ID within channel).
    ts: String,
    /// Message text content.
    text: Option<String>,
    /// User ID of the sender.
    user: Option<String>,
    /// Message subtype (e.g., "bot_message", "channel_join").
    subtype: Option<String>,
    /// Thread parent timestamp (if this message is in a thread).
    thread_ts: Option<String>,
}

#[derive(Debug, Deserialize)]
struct ResponseMetadata {
    next_cursor: Option<String>,
}

#[derive(Debug, Deserialize)]
struct UsersInfoResponse {
    ok: bool,
    user: Option<SlackUser>,
}

#[derive(Debug, Deserialize)]
struct SlackUser {
    name: String,
    real_name: Option<String>,
    profile: Option<SlackProfile>,
}

#[derive(Debug, Deserialize)]
struct SlackProfile {
    display_name: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn make_config(token: &str, channels: Vec<&str>) -> ConnectorConfig {
        let mut settings = HashMap::new();
        if !channels.is_empty() {
            settings.insert(
                "channels".to_string(),
                serde_json::json!(channels),
            );
        }
        ConnectorConfig {
            id: "slack-test".to_string(),
            name: "Test Slack".to_string(),
            connector_type: "slack".to_string(),
            auth: AuthConfig::OAuth2 {
                client_id: "client_id".to_string(),
                client_secret: "client_secret".to_string(),
                access_token: Some(token.to_string()),
                refresh_token: None,
            },
            namespace: "test".to_string(),
            settings,
            enabled: true,
        }
    }

    #[test]
    fn test_slack_connector_id_and_name() {
        let connector = SlackConnector::new();
        assert_eq!(connector.id(), "slack");
        assert_eq!(connector.name(), "Slack");
    }

    #[test]
    fn test_slack_config_schema() {
        let connector = SlackConnector::new();
        let schema = connector.config_schema();
        assert_eq!(schema["type"], "object");
        assert!(schema["properties"]["auth"].is_object());
        assert!(schema["properties"]["settings"]["properties"]["channels"].is_object());
    }

    #[test]
    fn test_validate_config_valid() {
        let connector = SlackConnector::new();
        let config = make_config("xoxb-test-token", vec![]);
        assert!(connector.validate_config(&config).is_ok());
    }

    #[test]
    fn test_validate_config_wrong_type() {
        let connector = SlackConnector::new();
        let mut config = make_config("xoxb-test-token", vec![]);
        config.connector_type = "github".to_string();
        assert!(connector.validate_config(&config).is_err());
    }

    #[test]
    fn test_validate_config_missing_token() {
        let connector = SlackConnector::new();
        let config = ConnectorConfig {
            id: "slack-test".to_string(),
            name: "Test Slack".to_string(),
            connector_type: "slack".to_string(),
            auth: AuthConfig::OAuth2 {
                client_id: "cid".to_string(),
                client_secret: "cs".to_string(),
                access_token: None,
                refresh_token: None,
            },
            namespace: "test".to_string(),
            settings: HashMap::new(),
            enabled: true,
        };
        let err = connector.validate_config(&config);
        assert!(err.is_err());
        assert!(err.unwrap_err().to_string().contains("access_token"));
    }

    #[test]
    fn test_validate_config_wrong_auth_type() {
        let connector = SlackConnector::new();
        let config = ConnectorConfig {
            id: "slack-test".to_string(),
            name: "Test Slack".to_string(),
            connector_type: "slack".to_string(),
            auth: AuthConfig::Token {
                token: "some-token".to_string(),
            },
            namespace: "test".to_string(),
            settings: HashMap::new(),
            enabled: true,
        };
        let err = connector.validate_config(&config);
        assert!(err.is_err());
        assert!(err.unwrap_err().to_string().contains("OAuth2"));
    }

    #[test]
    fn test_get_channels_from_settings() {
        let config = make_config("token", vec!["C01ABC", "C02DEF"]);
        let channels = SlackConnector::get_channels(&config);
        assert_eq!(channels, vec!["C01ABC", "C02DEF"]);
    }

    #[test]
    fn test_get_channels_empty_settings() {
        let config = make_config("token", vec![]);
        let channels = SlackConnector::get_channels(&config);
        assert!(channels.is_empty());
    }

    #[test]
    fn test_message_to_content_item_basic() {
        let connector = SlackConnector::new();
        let msg = SlackMessage {
            ts: "1700000000.123456".to_string(),
            text: Some("Hello, world!".to_string()),
            user: Some("U01ABC".to_string()),
            subtype: None,
            thread_ts: None,
        };

        let item = connector
            .message_to_content_item(&msg, "C01DEF", "general", "conn-1", Some("Alice"))
            .expect("should produce content item");

        assert_eq!(item.content, "[#general] Hello, world!");
        assert_eq!(item.source.connector_type, "slack");
        assert_eq!(item.source.connector_id, "conn-1");
        assert_eq!(item.source.source_id, "1700000000.123456");
        assert_eq!(item.source.author.as_deref(), Some("Alice"));
        assert_eq!(item.source.created_at, Some(1700000000));
        assert!(item.source.source_url.unwrap().contains("C01DEF"));
        assert!(item.media.is_none());
    }

    #[test]
    fn test_message_to_content_item_empty_text() {
        let connector = SlackConnector::new();
        let msg = SlackMessage {
            ts: "1700000000.000000".to_string(),
            text: Some("".to_string()),
            user: None,
            subtype: None,
            thread_ts: None,
        };

        let item = connector.message_to_content_item(&msg, "C01", "general", "conn-1", None);
        assert!(item.is_none());
    }

    #[test]
    fn test_message_to_content_item_no_text() {
        let connector = SlackConnector::new();
        let msg = SlackMessage {
            ts: "1700000000.000000".to_string(),
            text: None,
            user: None,
            subtype: Some("channel_join".to_string()),
            thread_ts: None,
        };

        let item = connector.message_to_content_item(&msg, "C01", "general", "conn-1", None);
        assert!(item.is_none());
    }

    #[test]
    fn test_message_to_content_item_with_thread() {
        let connector = SlackConnector::new();
        let msg = SlackMessage {
            ts: "1700000001.000000".to_string(),
            text: Some("Thread reply".to_string()),
            user: Some("U02".to_string()),
            subtype: None,
            thread_ts: Some("1700000000.000000".to_string()),
        };

        let item = connector
            .message_to_content_item(&msg, "C01", "general", "conn-1", None)
            .unwrap();

        assert!(item
            .source
            .extra
            .get("thread_ts")
            .is_some());
    }

    #[test]
    fn test_source_url_format() {
        let connector = SlackConnector::new();
        let msg = SlackMessage {
            ts: "1700000000.123456".to_string(),
            text: Some("test".to_string()),
            user: None,
            subtype: None,
            thread_ts: None,
        };

        let item = connector
            .message_to_content_item(&msg, "C01ABC", "general", "conn-1", None)
            .unwrap();

        assert_eq!(
            item.source.source_url.as_deref(),
            Some("https://slack.com/archives/C01ABC/p1700000000123456")
        );
    }

    #[tokio::test]
    async fn test_handle_webhook_url_verification() {
        let connector = SlackConnector::new();
        let config = make_config("token", vec![]);
        let payload = WebhookPayload {
            body: serde_json::to_vec(&serde_json::json!({
                "type": "url_verification",
                "challenge": "abc123"
            }))
            .unwrap(),
            headers: HashMap::new(),
            content_type: Some("application/json".to_string()),
        };

        let items = connector.handle_webhook(&config, payload).await.unwrap();
        assert!(items.is_empty());
    }

    #[tokio::test]
    async fn test_handle_webhook_message_event() {
        let connector = SlackConnector::new();
        let config = make_config("token", vec![]);
        let payload = WebhookPayload {
            body: serde_json::to_vec(&serde_json::json!({
                "event": {
                    "type": "message",
                    "text": "Hello from webhook",
                    "user": "U01ABC",
                    "channel": "C01DEF",
                    "ts": "1700000000.123456"
                }
            }))
            .unwrap(),
            headers: HashMap::new(),
            content_type: Some("application/json".to_string()),
        };

        let items = connector.handle_webhook(&config, payload).await.unwrap();
        assert_eq!(items.len(), 1);
        assert_eq!(items[0].content, "[#C01DEF] Hello from webhook");
        assert_eq!(items[0].source.author.as_deref(), Some("U01ABC"));
        assert_eq!(items[0].source.created_at, Some(1700000000));
    }

    #[tokio::test]
    async fn test_handle_webhook_non_message_event() {
        let connector = SlackConnector::new();
        let config = make_config("token", vec![]);
        let payload = WebhookPayload {
            body: serde_json::to_vec(&serde_json::json!({
                "event": {
                    "type": "reaction_added",
                    "user": "U01ABC",
                    "reaction": "thumbsup"
                }
            }))
            .unwrap(),
            headers: HashMap::new(),
            content_type: Some("application/json".to_string()),
        };

        let items = connector.handle_webhook(&config, payload).await.unwrap();
        assert!(items.is_empty());
    }

    #[tokio::test]
    async fn test_handle_webhook_message_subtype_skipped() {
        let connector = SlackConnector::new();
        let config = make_config("token", vec![]);
        let payload = WebhookPayload {
            body: serde_json::to_vec(&serde_json::json!({
                "event": {
                    "type": "message",
                    "subtype": "message_changed",
                    "text": "Edited message",
                    "channel": "C01",
                    "ts": "1700000000.000000"
                }
            }))
            .unwrap(),
            headers: HashMap::new(),
            content_type: Some("application/json".to_string()),
        };

        let items = connector.handle_webhook(&config, payload).await.unwrap();
        assert!(items.is_empty());
    }

    #[tokio::test]
    async fn test_handle_webhook_invalid_json() {
        let connector = SlackConnector::new();
        let config = make_config("token", vec![]);
        let payload = WebhookPayload {
            body: b"not json".to_vec(),
            headers: HashMap::new(),
            content_type: None,
        };

        let result = connector.handle_webhook(&config, payload).await;
        assert!(result.is_err());
    }

    #[test]
    fn test_default_constructor() {
        let connector = SlackConnector::default();
        assert_eq!(connector.id(), "slack");
    }
}
