//! Telegram connector — fetches messages from Telegram chats via the Bot API.
//!
//! Uses a Bot Token to authenticate with the Telegram Bot API.
//! Supports full sync (recent messages from configured chats) and
//! incremental sync via offset-based pagination (update_id).

use std::collections::HashMap;

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};

use crate::connector::{
    AuthConfig, Connector, ConnectorConfig, ContentItem, SourceMetadata, SyncCursor, SyncResult,
    WebhookPayload,
};

const TELEGRAM_API_BASE: &str = "https://api.telegram.org";

/// Telegram connector for fetching messages from chats and groups.
///
/// Requires a Bot Token obtained from [@BotFather](https://t.me/BotFather).
/// The bot must be added to the target chats/groups to read messages.
///
/// # Settings
///
/// - `chat_ids` (required): array of chat ID strings (numeric IDs or `@channel_username`)
/// - `max_messages`: maximum messages per chat (default: 1000)
/// - `include_service_messages`: whether to include service messages like member joins (default: false)
pub struct TelegramConnector {
    client: reqwest::Client,
}

impl TelegramConnector {
    /// Creates a new TelegramConnector with a default HTTP client.
    pub fn new() -> Self {
        Self {
            client: reqwest::Client::builder()
                .user_agent("ucotron-connector/0.1")
                .build()
                .unwrap_or_else(|_| reqwest::Client::new()),
        }
    }

    /// Creates a new TelegramConnector with a custom HTTP client.
    pub fn with_client(client: reqwest::Client) -> Self {
        Self { client }
    }

    /// Extracts the bot token from the connector config.
    fn get_token(config: &ConnectorConfig) -> Result<&str> {
        match &config.auth {
            AuthConfig::BotToken { token } => Ok(token.as_str()),
            _ => bail!("Telegram connector requires BotToken authentication"),
        }
    }

    /// Extracts configured chat IDs from settings.
    fn get_chat_ids(config: &ConnectorConfig) -> Vec<String> {
        config
            .settings
            .get("chat_ids")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| {
                        v.as_str()
                            .map(String::from)
                            .or_else(|| v.as_i64().map(|n| n.to_string()))
                    })
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Maximum number of messages to fetch per chat (default: 1000).
    fn max_messages(config: &ConnectorConfig) -> usize {
        config
            .settings
            .get("max_messages")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .unwrap_or(1000)
    }

    /// Whether to include service messages (default: false).
    fn include_service_messages(config: &ConnectorConfig) -> bool {
        config
            .settings
            .get("include_service_messages")
            .and_then(|v| v.as_bool())
            .unwrap_or(false)
    }

    /// Builds the API URL for a given method.
    fn api_url(token: &str, method: &str) -> String {
        format!("{}/bot{}/{}", TELEGRAM_API_BASE, token, method)
    }

    /// Calls getUpdates to fetch messages via long polling.
    /// Uses `offset` to paginate (returns updates with update_id >= offset).
    async fn get_updates(
        &self,
        token: &str,
        offset: Option<i64>,
        limit: usize,
    ) -> Result<Vec<TelegramUpdate>> {
        let url = Self::api_url(token, "getUpdates");

        let mut params = HashMap::new();
        params.insert("limit", serde_json::json!(limit.min(100)));
        params.insert("timeout", serde_json::json!(0)); // Non-blocking
        if let Some(off) = offset {
            params.insert("offset", serde_json::json!(off));
        }

        let resp = self
            .client
            .post(&url)
            .json(&params)
            .send()
            .await
            .context("Failed to call Telegram getUpdates")?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            bail!("Telegram API error in getUpdates: {} - {}", status, body);
        }

        let response: TelegramResponse<Vec<TelegramUpdate>> = resp
            .json()
            .await
            .context("Failed to parse Telegram getUpdates response")?;

        if !response.ok {
            bail!(
                "Telegram API returned error: {}",
                response.description.unwrap_or_default()
            );
        }

        Ok(response.result.unwrap_or_default())
    }

    /// Fetches chat history using getChatHistory-like approach.
    /// Since the Bot API doesn't have getChatHistory, we use getUpdates
    /// which only returns messages sent after the bot was added.
    /// For channels/supergroups, the bot must be an admin.
    async fn fetch_chat_messages(
        &self,
        token: &str,
        chat_id: &str,
        after_update_id: Option<i64>,
        max: usize,
    ) -> Result<Vec<(TelegramMessage, i64)>> {
        let mut all_messages = Vec::new();
        let mut current_offset = after_update_id.map(|id| id + 1);

        loop {
            if all_messages.len() >= max {
                break;
            }

            let remaining = max - all_messages.len();
            let limit = remaining.min(100);

            let updates = self.get_updates(token, current_offset, limit).await?;

            if updates.is_empty() {
                break;
            }

            let batch_count = updates.len();

            // Track offset for next page
            if let Some(last) = updates.last() {
                current_offset = Some(last.update_id + 1);
            }

            // Filter to messages for the target chat
            for update in updates {
                if let Some(msg) = update.message.or(update.channel_post) {
                    let msg_chat_id = msg.chat.id.to_string();
                    let matches = msg_chat_id == chat_id
                        || msg
                            .chat
                            .username
                            .as_deref()
                            .map(|u| format!("@{}", u) == chat_id)
                            .unwrap_or(false);

                    if matches {
                        all_messages.push((msg, update.update_id));
                    }
                }
            }

            if batch_count < limit {
                break;
            }
        }

        Ok(all_messages)
    }

    /// Converts a Telegram message to a ContentItem.
    fn message_to_content_item(
        &self,
        message: &TelegramMessage,
        connector_id: &str,
    ) -> ContentItem {
        let content = self.extract_message_text(message);
        let chat_id_str = message.chat.id.to_string();

        let mut extra = HashMap::new();
        extra.insert(
            "chat_id".to_string(),
            serde_json::Value::String(chat_id_str.clone()),
        );
        extra.insert(
            "chat_type".to_string(),
            serde_json::Value::String(message.chat.chat_type.clone()),
        );
        if let Some(ref title) = message.chat.title {
            extra.insert(
                "chat_title".to_string(),
                serde_json::Value::String(title.clone()),
            );
        }
        if let Some(ref reply) = message.reply_to_message {
            extra.insert(
                "reply_to_message_id".to_string(),
                serde_json::Value::Number(reply.message_id.into()),
            );
        }
        if let Some(ref forward_from) = message.forward_from {
            extra.insert(
                "forwarded_from".to_string(),
                serde_json::Value::String(
                    forward_from
                        .username
                        .clone()
                        .unwrap_or_else(|| forward_from.first_name.clone()),
                ),
            );
        }
        if message.photo.is_some() {
            extra.insert("has_photo".to_string(), serde_json::Value::Bool(true));
        }
        if message.document.is_some() {
            extra.insert("has_document".to_string(), serde_json::Value::Bool(true));
        }

        let author = message.from.as_ref().map(|u| {
            if let Some(ref uname) = u.username {
                format!("@{}", uname)
            } else {
                u.first_name.clone()
            }
        });

        ContentItem {
            content,
            source: SourceMetadata {
                connector_type: "telegram".to_string(),
                connector_id: connector_id.to_string(),
                source_id: message.message_id.to_string(),
                source_url: None, // Telegram doesn't have web URLs for private messages
                author,
                created_at: Some(message.date as u64),
                extra,
            },
            media: None,
        }
    }

    /// Extracts text content from a Telegram message, including captions.
    fn extract_message_text(&self, message: &TelegramMessage) -> String {
        if let Some(ref text) = message.text {
            return text.clone();
        }
        if let Some(ref caption) = message.caption {
            return caption.clone();
        }
        // Service messages or media-only
        if message.photo.is_some() {
            return "[Telegram] Photo".to_string();
        }
        if message.document.is_some() {
            return "[Telegram] Document".to_string();
        }
        if message.sticker.is_some() {
            return "[Telegram] Sticker".to_string();
        }
        if message.voice.is_some() {
            return "[Telegram] Voice message".to_string();
        }
        if message.video.is_some() {
            return "[Telegram] Video".to_string();
        }
        if message.audio.is_some() {
            return "[Telegram] Audio".to_string();
        }
        if let Some(ref new_members) = message.new_chat_members {
            let names: Vec<&str> = new_members.iter().map(|u| u.first_name.as_str()).collect();
            return format!("[Telegram] {} joined the chat", names.join(", "));
        }
        if message.left_chat_member.is_some() {
            return "[Telegram] Member left the chat".to_string();
        }
        "[Telegram] Unknown message type".to_string()
    }

    /// Checks if a message is a service message (join, leave, etc.).
    fn is_service_message(message: &TelegramMessage) -> bool {
        message.new_chat_members.is_some()
            || message.left_chat_member.is_some()
            || message.new_chat_title.is_some()
            || message.new_chat_photo.is_some()
            || message.delete_chat_photo.unwrap_or(false)
            || message.group_chat_created.unwrap_or(false)
            || message.pinned_message.is_some()
    }
}

impl Default for TelegramConnector {
    fn default() -> Self {
        Self::new()
    }
}

impl Connector for TelegramConnector {
    fn id(&self) -> &str {
        "telegram"
    }

    fn name(&self) -> &str {
        "Telegram"
    }

    fn config_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "auth": {
                    "type": "object",
                    "description": "Telegram Bot Token from @BotFather",
                    "properties": {
                        "token": { "type": "string", "description": "Bot Token (e.g., 123456:ABC-DEF1234ghIkl-zyx57W2v1u123ew11)" }
                    },
                    "required": ["token"]
                },
                "settings": {
                    "type": "object",
                    "properties": {
                        "chat_ids": {
                            "type": "array",
                            "items": { "type": "string" },
                            "description": "Chat IDs to sync (numeric IDs or @channel_username)"
                        },
                        "max_messages": {
                            "type": "integer",
                            "description": "Maximum messages per chat (default: 1000)"
                        },
                        "include_service_messages": {
                            "type": "boolean",
                            "description": "Include service messages like joins/leaves (default: false)"
                        }
                    },
                    "required": ["chat_ids"]
                }
            },
            "required": ["auth", "settings"]
        })
    }

    fn validate_config(&self, config: &ConnectorConfig) -> Result<()> {
        if config.connector_type != "telegram" {
            bail!(
                "Invalid connector type '{}', expected 'telegram'",
                config.connector_type
            );
        }
        Self::get_token(config)?;
        let chat_ids = Self::get_chat_ids(config);
        if chat_ids.is_empty() {
            bail!("Telegram connector requires at least one chat_id in settings");
        }
        Ok(())
    }

    async fn fetch(&self, config: &ConnectorConfig) -> Result<Vec<ContentItem>> {
        let token = Self::get_token(config)?;
        let chat_ids = Self::get_chat_ids(config);
        let max = Self::max_messages(config);
        let include_service = Self::include_service_messages(config);

        let mut items = Vec::new();

        for chat_id in &chat_ids {
            match self.fetch_chat_messages(token, chat_id, None, max).await {
                Ok(messages) => {
                    for (msg, _update_id) in &messages {
                        if !include_service && Self::is_service_message(msg) {
                            continue;
                        }
                        items.push(self.message_to_content_item(msg, &config.id));
                    }
                }
                Err(e) => {
                    eprintln!(
                        "Warning: failed to fetch messages from chat {}: {}",
                        chat_id, e
                    );
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
        let chat_ids = Self::get_chat_ids(config);
        let max = Self::max_messages(config);
        let include_service = Self::include_service_messages(config);

        // Parse cursor as the last update_id
        let after_update_id = cursor.value.as_ref().and_then(|v| v.parse::<i64>().ok());

        let mut items = Vec::new();
        let mut latest_update_id = after_update_id;

        for chat_id in &chat_ids {
            if let Ok(messages) = self
                .fetch_chat_messages(token, chat_id, after_update_id, max)
                .await
            {
                for (msg, update_id) in &messages {
                    if !include_service && Self::is_service_message(msg) {
                        continue;
                    }

                    // Track latest update_id for cursor
                    if latest_update_id.map_or(true, |current| *update_id > current) {
                        latest_update_id = Some(*update_id);
                    }

                    items.push(self.message_to_content_item(msg, &config.id));
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
                value: latest_update_id.map(|id| id.to_string()),
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
        let body: serde_json::Value =
            serde_json::from_slice(&payload.body).context("Invalid webhook JSON body")?;

        // Telegram webhook sends an Update object directly
        let update: TelegramUpdate =
            serde_json::from_value(body).context("Failed to parse Telegram Update")?;

        let include_service = Self::include_service_messages(config);

        if let Some(msg) = update.message.or(update.channel_post) {
            if !include_service && Self::is_service_message(&msg) {
                return Ok(Vec::new());
            }
            Ok(vec![self.message_to_content_item(&msg, &config.id)])
        } else {
            Ok(Vec::new())
        }
    }
}

// --- Telegram Bot API response types ---

#[derive(Debug, Clone, Deserialize, Serialize)]
struct TelegramResponse<T> {
    ok: bool,
    description: Option<String>,
    result: Option<T>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
struct TelegramUpdate {
    update_id: i64,
    message: Option<TelegramMessage>,
    channel_post: Option<TelegramMessage>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
struct TelegramMessage {
    message_id: i64,
    date: i64,
    chat: TelegramChat,
    from: Option<TelegramUser>,
    text: Option<String>,
    caption: Option<String>,
    reply_to_message: Option<Box<TelegramReplyMessage>>,
    forward_from: Option<TelegramUser>,
    photo: Option<Vec<serde_json::Value>>,
    document: Option<serde_json::Value>,
    sticker: Option<serde_json::Value>,
    voice: Option<serde_json::Value>,
    video: Option<serde_json::Value>,
    audio: Option<serde_json::Value>,
    new_chat_members: Option<Vec<TelegramUser>>,
    left_chat_member: Option<TelegramUser>,
    new_chat_title: Option<String>,
    new_chat_photo: Option<Vec<serde_json::Value>>,
    delete_chat_photo: Option<bool>,
    group_chat_created: Option<bool>,
    pinned_message: Option<Box<TelegramReplyMessage>>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
struct TelegramChat {
    id: i64,
    #[serde(rename = "type")]
    chat_type: String,
    title: Option<String>,
    username: Option<String>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
struct TelegramUser {
    id: i64,
    is_bot: bool,
    first_name: String,
    last_name: Option<String>,
    username: Option<String>,
}

/// Simplified reply message to avoid recursive types.
#[derive(Debug, Clone, Deserialize, Serialize)]
struct TelegramReplyMessage {
    message_id: i64,
    date: i64,
    chat: TelegramChat,
    from: Option<TelegramUser>,
    text: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn make_config(token: &str, chat_ids: Vec<&str>) -> ConnectorConfig {
        let mut settings = HashMap::new();
        settings.insert("chat_ids".to_string(), serde_json::json!(chat_ids));
        ConnectorConfig {
            id: "telegram-test".to_string(),
            name: "Test Telegram".to_string(),
            connector_type: "telegram".to_string(),
            auth: AuthConfig::BotToken {
                token: token.to_string(),
            },
            namespace: "test".to_string(),
            settings,
            enabled: true,
        }
    }

    fn make_message(
        msg_id: i64,
        text: Option<&str>,
        username: Option<&str>,
        chat_id: i64,
        chat_type: &str,
    ) -> TelegramMessage {
        TelegramMessage {
            message_id: msg_id,
            date: 1700000000,
            chat: TelegramChat {
                id: chat_id,
                chat_type: chat_type.to_string(),
                title: Some("Test Chat".to_string()),
                username: None,
            },
            from: Some(TelegramUser {
                id: 12345,
                is_bot: false,
                first_name: "TestUser".to_string(),
                last_name: None,
                username: username.map(String::from),
            }),
            text: text.map(String::from),
            caption: None,
            reply_to_message: None,
            forward_from: None,
            photo: None,
            document: None,
            sticker: None,
            voice: None,
            video: None,
            audio: None,
            new_chat_members: None,
            left_chat_member: None,
            new_chat_title: None,
            new_chat_photo: None,
            delete_chat_photo: None,
            group_chat_created: None,
            pinned_message: None,
        }
    }

    #[test]
    fn test_telegram_connector_id_and_name() {
        let connector = TelegramConnector::new();
        assert_eq!(connector.id(), "telegram");
        assert_eq!(connector.name(), "Telegram");
    }

    #[test]
    fn test_telegram_config_schema() {
        let connector = TelegramConnector::new();
        let schema = connector.config_schema();
        assert_eq!(schema["type"], "object");
        assert!(schema["properties"]["auth"].is_object());
        assert!(schema["properties"]["settings"]["properties"]["chat_ids"].is_object());
        assert!(schema["properties"]["settings"]["properties"]["max_messages"].is_object());
        assert!(
            schema["properties"]["settings"]["properties"]["include_service_messages"].is_object()
        );
    }

    #[test]
    fn test_validate_config_valid() {
        let connector = TelegramConnector::new();
        let config = make_config("123456:ABC-DEF", vec!["-1001234567890"]);
        assert!(connector.validate_config(&config).is_ok());
    }

    #[test]
    fn test_validate_config_wrong_type() {
        let connector = TelegramConnector::new();
        let mut config = make_config("token", vec!["123"]);
        config.connector_type = "discord".to_string();
        assert!(connector.validate_config(&config).is_err());
    }

    #[test]
    fn test_validate_config_wrong_auth_type() {
        let connector = TelegramConnector::new();
        let config = ConnectorConfig {
            id: "tg-test".to_string(),
            name: "Test".to_string(),
            connector_type: "telegram".to_string(),
            auth: AuthConfig::Token {
                token: "not_a_bot_token".to_string(),
            },
            namespace: "test".to_string(),
            settings: {
                let mut s = HashMap::new();
                s.insert("chat_ids".to_string(), serde_json::json!(["123"]));
                s
            },
            enabled: true,
        };
        let err = connector.validate_config(&config);
        assert!(err.is_err());
        assert!(err.unwrap_err().to_string().contains("BotToken"));
    }

    #[test]
    fn test_validate_config_no_chat_ids() {
        let connector = TelegramConnector::new();
        let config = make_config("token", vec![]);
        let err = connector.validate_config(&config);
        assert!(err.is_err());
        assert!(err.unwrap_err().to_string().contains("chat_id"));
    }

    #[test]
    fn test_message_to_content_item_text() {
        let connector = TelegramConnector::new();
        let msg = make_message(42, Some("Hello from Telegram!"), Some("testuser"), -100123, "group");

        let item = connector.message_to_content_item(&msg, "conn-1");

        assert_eq!(item.content, "Hello from Telegram!");
        assert_eq!(item.source.connector_type, "telegram");
        assert_eq!(item.source.connector_id, "conn-1");
        assert_eq!(item.source.source_id, "42");
        assert_eq!(item.source.author.as_deref(), Some("@testuser"));
        assert_eq!(item.source.created_at, Some(1700000000));
        assert_eq!(
            item.source.extra.get("chat_id").unwrap(),
            &serde_json::Value::String("-100123".to_string())
        );
        assert_eq!(
            item.source.extra.get("chat_type").unwrap(),
            &serde_json::Value::String("group".to_string())
        );
        assert_eq!(
            item.source.extra.get("chat_title").unwrap(),
            &serde_json::Value::String("Test Chat".to_string())
        );
        assert!(item.source.source_url.is_none());
        assert!(item.media.is_none());
    }

    #[test]
    fn test_message_to_content_item_no_username() {
        let connector = TelegramConnector::new();
        let msg = make_message(43, Some("Hi"), None, -100123, "group");

        let item = connector.message_to_content_item(&msg, "conn-1");
        assert_eq!(item.source.author.as_deref(), Some("TestUser"));
    }

    #[test]
    fn test_message_to_content_item_caption() {
        let connector = TelegramConnector::new();
        let mut msg = make_message(44, None, Some("user"), -100123, "group");
        msg.caption = Some("Photo caption".to_string());
        msg.photo = Some(vec![serde_json::json!({"file_id": "abc"})]);

        let item = connector.message_to_content_item(&msg, "conn-1");
        assert_eq!(item.content, "Photo caption");
        assert_eq!(
            item.source.extra.get("has_photo").unwrap(),
            &serde_json::Value::Bool(true)
        );
    }

    #[test]
    fn test_message_to_content_item_photo_no_caption() {
        let connector = TelegramConnector::new();
        let mut msg = make_message(45, None, Some("user"), -100123, "group");
        msg.photo = Some(vec![serde_json::json!({"file_id": "abc"})]);

        let item = connector.message_to_content_item(&msg, "conn-1");
        assert_eq!(item.content, "[Telegram] Photo");
    }

    #[test]
    fn test_message_to_content_item_document() {
        let connector = TelegramConnector::new();
        let mut msg = make_message(46, None, Some("user"), -100123, "group");
        msg.document = Some(serde_json::json!({"file_id": "doc1", "file_name": "report.pdf"}));

        let item = connector.message_to_content_item(&msg, "conn-1");
        assert_eq!(item.content, "[Telegram] Document");
        assert_eq!(
            item.source.extra.get("has_document").unwrap(),
            &serde_json::Value::Bool(true)
        );
    }

    #[test]
    fn test_message_to_content_item_sticker() {
        let connector = TelegramConnector::new();
        let mut msg = make_message(47, None, Some("user"), -100123, "group");
        msg.sticker = Some(serde_json::json!({"file_id": "stk1"}));

        let item = connector.message_to_content_item(&msg, "conn-1");
        assert_eq!(item.content, "[Telegram] Sticker");
    }

    #[test]
    fn test_message_to_content_item_voice() {
        let connector = TelegramConnector::new();
        let mut msg = make_message(48, None, Some("user"), -100123, "group");
        msg.voice = Some(serde_json::json!({"file_id": "v1", "duration": 5}));

        let item = connector.message_to_content_item(&msg, "conn-1");
        assert_eq!(item.content, "[Telegram] Voice message");
    }

    #[test]
    fn test_message_to_content_item_with_reply() {
        let connector = TelegramConnector::new();
        let mut msg = make_message(49, Some("Reply text"), Some("user"), -100123, "group");
        msg.reply_to_message = Some(Box::new(TelegramReplyMessage {
            message_id: 40,
            date: 1699999900,
            chat: TelegramChat {
                id: -100123,
                chat_type: "group".to_string(),
                title: Some("Test Chat".to_string()),
                username: None,
            },
            from: None,
            text: Some("Original".to_string()),
        }));

        let item = connector.message_to_content_item(&msg, "conn-1");
        assert_eq!(item.content, "Reply text");
        assert_eq!(
            item.source.extra.get("reply_to_message_id").unwrap(),
            &serde_json::Value::Number(40.into())
        );
    }

    #[test]
    fn test_message_to_content_item_forwarded() {
        let connector = TelegramConnector::new();
        let mut msg = make_message(50, Some("Forwarded text"), Some("user"), -100123, "group");
        msg.forward_from = Some(TelegramUser {
            id: 99999,
            is_bot: false,
            first_name: "OrigAuthor".to_string(),
            last_name: None,
            username: Some("orig_author".to_string()),
        });

        let item = connector.message_to_content_item(&msg, "conn-1");
        assert_eq!(
            item.source.extra.get("forwarded_from").unwrap(),
            &serde_json::Value::String("orig_author".to_string())
        );
    }

    #[test]
    fn test_extract_message_text_new_members() {
        let connector = TelegramConnector::new();
        let mut msg = make_message(51, None, Some("user"), -100123, "group");
        msg.new_chat_members = Some(vec![TelegramUser {
            id: 111,
            is_bot: false,
            first_name: "Alice".to_string(),
            last_name: None,
            username: None,
        }]);

        let text = connector.extract_message_text(&msg);
        assert!(text.contains("Alice"));
        assert!(text.contains("joined the chat"));
    }

    #[test]
    fn test_extract_message_text_left_member() {
        let connector = TelegramConnector::new();
        let mut msg = make_message(52, None, Some("user"), -100123, "group");
        msg.left_chat_member = Some(TelegramUser {
            id: 111,
            is_bot: false,
            first_name: "Bob".to_string(),
            last_name: None,
            username: None,
        });

        let text = connector.extract_message_text(&msg);
        assert!(text.contains("left the chat"));
    }

    #[test]
    fn test_is_service_message() {
        let mut msg = make_message(53, None, Some("user"), -100123, "group");
        assert!(!TelegramConnector::is_service_message(&msg));

        msg.new_chat_members = Some(vec![]);
        assert!(TelegramConnector::is_service_message(&msg));
    }

    #[test]
    fn test_is_service_message_left_member() {
        let mut msg = make_message(54, None, Some("user"), -100123, "group");
        msg.left_chat_member = Some(TelegramUser {
            id: 1,
            is_bot: false,
            first_name: "X".to_string(),
            last_name: None,
            username: None,
        });
        assert!(TelegramConnector::is_service_message(&msg));
    }

    #[test]
    fn test_is_service_message_pinned() {
        let mut msg = make_message(55, None, Some("user"), -100123, "group");
        msg.pinned_message = Some(Box::new(TelegramReplyMessage {
            message_id: 10,
            date: 1700000000,
            chat: TelegramChat {
                id: -100123,
                chat_type: "group".to_string(),
                title: None,
                username: None,
            },
            from: None,
            text: Some("Pinned".to_string()),
        }));
        assert!(TelegramConnector::is_service_message(&msg));
    }

    #[test]
    fn test_get_chat_ids_strings() {
        let config = make_config("token", vec!["-1001234567890", "@mychannel"]);
        let ids = TelegramConnector::get_chat_ids(&config);
        assert_eq!(ids, vec!["-1001234567890", "@mychannel"]);
    }

    #[test]
    fn test_get_chat_ids_numeric() {
        let mut config = make_config("token", vec![]);
        config.settings.insert(
            "chat_ids".to_string(),
            serde_json::json!([-1001234567890i64, -1009876543210i64]),
        );
        let ids = TelegramConnector::get_chat_ids(&config);
        assert_eq!(ids, vec!["-1001234567890", "-1009876543210"]);
    }

    #[test]
    fn test_get_chat_ids_empty() {
        let mut config = make_config("token", vec![]);
        config.settings.remove("chat_ids");
        let ids = TelegramConnector::get_chat_ids(&config);
        assert!(ids.is_empty());
    }

    #[test]
    fn test_max_messages_default() {
        let config = make_config("token", vec!["123"]);
        assert_eq!(TelegramConnector::max_messages(&config), 1000);
    }

    #[test]
    fn test_max_messages_custom() {
        let mut config = make_config("token", vec!["123"]);
        config
            .settings
            .insert("max_messages".to_string(), serde_json::json!(500));
        assert_eq!(TelegramConnector::max_messages(&config), 500);
    }

    #[test]
    fn test_include_service_messages_default() {
        let config = make_config("token", vec!["123"]);
        assert!(!TelegramConnector::include_service_messages(&config));
    }

    #[test]
    fn test_include_service_messages_custom() {
        let mut config = make_config("token", vec!["123"]);
        config.settings.insert(
            "include_service_messages".to_string(),
            serde_json::json!(true),
        );
        assert!(TelegramConnector::include_service_messages(&config));
    }

    #[test]
    fn test_default_constructor() {
        let connector = TelegramConnector::default();
        assert_eq!(connector.id(), "telegram");
    }

    #[test]
    fn test_api_url() {
        let url = TelegramConnector::api_url("123456:ABC", "getUpdates");
        assert_eq!(url, "https://api.telegram.org/bot123456:ABC/getUpdates");
    }

    #[tokio::test]
    async fn test_handle_webhook_message() {
        let connector = TelegramConnector::new();
        let config = make_config("token", vec!["-100123"]);
        let payload = WebhookPayload {
            body: serde_json::to_vec(&serde_json::json!({
                "update_id": 12345678,
                "message": {
                    "message_id": 100,
                    "date": 1700000000,
                    "chat": {
                        "id": -100123,
                        "type": "group",
                        "title": "Test Group"
                    },
                    "from": {
                        "id": 999,
                        "is_bot": false,
                        "first_name": "WebhookUser",
                        "username": "whuser"
                    },
                    "text": "Hello from webhook!"
                }
            }))
            .unwrap(),
            headers: HashMap::new(),
            content_type: Some("application/json".to_string()),
        };

        let items = connector.handle_webhook(&config, payload).await.unwrap();
        assert_eq!(items.len(), 1);
        assert_eq!(items[0].content, "Hello from webhook!");
        assert_eq!(items[0].source.source_id, "100");
        assert_eq!(items[0].source.author.as_deref(), Some("@whuser"));
    }

    #[tokio::test]
    async fn test_handle_webhook_channel_post() {
        let connector = TelegramConnector::new();
        let config = make_config("token", vec!["@mychannel"]);
        let payload = WebhookPayload {
            body: serde_json::to_vec(&serde_json::json!({
                "update_id": 12345679,
                "channel_post": {
                    "message_id": 200,
                    "date": 1700000000,
                    "chat": {
                        "id": -1001234,
                        "type": "channel",
                        "title": "My Channel",
                        "username": "mychannel"
                    },
                    "text": "Channel announcement"
                }
            }))
            .unwrap(),
            headers: HashMap::new(),
            content_type: Some("application/json".to_string()),
        };

        let items = connector.handle_webhook(&config, payload).await.unwrap();
        assert_eq!(items.len(), 1);
        assert_eq!(items[0].content, "Channel announcement");
        assert_eq!(items[0].source.source_id, "200");
    }

    #[tokio::test]
    async fn test_handle_webhook_service_message_filtered() {
        let connector = TelegramConnector::new();
        let config = make_config("token", vec!["-100123"]);
        let payload = WebhookPayload {
            body: serde_json::to_vec(&serde_json::json!({
                "update_id": 12345680,
                "message": {
                    "message_id": 300,
                    "date": 1700000000,
                    "chat": {
                        "id": -100123,
                        "type": "group",
                        "title": "Test Group"
                    },
                    "from": {
                        "id": 111,
                        "is_bot": false,
                        "first_name": "NewUser"
                    },
                    "new_chat_members": [{
                        "id": 111,
                        "is_bot": false,
                        "first_name": "NewUser"
                    }]
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
    async fn test_handle_webhook_service_message_included() {
        let connector = TelegramConnector::new();
        let mut config = make_config("token", vec!["-100123"]);
        config.settings.insert(
            "include_service_messages".to_string(),
            serde_json::json!(true),
        );
        let payload = WebhookPayload {
            body: serde_json::to_vec(&serde_json::json!({
                "update_id": 12345681,
                "message": {
                    "message_id": 301,
                    "date": 1700000000,
                    "chat": {
                        "id": -100123,
                        "type": "group",
                        "title": "Test Group"
                    },
                    "from": {
                        "id": 111,
                        "is_bot": false,
                        "first_name": "NewUser"
                    },
                    "new_chat_members": [{
                        "id": 111,
                        "is_bot": false,
                        "first_name": "NewUser"
                    }]
                }
            }))
            .unwrap(),
            headers: HashMap::new(),
            content_type: Some("application/json".to_string()),
        };

        let items = connector.handle_webhook(&config, payload).await.unwrap();
        assert_eq!(items.len(), 1);
        assert!(items[0].content.contains("NewUser"));
    }

    #[tokio::test]
    async fn test_handle_webhook_non_message_update() {
        let connector = TelegramConnector::new();
        let config = make_config("token", vec!["-100123"]);
        let payload = WebhookPayload {
            body: serde_json::to_vec(&serde_json::json!({
                "update_id": 12345682,
                "callback_query": {
                    "id": "cb1",
                    "from": { "id": 1, "is_bot": false, "first_name": "X" },
                    "data": "button_click"
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
        let connector = TelegramConnector::new();
        let config = make_config("token", vec!["-100123"]);
        let payload = WebhookPayload {
            body: b"not json".to_vec(),
            headers: HashMap::new(),
            content_type: None,
        };

        let result = connector.handle_webhook(&config, payload).await;
        assert!(result.is_err());
    }

    #[test]
    fn test_video_message_text() {
        let connector = TelegramConnector::new();
        let mut msg = make_message(60, None, Some("user"), -100123, "group");
        msg.video = Some(serde_json::json!({"file_id": "vid1"}));

        let text = connector.extract_message_text(&msg);
        assert_eq!(text, "[Telegram] Video");
    }

    #[test]
    fn test_audio_message_text() {
        let connector = TelegramConnector::new();
        let mut msg = make_message(61, None, Some("user"), -100123, "group");
        msg.audio = Some(serde_json::json!({"file_id": "aud1"}));

        let text = connector.extract_message_text(&msg);
        assert_eq!(text, "[Telegram] Audio");
    }

    #[test]
    fn test_forwarded_without_username() {
        let connector = TelegramConnector::new();
        let mut msg = make_message(62, Some("Fwd"), Some("user"), -100123, "group");
        msg.forward_from = Some(TelegramUser {
            id: 88888,
            is_bot: false,
            first_name: "NoUsername".to_string(),
            last_name: None,
            username: None,
        });

        let item = connector.message_to_content_item(&msg, "conn-1");
        assert_eq!(
            item.source.extra.get("forwarded_from").unwrap(),
            &serde_json::Value::String("NoUsername".to_string())
        );
    }
}
