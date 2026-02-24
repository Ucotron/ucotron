//! Discord connector — fetches messages from Discord channels via the REST API.
//!
//! Uses a Bot Token to authenticate with the Discord API.
//! Supports full sync (all messages from configured channels) and
//! incremental sync via snowflake ID-based pagination.

use std::collections::HashMap;

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};

use crate::connector::{
    AuthConfig, Connector, ConnectorConfig, ContentItem, SourceMetadata, SyncCursor, SyncResult,
    WebhookPayload,
};

const DISCORD_API_BASE: &str = "https://discord.com/api/v10";

/// Discord connector for fetching messages from channels and guilds.
///
/// Requires a Bot Token with the following intents/permissions:
/// - `MESSAGE_CONTENT` intent for reading message content
/// - `Read Message History` permission in target channels
/// - `View Channel` permission in target guilds
pub struct DiscordConnector {
    client: reqwest::Client,
}

impl DiscordConnector {
    /// Creates a new DiscordConnector with a default HTTP client.
    pub fn new() -> Self {
        Self {
            client: reqwest::Client::builder()
                .user_agent("ucotron-connector/0.1 (Bot)")
                .build()
                .unwrap_or_else(|_| reqwest::Client::new()),
        }
    }

    /// Creates a new DiscordConnector with a custom HTTP client.
    pub fn with_client(client: reqwest::Client) -> Self {
        Self { client }
    }

    /// Extracts the bot token from the connector config.
    fn get_token(config: &ConnectorConfig) -> Result<&str> {
        match &config.auth {
            AuthConfig::BotToken { token } => Ok(token.as_str()),
            _ => bail!("Discord connector requires BotToken authentication"),
        }
    }

    /// Extracts configured guild IDs from settings.
    fn get_guild_ids(config: &ConnectorConfig) -> Vec<String> {
        config
            .settings
            .get("guild_ids")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str().map(String::from))
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Extracts configured channel IDs from settings.
    fn get_channel_ids(config: &ConnectorConfig) -> Vec<String> {
        config
            .settings
            .get("channel_ids")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str().map(String::from))
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Checks if thread messages should be fetched (default: false).
    fn include_threads(config: &ConnectorConfig) -> bool {
        config
            .settings
            .get("include_threads")
            .and_then(|v| v.as_bool())
            .unwrap_or(false)
    }

    /// Maximum number of messages to fetch per channel (default: 1000).
    fn max_messages(config: &ConnectorConfig) -> usize {
        config
            .settings
            .get("max_messages")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .unwrap_or(1000)
    }

    /// Fetches text channels from a guild.
    async fn fetch_guild_channels(
        &self,
        token: &str,
        guild_id: &str,
    ) -> Result<Vec<DiscordChannel>> {
        let url = format!("{}/guilds/{}/channels", DISCORD_API_BASE, guild_id);

        let resp = self
            .client
            .get(&url)
            .header("Authorization", format!("Bot {}", token))
            .send()
            .await
            .context("Failed to fetch Discord guild channels")?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            bail!(
                "Discord API error fetching guild {} channels: {} - {}",
                guild_id,
                status,
                body
            );
        }

        let channels: Vec<DiscordChannel> = resp
            .json()
            .await
            .context("Failed to parse Discord channels response")?;

        // Filter to text channels (type 0) and announcement channels (type 5)
        Ok(channels
            .into_iter()
            .filter(|c| c.channel_type == 0 || c.channel_type == 5)
            .collect())
    }

    /// Fetches messages from a channel with pagination.
    /// When `after` is provided, only returns messages after that snowflake ID.
    async fn fetch_channel_messages(
        &self,
        token: &str,
        channel_id: &str,
        after: Option<&str>,
        max: usize,
    ) -> Result<Vec<DiscordMessage>> {
        let mut all_messages = Vec::new();
        let mut current_after = after.map(String::from);

        loop {
            if all_messages.len() >= max {
                break;
            }

            let remaining = max - all_messages.len();
            let limit = remaining.min(100);

            let mut url = format!(
                "{}/channels/{}/messages?limit={}",
                DISCORD_API_BASE, channel_id, limit
            );
            if let Some(ref after_id) = current_after {
                url.push_str(&format!("&after={}", after_id));
            }

            let resp = self
                .client
                .get(&url)
                .header("Authorization", format!("Bot {}", token))
                .send()
                .await
                .context("Failed to fetch Discord messages")?;

            if !resp.status().is_success() {
                let status = resp.status();
                let body = resp.text().await.unwrap_or_default();
                bail!(
                    "Discord API error fetching messages from {}: {} - {}",
                    channel_id,
                    status,
                    body
                );
            }

            let mut messages: Vec<DiscordMessage> = resp
                .json()
                .await
                .context("Failed to parse Discord messages response")?;

            let count = messages.len();

            // Discord returns newest first; sort oldest first for consistent cursor tracking
            messages.sort_by(|a, b| a.id.cmp(&b.id));

            if let Some(last) = messages.last() {
                current_after = Some(last.id.clone());
            }

            all_messages.extend(messages);

            if count < limit {
                break;
            }
        }

        Ok(all_messages)
    }

    /// Fetches active threads in a channel.
    async fn fetch_active_threads(
        &self,
        token: &str,
        guild_id: &str,
    ) -> Result<Vec<DiscordChannel>> {
        let url = format!("{}/guilds/{}/threads/active", DISCORD_API_BASE, guild_id);

        let resp = self
            .client
            .get(&url)
            .header("Authorization", format!("Bot {}", token))
            .send()
            .await
            .context("Failed to fetch Discord active threads")?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            bail!(
                "Discord API error fetching threads for guild {}: {} - {}",
                guild_id,
                status,
                body
            );
        }

        let response: ThreadListResponse = resp
            .json()
            .await
            .context("Failed to parse Discord threads response")?;

        Ok(response.threads)
    }

    /// Converts a Discord message to a ContentItem.
    fn message_to_content_item(
        &self,
        message: &DiscordMessage,
        channel_id: &str,
        guild_id: Option<&str>,
        connector_id: &str,
    ) -> ContentItem {
        let content = if message.content.is_empty() {
            // For embeds or attachments with no text
            format!("[Discord] {} (embed/attachment)", message.author.username)
        } else {
            message.content.clone()
        };

        let created_at = parse_discord_timestamp(&message.timestamp);

        let mut extra = HashMap::new();
        extra.insert(
            "channel_id".to_string(),
            serde_json::Value::String(channel_id.to_string()),
        );
        if let Some(gid) = guild_id {
            extra.insert(
                "guild_id".to_string(),
                serde_json::Value::String(gid.to_string()),
            );
        }
        if !message.attachments.is_empty() {
            extra.insert(
                "attachment_count".to_string(),
                serde_json::Value::Number(message.attachments.len().into()),
            );
        }
        if !message.embeds.is_empty() {
            extra.insert(
                "embed_count".to_string(),
                serde_json::Value::Number(message.embeds.len().into()),
            );
        }
        if message.message_type != 0 {
            extra.insert(
                "message_type".to_string(),
                serde_json::Value::Number(message.message_type.into()),
            );
        }

        let source_url = guild_id.map(|gid| {
            format!(
                "https://discord.com/channels/{}/{}/{}",
                gid, channel_id, message.id
            )
        });

        ContentItem {
            content,
            source: SourceMetadata {
                connector_type: "discord".to_string(),
                connector_id: connector_id.to_string(),
                source_id: message.id.clone(),
                source_url,
                author: Some(message.author.username.clone()),
                created_at,
                extra,
            },
            media: None,
        }
    }
}

impl Default for DiscordConnector {
    fn default() -> Self {
        Self::new()
    }
}

impl Connector for DiscordConnector {
    fn id(&self) -> &str {
        "discord"
    }

    fn name(&self) -> &str {
        "Discord"
    }

    fn config_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "auth": {
                    "type": "object",
                    "description": "Discord Bot Token credentials",
                    "properties": {
                        "token": { "type": "string", "description": "Discord Bot Token" }
                    },
                    "required": ["token"]
                },
                "settings": {
                    "type": "object",
                    "properties": {
                        "guild_ids": {
                            "type": "array",
                            "items": { "type": "string" },
                            "description": "Guild (server) IDs to sync"
                        },
                        "channel_ids": {
                            "type": "array",
                            "items": { "type": "string" },
                            "description": "Specific channel IDs to sync (overrides guild-level discovery)"
                        },
                        "include_threads": {
                            "type": "boolean",
                            "description": "Whether to fetch thread messages (default: false)"
                        },
                        "max_messages": {
                            "type": "integer",
                            "description": "Maximum messages per channel (default: 1000)"
                        }
                    }
                }
            },
            "required": ["auth", "settings"]
        })
    }

    fn validate_config(&self, config: &ConnectorConfig) -> Result<()> {
        if config.connector_type != "discord" {
            bail!(
                "Invalid connector type '{}', expected 'discord'",
                config.connector_type
            );
        }
        Self::get_token(config)?;
        let guilds = Self::get_guild_ids(config);
        let channels = Self::get_channel_ids(config);
        if guilds.is_empty() && channels.is_empty() {
            bail!("Discord connector requires at least one guild_id or channel_id in settings");
        }
        Ok(())
    }

    async fn fetch(&self, config: &ConnectorConfig) -> Result<Vec<ContentItem>> {
        let token = Self::get_token(config)?;
        let guild_ids = Self::get_guild_ids(config);
        let explicit_channel_ids = Self::get_channel_ids(config);
        let fetch_threads = Self::include_threads(config);
        let max = Self::max_messages(config);

        let mut items = Vec::new();

        // Collect all channel IDs to fetch from
        let mut channels_to_fetch: Vec<(String, Option<String>)> = Vec::new(); // (channel_id, guild_id)

        // Add explicit channel IDs (no guild context)
        for ch_id in &explicit_channel_ids {
            channels_to_fetch.push((ch_id.clone(), None));
        }

        // Discover channels from guilds
        for guild_id in &guild_ids {
            match self.fetch_guild_channels(token, guild_id).await {
                Ok(channels) => {
                    for ch in &channels {
                        channels_to_fetch.push((ch.id.clone(), Some(guild_id.clone())));
                    }

                    // Fetch active threads if enabled
                    if fetch_threads {
                        if let Ok(threads) = self.fetch_active_threads(token, guild_id).await {
                            for thread in &threads {
                                channels_to_fetch.push((thread.id.clone(), Some(guild_id.clone())));
                            }
                        }
                    }
                }
                Err(e) => {
                    eprintln!(
                        "Warning: failed to fetch channels for guild {}: {}",
                        guild_id, e
                    );
                }
            }
        }

        // Fetch messages from all channels
        for (channel_id, guild_id) in &channels_to_fetch {
            match self
                .fetch_channel_messages(token, channel_id, None, max)
                .await
            {
                Ok(messages) => {
                    for msg in &messages {
                        // Skip bot messages and system messages by default
                        if msg.author.bot.unwrap_or(false) {
                            continue;
                        }
                        items.push(self.message_to_content_item(
                            msg,
                            channel_id,
                            guild_id.as_deref(),
                            &config.id,
                        ));
                    }
                }
                Err(e) => {
                    eprintln!(
                        "Warning: failed to fetch messages from channel {}: {}",
                        channel_id, e
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
        let guild_ids = Self::get_guild_ids(config);
        let explicit_channel_ids = Self::get_channel_ids(config);
        let fetch_threads = Self::include_threads(config);
        let max = Self::max_messages(config);

        // Use cursor value as the last message snowflake ID
        let after = cursor.value.as_deref();

        let mut items = Vec::new();
        let mut latest_id: Option<String> = cursor.value.clone();

        // Collect channels
        let mut channels_to_fetch: Vec<(String, Option<String>)> = Vec::new();

        for ch_id in &explicit_channel_ids {
            channels_to_fetch.push((ch_id.clone(), None));
        }

        for guild_id in &guild_ids {
            if let Ok(channels) = self.fetch_guild_channels(token, guild_id).await {
                for ch in &channels {
                    channels_to_fetch.push((ch.id.clone(), Some(guild_id.clone())));
                }
                if fetch_threads {
                    if let Ok(threads) = self.fetch_active_threads(token, guild_id).await {
                        for thread in &threads {
                            channels_to_fetch.push((thread.id.clone(), Some(guild_id.clone())));
                        }
                    }
                }
            }
        }

        for (channel_id, guild_id) in &channels_to_fetch {
            if let Ok(messages) = self
                .fetch_channel_messages(token, channel_id, after, max)
                .await
            {
                for msg in &messages {
                    if msg.author.bot.unwrap_or(false) {
                        continue;
                    }

                    // Track the latest message ID for cursor
                    if latest_id.as_ref().is_none_or(|current| &msg.id > current) {
                        latest_id = Some(msg.id.clone());
                    }

                    items.push(self.message_to_content_item(
                        msg,
                        channel_id,
                        guild_id.as_deref(),
                        &config.id,
                    ));
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
                value: latest_id,
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

        // Discord Interactions/Webhooks send a `type` field
        // Type 1 = PING (respond with type 1 for verification)
        // For gateway events forwarded via webhook, check `t` field
        let event_type = body.get("t").and_then(|v| v.as_str()).unwrap_or_default();

        match event_type {
            "MESSAGE_CREATE" => {
                let data = body
                    .get("d")
                    .context("Missing 'd' field in Discord event")?;

                let message: DiscordMessage = serde_json::from_value(data.clone())
                    .context("Failed to parse Discord message from webhook")?;

                // Skip bot messages
                if message.author.bot.unwrap_or(false) {
                    return Ok(Vec::new());
                }

                let channel_id = data
                    .get("channel_id")
                    .and_then(|v| v.as_str())
                    .unwrap_or_default();
                let guild_id = data.get("guild_id").and_then(|v| v.as_str());

                Ok(vec![self.message_to_content_item(
                    &message, channel_id, guild_id, &config.id,
                )])
            }
            _ => Ok(Vec::new()),
        }
    }
}

/// Parses a Discord ISO 8601 timestamp (e.g., "2024-06-15T10:30:00.000000+00:00") to Unix seconds.
fn parse_discord_timestamp(ts: &str) -> Option<u64> {
    // Discord timestamps can be "2024-06-15T10:30:00.000000+00:00" or "2024-06-15T10:30:00Z"
    // Strip everything after seconds (fractional seconds and timezone)
    let ts = ts.split('.').next().unwrap_or(ts);
    let ts = ts.split('+').next().unwrap_or(ts);
    let ts = ts.trim_end_matches('Z');

    let parts: Vec<&str> = ts.split('T').collect();
    if parts.len() != 2 {
        return None;
    }

    let date_parts: Vec<u64> = parts[0].split('-').filter_map(|s| s.parse().ok()).collect();
    let time_parts: Vec<u64> = parts[1].split(':').filter_map(|s| s.parse().ok()).collect();

    if date_parts.len() != 3 || time_parts.len() != 3 {
        return None;
    }

    let (year, month, day) = (date_parts[0], date_parts[1], date_parts[2]);
    let (hour, min, sec) = (time_parts[0], time_parts[1], time_parts[2]);

    let mut days: i64 = 0;
    for y in 1970..year {
        days += if is_leap_year(y) { 366 } else { 365 };
    }

    let month_days = [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];
    for m in 1..month {
        days += month_days[m as usize] as i64;
        if m == 2 && is_leap_year(year) {
            days += 1;
        }
    }
    days += day as i64 - 1;

    let total_secs = days as u64 * 86400 + hour * 3600 + min * 60 + sec;
    Some(total_secs)
}

fn is_leap_year(year: u64) -> bool {
    (year.is_multiple_of(4) && !year.is_multiple_of(100)) || year.is_multiple_of(400)
}

// --- Discord API response types ---

#[derive(Debug, Clone, Deserialize, Serialize)]
struct DiscordChannel {
    id: String,
    #[serde(rename = "type")]
    channel_type: u8,
    name: Option<String>,
    guild_id: Option<String>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
struct DiscordMessage {
    id: String,
    content: String,
    timestamp: String,
    author: DiscordUser,
    #[serde(default)]
    attachments: Vec<DiscordAttachment>,
    #[serde(default)]
    embeds: Vec<serde_json::Value>,
    #[serde(default, rename = "type")]
    message_type: u8,
    channel_id: Option<String>,
    guild_id: Option<String>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
struct DiscordUser {
    id: String,
    username: String,
    #[serde(default)]
    bot: Option<bool>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
struct DiscordAttachment {
    id: String,
    filename: String,
    url: String,
    size: u64,
    content_type: Option<String>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
struct ThreadListResponse {
    threads: Vec<DiscordChannel>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn make_config(token: &str, guild_ids: Vec<&str>, channel_ids: Vec<&str>) -> ConnectorConfig {
        let mut settings = HashMap::new();
        if !guild_ids.is_empty() {
            settings.insert("guild_ids".to_string(), serde_json::json!(guild_ids));
        }
        if !channel_ids.is_empty() {
            settings.insert("channel_ids".to_string(), serde_json::json!(channel_ids));
        }
        ConnectorConfig {
            id: "discord-test".to_string(),
            name: "Test Discord".to_string(),
            connector_type: "discord".to_string(),
            auth: AuthConfig::BotToken {
                token: token.to_string(),
            },
            namespace: "test".to_string(),
            settings,
            enabled: true,
        }
    }

    fn make_message(id: &str, content: &str, username: &str, is_bot: bool) -> DiscordMessage {
        DiscordMessage {
            id: id.to_string(),
            content: content.to_string(),
            timestamp: "2024-06-15T10:30:00.000000+00:00".to_string(),
            author: DiscordUser {
                id: "user123".to_string(),
                username: username.to_string(),
                bot: Some(is_bot),
            },
            attachments: Vec::new(),
            embeds: Vec::new(),
            message_type: 0,
            channel_id: Some("ch123".to_string()),
            guild_id: Some("guild456".to_string()),
        }
    }

    #[test]
    fn test_discord_connector_id_and_name() {
        let connector = DiscordConnector::new();
        assert_eq!(connector.id(), "discord");
        assert_eq!(connector.name(), "Discord");
    }

    #[test]
    fn test_discord_config_schema() {
        let connector = DiscordConnector::new();
        let schema = connector.config_schema();
        assert_eq!(schema["type"], "object");
        assert!(schema["properties"]["auth"].is_object());
        assert!(schema["properties"]["settings"]["properties"]["guild_ids"].is_object());
        assert!(schema["properties"]["settings"]["properties"]["channel_ids"].is_object());
        assert!(schema["properties"]["settings"]["properties"]["include_threads"].is_object());
        assert!(schema["properties"]["settings"]["properties"]["max_messages"].is_object());
    }

    #[test]
    fn test_validate_config_valid_guild() {
        let connector = DiscordConnector::new();
        let config = make_config("bot_token_123", vec!["123456789"], vec![]);
        assert!(connector.validate_config(&config).is_ok());
    }

    #[test]
    fn test_validate_config_valid_channel() {
        let connector = DiscordConnector::new();
        let config = make_config("bot_token_123", vec![], vec!["987654321"]);
        assert!(connector.validate_config(&config).is_ok());
    }

    #[test]
    fn test_validate_config_wrong_type() {
        let connector = DiscordConnector::new();
        let mut config = make_config("bot_token_123", vec!["123"], vec![]);
        config.connector_type = "slack".to_string();
        assert!(connector.validate_config(&config).is_err());
    }

    #[test]
    fn test_validate_config_wrong_auth_type() {
        let connector = DiscordConnector::new();
        let config = ConnectorConfig {
            id: "dc-test".to_string(),
            name: "Test".to_string(),
            connector_type: "discord".to_string(),
            auth: AuthConfig::Token {
                token: "not_a_bot_token".to_string(),
            },
            namespace: "test".to_string(),
            settings: {
                let mut s = HashMap::new();
                s.insert("guild_ids".to_string(), serde_json::json!(["123"]));
                s
            },
            enabled: true,
        };
        let err = connector.validate_config(&config);
        assert!(err.is_err());
        assert!(err.unwrap_err().to_string().contains("BotToken"));
    }

    #[test]
    fn test_validate_config_no_guilds_or_channels() {
        let connector = DiscordConnector::new();
        let config = make_config("bot_token_123", vec![], vec![]);
        let err = connector.validate_config(&config);
        assert!(err.is_err());
        assert!(err
            .unwrap_err()
            .to_string()
            .contains("guild_id or channel_id"));
    }

    #[test]
    fn test_message_to_content_item() {
        let connector = DiscordConnector::new();
        let msg = make_message("msg001", "Hello from Discord!", "testuser", false);

        let item = connector.message_to_content_item(&msg, "ch123", Some("guild456"), "conn-1");

        assert_eq!(item.content, "Hello from Discord!");
        assert_eq!(item.source.connector_type, "discord");
        assert_eq!(item.source.connector_id, "conn-1");
        assert_eq!(item.source.source_id, "msg001");
        assert_eq!(item.source.author.as_deref(), Some("testuser"));
        assert_eq!(
            item.source.source_url.as_deref(),
            Some("https://discord.com/channels/guild456/ch123/msg001")
        );
        assert_eq!(
            item.source.extra.get("channel_id").unwrap(),
            &serde_json::Value::String("ch123".to_string())
        );
        assert_eq!(
            item.source.extra.get("guild_id").unwrap(),
            &serde_json::Value::String("guild456".to_string())
        );
        assert!(item.media.is_none());
    }

    #[test]
    fn test_message_to_content_item_empty_content() {
        let connector = DiscordConnector::new();
        let msg = make_message("msg002", "", "embedbot", false);

        let item = connector.message_to_content_item(&msg, "ch123", Some("guild456"), "conn-1");

        assert!(item.content.contains("embedbot"));
        assert!(item.content.contains("embed/attachment"));
    }

    #[test]
    fn test_message_to_content_item_no_guild() {
        let connector = DiscordConnector::new();
        let msg = make_message("msg003", "DM message", "friend", false);

        let item = connector.message_to_content_item(&msg, "ch_dm", None, "conn-1");

        assert!(item.source.source_url.is_none());
        assert!(!item.source.extra.contains_key("guild_id"));
    }

    #[test]
    fn test_message_to_content_item_with_attachments() {
        let connector = DiscordConnector::new();
        let mut msg = make_message("msg004", "Check this image", "uploader", false);
        msg.attachments.push(DiscordAttachment {
            id: "att1".to_string(),
            filename: "photo.png".to_string(),
            url: "https://cdn.discordapp.com/attachments/photo.png".to_string(),
            size: 1024,
            content_type: Some("image/png".to_string()),
        });

        let item = connector.message_to_content_item(&msg, "ch123", Some("guild456"), "conn-1");

        assert_eq!(
            item.source.extra.get("attachment_count").unwrap(),
            &serde_json::Value::Number(1.into())
        );
    }

    #[test]
    fn test_message_to_content_item_with_embeds() {
        let connector = DiscordConnector::new();
        let mut msg = make_message("msg005", "Link preview", "sharer", false);
        msg.embeds.push(serde_json::json!({"title": "Some Embed"}));

        let item = connector.message_to_content_item(&msg, "ch123", Some("guild456"), "conn-1");

        assert_eq!(
            item.source.extra.get("embed_count").unwrap(),
            &serde_json::Value::Number(1.into())
        );
    }

    #[test]
    fn test_parse_discord_timestamp_with_offset() {
        let ts = parse_discord_timestamp("2024-06-15T10:30:00.000000+00:00");
        assert!(ts.is_some());
        assert!(ts.unwrap() > 1704067200); // After 2024-01-01
    }

    #[test]
    fn test_parse_discord_timestamp_with_z() {
        let ts = parse_discord_timestamp("2024-01-01T00:00:00Z");
        assert!(ts.is_some());
        assert_eq!(ts.unwrap(), 1704067200);
    }

    #[test]
    fn test_parse_discord_timestamp_invalid() {
        assert!(parse_discord_timestamp("not-a-date").is_none());
        assert!(parse_discord_timestamp("").is_none());
    }

    #[test]
    fn test_get_guild_ids() {
        let config = make_config("token", vec!["guild1", "guild2"], vec![]);
        let guilds = DiscordConnector::get_guild_ids(&config);
        assert_eq!(guilds, vec!["guild1", "guild2"]);
    }

    #[test]
    fn test_get_channel_ids() {
        let config = make_config("token", vec![], vec!["ch1", "ch2"]);
        let channels = DiscordConnector::get_channel_ids(&config);
        assert_eq!(channels, vec!["ch1", "ch2"]);
    }

    #[test]
    fn test_include_threads_default() {
        let config = make_config("token", vec!["g1"], vec![]);
        assert!(!DiscordConnector::include_threads(&config));
    }

    #[test]
    fn test_include_threads_custom() {
        let mut config = make_config("token", vec!["g1"], vec![]);
        config
            .settings
            .insert("include_threads".to_string(), serde_json::json!(true));
        assert!(DiscordConnector::include_threads(&config));
    }

    #[test]
    fn test_max_messages_default() {
        let config = make_config("token", vec!["g1"], vec![]);
        assert_eq!(DiscordConnector::max_messages(&config), 1000);
    }

    #[test]
    fn test_max_messages_custom() {
        let mut config = make_config("token", vec!["g1"], vec![]);
        config
            .settings
            .insert("max_messages".to_string(), serde_json::json!(500));
        assert_eq!(DiscordConnector::max_messages(&config), 500);
    }

    #[test]
    fn test_default_constructor() {
        let connector = DiscordConnector::default();
        assert_eq!(connector.id(), "discord");
    }

    #[tokio::test]
    async fn test_handle_webhook_message_create() {
        let connector = DiscordConnector::new();
        let config = make_config("token", vec!["guild1"], vec![]);
        let payload = WebhookPayload {
            body: serde_json::to_vec(&serde_json::json!({
                "t": "MESSAGE_CREATE",
                "d": {
                    "id": "msg999",
                    "content": "Hello from webhook!",
                    "timestamp": "2024-06-15T10:30:00.000000+00:00",
                    "author": {
                        "id": "user1",
                        "username": "webhook_user",
                        "bot": false
                    },
                    "channel_id": "ch100",
                    "guild_id": "guild1",
                    "attachments": [],
                    "embeds": [],
                    "type": 0
                }
            }))
            .unwrap(),
            headers: HashMap::new(),
            content_type: Some("application/json".to_string()),
        };

        let items = connector.handle_webhook(&config, payload).await.unwrap();
        assert_eq!(items.len(), 1);
        assert_eq!(items[0].content, "Hello from webhook!");
        assert_eq!(items[0].source.source_id, "msg999");
        assert_eq!(items[0].source.author.as_deref(), Some("webhook_user"));
    }

    #[tokio::test]
    async fn test_handle_webhook_bot_message_skipped() {
        let connector = DiscordConnector::new();
        let config = make_config("token", vec!["guild1"], vec![]);
        let payload = WebhookPayload {
            body: serde_json::to_vec(&serde_json::json!({
                "t": "MESSAGE_CREATE",
                "d": {
                    "id": "msg888",
                    "content": "I am a bot message",
                    "timestamp": "2024-06-15T10:30:00Z",
                    "author": {
                        "id": "bot1",
                        "username": "some_bot",
                        "bot": true
                    },
                    "channel_id": "ch100",
                    "guild_id": "guild1",
                    "attachments": [],
                    "embeds": [],
                    "type": 0
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
    async fn test_handle_webhook_unknown_event() {
        let connector = DiscordConnector::new();
        let config = make_config("token", vec!["guild1"], vec![]);
        let payload = WebhookPayload {
            body: serde_json::to_vec(&serde_json::json!({
                "t": "GUILD_MEMBER_ADD",
                "d": { "user": { "id": "u1" } }
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
        let connector = DiscordConnector::new();
        let config = make_config("token", vec!["guild1"], vec![]);
        let payload = WebhookPayload {
            body: b"not json".to_vec(),
            headers: HashMap::new(),
            content_type: None,
        };

        let result = connector.handle_webhook(&config, payload).await;
        assert!(result.is_err());
    }

    #[test]
    fn test_is_leap_year() {
        assert!(is_leap_year(2000));
        assert!(is_leap_year(2024));
        assert!(!is_leap_year(1900));
        assert!(!is_leap_year(2023));
    }
}
