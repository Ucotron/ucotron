//! Notion connector — fetches pages from Notion databases via the REST API.
//!
//! Uses internal integration tokens (API keys) to authenticate with the Notion API.
//! Supports full sync (all pages from configured databases) and
//! incremental sync via `last_edited_time` filtering.

use std::collections::HashMap;

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};

use crate::connector::{
    AuthConfig, Connector, ConnectorConfig, ContentItem, SourceMetadata, SyncCursor, SyncResult,
    WebhookPayload,
};

const NOTION_API_BASE: &str = "https://api.notion.com/v1";
const NOTION_API_VERSION: &str = "2022-06-28";

/// Notion connector for fetching pages from Notion databases.
///
/// Requires an internal integration token (API key) created at
/// <https://www.notion.so/my-integrations>. The integration must
/// have access to the databases specified in the connector config.
pub struct NotionConnector {
    client: reqwest::Client,
}

impl NotionConnector {
    /// Creates a new NotionConnector with a default HTTP client.
    pub fn new() -> Self {
        Self {
            client: reqwest::Client::builder()
                .user_agent("ucotron-connector/0.1")
                .build()
                .unwrap_or_else(|_| reqwest::Client::new()),
        }
    }

    /// Creates a new NotionConnector with a custom HTTP client.
    pub fn with_client(client: reqwest::Client) -> Self {
        Self { client }
    }

    /// Extracts the API key from the connector config.
    fn get_api_key(config: &ConnectorConfig) -> Result<&str> {
        match &config.auth {
            AuthConfig::ApiKey { key } => Ok(key.as_str()),
            _ => bail!(
                "Notion connector requires ApiKey authentication (internal integration token)"
            ),
        }
    }

    /// Extracts configured database IDs from settings.
    fn get_database_ids(config: &ConnectorConfig) -> Vec<String> {
        config
            .settings
            .get("database_ids")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str().map(String::from))
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Whether to include page properties in the content (default: true).
    fn include_properties(config: &ConnectorConfig) -> bool {
        config
            .settings
            .get("include_properties")
            .and_then(|v| v.as_bool())
            .unwrap_or(true)
    }

    /// Whether to fetch block content (page body) (default: true).
    fn include_body(config: &ConnectorConfig) -> bool {
        config
            .settings
            .get("include_body")
            .and_then(|v| v.as_bool())
            .unwrap_or(true)
    }

    /// Queries pages from a Notion database with cursor-based pagination.
    /// When `last_edited_after` is provided, only returns pages edited after that timestamp.
    async fn query_database(
        &self,
        api_key: &str,
        database_id: &str,
        last_edited_after: Option<&str>,
    ) -> Result<Vec<NotionPage>> {
        let mut all_pages = Vec::new();
        let mut start_cursor: Option<String> = None;

        loop {
            let mut payload = serde_json::json!({
                "page_size": 100
            });

            if let Some(cursor) = start_cursor.take() {
                payload["start_cursor"] = serde_json::json!(cursor);
            }

            // Add filter for incremental sync
            if let Some(after) = last_edited_after {
                payload["filter"] = serde_json::json!({
                    "timestamp": "last_edited_time",
                    "last_edited_time": {
                        "after": after
                    }
                });
            }

            let resp = self
                .client
                .post(format!(
                    "{}/databases/{}/query",
                    NOTION_API_BASE, database_id
                ))
                .bearer_auth(api_key)
                .header("Notion-Version", NOTION_API_VERSION)
                .header("Content-Type", "application/json")
                .json(&payload)
                .send()
                .await
                .context("Failed to query Notion database")?;

            if !resp.status().is_success() {
                let status = resp.status();
                let body = resp.text().await.unwrap_or_default();
                bail!(
                    "Notion API error querying database {}: {} - {}",
                    database_id,
                    status,
                    body
                );
            }

            let data: NotionQueryResponse = resp
                .json()
                .await
                .context("Failed to parse Notion database query response")?;

            all_pages.extend(data.results);

            if !data.has_more {
                break;
            }
            start_cursor = data.next_cursor;
        }

        Ok(all_pages)
    }

    /// Fetches all blocks (content) from a Notion page.
    async fn fetch_page_blocks(&self, api_key: &str, page_id: &str) -> Result<Vec<NotionBlock>> {
        let mut all_blocks = Vec::new();
        let mut start_cursor: Option<String> = None;

        loop {
            let mut url = format!(
                "{}/blocks/{}/children?page_size=100",
                NOTION_API_BASE, page_id
            );
            if let Some(cursor) = start_cursor.take() {
                url.push_str(&format!("&start_cursor={}", cursor));
            }

            let resp = self
                .client
                .get(&url)
                .bearer_auth(api_key)
                .header("Notion-Version", NOTION_API_VERSION)
                .send()
                .await
                .context("Failed to fetch Notion page blocks")?;

            if !resp.status().is_success() {
                let status = resp.status();
                let body = resp.text().await.unwrap_or_default();
                bail!(
                    "Notion API error fetching blocks for page {}: {} - {}",
                    page_id,
                    status,
                    body
                );
            }

            let data: NotionBlocksResponse = resp
                .json()
                .await
                .context("Failed to parse Notion blocks response")?;

            all_blocks.extend(data.results);

            if !data.has_more {
                break;
            }
            start_cursor = data.next_cursor;
        }

        Ok(all_blocks)
    }

    /// Converts a Notion page (with optional blocks) into a ContentItem.
    fn page_to_content_item(
        &self,
        page: &NotionPage,
        blocks: Option<&[NotionBlock]>,
        database_id: &str,
        connector_id: &str,
        include_props: bool,
    ) -> ContentItem {
        let title = extract_page_title(&page.properties);
        let mut parts = Vec::new();

        // Page title
        if !title.is_empty() {
            parts.push(title.clone());
        }

        // Properties (if enabled)
        if include_props {
            let prop_text = extract_properties_text(&page.properties);
            if !prop_text.is_empty() {
                parts.push(prop_text);
            }
        }

        // Block content (page body)
        if let Some(blocks) = blocks {
            let body_text = blocks_to_plain_text(blocks);
            if !body_text.is_empty() {
                parts.push(body_text);
            }
        }

        let content = parts.join("\n\n");
        // Cap at 5000 chars to prevent oversized content items
        let content = if content.len() > 5000 {
            content.chars().take(5000).collect::<String>()
        } else {
            content
        };

        let created_at = parse_notion_timestamp(&page.created_time);

        let mut extra = HashMap::new();
        extra.insert(
            "page_id".to_string(),
            serde_json::Value::String(page.id.clone()),
        );
        extra.insert(
            "database_id".to_string(),
            serde_json::Value::String(database_id.to_string()),
        );
        if !title.is_empty() {
            extra.insert(
                "title".to_string(),
                serde_json::Value::String(title.clone()),
            );
        }
        if let Some(ref parent) = page.parent {
            if let Some(ref db_id) = parent.database_id {
                extra.insert(
                    "parent_database".to_string(),
                    serde_json::Value::String(db_id.clone()),
                );
            }
        }

        let page_url = page
            .url
            .clone()
            .unwrap_or_else(|| format!("https://notion.so/{}", page.id.replace('-', "")));

        ContentItem {
            content,
            source: SourceMetadata {
                connector_type: "notion".to_string(),
                connector_id: connector_id.to_string(),
                source_id: page.id.clone(),
                source_url: Some(page_url),
                author: page.created_by.as_ref().and_then(|u| u.name.clone()),
                created_at,
                extra,
            },
            media: None,
        }
    }
}

impl Default for NotionConnector {
    fn default() -> Self {
        Self::new()
    }
}

impl Connector for NotionConnector {
    fn id(&self) -> &str {
        "notion"
    }

    fn name(&self) -> &str {
        "Notion"
    }

    fn config_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "auth": {
                    "type": "object",
                    "description": "Notion internal integration credentials",
                    "properties": {
                        "key": { "type": "string", "description": "Notion integration secret (secret_...)" }
                    },
                    "required": ["key"]
                },
                "settings": {
                    "type": "object",
                    "properties": {
                        "database_ids": {
                            "type": "array",
                            "items": { "type": "string" },
                            "description": "Notion database IDs to sync"
                        },
                        "include_properties": {
                            "type": "boolean",
                            "description": "Whether to include page properties in content (default: true)"
                        },
                        "include_body": {
                            "type": "boolean",
                            "description": "Whether to fetch page body/blocks (default: true)"
                        }
                    }
                }
            },
            "required": ["auth", "settings"]
        })
    }

    fn validate_config(&self, config: &ConnectorConfig) -> Result<()> {
        if config.connector_type != "notion" {
            bail!(
                "Invalid connector type '{}', expected 'notion'",
                config.connector_type
            );
        }
        Self::get_api_key(config)?;
        let db_ids = Self::get_database_ids(config);
        if db_ids.is_empty() {
            bail!("Notion connector requires at least one database ID in settings.database_ids");
        }
        for db_id in &db_ids {
            if !is_valid_notion_id(db_id) {
                bail!(
                    "Invalid Notion database ID format '{}': expected UUID (with or without hyphens, 32 hex chars)",
                    db_id
                );
            }
        }
        Ok(())
    }

    async fn fetch(&self, config: &ConnectorConfig) -> Result<Vec<ContentItem>> {
        let api_key = Self::get_api_key(config)?;
        let database_ids = Self::get_database_ids(config);
        let include_props = Self::include_properties(config);
        let fetch_body = Self::include_body(config);

        let mut items = Vec::new();

        for db_id in &database_ids {
            let pages = self.query_database(api_key, db_id, None).await?;

            for page in &pages {
                let blocks = if fetch_body {
                    match self.fetch_page_blocks(api_key, &page.id).await {
                        Ok(b) => Some(b),
                        Err(e) => {
                            eprintln!(
                                "Warning: failed to fetch blocks for page {}: {}",
                                page.id, e
                            );
                            None
                        }
                    }
                } else {
                    None
                };

                items.push(self.page_to_content_item(
                    page,
                    blocks.as_deref(),
                    db_id,
                    &config.id,
                    include_props,
                ));
            }
        }

        Ok(items)
    }

    async fn sync_incremental(
        &self,
        config: &ConnectorConfig,
        cursor: &SyncCursor,
    ) -> Result<SyncResult> {
        let api_key = Self::get_api_key(config)?;
        let database_ids = Self::get_database_ids(config);
        let include_props = Self::include_properties(config);
        let fetch_body = Self::include_body(config);

        // Use cursor value as ISO 8601 timestamp for `last_edited_time` filter
        let last_edited_after = cursor.value.as_deref();

        let mut items = Vec::new();
        let mut latest_edited: Option<String> = cursor.value.clone();

        for db_id in &database_ids {
            let pages = self
                .query_database(api_key, db_id, last_edited_after)
                .await?;

            for page in &pages {
                // Track the most recent last_edited_time across all pages
                if latest_edited
                    .as_ref()
                    .is_none_or(|current| &page.last_edited_time > current)
                {
                    latest_edited = Some(page.last_edited_time.clone());
                }

                let blocks = if fetch_body {
                    self.fetch_page_blocks(api_key, &page.id).await.ok()
                } else {
                    None
                };

                items.push(self.page_to_content_item(
                    page,
                    blocks.as_deref(),
                    db_id,
                    &config.id,
                    include_props,
                ));
            }
        }

        let now_secs = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        Ok(SyncResult {
            items,
            cursor: SyncCursor {
                value: latest_edited,
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

        // Notion webhooks (via automation or third-party) typically send page data
        // Check for page_id in the payload
        let page_id = body
            .get("page_id")
            .or_else(|| body.get("data").and_then(|d| d.get("page_id")))
            .and_then(|v| v.as_str());

        let Some(page_id) = page_id else {
            // Try to handle as a Notion automation trigger with page object
            if let Some(page_obj) = body.get("page") {
                let page: NotionPage = serde_json::from_value(page_obj.clone())
                    .context("Failed to parse page from webhook")?;

                let db_id = page
                    .parent
                    .as_ref()
                    .and_then(|p| p.database_id.as_deref())
                    .unwrap_or("unknown");

                let include_props = Self::include_properties(config);
                return Ok(vec![self.page_to_content_item(
                    &page,
                    None,
                    db_id,
                    &config.id,
                    include_props,
                )]);
            }
            return Ok(Vec::new());
        };

        // Fetch the full page data using the page ID from the webhook
        let api_key = Self::get_api_key(config)?;
        let fetch_body = Self::include_body(config);
        let include_props = Self::include_properties(config);

        let resp = self
            .client
            .get(format!("{}/pages/{}", NOTION_API_BASE, page_id))
            .bearer_auth(api_key)
            .header("Notion-Version", NOTION_API_VERSION)
            .send()
            .await
            .context("Failed to fetch page from webhook page_id")?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body_text = resp.text().await.unwrap_or_default();
            bail!(
                "Notion API error fetching page {}: {} - {}",
                page_id,
                status,
                body_text
            );
        }

        let page: NotionPage = resp
            .json()
            .await
            .context("Failed to parse Notion page response")?;

        let blocks = if fetch_body {
            self.fetch_page_blocks(api_key, &page.id).await.ok()
        } else {
            None
        };

        let db_id = page
            .parent
            .as_ref()
            .and_then(|p| p.database_id.as_deref())
            .unwrap_or("unknown");

        Ok(vec![self.page_to_content_item(
            &page,
            blocks.as_deref(),
            db_id,
            &config.id,
            include_props,
        )])
    }
}

/// Validates a Notion ID (UUID with or without hyphens, 32 hex chars).
fn is_valid_notion_id(id: &str) -> bool {
    let hex_only: String = id.chars().filter(|c| *c != '-').collect();
    hex_only.len() == 32 && hex_only.chars().all(|c| c.is_ascii_hexdigit())
}

/// Parses a Notion ISO 8601 timestamp to Unix seconds.
/// Notion timestamps are like "2024-01-15T10:30:00.000Z" or "2024-01-15".
fn parse_notion_timestamp(ts: &str) -> Option<u64> {
    // Strip timezone suffix and fractional seconds
    let ts = ts.trim_end_matches('Z');
    let ts = if let Some(dot_pos) = ts.rfind('.') {
        // Check that the dot is in the time part (after 'T'), not in the date
        if ts[..dot_pos].contains('T') {
            &ts[..dot_pos]
        } else {
            ts
        }
    } else {
        ts
    };

    let parts: Vec<&str> = ts.split('T').collect();
    let date_parts: Vec<u64> = parts[0].split('-').filter_map(|s| s.parse().ok()).collect();

    if date_parts.len() != 3 {
        return None;
    }

    let (year, month, day) = (date_parts[0], date_parts[1], date_parts[2]);

    let (hour, min, sec) = if parts.len() == 2 {
        let time_parts: Vec<u64> = parts[1].split(':').filter_map(|s| s.parse().ok()).collect();
        if time_parts.len() != 3 {
            return None;
        }
        (time_parts[0], time_parts[1], time_parts[2])
    } else {
        (0, 0, 0)
    };

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

/// Extracts the page title from Notion properties.
/// Looks for the "title" property type or "Name" property.
fn extract_page_title(properties: &HashMap<String, NotionPropertyValue>) -> String {
    // First, look for any property with type "title"
    for value in properties.values() {
        if let Some(ref title_parts) = value.title {
            let text: String = title_parts
                .iter()
                .filter_map(|rt| rt.plain_text.as_deref())
                .collect::<Vec<_>>()
                .join("");
            if !text.is_empty() {
                return text;
            }
        }
    }
    String::new()
}

/// Extracts non-title properties as readable text.
fn extract_properties_text(properties: &HashMap<String, NotionPropertyValue>) -> String {
    let mut lines = Vec::new();

    for (key, value) in properties {
        // Skip title properties (already used as page title)
        if value.title.is_some() {
            continue;
        }

        if let Some(text) = property_to_text(value) {
            if !text.is_empty() {
                lines.push(format!("{}: {}", key, text));
            }
        }
    }

    lines.sort(); // Deterministic ordering
    lines.join("\n")
}

/// Converts a Notion property value to plain text.
fn property_to_text(value: &NotionPropertyValue) -> Option<String> {
    // Rich text
    if let Some(ref rich_text) = value.rich_text {
        let text: String = rich_text
            .iter()
            .filter_map(|rt| rt.plain_text.as_deref())
            .collect::<Vec<_>>()
            .join("");
        if !text.is_empty() {
            return Some(text);
        }
    }

    // Number
    if let Some(num) = value.number {
        return Some(num.to_string());
    }

    // Select
    if let Some(ref select) = value.select {
        return Some(select.name.clone());
    }

    // Multi-select
    if let Some(ref multi_select) = value.multi_select {
        let names: Vec<&str> = multi_select.iter().map(|s| s.name.as_str()).collect();
        if !names.is_empty() {
            return Some(names.join(", "));
        }
    }

    // Date
    if let Some(ref date) = value.date {
        let mut text = date.start.clone();
        if let Some(ref end) = date.end {
            text.push_str(" → ");
            text.push_str(end);
        }
        return Some(text);
    }

    // Checkbox
    if let Some(checked) = value.checkbox {
        return Some(if checked { "Yes" } else { "No" }.to_string());
    }

    // URL
    if let Some(ref url) = value.url {
        return Some(url.clone());
    }

    // Email
    if let Some(ref email) = value.email {
        return Some(email.clone());
    }

    // Phone number
    if let Some(ref phone) = value.phone_number {
        return Some(phone.clone());
    }

    // Status
    if let Some(ref status) = value.status {
        return Some(status.name.clone());
    }

    None
}

/// Converts Notion blocks to plain text.
fn blocks_to_plain_text(blocks: &[NotionBlock]) -> String {
    let mut lines = Vec::new();

    for block in blocks {
        if let Some(text) = block_to_text(block) {
            if !text.is_empty() {
                lines.push(text);
            }
        }
    }

    lines.join("\n")
}

/// Converts a single Notion block to plain text.
fn block_to_text(block: &NotionBlock) -> Option<String> {
    let block_type = block.block_type.as_str();

    match block_type {
        "paragraph" => extract_rich_text_from_block(&block.paragraph),
        "heading_1" => extract_rich_text_from_block(&block.heading_1).map(|t| format!("# {}", t)),
        "heading_2" => extract_rich_text_from_block(&block.heading_2).map(|t| format!("## {}", t)),
        "heading_3" => extract_rich_text_from_block(&block.heading_3).map(|t| format!("### {}", t)),
        "bulleted_list_item" => {
            extract_rich_text_from_block(&block.bulleted_list_item).map(|t| format!("- {}", t))
        }
        "numbered_list_item" => {
            extract_rich_text_from_block(&block.numbered_list_item).map(|t| format!("1. {}", t))
        }
        "to_do" => {
            let checked = block
                .to_do
                .as_ref()
                .and_then(|td| td.get("checked"))
                .and_then(|v| v.as_bool())
                .unwrap_or(false);
            let marker = if checked { "[x]" } else { "[ ]" };
            extract_rich_text_from_block(&block.to_do).map(|t| format!("- {} {}", marker, t))
        }
        "toggle" => extract_rich_text_from_block(&block.toggle),
        "quote" => extract_rich_text_from_block(&block.quote).map(|t| format!("> {}", t)),
        "callout" => extract_rich_text_from_block(&block.callout),
        "code" => {
            let lang = block
                .code
                .as_ref()
                .and_then(|c| c.get("language"))
                .and_then(|v| v.as_str())
                .unwrap_or("text");
            extract_rich_text_from_block(&block.code).map(|t| format!("```{}\n{}\n```", lang, t))
        }
        "divider" => Some("---".to_string()),
        "table_of_contents" | "breadcrumb" | "column_list" | "column" => None,
        _ => None,
    }
}

/// Extracts rich text content from a block's data object.
fn extract_rich_text_from_block(data: &Option<serde_json::Value>) -> Option<String> {
    let data = data.as_ref()?;
    let rich_text = data.get("rich_text")?.as_array()?;

    let text: String = rich_text
        .iter()
        .filter_map(|rt| rt.get("plain_text").and_then(|v| v.as_str()))
        .collect::<Vec<_>>()
        .join("");

    if text.is_empty() {
        None
    } else {
        Some(text)
    }
}

// --- Notion API response types ---

#[derive(Debug, Clone, Deserialize, Serialize)]
struct NotionQueryResponse {
    results: Vec<NotionPage>,
    has_more: bool,
    next_cursor: Option<String>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
struct NotionPage {
    id: String,
    created_time: String,
    last_edited_time: String,
    #[serde(default)]
    url: Option<String>,
    #[serde(default)]
    parent: Option<NotionParent>,
    #[serde(default)]
    created_by: Option<NotionUser>,
    #[serde(default)]
    properties: HashMap<String, NotionPropertyValue>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
struct NotionParent {
    #[serde(default)]
    database_id: Option<String>,
    #[serde(default)]
    page_id: Option<String>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
struct NotionUser {
    id: String,
    #[serde(default)]
    name: Option<String>,
}

#[derive(Debug, Clone, Deserialize, Serialize, Default)]
struct NotionPropertyValue {
    #[serde(default)]
    title: Option<Vec<NotionRichText>>,
    #[serde(default)]
    rich_text: Option<Vec<NotionRichText>>,
    #[serde(default)]
    number: Option<f64>,
    #[serde(default)]
    select: Option<NotionSelectOption>,
    #[serde(default)]
    multi_select: Option<Vec<NotionSelectOption>>,
    #[serde(default)]
    date: Option<NotionDateValue>,
    #[serde(default)]
    checkbox: Option<bool>,
    #[serde(default)]
    url: Option<String>,
    #[serde(default)]
    email: Option<String>,
    #[serde(default)]
    phone_number: Option<String>,
    #[serde(default)]
    status: Option<NotionSelectOption>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
struct NotionRichText {
    #[serde(default)]
    plain_text: Option<String>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
struct NotionSelectOption {
    name: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
struct NotionDateValue {
    start: String,
    #[serde(default)]
    end: Option<String>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
struct NotionBlocksResponse {
    results: Vec<NotionBlock>,
    has_more: bool,
    next_cursor: Option<String>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
struct NotionBlock {
    #[serde(rename = "type")]
    block_type: String,
    #[serde(default)]
    paragraph: Option<serde_json::Value>,
    #[serde(default)]
    heading_1: Option<serde_json::Value>,
    #[serde(default)]
    heading_2: Option<serde_json::Value>,
    #[serde(default)]
    heading_3: Option<serde_json::Value>,
    #[serde(default)]
    bulleted_list_item: Option<serde_json::Value>,
    #[serde(default)]
    numbered_list_item: Option<serde_json::Value>,
    #[serde(default)]
    to_do: Option<serde_json::Value>,
    #[serde(default)]
    toggle: Option<serde_json::Value>,
    #[serde(default)]
    quote: Option<serde_json::Value>,
    #[serde(default)]
    callout: Option<serde_json::Value>,
    #[serde(default)]
    code: Option<serde_json::Value>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn make_config(api_key: &str, db_ids: Vec<&str>) -> ConnectorConfig {
        let mut settings = HashMap::new();
        settings.insert("database_ids".to_string(), serde_json::json!(db_ids));
        ConnectorConfig {
            id: "notion-test".to_string(),
            name: "Test Notion".to_string(),
            connector_type: "notion".to_string(),
            auth: AuthConfig::ApiKey {
                key: api_key.to_string(),
            },
            namespace: "test".to_string(),
            settings,
            enabled: true,
        }
    }

    fn make_page(id: &str, title: &str) -> NotionPage {
        let mut properties = HashMap::new();
        properties.insert(
            "Name".to_string(),
            NotionPropertyValue {
                title: Some(vec![NotionRichText {
                    plain_text: Some(title.to_string()),
                }]),
                ..Default::default()
            },
        );
        properties.insert(
            "Status".to_string(),
            NotionPropertyValue {
                status: Some(NotionSelectOption {
                    name: "In Progress".to_string(),
                }),
                ..Default::default()
            },
        );
        properties.insert(
            "Tags".to_string(),
            NotionPropertyValue {
                multi_select: Some(vec![
                    NotionSelectOption {
                        name: "rust".to_string(),
                    },
                    NotionSelectOption {
                        name: "memory".to_string(),
                    },
                ]),
                ..Default::default()
            },
        );

        NotionPage {
            id: id.to_string(),
            created_time: "2024-06-15T10:30:00.000Z".to_string(),
            last_edited_time: "2024-06-16T12:00:00.000Z".to_string(),
            url: Some(format!("https://notion.so/{}", id.replace('-', ""))),
            parent: Some(NotionParent {
                database_id: Some("db-123".to_string()),
                page_id: None,
            }),
            created_by: Some(NotionUser {
                id: "user-1".to_string(),
                name: Some("Alice".to_string()),
            }),
            properties,
        }
    }

    fn make_blocks() -> Vec<NotionBlock> {
        vec![
            NotionBlock {
                block_type: "heading_1".to_string(),
                heading_1: Some(serde_json::json!({
                    "rich_text": [{ "plain_text": "Introduction" }]
                })),
                paragraph: None,
                heading_2: None,
                heading_3: None,
                bulleted_list_item: None,
                numbered_list_item: None,
                to_do: None,
                toggle: None,
                quote: None,
                callout: None,
                code: None,
            },
            NotionBlock {
                block_type: "paragraph".to_string(),
                paragraph: Some(serde_json::json!({
                    "rich_text": [{ "plain_text": "This is a paragraph about memory systems." }]
                })),
                heading_1: None,
                heading_2: None,
                heading_3: None,
                bulleted_list_item: None,
                numbered_list_item: None,
                to_do: None,
                toggle: None,
                quote: None,
                callout: None,
                code: None,
            },
            NotionBlock {
                block_type: "bulleted_list_item".to_string(),
                bulleted_list_item: Some(serde_json::json!({
                    "rich_text": [{ "plain_text": "First item" }]
                })),
                paragraph: None,
                heading_1: None,
                heading_2: None,
                heading_3: None,
                numbered_list_item: None,
                to_do: None,
                toggle: None,
                quote: None,
                callout: None,
                code: None,
            },
            NotionBlock {
                block_type: "code".to_string(),
                code: Some(serde_json::json!({
                    "rich_text": [{ "plain_text": "fn main() {}" }],
                    "language": "rust"
                })),
                paragraph: None,
                heading_1: None,
                heading_2: None,
                heading_3: None,
                bulleted_list_item: None,
                numbered_list_item: None,
                to_do: None,
                toggle: None,
                quote: None,
                callout: None,
            },
        ]
    }

    #[test]
    fn test_notion_connector_id_and_name() {
        let connector = NotionConnector::new();
        assert_eq!(connector.id(), "notion");
        assert_eq!(connector.name(), "Notion");
    }

    #[test]
    fn test_notion_config_schema() {
        let connector = NotionConnector::new();
        let schema = connector.config_schema();
        assert_eq!(schema["type"], "object");
        assert!(schema["properties"]["auth"].is_object());
        assert!(schema["properties"]["settings"]["properties"]["database_ids"].is_object());
        assert!(schema["properties"]["settings"]["properties"]["include_properties"].is_object());
        assert!(schema["properties"]["settings"]["properties"]["include_body"].is_object());
    }

    #[test]
    fn test_validate_config_valid() {
        let connector = NotionConnector::new();
        let config = make_config("secret_test123", vec!["a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4"]);
        assert!(connector.validate_config(&config).is_ok());
    }

    #[test]
    fn test_validate_config_valid_uuid_with_hyphens() {
        let connector = NotionConnector::new();
        let config = make_config(
            "secret_test123",
            vec!["a1b2c3d4-e5f6-a1b2-c3d4-e5f6a1b2c3d4"],
        );
        assert!(connector.validate_config(&config).is_ok());
    }

    #[test]
    fn test_validate_config_wrong_type() {
        let connector = NotionConnector::new();
        let mut config = make_config("secret_test123", vec!["a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4"]);
        config.connector_type = "github".to_string();
        let err = connector.validate_config(&config);
        assert!(err.is_err());
        assert!(err.unwrap_err().to_string().contains("notion"));
    }

    #[test]
    fn test_validate_config_wrong_auth_type() {
        let connector = NotionConnector::new();
        let config = ConnectorConfig {
            id: "notion-test".to_string(),
            name: "Test".to_string(),
            connector_type: "notion".to_string(),
            auth: AuthConfig::Token {
                token: "not-an-api-key".to_string(),
            },
            namespace: "test".to_string(),
            settings: {
                let mut s = HashMap::new();
                s.insert(
                    "database_ids".to_string(),
                    serde_json::json!(["a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4"]),
                );
                s
            },
            enabled: true,
        };
        let err = connector.validate_config(&config);
        assert!(err.is_err());
        assert!(err.unwrap_err().to_string().contains("ApiKey"));
    }

    #[test]
    fn test_validate_config_no_database_ids() {
        let connector = NotionConnector::new();
        let config = make_config("secret_test123", vec![]);
        let err = connector.validate_config(&config);
        assert!(err.is_err());
        assert!(err.unwrap_err().to_string().contains("database ID"));
    }

    #[test]
    fn test_validate_config_invalid_id_format() {
        let connector = NotionConnector::new();
        let config = make_config("secret_test123", vec!["not-a-valid-id"]);
        let err = connector.validate_config(&config);
        assert!(err.is_err());
        assert!(err
            .unwrap_err()
            .to_string()
            .contains("Invalid Notion database ID"));
    }

    #[test]
    fn test_validate_config_id_too_short() {
        let connector = NotionConnector::new();
        let config = make_config("secret_test123", vec!["a1b2c3"]);
        assert!(connector.validate_config(&config).is_err());
    }

    #[test]
    fn test_is_valid_notion_id() {
        // Valid: 32 hex chars without hyphens
        assert!(is_valid_notion_id("a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4"));
        // Valid: UUID format with hyphens
        assert!(is_valid_notion_id("a1b2c3d4-e5f6-a1b2-c3d4-e5f6a1b2c3d4"));
        // Invalid: too short
        assert!(!is_valid_notion_id("a1b2c3"));
        // Invalid: non-hex characters
        assert!(!is_valid_notion_id("zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz"));
        // Invalid: empty
        assert!(!is_valid_notion_id(""));
    }

    #[test]
    fn test_page_to_content_item_with_blocks() {
        let connector = NotionConnector::new();
        let page = make_page("page-123-abc", "My Test Page");
        let blocks = make_blocks();

        let item = connector.page_to_content_item(&page, Some(&blocks), "db-1", "conn-1", true);

        assert!(item.content.contains("My Test Page"));
        assert!(item.content.contains("# Introduction"));
        assert!(item.content.contains("paragraph about memory"));
        assert!(item.content.contains("- First item"));
        assert!(item.content.contains("```rust"));
        assert!(item.content.contains("fn main() {}"));
        assert!(item.content.contains("Status: In Progress"));
        assert!(
            item.content.contains("Tags: memory, rust")
                || item.content.contains("Tags: rust, memory")
        );
        assert_eq!(item.source.connector_type, "notion");
        assert_eq!(item.source.connector_id, "conn-1");
        assert_eq!(item.source.source_id, "page-123-abc");
        assert_eq!(item.source.author.as_deref(), Some("Alice"));
        assert!(item.source.source_url.is_some());
        assert_eq!(
            item.source.extra.get("database_id").unwrap(),
            &serde_json::Value::String("db-1".to_string())
        );
        assert!(item.media.is_none());
    }

    #[test]
    fn test_page_to_content_item_without_blocks() {
        let connector = NotionConnector::new();
        let page = make_page("page-456", "Another Page");

        let item = connector.page_to_content_item(&page, None, "db-1", "conn-1", true);

        assert!(item.content.contains("Another Page"));
        assert!(item.content.contains("Status: In Progress"));
        // No block content
        assert!(!item.content.contains("Introduction"));
    }

    #[test]
    fn test_page_to_content_item_no_properties() {
        let connector = NotionConnector::new();
        let page = make_page("page-789", "Props Test");

        let item = connector.page_to_content_item(&page, None, "db-1", "conn-1", false);

        assert!(item.content.contains("Props Test"));
        // Properties should not be included
        assert!(!item.content.contains("Status"));
        assert!(!item.content.contains("Tags"));
    }

    #[test]
    fn test_extract_page_title() {
        let mut properties = HashMap::new();
        properties.insert(
            "Name".to_string(),
            NotionPropertyValue {
                title: Some(vec![NotionRichText {
                    plain_text: Some("My Title".to_string()),
                }]),
                ..Default::default()
            },
        );
        assert_eq!(extract_page_title(&properties), "My Title");
    }

    #[test]
    fn test_extract_page_title_empty() {
        let properties = HashMap::new();
        assert_eq!(extract_page_title(&properties), "");
    }

    #[test]
    fn test_extract_properties_text() {
        let mut properties = HashMap::new();
        properties.insert(
            "Priority".to_string(),
            NotionPropertyValue {
                select: Some(NotionSelectOption {
                    name: "High".to_string(),
                }),
                ..Default::default()
            },
        );
        properties.insert(
            "Score".to_string(),
            NotionPropertyValue {
                number: Some(42.5),
                ..Default::default()
            },
        );
        properties.insert(
            "Done".to_string(),
            NotionPropertyValue {
                checkbox: Some(true),
                ..Default::default()
            },
        );

        let text = extract_properties_text(&properties);
        assert!(text.contains("Priority: High"));
        assert!(text.contains("Score: 42.5"));
        assert!(text.contains("Done: Yes"));
    }

    #[test]
    fn test_property_to_text_rich_text() {
        let val = NotionPropertyValue {
            rich_text: Some(vec![
                NotionRichText {
                    plain_text: Some("Hello ".to_string()),
                },
                NotionRichText {
                    plain_text: Some("World".to_string()),
                },
            ]),
            ..Default::default()
        };
        assert_eq!(property_to_text(&val), Some("Hello World".to_string()));
    }

    #[test]
    fn test_property_to_text_date_range() {
        let val = NotionPropertyValue {
            date: Some(NotionDateValue {
                start: "2024-01-01".to_string(),
                end: Some("2024-12-31".to_string()),
            }),
            ..Default::default()
        };
        assert_eq!(
            property_to_text(&val),
            Some("2024-01-01 → 2024-12-31".to_string())
        );
    }

    #[test]
    fn test_property_to_text_url() {
        let val = NotionPropertyValue {
            url: Some("https://example.com".to_string()),
            ..Default::default()
        };
        assert_eq!(
            property_to_text(&val),
            Some("https://example.com".to_string())
        );
    }

    #[test]
    fn test_property_to_text_email() {
        let val = NotionPropertyValue {
            email: Some("test@example.com".to_string()),
            ..Default::default()
        };
        assert_eq!(property_to_text(&val), Some("test@example.com".to_string()));
    }

    #[test]
    fn test_property_to_text_phone() {
        let val = NotionPropertyValue {
            phone_number: Some("+1-555-0123".to_string()),
            ..Default::default()
        };
        assert_eq!(property_to_text(&val), Some("+1-555-0123".to_string()));
    }

    #[test]
    fn test_blocks_to_plain_text() {
        let blocks = make_blocks();
        let text = blocks_to_plain_text(&blocks);

        assert!(text.contains("# Introduction"));
        assert!(text.contains("This is a paragraph about memory systems."));
        assert!(text.contains("- First item"));
        assert!(text.contains("```rust\nfn main() {}\n```"));
    }

    #[test]
    fn test_block_to_text_heading_2() {
        let block = NotionBlock {
            block_type: "heading_2".to_string(),
            heading_2: Some(serde_json::json!({
                "rich_text": [{ "plain_text": "Section Two" }]
            })),
            paragraph: None,
            heading_1: None,
            heading_3: None,
            bulleted_list_item: None,
            numbered_list_item: None,
            to_do: None,
            toggle: None,
            quote: None,
            callout: None,
            code: None,
        };
        assert_eq!(block_to_text(&block), Some("## Section Two".to_string()));
    }

    #[test]
    fn test_block_to_text_to_do() {
        let block = NotionBlock {
            block_type: "to_do".to_string(),
            to_do: Some(serde_json::json!({
                "rich_text": [{ "plain_text": "Buy groceries" }],
                "checked": true
            })),
            paragraph: None,
            heading_1: None,
            heading_2: None,
            heading_3: None,
            bulleted_list_item: None,
            numbered_list_item: None,
            toggle: None,
            quote: None,
            callout: None,
            code: None,
        };
        assert_eq!(
            block_to_text(&block),
            Some("- [x] Buy groceries".to_string())
        );
    }

    #[test]
    fn test_block_to_text_quote() {
        let block = NotionBlock {
            block_type: "quote".to_string(),
            quote: Some(serde_json::json!({
                "rich_text": [{ "plain_text": "Be the change." }]
            })),
            paragraph: None,
            heading_1: None,
            heading_2: None,
            heading_3: None,
            bulleted_list_item: None,
            numbered_list_item: None,
            to_do: None,
            toggle: None,
            callout: None,
            code: None,
        };
        assert_eq!(block_to_text(&block), Some("> Be the change.".to_string()));
    }

    #[test]
    fn test_block_to_text_divider() {
        let block = NotionBlock {
            block_type: "divider".to_string(),
            paragraph: None,
            heading_1: None,
            heading_2: None,
            heading_3: None,
            bulleted_list_item: None,
            numbered_list_item: None,
            to_do: None,
            toggle: None,
            quote: None,
            callout: None,
            code: None,
        };
        assert_eq!(block_to_text(&block), Some("---".to_string()));
    }

    #[test]
    fn test_block_to_text_unknown_type() {
        let block = NotionBlock {
            block_type: "unsupported_block".to_string(),
            paragraph: None,
            heading_1: None,
            heading_2: None,
            heading_3: None,
            bulleted_list_item: None,
            numbered_list_item: None,
            to_do: None,
            toggle: None,
            quote: None,
            callout: None,
            code: None,
        };
        assert_eq!(block_to_text(&block), None);
    }

    #[test]
    fn test_parse_notion_timestamp_full() {
        let ts = parse_notion_timestamp("2024-01-01T00:00:00.000Z");
        assert!(ts.is_some());
        assert_eq!(ts.unwrap(), 1704067200);
    }

    #[test]
    fn test_parse_notion_timestamp_no_millis() {
        let ts = parse_notion_timestamp("2024-01-01T00:00:00Z");
        assert!(ts.is_some());
        assert_eq!(ts.unwrap(), 1704067200);
    }

    #[test]
    fn test_parse_notion_timestamp_date_only() {
        let ts = parse_notion_timestamp("2024-01-01");
        assert!(ts.is_some());
        assert_eq!(ts.unwrap(), 1704067200);
    }

    #[test]
    fn test_parse_notion_timestamp_invalid() {
        assert!(parse_notion_timestamp("not-a-date").is_none());
        assert!(parse_notion_timestamp("").is_none());
    }

    #[test]
    fn test_get_database_ids_from_settings() {
        let config = make_config("key", vec!["db1", "db2"]);
        let ids = NotionConnector::get_database_ids(&config);
        assert_eq!(ids, vec!["db1", "db2"]);
    }

    #[test]
    fn test_get_database_ids_empty_settings() {
        let mut config = make_config("key", vec![]);
        config.settings.clear();
        let ids = NotionConnector::get_database_ids(&config);
        assert!(ids.is_empty());
    }

    #[test]
    fn test_include_flags_defaults() {
        let config = make_config("key", vec!["a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4"]);
        assert!(NotionConnector::include_properties(&config));
        assert!(NotionConnector::include_body(&config));
    }

    #[test]
    fn test_include_flags_custom() {
        let mut config = make_config("key", vec!["a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4"]);
        config
            .settings
            .insert("include_properties".to_string(), serde_json::json!(false));
        config
            .settings
            .insert("include_body".to_string(), serde_json::json!(false));

        assert!(!NotionConnector::include_properties(&config));
        assert!(!NotionConnector::include_body(&config));
    }

    #[test]
    fn test_default_constructor() {
        let connector = NotionConnector::default();
        assert_eq!(connector.id(), "notion");
    }

    #[test]
    fn test_content_item_url_fallback() {
        let connector = NotionConnector::new();
        let mut page = make_page("a1b2c3d4-e5f6-a1b2-c3d4-e5f6a1b2c3d4", "Test");
        page.url = None;

        let item = connector.page_to_content_item(&page, None, "db-1", "conn-1", false);

        // Should generate URL from page ID with hyphens removed
        assert_eq!(
            item.source.source_url.as_deref(),
            Some("https://notion.so/a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4")
        );
    }

    #[test]
    fn test_content_truncation() {
        let connector = NotionConnector::new();
        let page = make_page("page-1", "Test");

        // Create a very long block
        let long_text = "x".repeat(6000);
        let blocks = vec![NotionBlock {
            block_type: "paragraph".to_string(),
            paragraph: Some(serde_json::json!({
                "rich_text": [{ "plain_text": long_text }]
            })),
            heading_1: None,
            heading_2: None,
            heading_3: None,
            bulleted_list_item: None,
            numbered_list_item: None,
            to_do: None,
            toggle: None,
            quote: None,
            callout: None,
            code: None,
        }];

        let item = connector.page_to_content_item(&page, Some(&blocks), "db-1", "conn-1", false);
        assert!(item.content.len() <= 5000);
    }

    #[tokio::test]
    async fn test_handle_webhook_with_page_object() {
        let connector = NotionConnector::new();
        let config = make_config("secret_key", vec!["a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4"]);

        let page_json = serde_json::json!({
            "id": "page-webhook-1",
            "created_time": "2024-06-15T10:00:00.000Z",
            "last_edited_time": "2024-06-15T12:00:00.000Z",
            "url": "https://notion.so/pagewebhook1",
            "parent": { "database_id": "db-abc" },
            "created_by": { "id": "user-1", "name": "Bob" },
            "properties": {
                "Title": {
                    "title": [{ "plain_text": "Webhook Page" }]
                }
            }
        });

        let payload = WebhookPayload {
            body: serde_json::to_vec(&serde_json::json!({
                "page": page_json
            }))
            .unwrap(),
            headers: HashMap::new(),
            content_type: Some("application/json".to_string()),
        };

        let items = connector.handle_webhook(&config, payload).await.unwrap();
        assert_eq!(items.len(), 1);
        assert!(items[0].content.contains("Webhook Page"));
        assert_eq!(items[0].source.source_id, "page-webhook-1");
        assert_eq!(items[0].source.author.as_deref(), Some("Bob"));
    }

    #[tokio::test]
    async fn test_handle_webhook_no_page_data() {
        let connector = NotionConnector::new();
        let config = make_config("secret_key", vec!["a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4"]);

        let payload = WebhookPayload {
            body: serde_json::to_vec(&serde_json::json!({
                "event_type": "unknown_event"
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
        let connector = NotionConnector::new();
        let config = make_config("secret_key", vec!["a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4"]);

        let payload = WebhookPayload {
            body: b"not json".to_vec(),
            headers: HashMap::new(),
            content_type: None,
        };

        let result = connector.handle_webhook(&config, payload).await;
        assert!(result.is_err());
    }

    #[test]
    fn test_numbered_list_item() {
        let block = NotionBlock {
            block_type: "numbered_list_item".to_string(),
            numbered_list_item: Some(serde_json::json!({
                "rich_text": [{ "plain_text": "Step one" }]
            })),
            paragraph: None,
            heading_1: None,
            heading_2: None,
            heading_3: None,
            bulleted_list_item: None,
            to_do: None,
            toggle: None,
            quote: None,
            callout: None,
            code: None,
        };
        assert_eq!(block_to_text(&block), Some("1. Step one".to_string()));
    }

    #[test]
    fn test_heading_3() {
        let block = NotionBlock {
            block_type: "heading_3".to_string(),
            heading_3: Some(serde_json::json!({
                "rich_text": [{ "plain_text": "Subsection" }]
            })),
            paragraph: None,
            heading_1: None,
            heading_2: None,
            bulleted_list_item: None,
            numbered_list_item: None,
            to_do: None,
            toggle: None,
            quote: None,
            callout: None,
            code: None,
        };
        assert_eq!(block_to_text(&block), Some("### Subsection".to_string()));
    }
}
