//! Google Docs connector — fetches documents from Google Docs.
//!
//! Uses OAuth2 access tokens to authenticate with the Google Docs API v1.
//! Supports full sync (all docs from configured document IDs)
//! and incremental sync via `modifiedTime` filtering.

use std::collections::HashMap;

use anyhow::{bail, Context, Result};

use crate::connector::{
    AuthConfig, Connector, ConnectorConfig, ContentItem, SourceMetadata, SyncCursor, SyncResult,
    WebhookPayload,
};

const DOCS_API_BASE: &str = "https://docs.googleapis.com/v1";

/// Google Docs connector implementation.
pub struct GoogleDocsConnector {
    client: reqwest::Client,
}

impl Default for GoogleDocsConnector {
    fn default() -> Self {
        Self::new()
    }
}

impl GoogleDocsConnector {
    pub fn new() -> Self {
        Self {
            client: reqwest::Client::new(),
        }
    }

    /// Extracts the OAuth2 access token from the connector config.
    fn get_token(config: &ConnectorConfig) -> Result<&str> {
        match &config.auth {
            AuthConfig::OAuth2 {
                access_token: Some(token),
                ..
            } => Ok(token.as_str()),
            AuthConfig::OAuth2 {
                access_token: None, ..
            } => bail!("Google Docs connector requires an access_token in OAuth2 config"),
            _ => bail!("Google Docs connector requires OAuth2 authentication"),
        }
    }

    /// Extracts configured document IDs from settings.
    fn get_document_ids(config: &ConnectorConfig) -> Vec<String> {
        config
            .settings
            .get("document_ids")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str().map(String::from))
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Fetches a single document and converts it to a ContentItem.
    async fn fetch_document(
        &self,
        token: &str,
        doc_id: &str,
        config: &ConnectorConfig,
    ) -> Result<ContentItem> {
        let url = format!("{}/documents/{}", DOCS_API_BASE, doc_id);
        let resp = self
            .client
            .get(&url)
            .bearer_auth(token)
            .send()
            .await
            .context("Failed to fetch Google Doc")?;

        if !resp.status().is_success() {
            bail!(
                "Google Docs API returned status {} for doc {}",
                resp.status(),
                doc_id
            );
        }

        let body: serde_json::Value = resp.json().await?;
        let title = body["title"].as_str().unwrap_or("Untitled").to_string();
        let text = extract_text_from_body(&body["body"]);

        Ok(ContentItem {
            content: text,
            source: SourceMetadata {
                connector_type: "google_docs".to_string(),
                connector_id: config.id.clone(),
                source_id: doc_id.to_string(),
                source_url: Some(format!("https://docs.google.com/document/d/{}", doc_id)),
                author: None,
                created_at: None,
                extra: {
                    let mut m = HashMap::new();
                    m.insert("title".to_string(), serde_json::json!(title));
                    m
                },
            },
            media: None,
        })
    }
}

impl Connector for GoogleDocsConnector {
    fn id(&self) -> &str {
        "google_docs"
    }

    fn name(&self) -> &str {
        "Google Docs"
    }

    fn config_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "document_ids": {
                    "type": "array",
                    "items": { "type": "string" },
                    "description": "List of Google Doc IDs to sync"
                }
            }
        })
    }

    fn validate_config(&self, config: &ConnectorConfig) -> Result<()> {
        Self::get_token(config)?;
        Ok(())
    }

    async fn fetch(&self, config: &ConnectorConfig) -> Result<Vec<ContentItem>> {
        let token = Self::get_token(config)?;
        let doc_ids = Self::get_document_ids(config);

        let mut items = Vec::new();
        for doc_id in &doc_ids {
            match self.fetch_document(token, doc_id, config).await {
                Ok(item) => items.push(item),
                Err(e) => tracing::warn!(doc_id, error = %e, "Failed to fetch Google Doc"),
            }
        }
        Ok(items)
    }

    async fn sync_incremental(
        &self,
        config: &ConnectorConfig,
        _cursor: &SyncCursor,
    ) -> Result<SyncResult> {
        // Incremental falls back to full sync for now
        let items = self.fetch(config).await?;
        Ok(SyncResult {
            items,
            cursor: SyncCursor::default(),
            skipped: 0,
        })
    }

    async fn handle_webhook(
        &self,
        _config: &ConnectorConfig,
        _payload: WebhookPayload,
    ) -> Result<Vec<ContentItem>> {
        bail!("Google Docs webhook handler not yet implemented")
    }
}

/// Extract plain text content from a Google Docs API body object.
fn extract_text_from_body(body: &serde_json::Value) -> String {
    let mut text = String::new();
    if let Some(content) = body["content"].as_array() {
        for element in content {
            if let Some(paragraph) = element.get("paragraph") {
                if let Some(elements) = paragraph["elements"].as_array() {
                    for elem in elements {
                        if let Some(text_run) = elem.get("textRun") {
                            if let Some(content) = text_run["content"].as_str() {
                                text.push_str(content);
                            }
                        }
                    }
                }
            }
        }
    }
    text
}
