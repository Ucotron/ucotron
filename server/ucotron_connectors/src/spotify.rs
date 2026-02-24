//! Spotify connector — fetches podcast episode metadata and descriptions.
//!
//! Uses the Spotify Web API to fetch podcast (show) episodes.
//! Supports fetching by show IDs or individual episode IDs.
//! Note: The Spotify Web API does not expose episode transcripts;
//! episode descriptions and metadata are ingested instead.

use std::collections::HashMap;

use anyhow::{bail, Context, Result};
use serde::Deserialize;

use crate::connector::{
    AuthConfig, Connector, ConnectorConfig, ContentItem, SourceMetadata, SyncCursor, SyncResult,
    WebhookPayload,
};

const SPOTIFY_API_BASE: &str = "https://api.spotify.com/v1";

/// Spotify connector for fetching podcast episode metadata and descriptions.
///
/// Requires a Spotify Web API access token (OAuth2). Fetches:
/// - Show (podcast) metadata
/// - Episode metadata (title, description, duration, release date)
///
/// # Settings
///
/// - `show_ids`: Array of Spotify show IDs to fetch episodes from
/// - `episode_ids`: Array of individual Spotify episode IDs to fetch
/// - `market`: ISO 3166-1 alpha-2 market code (default: "US")
/// - `max_episodes`: Maximum episodes per show (default: 50)
pub struct SpotifyConnector {
    client: reqwest::Client,
}

impl SpotifyConnector {
    /// Creates a new SpotifyConnector with a default HTTP client.
    pub fn new() -> Self {
        Self {
            client: reqwest::Client::new(),
        }
    }

    /// Creates a new SpotifyConnector with a custom HTTP client.
    pub fn with_client(client: reqwest::Client) -> Self {
        Self { client }
    }

    /// Extracts the OAuth2 access token from the connector config.
    fn get_access_token(config: &ConnectorConfig) -> Result<&str> {
        match &config.auth {
            AuthConfig::OAuth2 {
                access_token: Some(token),
                ..
            } => Ok(token.as_str()),
            AuthConfig::Token { token } => Ok(token.as_str()),
            _ => bail!("Spotify connector requires OAuth2 authentication with an access_token, or Token authentication"),
        }
    }

    /// Extracts show IDs from settings.
    fn get_show_ids(config: &ConnectorConfig) -> Vec<String> {
        config
            .settings
            .get("show_ids")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str().map(String::from))
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Extracts episode IDs from settings.
    fn get_episode_ids(config: &ConnectorConfig) -> Vec<String> {
        config
            .settings
            .get("episode_ids")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str().map(String::from))
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Extracts the market setting (default: "US").
    fn get_market(config: &ConnectorConfig) -> String {
        config
            .settings
            .get("market")
            .and_then(|v| v.as_str())
            .unwrap_or("US")
            .to_string()
    }

    /// Extracts the max episodes per show setting (default: 50).
    fn get_max_episodes(config: &ConnectorConfig) -> u32 {
        config
            .settings
            .get("max_episodes")
            .and_then(|v| v.as_u64())
            .map(|v| v.min(50) as u32)
            .unwrap_or(50)
    }

    /// Fetches episodes from a specific show with pagination.
    async fn fetch_show_episodes(
        &self,
        token: &str,
        show_id: &str,
        market: &str,
        max_episodes: u32,
    ) -> Result<(Option<ShowInfo>, Vec<EpisodeItem>)> {
        // First, get the show details
        let show_resp: ShowResponse = self
            .client
            .get(format!("{}/shows/{}", SPOTIFY_API_BASE, show_id))
            .bearer_auth(token)
            .query(&[("market", market)])
            .send()
            .await
            .context("Failed to call Spotify shows API")?
            .json()
            .await
            .context("Failed to parse Spotify shows response")?;

        if let Some(error) = show_resp.error {
            bail!(
                "Spotify API error: {} (status {})",
                error.message,
                error.status
            );
        }

        let show_info = ShowInfo {
            name: show_resp.name.unwrap_or_default(),
            publisher: show_resp.publisher,
            description: show_resp.description,
        };

        // Fetch episodes with pagination
        let mut all_episodes = Vec::new();
        let mut offset = 0u32;
        let mut remaining = max_episodes;

        loop {
            let limit = remaining.min(50);
            let resp: EpisodeListResponse = self
                .client
                .get(format!("{}/shows/{}/episodes", SPOTIFY_API_BASE, show_id))
                .bearer_auth(token)
                .query(&[
                    ("market", market),
                    ("limit", &limit.to_string()),
                    ("offset", &offset.to_string()),
                ])
                .send()
                .await
                .context("Failed to call Spotify show episodes API")?
                .json()
                .await
                .context("Failed to parse Spotify show episodes response")?;

            if let Some(error) = resp.error {
                bail!(
                    "Spotify API error: {} (status {})",
                    error.message,
                    error.status
                );
            }

            let items = resp.items.unwrap_or_default();
            let fetched = items.len() as u32;

            all_episodes.extend(items);
            remaining = remaining.saturating_sub(fetched);

            if fetched == 0 || remaining == 0 || resp.next.is_none() {
                break;
            }

            offset += fetched;
        }

        Ok((Some(show_info), all_episodes))
    }

    /// Fetches details for specific episode IDs (up to 50 per request).
    async fn fetch_episodes_by_ids(
        &self,
        token: &str,
        episode_ids: &[String],
        market: &str,
    ) -> Result<Vec<EpisodeItem>> {
        if episode_ids.is_empty() {
            return Ok(Vec::new());
        }

        let mut all_episodes = Vec::new();

        // Spotify API accepts up to 50 IDs per request
        for chunk in episode_ids.chunks(50) {
            let ids = chunk.join(",");
            let resp: EpisodesResponse = self
                .client
                .get(format!("{}/episodes", SPOTIFY_API_BASE))
                .bearer_auth(token)
                .query(&[("ids", ids.as_str()), ("market", market)])
                .send()
                .await
                .context("Failed to call Spotify episodes API")?
                .json()
                .await
                .context("Failed to parse Spotify episodes response")?;

            if let Some(error) = resp.error {
                bail!(
                    "Spotify API error: {} (status {})",
                    error.message,
                    error.status
                );
            }

            all_episodes.extend(resp.episodes.unwrap_or_default());
        }

        Ok(all_episodes)
    }

    /// Converts an episode into a ContentItem.
    fn episode_to_content_item(
        episode: &EpisodeItem,
        show_info: Option<&ShowInfo>,
        connector_id: &str,
    ) -> ContentItem {
        let mut content_parts = Vec::new();
        content_parts.push(format!("Title: {}", episode.name));

        // Include show name if available
        let show_name = show_info
            .map(|s| s.name.clone())
            .or_else(|| episode.show.as_ref().map(|s| s.name.clone()));

        if let Some(ref show) = show_name {
            content_parts.push(format!("Show: {}", show));
        }

        if let Some(ref desc) = episode.description {
            if !desc.is_empty() {
                // Truncate long descriptions
                let desc = if desc.len() > 1000 {
                    format!("{}...", &desc[..1000])
                } else {
                    desc.clone()
                };
                content_parts.push(format!("Description: {}", desc));
            }
        }

        // Include HTML description stripped content if available and different
        if let Some(ref html_desc) = episode.html_description {
            let plain = strip_html_tags(html_desc);
            if plain.len() > episode.description.as_deref().unwrap_or("").len() + 50 {
                // HTML description has significantly more content
                let plain = if plain.len() > 2000 {
                    format!("{}...", &plain[..2000])
                } else {
                    plain
                };
                content_parts.push(format!("Full Description: {}", plain));
            }
        }

        if let Some(duration_ms) = episode.duration_ms {
            let minutes = duration_ms / 60_000;
            let seconds = (duration_ms % 60_000) / 1000;
            content_parts.push(format!("Duration: {}:{:02}", minutes, seconds));
        }

        let content = content_parts.join("\n\n");

        // Parse release_date to Unix timestamp
        let created_at = episode.release_date.as_deref().and_then(parse_release_date);

        let mut extra = HashMap::new();
        extra.insert(
            "episode_id".to_string(),
            serde_json::Value::String(episode.id.clone()),
        );
        if let Some(ref show) = episode.show {
            extra.insert(
                "show_id".to_string(),
                serde_json::Value::String(show.id.clone()),
            );
            extra.insert(
                "show_name".to_string(),
                serde_json::Value::String(show.name.clone()),
            );
        } else if let Some(info) = show_info {
            extra.insert(
                "show_name".to_string(),
                serde_json::Value::String(info.name.clone()),
            );
        }
        if let Some(duration_ms) = episode.duration_ms {
            extra.insert(
                "duration_ms".to_string(),
                serde_json::Value::Number(serde_json::Number::from(duration_ms)),
            );
        }
        if let Some(ref lang) = episode.language {
            extra.insert(
                "language".to_string(),
                serde_json::Value::String(lang.clone()),
            );
        }
        extra.insert(
            "explicit".to_string(),
            serde_json::Value::Bool(episode.explicit.unwrap_or(false)),
        );
        if let Some(ref release_date) = episode.release_date {
            extra.insert(
                "release_date".to_string(),
                serde_json::Value::String(release_date.clone()),
            );
        }

        // Determine publisher from show info
        let author = show_info
            .and_then(|s| s.publisher.clone())
            .or_else(|| episode.show.as_ref().and_then(|s| s.publisher.clone()));

        // Build source URL
        let source_url = episode
            .external_urls
            .as_ref()
            .and_then(|urls| urls.spotify.clone())
            .unwrap_or_else(|| format!("https://open.spotify.com/episode/{}", episode.id));

        ContentItem {
            content,
            source: SourceMetadata {
                connector_type: "spotify".to_string(),
                connector_id: connector_id.to_string(),
                source_id: episode.id.clone(),
                source_url: Some(source_url),
                author,
                created_at,
                extra,
            },
            media: None,
        }
    }
}

impl Default for SpotifyConnector {
    fn default() -> Self {
        Self::new()
    }
}

impl Connector for SpotifyConnector {
    fn id(&self) -> &str {
        "spotify"
    }

    fn name(&self) -> &str {
        "Spotify"
    }

    fn config_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "auth": {
                    "type": "object",
                    "description": "OAuth2 credentials with access_token, or Token authentication",
                    "properties": {
                        "access_token": { "type": "string", "description": "Spotify Web API access token" }
                    },
                    "required": ["access_token"]
                },
                "settings": {
                    "type": "object",
                    "properties": {
                        "show_ids": {
                            "type": "array",
                            "items": { "type": "string" },
                            "description": "Spotify show (podcast) IDs to fetch episodes from"
                        },
                        "episode_ids": {
                            "type": "array",
                            "items": { "type": "string" },
                            "description": "Individual Spotify episode IDs to fetch"
                        },
                        "market": {
                            "type": "string",
                            "description": "ISO 3166-1 alpha-2 market code (default: US)"
                        },
                        "max_episodes": {
                            "type": "integer",
                            "description": "Maximum episodes per show (default: 50, max: 50)"
                        }
                    }
                }
            },
            "required": ["auth"]
        })
    }

    fn validate_config(&self, config: &ConnectorConfig) -> Result<()> {
        if config.connector_type != "spotify" {
            bail!(
                "Invalid connector type '{}', expected 'spotify'",
                config.connector_type
            );
        }
        Self::get_access_token(config)?;

        // Must have at least one source: show_ids or episode_ids
        let has_shows = !Self::get_show_ids(config).is_empty();
        let has_episodes = !Self::get_episode_ids(config).is_empty();

        if !has_shows && !has_episodes {
            bail!(
                "Spotify connector requires at least one of: show_ids or episode_ids in settings"
            );
        }

        Ok(())
    }

    async fn fetch(&self, config: &ConnectorConfig) -> Result<Vec<ContentItem>> {
        let token = Self::get_access_token(config)?;
        let market = Self::get_market(config);
        let max_episodes = Self::get_max_episodes(config);

        let mut items = Vec::new();

        // Fetch episodes from shows
        for show_id in Self::get_show_ids(config) {
            let (show_info, episodes) = self
                .fetch_show_episodes(token, &show_id, &market, max_episodes)
                .await?;

            for episode in &episodes {
                items.push(Self::episode_to_content_item(
                    episode,
                    show_info.as_ref(),
                    &config.id,
                ));
            }
        }

        // Fetch individual episodes
        let episode_ids = Self::get_episode_ids(config);
        if !episode_ids.is_empty() {
            let episodes = self
                .fetch_episodes_by_ids(token, &episode_ids, &market)
                .await?;

            for episode in &episodes {
                items.push(Self::episode_to_content_item(episode, None, &config.id));
            }
        }

        // Deduplicate by episode ID
        let mut seen = std::collections::HashSet::new();
        items.retain(|item| seen.insert(item.source.source_id.clone()));

        Ok(items)
    }

    async fn sync_incremental(
        &self,
        config: &ConnectorConfig,
        cursor: &SyncCursor,
    ) -> Result<SyncResult> {
        // Fetch all episodes
        let all_items = self.fetch(config).await?;

        // Filter by release date after the cursor
        let after_date = cursor.value.as_deref();
        let mut latest_date = cursor.value.clone();

        let mut items = Vec::new();
        let mut skipped = 0usize;

        for item in all_items {
            let release_date = item
                .source
                .extra
                .get("release_date")
                .and_then(|v| v.as_str())
                .map(String::from);

            if let Some(after) = after_date {
                if let Some(ref date) = release_date {
                    if date.as_str() <= after {
                        skipped += 1;
                        continue;
                    }
                }
            }

            // Track the most recent release date
            if let Some(ref date) = release_date {
                if latest_date
                    .as_ref()
                    .is_none_or(|current: &String| date.as_str() > current.as_str())
                {
                    latest_date = Some(date.clone());
                }
            }

            items.push(item);
        }

        let now_secs = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        Ok(SyncResult {
            items,
            cursor: SyncCursor {
                value: latest_date,
                last_sync: Some(now_secs),
            },
            skipped,
        })
    }

    async fn handle_webhook(
        &self,
        _config: &ConnectorConfig,
        _payload: WebhookPayload,
    ) -> Result<Vec<ContentItem>> {
        // Spotify does not support webhooks for podcast updates.
        // Return empty — use polling via sync_incremental instead.
        Ok(Vec::new())
    }
}

// ---------------------------------------------------------------------------
// Spotify API response types
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
struct ApiError {
    status: u32,
    message: String,
}

#[derive(Debug, Deserialize)]
struct ShowResponse {
    #[serde(default)]
    name: Option<String>,
    publisher: Option<String>,
    description: Option<String>,
    error: Option<ApiError>,
}

/// Internal show info for passing between methods.
#[derive(Debug, Clone)]
struct ShowInfo {
    name: String,
    publisher: Option<String>,
    #[allow(dead_code)]
    description: Option<String>,
}

#[derive(Debug, Deserialize)]
struct EpisodeListResponse {
    #[serde(default)]
    items: Option<Vec<EpisodeItem>>,
    next: Option<String>,
    error: Option<ApiError>,
}

#[derive(Debug, Deserialize)]
struct EpisodesResponse {
    #[serde(default)]
    episodes: Option<Vec<EpisodeItem>>,
    error: Option<ApiError>,
}

#[derive(Debug, Deserialize)]
struct EpisodeItem {
    id: String,
    name: String,
    description: Option<String>,
    html_description: Option<String>,
    duration_ms: Option<u64>,
    release_date: Option<String>,
    language: Option<String>,
    explicit: Option<bool>,
    external_urls: Option<ExternalUrls>,
    show: Option<EpisodeShow>,
}

#[derive(Debug, Deserialize)]
struct ExternalUrls {
    spotify: Option<String>,
}

#[derive(Debug, Deserialize)]
struct EpisodeShow {
    id: String,
    name: String,
    publisher: Option<String>,
}

// ---------------------------------------------------------------------------
// Helper functions
// ---------------------------------------------------------------------------

/// Parses a Spotify release date string (YYYY-MM-DD or YYYY) to Unix timestamp.
fn parse_release_date(date: &str) -> Option<u64> {
    // Spotify returns dates in YYYY-MM-DD, YYYY-MM, or YYYY format
    let parts: Vec<&str> = date.split('-').collect();
    let year: u32 = parts.first()?.parse().ok()?;
    let month: u32 = parts.get(1).and_then(|m| m.parse().ok()).unwrap_or(1);
    let day: u32 = parts.get(2).and_then(|d| d.parse().ok()).unwrap_or(1);

    // Simple conversion: days since Unix epoch (1970-01-01)
    // Using a simplified calendar (not accounting for leap seconds)
    if year < 1970 || !(1..=12).contains(&month) || !(1..=31).contains(&day) {
        return None;
    }

    let mut total_days: u64 = 0;
    for y in 1970..year {
        total_days += if is_leap_year(y) { 366 } else { 365 };
    }

    let days_in_months = [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];
    for m in 1..month {
        total_days += days_in_months[m as usize] as u64;
        if m == 2 && is_leap_year(year) {
            total_days += 1;
        }
    }

    total_days += (day - 1) as u64;

    Some(total_days * 86400)
}

/// Checks if a year is a leap year.
fn is_leap_year(year: u32) -> bool {
    (year.is_multiple_of(4) && !year.is_multiple_of(100)) || year.is_multiple_of(400)
}

/// Strips HTML tags from a string, returning plain text.
fn strip_html_tags(html: &str) -> String {
    let mut result = String::with_capacity(html.len());
    let mut in_tag = false;

    for ch in html.chars() {
        match ch {
            '<' => in_tag = true,
            '>' => in_tag = false,
            _ if !in_tag => result.push(ch),
            _ => {}
        }
    }

    // Decode common HTML entities
    result
        .replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", "\"")
        .replace("&apos;", "'")
        .replace("&#39;", "'")
        .replace("&nbsp;", " ")
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_config(
        auth: AuthConfig,
        settings: HashMap<String, serde_json::Value>,
    ) -> ConnectorConfig {
        ConnectorConfig {
            id: "spotify-test".to_string(),
            name: "Test Spotify".to_string(),
            connector_type: "spotify".to_string(),
            auth,
            namespace: "test".to_string(),
            settings,
            enabled: true,
        }
    }

    fn oauth_auth() -> AuthConfig {
        AuthConfig::OAuth2 {
            client_id: "client123".to_string(),
            client_secret: "secret456".to_string(),
            access_token: Some("test_token_abc".to_string()),
            refresh_token: Some("refresh_xyz".to_string()),
        }
    }

    fn token_auth() -> AuthConfig {
        AuthConfig::Token {
            token: "test_bearer_token".to_string(),
        }
    }

    // --- Trait method tests ---

    #[test]
    fn test_connector_id() {
        let c = SpotifyConnector::new();
        assert_eq!(c.id(), "spotify");
    }

    #[test]
    fn test_connector_name() {
        let c = SpotifyConnector::new();
        assert_eq!(c.name(), "Spotify");
    }

    #[test]
    fn test_config_schema_has_required_fields() {
        let c = SpotifyConnector::new();
        let schema = c.config_schema();
        assert_eq!(schema["type"], "object");
        assert!(schema["properties"]["settings"]["properties"]["show_ids"].is_object());
        assert!(schema["properties"]["settings"]["properties"]["episode_ids"].is_object());
        assert!(schema["properties"]["settings"]["properties"]["market"].is_object());
        assert!(schema["properties"]["settings"]["properties"]["max_episodes"].is_object());
    }

    // --- Validation tests ---

    #[test]
    fn test_validate_valid_config_with_shows() {
        let c = SpotifyConnector::new();
        let mut settings = HashMap::new();
        settings.insert(
            "show_ids".to_string(),
            serde_json::json!(["show1", "show2"]),
        );
        let config = make_config(oauth_auth(), settings);
        assert!(c.validate_config(&config).is_ok());
    }

    #[test]
    fn test_validate_valid_config_with_episodes() {
        let c = SpotifyConnector::new();
        let mut settings = HashMap::new();
        settings.insert("episode_ids".to_string(), serde_json::json!(["ep1", "ep2"]));
        let config = make_config(token_auth(), settings);
        assert!(c.validate_config(&config).is_ok());
    }

    #[test]
    fn test_validate_rejects_wrong_connector_type() {
        let c = SpotifyConnector::new();
        let mut settings = HashMap::new();
        settings.insert("show_ids".to_string(), serde_json::json!(["show1"]));
        let mut config = make_config(oauth_auth(), settings);
        config.connector_type = "youtube".to_string();
        let err = c.validate_config(&config).unwrap_err();
        assert!(err.to_string().contains("expected 'spotify'"));
    }

    #[test]
    fn test_validate_rejects_no_auth() {
        let c = SpotifyConnector::new();
        let mut settings = HashMap::new();
        settings.insert("show_ids".to_string(), serde_json::json!(["show1"]));
        let config = make_config(AuthConfig::None, settings);
        assert!(c.validate_config(&config).is_err());
    }

    #[test]
    fn test_validate_rejects_no_sources() {
        let c = SpotifyConnector::new();
        let config = make_config(oauth_auth(), HashMap::new());
        let err = c.validate_config(&config).unwrap_err();
        assert!(err.to_string().contains("show_ids or episode_ids"));
    }

    #[test]
    fn test_validate_rejects_empty_arrays() {
        let c = SpotifyConnector::new();
        let mut settings = HashMap::new();
        settings.insert("show_ids".to_string(), serde_json::json!([]));
        settings.insert("episode_ids".to_string(), serde_json::json!([]));
        let config = make_config(oauth_auth(), settings);
        assert!(c.validate_config(&config).is_err());
    }

    // --- Auth extraction tests ---

    #[test]
    fn test_get_access_token_oauth2() {
        let mut settings = HashMap::new();
        settings.insert("show_ids".to_string(), serde_json::json!(["s1"]));
        let config = make_config(oauth_auth(), settings);
        let token = SpotifyConnector::get_access_token(&config).unwrap();
        assert_eq!(token, "test_token_abc");
    }

    #[test]
    fn test_get_access_token_token() {
        let mut settings = HashMap::new();
        settings.insert("show_ids".to_string(), serde_json::json!(["s1"]));
        let config = make_config(token_auth(), settings);
        let token = SpotifyConnector::get_access_token(&config).unwrap();
        assert_eq!(token, "test_bearer_token");
    }

    #[test]
    fn test_get_access_token_oauth2_missing_token() {
        let auth = AuthConfig::OAuth2 {
            client_id: "c".to_string(),
            client_secret: "s".to_string(),
            access_token: None,
            refresh_token: None,
        };
        let mut settings = HashMap::new();
        settings.insert("show_ids".to_string(), serde_json::json!(["s1"]));
        let config = make_config(auth, settings);
        assert!(SpotifyConnector::get_access_token(&config).is_err());
    }

    // --- Settings extraction tests ---

    #[test]
    fn test_get_show_ids() {
        let mut settings = HashMap::new();
        settings.insert(
            "show_ids".to_string(),
            serde_json::json!(["abc123", "def456"]),
        );
        let config = make_config(oauth_auth(), settings);
        let ids = SpotifyConnector::get_show_ids(&config);
        assert_eq!(ids, vec!["abc123", "def456"]);
    }

    #[test]
    fn test_get_show_ids_empty() {
        let config = make_config(oauth_auth(), HashMap::new());
        let ids = SpotifyConnector::get_show_ids(&config);
        assert!(ids.is_empty());
    }

    #[test]
    fn test_get_episode_ids() {
        let mut settings = HashMap::new();
        settings.insert(
            "episode_ids".to_string(),
            serde_json::json!(["ep1", "ep2", "ep3"]),
        );
        let config = make_config(oauth_auth(), settings);
        let ids = SpotifyConnector::get_episode_ids(&config);
        assert_eq!(ids, vec!["ep1", "ep2", "ep3"]);
    }

    #[test]
    fn test_get_market_default() {
        let config = make_config(oauth_auth(), HashMap::new());
        assert_eq!(SpotifyConnector::get_market(&config), "US");
    }

    #[test]
    fn test_get_market_custom() {
        let mut settings = HashMap::new();
        settings.insert("market".to_string(), serde_json::json!("GB"));
        let config = make_config(oauth_auth(), settings);
        assert_eq!(SpotifyConnector::get_market(&config), "GB");
    }

    #[test]
    fn test_get_max_episodes_default() {
        let config = make_config(oauth_auth(), HashMap::new());
        assert_eq!(SpotifyConnector::get_max_episodes(&config), 50);
    }

    #[test]
    fn test_get_max_episodes_custom() {
        let mut settings = HashMap::new();
        settings.insert("max_episodes".to_string(), serde_json::json!(25));
        let config = make_config(oauth_auth(), settings);
        assert_eq!(SpotifyConnector::get_max_episodes(&config), 25);
    }

    #[test]
    fn test_get_max_episodes_capped_at_50() {
        let mut settings = HashMap::new();
        settings.insert("max_episodes".to_string(), serde_json::json!(200));
        let config = make_config(oauth_auth(), settings);
        assert_eq!(SpotifyConnector::get_max_episodes(&config), 50);
    }

    // --- Content item creation tests ---

    #[test]
    fn test_episode_to_content_item_full() {
        let episode = EpisodeItem {
            id: "ep123".to_string(),
            name: "My Podcast Episode".to_string(),
            description: Some("A great episode about Rust.".to_string()),
            html_description: None,
            duration_ms: Some(1_800_000), // 30 minutes
            release_date: Some("2025-06-15".to_string()),
            language: Some("en".to_string()),
            explicit: Some(false),
            external_urls: Some(ExternalUrls {
                spotify: Some("https://open.spotify.com/episode/ep123".to_string()),
            }),
            show: Some(EpisodeShow {
                id: "show456".to_string(),
                name: "Rust Radio".to_string(),
                publisher: Some("Rust Foundation".to_string()),
            }),
        };

        let item = SpotifyConnector::episode_to_content_item(&episode, None, "conn-1");
        assert!(item.content.contains("Title: My Podcast Episode"));
        assert!(item.content.contains("Show: Rust Radio"));
        assert!(item
            .content
            .contains("Description: A great episode about Rust."));
        assert!(item.content.contains("Duration: 30:00"));
        assert_eq!(item.source.connector_type, "spotify");
        assert_eq!(item.source.source_id, "ep123");
        assert_eq!(
            item.source.source_url,
            Some("https://open.spotify.com/episode/ep123".to_string())
        );
        assert_eq!(item.source.author, Some("Rust Foundation".to_string()));
        assert!(item.source.created_at.is_some());
        assert_eq!(
            item.source.extra.get("show_id").and_then(|v| v.as_str()),
            Some("show456")
        );
        assert_eq!(
            item.source
                .extra
                .get("duration_ms")
                .and_then(|v| v.as_u64()),
            Some(1_800_000)
        );
    }

    #[test]
    fn test_episode_to_content_item_with_show_info() {
        let episode = EpisodeItem {
            id: "ep789".to_string(),
            name: "Episode Title".to_string(),
            description: Some("Desc".to_string()),
            html_description: None,
            duration_ms: None,
            release_date: None,
            language: None,
            explicit: None,
            external_urls: None,
            show: None,
        };

        let show_info = ShowInfo {
            name: "My Show".to_string(),
            publisher: Some("Publisher Co".to_string()),
            description: Some("Show desc".to_string()),
        };

        let item = SpotifyConnector::episode_to_content_item(&episode, Some(&show_info), "conn-2");
        assert!(item.content.contains("Show: My Show"));
        assert_eq!(item.source.author, Some("Publisher Co".to_string()));
        // Fallback URL when no external_urls
        assert_eq!(
            item.source.source_url,
            Some("https://open.spotify.com/episode/ep789".to_string())
        );
    }

    #[test]
    fn test_episode_to_content_item_minimal() {
        let episode = EpisodeItem {
            id: "ep_min".to_string(),
            name: "Minimal".to_string(),
            description: None,
            html_description: None,
            duration_ms: None,
            release_date: None,
            language: None,
            explicit: None,
            external_urls: None,
            show: None,
        };

        let item = SpotifyConnector::episode_to_content_item(&episode, None, "conn-3");
        assert_eq!(item.content, "Title: Minimal");
        assert_eq!(item.source.source_id, "ep_min");
        assert!(item.source.created_at.is_none());
        assert!(item.media.is_none());
    }

    #[test]
    fn test_episode_long_description_truncated() {
        let long_desc = "A".repeat(2000);
        let episode = EpisodeItem {
            id: "ep_long".to_string(),
            name: "Long".to_string(),
            description: Some(long_desc),
            html_description: None,
            duration_ms: None,
            release_date: None,
            language: None,
            explicit: None,
            external_urls: None,
            show: None,
        };

        let item = SpotifyConnector::episode_to_content_item(&episode, None, "conn-4");
        // Description should be truncated to 1000 chars + "..."
        assert!(item.content.contains("..."));
        assert!(item.content.len() < 2200);
    }

    // --- Helper function tests ---

    #[test]
    fn test_parse_release_date_full() {
        let ts = parse_release_date("2024-01-15").unwrap();
        // 2024-01-15 should be a valid timestamp
        assert!(ts > 0);
        // Rough check: 2024 is ~54 years after 1970
        assert!(ts > 54 * 365 * 86400);
        assert!(ts < 55 * 365 * 86400);
    }

    #[test]
    fn test_parse_release_date_year_month() {
        let ts = parse_release_date("2023-06").unwrap();
        assert!(ts > 0);
    }

    #[test]
    fn test_parse_release_date_year_only() {
        let ts = parse_release_date("2020").unwrap();
        assert!(ts > 0);
    }

    #[test]
    fn test_parse_release_date_invalid() {
        assert!(parse_release_date("not-a-date").is_none());
        assert!(parse_release_date("").is_none());
    }

    #[test]
    fn test_parse_release_date_before_epoch() {
        assert!(parse_release_date("1960-01-01").is_none());
    }

    #[test]
    fn test_is_leap_year() {
        assert!(is_leap_year(2000)); // divisible by 400
        assert!(is_leap_year(2024)); // divisible by 4
        assert!(!is_leap_year(1900)); // divisible by 100 but not 400
        assert!(!is_leap_year(2023)); // not divisible by 4
    }

    #[test]
    fn test_strip_html_tags() {
        assert_eq!(strip_html_tags("<p>Hello <b>world</b></p>"), "Hello world");
        assert_eq!(strip_html_tags("no tags here"), "no tags here");
        assert_eq!(
            strip_html_tags("<a href=\"url\">link</a> &amp; more"),
            "link & more"
        );
        assert_eq!(strip_html_tags(""), "");
    }

    #[test]
    fn test_strip_html_tags_entities() {
        assert_eq!(
            strip_html_tags("&lt;code&gt; &amp; &quot;test&quot;"),
            "<code> & \"test\""
        );
    }

    // --- Webhook test ---

    #[tokio::test]
    async fn test_webhook_returns_empty() {
        let c = SpotifyConnector::new();
        let mut settings = HashMap::new();
        settings.insert("show_ids".to_string(), serde_json::json!(["s1"]));
        let config = make_config(oauth_auth(), settings);
        let payload = WebhookPayload {
            body: b"{}".to_vec(),
            headers: HashMap::new(),
            content_type: Some("application/json".to_string()),
        };
        let result = c.handle_webhook(&config, payload).await.unwrap();
        assert!(result.is_empty());
    }

    // --- Default trait test ---

    #[test]
    fn test_default_creates_connector() {
        let c = SpotifyConnector::default();
        assert_eq!(c.id(), "spotify");
    }
}
