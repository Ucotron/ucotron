//! YouTube connector — fetches video metadata and captions/transcripts.
//!
//! Uses the YouTube Data API v3 for video metadata and captions.
//! Supports fetching auto-generated and manual captions for transcript ingestion.

use std::collections::HashMap;

use anyhow::{bail, Context, Result};
use serde::Deserialize;

use crate::connector::{
    AuthConfig, Connector, ConnectorConfig, ContentItem, SourceMetadata, SyncCursor, SyncResult,
    WebhookPayload,
};

const YOUTUBE_API_BASE: &str = "https://www.googleapis.com/youtube/v3";

/// YouTube connector for fetching video transcripts and metadata.
///
/// Requires a YouTube Data API v3 key. Fetches:
/// - Video metadata (title, description, channel, published date)
/// - Captions/transcripts (auto-generated or manual)
///
/// # Settings
///
/// - `video_ids`: Array of YouTube video IDs to fetch
/// - `channel_id`: YouTube channel ID to fetch all videos from
/// - `playlist_id`: YouTube playlist ID to fetch videos from
/// - `max_results`: Maximum number of videos per channel/playlist (default: 50)
/// - `language`: Preferred caption language (default: "en")
/// - `include_auto_captions`: Whether to include auto-generated captions (default: true)
pub struct YouTubeConnector {
    client: reqwest::Client,
}

impl YouTubeConnector {
    /// Creates a new YouTubeConnector with a default HTTP client.
    pub fn new() -> Self {
        Self {
            client: reqwest::Client::new(),
        }
    }

    /// Creates a new YouTubeConnector with a custom HTTP client.
    pub fn with_client(client: reqwest::Client) -> Self {
        Self { client }
    }

    /// Extracts the API key from the connector config.
    fn get_api_key(config: &ConnectorConfig) -> Result<&str> {
        match &config.auth {
            AuthConfig::ApiKey { key } => Ok(key.as_str()),
            _ => bail!("YouTube connector requires ApiKey authentication"),
        }
    }

    /// Extracts video IDs from settings.
    fn get_video_ids(config: &ConnectorConfig) -> Vec<String> {
        config
            .settings
            .get("video_ids")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str().map(String::from))
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Extracts the channel ID from settings.
    fn get_channel_id(config: &ConnectorConfig) -> Option<String> {
        config
            .settings
            .get("channel_id")
            .and_then(|v| v.as_str())
            .map(String::from)
    }

    /// Extracts the playlist ID from settings.
    fn get_playlist_id(config: &ConnectorConfig) -> Option<String> {
        config
            .settings
            .get("playlist_id")
            .and_then(|v| v.as_str())
            .map(String::from)
    }

    /// Extracts the max results setting (default: 50).
    fn get_max_results(config: &ConnectorConfig) -> u32 {
        config
            .settings
            .get("max_results")
            .and_then(|v| v.as_u64())
            .map(|v| v.min(50) as u32)
            .unwrap_or(50)
    }

    /// Extracts the preferred caption language (default: "en").
    fn get_language(config: &ConnectorConfig) -> String {
        config
            .settings
            .get("language")
            .and_then(|v| v.as_str())
            .unwrap_or("en")
            .to_string()
    }

    /// Checks if auto-generated captions should be included (default: true).
    fn include_auto_captions(config: &ConnectorConfig) -> bool {
        config
            .settings
            .get("include_auto_captions")
            .and_then(|v| v.as_bool())
            .unwrap_or(true)
    }

    /// Fetches video details (metadata) for a list of video IDs.
    async fn fetch_video_details(
        &self,
        api_key: &str,
        video_ids: &[String],
    ) -> Result<Vec<VideoDetails>> {
        if video_ids.is_empty() {
            return Ok(Vec::new());
        }

        let mut all_details = Vec::new();

        // YouTube API accepts up to 50 IDs per request
        for chunk in video_ids.chunks(50) {
            let ids = chunk.join(",");
            let resp: VideoListResponse = self
                .client
                .get(format!("{}/videos", YOUTUBE_API_BASE))
                .query(&[
                    ("key", api_key),
                    ("id", &ids),
                    ("part", "snippet,contentDetails"),
                ])
                .send()
                .await
                .context("Failed to call YouTube videos.list API")?
                .json()
                .await
                .context("Failed to parse YouTube videos.list response")?;

            if let Some(error) = resp.error {
                bail!("YouTube API error: {} ({})", error.message, error.code);
            }

            all_details.extend(resp.items.unwrap_or_default());
        }

        Ok(all_details)
    }

    /// Fetches video IDs from a channel's uploads.
    async fn fetch_channel_videos(
        &self,
        api_key: &str,
        channel_id: &str,
        max_results: u32,
    ) -> Result<Vec<String>> {
        // First, get the channel's uploads playlist ID
        let channel_resp: ChannelListResponse = self
            .client
            .get(format!("{}/channels", YOUTUBE_API_BASE))
            .query(&[
                ("key", api_key),
                ("id", channel_id),
                ("part", "contentDetails"),
            ])
            .send()
            .await
            .context("Failed to call YouTube channels.list API")?
            .json()
            .await
            .context("Failed to parse YouTube channels.list response")?;

        if let Some(error) = channel_resp.error {
            bail!("YouTube API error: {} ({})", error.message, error.code);
        }

        let uploads_playlist_id = channel_resp
            .items
            .and_then(|items| items.into_iter().next())
            .and_then(|ch| ch.content_details)
            .and_then(|cd| cd.related_playlists)
            .and_then(|rp| rp.uploads)
            .context("Channel has no uploads playlist")?;

        self.fetch_playlist_videos(api_key, &uploads_playlist_id, max_results)
            .await
    }

    /// Fetches video IDs from a playlist.
    async fn fetch_playlist_videos(
        &self,
        api_key: &str,
        playlist_id: &str,
        max_results: u32,
    ) -> Result<Vec<String>> {
        let mut video_ids = Vec::new();
        let mut page_token: Option<String> = None;
        let mut remaining = max_results;

        loop {
            let page_size = remaining.min(50);
            let mut params = vec![
                ("key".to_string(), api_key.to_string()),
                ("playlistId".to_string(), playlist_id.to_string()),
                ("part".to_string(), "contentDetails".to_string()),
                ("maxResults".to_string(), page_size.to_string()),
            ];
            if let Some(ref token) = page_token {
                params.push(("pageToken".to_string(), token.clone()));
            }

            let resp: PlaylistItemListResponse = self
                .client
                .get(format!("{}/playlistItems", YOUTUBE_API_BASE))
                .query(&params)
                .send()
                .await
                .context("Failed to call YouTube playlistItems.list API")?
                .json()
                .await
                .context("Failed to parse YouTube playlistItems.list response")?;

            if let Some(error) = resp.error {
                bail!("YouTube API error: {} ({})", error.message, error.code);
            }

            for item in resp.items.unwrap_or_default() {
                if let Some(cd) = item.content_details {
                    video_ids.push(cd.video_id);
                    remaining = remaining.saturating_sub(1);
                    if remaining == 0 {
                        return Ok(video_ids);
                    }
                }
            }

            match resp.next_page_token {
                Some(token) if remaining > 0 => page_token = Some(token),
                _ => break,
            }
        }

        Ok(video_ids)
    }

    /// Fetches available caption tracks for a video.
    ///
    /// Note: This lists available tracks but downloading requires OAuth for third-party content.
    /// The `download_captions` method uses the public timedtext endpoint instead.
    #[allow(dead_code)]
    async fn fetch_caption_tracks(
        &self,
        api_key: &str,
        video_id: &str,
    ) -> Result<Vec<CaptionTrack>> {
        let resp: CaptionListResponse = self
            .client
            .get(format!("{}/captions", YOUTUBE_API_BASE))
            .query(&[("key", api_key), ("videoId", video_id), ("part", "snippet")])
            .send()
            .await
            .context("Failed to call YouTube captions.list API")?
            .json()
            .await
            .context("Failed to parse YouTube captions.list response")?;

        if let Some(error) = resp.error {
            bail!("YouTube API error: {} ({})", error.message, error.code);
        }

        Ok(resp.items.unwrap_or_default())
    }

    /// Downloads caption text for a specific caption track.
    ///
    /// Falls back to the timedtext endpoint for auto-generated captions
    /// since the captions.download endpoint requires OAuth for third-party content.
    async fn download_captions(&self, video_id: &str, language: &str) -> Result<Option<String>> {
        // Use the public timedtext endpoint (works without OAuth for public videos)
        let resp = self
            .client
            .get("https://www.youtube.com/api/timedtext")
            .query(&[("v", video_id), ("lang", language), ("fmt", "srv3")])
            .send()
            .await
            .context("Failed to fetch captions from timedtext API")?;

        if !resp.status().is_success() {
            // Try auto-generated captions with asr_langs
            let resp = self
                .client
                .get("https://www.youtube.com/api/timedtext")
                .query(&[
                    ("v", video_id),
                    ("lang", language),
                    ("fmt", "srv3"),
                    ("kind", "asr"),
                ])
                .send()
                .await
                .context("Failed to fetch auto-generated captions")?;

            if !resp.status().is_success() {
                return Ok(None);
            }

            let xml = resp
                .text()
                .await
                .context("Failed to read auto-generated caption response")?;

            if xml.trim().is_empty() {
                return Ok(None);
            }

            return Ok(Some(Self::parse_srv3_captions(&xml)));
        }

        let xml = resp
            .text()
            .await
            .context("Failed to read caption response")?;

        if xml.trim().is_empty() {
            return Ok(None);
        }

        Ok(Some(Self::parse_srv3_captions(&xml)))
    }

    /// Parses srv3 (XML) caption format into plain text.
    ///
    /// srv3 format: `<transcript><text start="0" dur="5.2">Hello</text>...</transcript>`
    fn parse_srv3_captions(xml: &str) -> String {
        let mut lines = Vec::new();
        // Simple XML parsing — extract text content between <text ...> and </text> tags
        let mut pos = 0;
        while let Some(start_tag_begin) = xml[pos..].find("<text ") {
            let abs_start = pos + start_tag_begin;
            if let Some(start_tag_end) = xml[abs_start..].find('>') {
                let content_start = abs_start + start_tag_end + 1;
                if let Some(end_tag) = xml[content_start..].find("</text>") {
                    let text = &xml[content_start..content_start + end_tag];
                    let decoded = Self::decode_xml_entities(text);
                    let decoded = decoded.trim();
                    if !decoded.is_empty() {
                        lines.push(decoded.to_string());
                    }
                    pos = content_start + end_tag + 7; // 7 = "</text>".len()
                } else {
                    break;
                }
            } else {
                break;
            }
        }

        lines.join(" ")
    }

    /// Decodes common XML/HTML entities.
    fn decode_xml_entities(text: &str) -> String {
        text.replace("&amp;", "&")
            .replace("&lt;", "<")
            .replace("&gt;", ">")
            .replace("&quot;", "\"")
            .replace("&apos;", "'")
            .replace("&#39;", "'")
    }

    /// Converts video details and transcript into a ContentItem.
    fn video_to_content_item(
        video: &VideoDetails,
        transcript: Option<&str>,
        connector_id: &str,
    ) -> ContentItem {
        let snippet = &video.snippet;
        let video_id = &video.id;

        // Build content: title + description + transcript
        let mut content_parts = Vec::new();
        content_parts.push(format!("Title: {}", snippet.title));

        if let Some(ref channel) = snippet.channel_title {
            content_parts.push(format!("Channel: {}", channel));
        }

        if !snippet.description.is_empty() {
            // Truncate long descriptions
            let desc = if snippet.description.len() > 500 {
                format!("{}...", &snippet.description[..500])
            } else {
                snippet.description.clone()
            };
            content_parts.push(format!("Description: {}", desc));
        }

        if let Some(transcript) = transcript {
            content_parts.push(format!("Transcript: {}", transcript));
        }

        let content = content_parts.join("\n\n");

        // Parse published_at timestamp
        let created_at = parse_iso8601_timestamp(&snippet.published_at);

        let mut extra = HashMap::new();
        extra.insert(
            "video_id".to_string(),
            serde_json::Value::String(video_id.clone()),
        );
        if let Some(ref channel_id) = snippet.channel_id {
            extra.insert(
                "channel_id".to_string(),
                serde_json::Value::String(channel_id.clone()),
            );
        }
        if let Some(ref channel) = snippet.channel_title {
            extra.insert(
                "channel_title".to_string(),
                serde_json::Value::String(channel.clone()),
            );
        }
        if let Some(ref duration) = video
            .content_details
            .as_ref()
            .and_then(|cd| cd.duration.as_deref())
        {
            extra.insert(
                "duration".to_string(),
                serde_json::Value::String(duration.to_string()),
            );
        }
        extra.insert(
            "has_transcript".to_string(),
            serde_json::Value::Bool(transcript.is_some()),
        );

        ContentItem {
            content,
            source: SourceMetadata {
                connector_type: "youtube".to_string(),
                connector_id: connector_id.to_string(),
                source_id: video_id.clone(),
                source_url: Some(format!("https://www.youtube.com/watch?v={}", video_id)),
                author: snippet.channel_title.clone(),
                created_at,
                extra,
            },
            media: None,
        }
    }
}

impl Default for YouTubeConnector {
    fn default() -> Self {
        Self::new()
    }
}

impl Connector for YouTubeConnector {
    fn id(&self) -> &str {
        "youtube"
    }

    fn name(&self) -> &str {
        "YouTube"
    }

    fn config_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "auth": {
                    "type": "object",
                    "description": "API key credentials",
                    "properties": {
                        "key": { "type": "string", "description": "YouTube Data API v3 key" }
                    },
                    "required": ["key"]
                },
                "settings": {
                    "type": "object",
                    "properties": {
                        "video_ids": {
                            "type": "array",
                            "items": { "type": "string" },
                            "description": "YouTube video IDs to fetch"
                        },
                        "channel_id": {
                            "type": "string",
                            "description": "YouTube channel ID to fetch all videos from"
                        },
                        "playlist_id": {
                            "type": "string",
                            "description": "YouTube playlist ID to fetch videos from"
                        },
                        "max_results": {
                            "type": "integer",
                            "description": "Maximum number of videos per channel/playlist (default: 50)"
                        },
                        "language": {
                            "type": "string",
                            "description": "Preferred caption language code (default: en)"
                        },
                        "include_auto_captions": {
                            "type": "boolean",
                            "description": "Whether to include auto-generated captions (default: true)"
                        }
                    }
                }
            },
            "required": ["auth"]
        })
    }

    fn validate_config(&self, config: &ConnectorConfig) -> Result<()> {
        if config.connector_type != "youtube" {
            bail!(
                "Invalid connector type '{}', expected 'youtube'",
                config.connector_type
            );
        }
        Self::get_api_key(config)?;

        // Must have at least one source: video_ids, channel_id, or playlist_id
        let has_videos = !Self::get_video_ids(config).is_empty();
        let has_channel = Self::get_channel_id(config).is_some();
        let has_playlist = Self::get_playlist_id(config).is_some();

        if !has_videos && !has_channel && !has_playlist {
            bail!("YouTube connector requires at least one of: video_ids, channel_id, or playlist_id in settings");
        }

        Ok(())
    }

    async fn fetch(&self, config: &ConnectorConfig) -> Result<Vec<ContentItem>> {
        let api_key = Self::get_api_key(config)?;
        let language = Self::get_language(config);
        let max_results = Self::get_max_results(config);

        // Collect video IDs from all sources
        let mut video_ids = Self::get_video_ids(config);

        if let Some(channel_id) = Self::get_channel_id(config) {
            let channel_videos = self
                .fetch_channel_videos(api_key, &channel_id, max_results)
                .await?;
            video_ids.extend(channel_videos);
        }

        if let Some(playlist_id) = Self::get_playlist_id(config) {
            let playlist_videos = self
                .fetch_playlist_videos(api_key, &playlist_id, max_results)
                .await?;
            video_ids.extend(playlist_videos);
        }

        // Deduplicate video IDs
        video_ids.sort();
        video_ids.dedup();

        if video_ids.is_empty() {
            return Ok(Vec::new());
        }

        // Fetch video details
        let details = self.fetch_video_details(api_key, &video_ids).await?;

        // Fetch transcripts and build content items
        let mut items = Vec::new();
        for video in &details {
            // Try to get captions
            let transcript = self
                .download_captions(&video.id, &language)
                .await
                .unwrap_or(None);

            // If no captions in preferred language and auto-captions enabled, try without language preference
            let transcript = if transcript.is_none() && Self::include_auto_captions(config) {
                // Try English as fallback
                if language != "en" {
                    self.download_captions(&video.id, "en")
                        .await
                        .unwrap_or(None)
                } else {
                    None
                }
            } else {
                transcript
            };

            items.push(Self::video_to_content_item(
                video,
                transcript.as_deref(),
                &config.id,
            ));
        }

        Ok(items)
    }

    async fn sync_incremental(
        &self,
        config: &ConnectorConfig,
        cursor: &SyncCursor,
    ) -> Result<SyncResult> {
        let api_key = Self::get_api_key(config)?;
        let language = Self::get_language(config);
        let max_results = Self::get_max_results(config);

        // For incremental sync, we filter by published date after the cursor
        let published_after = cursor.value.as_deref();

        let mut video_ids = Self::get_video_ids(config);

        // For channel/playlist, we fetch and filter by date
        if let Some(channel_id) = Self::get_channel_id(config) {
            let channel_videos = self
                .fetch_channel_videos(api_key, &channel_id, max_results)
                .await?;
            video_ids.extend(channel_videos);
        }

        if let Some(playlist_id) = Self::get_playlist_id(config) {
            let playlist_videos = self
                .fetch_playlist_videos(api_key, &playlist_id, max_results)
                .await?;
            video_ids.extend(playlist_videos);
        }

        video_ids.sort();
        video_ids.dedup();

        if video_ids.is_empty() {
            let now_secs = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs();

            return Ok(SyncResult {
                items: Vec::new(),
                cursor: SyncCursor {
                    value: cursor.value.clone(),
                    last_sync: Some(now_secs),
                },
                skipped: 0,
            });
        }

        let details = self.fetch_video_details(api_key, &video_ids).await?;

        // Filter by published date if cursor has a value
        let filtered_details: Vec<&VideoDetails> = if let Some(after) = published_after {
            details
                .iter()
                .filter(|v| v.snippet.published_at.as_str() > after)
                .collect()
        } else {
            details.iter().collect()
        };

        let skipped = details.len() - filtered_details.len();

        let mut items = Vec::new();
        let mut latest_published = cursor.value.clone();

        for video in &filtered_details {
            let transcript = self
                .download_captions(&video.id, &language)
                .await
                .unwrap_or(None);

            let transcript = if transcript.is_none() && Self::include_auto_captions(config) {
                if language != "en" {
                    self.download_captions(&video.id, "en")
                        .await
                        .unwrap_or(None)
                } else {
                    None
                }
            } else {
                transcript
            };

            // Track the most recent published_at
            if latest_published
                .as_ref()
                .is_none_or(|current| video.snippet.published_at.as_str() > current.as_str())
            {
                latest_published = Some(video.snippet.published_at.clone());
            }

            items.push(Self::video_to_content_item(
                video,
                transcript.as_deref(),
                &config.id,
            ));
        }

        let now_secs = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        Ok(SyncResult {
            items,
            cursor: SyncCursor {
                value: latest_published,
                last_sync: Some(now_secs),
            },
            skipped,
        })
    }

    async fn handle_webhook(
        &self,
        config: &ConnectorConfig,
        payload: WebhookPayload,
    ) -> Result<Vec<ContentItem>> {
        // YouTube PubSubHubbub (WebSub) delivers Atom XML for new video notifications
        let body_str = String::from_utf8_lossy(&payload.body);

        // Check for hub verification challenge (GET request echoed back)
        if let Some(challenge) = payload.headers.get("hub.challenge") {
            let _ = challenge;
            return Ok(Vec::new());
        }

        // Parse Atom XML feed to extract video ID
        let video_id = extract_video_id_from_atom(&body_str);

        if let Some(video_id) = video_id {
            let api_key = Self::get_api_key(config)?;
            let language = Self::get_language(config);

            let details = self
                .fetch_video_details(api_key, std::slice::from_ref(&video_id))
                .await?;

            if let Some(video) = details.first() {
                let transcript = self
                    .download_captions(&video_id, &language)
                    .await
                    .unwrap_or(None);

                return Ok(vec![Self::video_to_content_item(
                    video,
                    transcript.as_deref(),
                    &config.id,
                )]);
            }
        }

        Ok(Vec::new())
    }
}

/// Extracts a video ID from a YouTube Atom/RSS notification.
fn extract_video_id_from_atom(xml: &str) -> Option<String> {
    // Look for <yt:videoId>VIDEO_ID</yt:videoId>
    let start_tag = "<yt:videoId>";
    let end_tag = "</yt:videoId>";

    if let Some(start) = xml.find(start_tag) {
        let content_start = start + start_tag.len();
        if let Some(end) = xml[content_start..].find(end_tag) {
            let video_id = xml[content_start..content_start + end].trim();
            if !video_id.is_empty() {
                return Some(video_id.to_string());
            }
        }
    }

    None
}

/// Parses ISO 8601 timestamp (e.g., "2024-01-15T10:30:00Z") to Unix seconds.
fn parse_iso8601_timestamp(ts: &str) -> Option<u64> {
    // Simple parser for YouTube's ISO 8601 format: YYYY-MM-DDTHH:MM:SSZ
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

    // Simple days-since-epoch calculation (approximate, ignoring leap seconds)
    let mut days: u64 = 0;
    for y in 1970..year {
        days += if is_leap_year(y) { 366 } else { 365 };
    }
    let days_in_months = [
        31,
        28 + if is_leap_year(year) { 1 } else { 0 },
        31,
        30,
        31,
        30,
        31,
        31,
        30,
        31,
        30,
        31,
    ];
    for d in days_in_months
        .iter()
        .take((month.saturating_sub(1)) as usize)
    {
        days += d;
    }
    days += day.saturating_sub(1);

    Some(days * 86400 + hour * 3600 + min * 60 + sec)
}

fn is_leap_year(year: u64) -> bool {
    (year.is_multiple_of(4) && !year.is_multiple_of(100)) || year.is_multiple_of(400)
}

// --- YouTube Data API v3 response types ---

#[derive(Debug, Deserialize)]
struct ApiError {
    code: u32,
    message: String,
}

#[derive(Debug, Deserialize)]
struct VideoListResponse {
    #[serde(default)]
    items: Option<Vec<VideoDetails>>,
    error: Option<ApiError>,
}

#[derive(Debug, Deserialize)]
struct VideoDetails {
    id: String,
    snippet: VideoSnippet,
    #[serde(rename = "contentDetails")]
    content_details: Option<VideoContentDetails>,
}

#[derive(Debug, Deserialize)]
struct VideoSnippet {
    title: String,
    description: String,
    #[serde(rename = "publishedAt")]
    published_at: String,
    #[serde(rename = "channelId")]
    channel_id: Option<String>,
    #[serde(rename = "channelTitle")]
    channel_title: Option<String>,
}

#[derive(Debug, Deserialize)]
struct VideoContentDetails {
    duration: Option<String>,
}

#[derive(Debug, Deserialize)]
struct ChannelListResponse {
    #[serde(default)]
    items: Option<Vec<ChannelDetails>>,
    error: Option<ApiError>,
}

#[derive(Debug, Deserialize)]
struct ChannelDetails {
    #[serde(rename = "contentDetails")]
    content_details: Option<ChannelContentDetails>,
}

#[derive(Debug, Deserialize)]
struct ChannelContentDetails {
    #[serde(rename = "relatedPlaylists")]
    related_playlists: Option<RelatedPlaylists>,
}

#[derive(Debug, Deserialize)]
struct RelatedPlaylists {
    uploads: Option<String>,
}

#[derive(Debug, Deserialize)]
struct PlaylistItemListResponse {
    #[serde(default)]
    items: Option<Vec<PlaylistItem>>,
    #[serde(rename = "nextPageToken")]
    next_page_token: Option<String>,
    error: Option<ApiError>,
}

#[derive(Debug, Deserialize)]
struct PlaylistItem {
    #[serde(rename = "contentDetails")]
    content_details: Option<PlaylistItemContentDetails>,
}

#[derive(Debug, Deserialize)]
struct PlaylistItemContentDetails {
    #[serde(rename = "videoId")]
    video_id: String,
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct CaptionListResponse {
    #[serde(default)]
    items: Option<Vec<CaptionTrack>>,
    error: Option<ApiError>,
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct CaptionTrack {
    id: String,
    snippet: Option<CaptionSnippet>,
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct CaptionSnippet {
    language: Option<String>,
    name: Option<String>,
    #[serde(rename = "trackKind")]
    track_kind: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn make_config(api_key: &str, settings: HashMap<String, serde_json::Value>) -> ConnectorConfig {
        ConnectorConfig {
            id: "youtube-test".to_string(),
            name: "Test YouTube".to_string(),
            connector_type: "youtube".to_string(),
            auth: AuthConfig::ApiKey {
                key: api_key.to_string(),
            },
            namespace: "test".to_string(),
            settings,
            enabled: true,
        }
    }

    fn make_video_config(api_key: &str, video_ids: Vec<&str>) -> ConnectorConfig {
        let mut settings = HashMap::new();
        settings.insert("video_ids".to_string(), serde_json::json!(video_ids));
        make_config(api_key, settings)
    }

    fn make_channel_config(api_key: &str, channel_id: &str) -> ConnectorConfig {
        let mut settings = HashMap::new();
        settings.insert("channel_id".to_string(), serde_json::json!(channel_id));
        make_config(api_key, settings)
    }

    #[test]
    fn test_youtube_connector_id_and_name() {
        let connector = YouTubeConnector::new();
        assert_eq!(connector.id(), "youtube");
        assert_eq!(connector.name(), "YouTube");
    }

    #[test]
    fn test_youtube_config_schema() {
        let connector = YouTubeConnector::new();
        let schema = connector.config_schema();
        assert_eq!(schema["type"], "object");
        assert!(schema["properties"]["auth"].is_object());
        assert!(schema["properties"]["settings"]["properties"]["video_ids"].is_object());
        assert!(schema["properties"]["settings"]["properties"]["channel_id"].is_object());
        assert!(schema["properties"]["settings"]["properties"]["playlist_id"].is_object());
        assert!(schema["properties"]["settings"]["properties"]["language"].is_object());
        assert!(
            schema["properties"]["settings"]["properties"]["include_auto_captions"].is_object()
        );
    }

    #[test]
    fn test_validate_config_valid_with_video_ids() {
        let connector = YouTubeConnector::new();
        let config = make_video_config("AIza_test_key", vec!["dQw4w9WgXcQ"]);
        assert!(connector.validate_config(&config).is_ok());
    }

    #[test]
    fn test_validate_config_valid_with_channel_id() {
        let connector = YouTubeConnector::new();
        let config = make_channel_config("AIza_test_key", "UCxxxxxxxxxxxxxxxx");
        assert!(connector.validate_config(&config).is_ok());
    }

    #[test]
    fn test_validate_config_valid_with_playlist_id() {
        let connector = YouTubeConnector::new();
        let mut settings = HashMap::new();
        settings.insert("playlist_id".to_string(), serde_json::json!("PLxxxxxxxx"));
        let config = make_config("AIza_test_key", settings);
        assert!(connector.validate_config(&config).is_ok());
    }

    #[test]
    fn test_validate_config_wrong_type() {
        let connector = YouTubeConnector::new();
        let mut config = make_video_config("key", vec!["id"]);
        config.connector_type = "github".to_string();
        assert!(connector.validate_config(&config).is_err());
    }

    #[test]
    fn test_validate_config_wrong_auth_type() {
        let connector = YouTubeConnector::new();
        let config = ConnectorConfig {
            id: "yt".to_string(),
            name: "Test".to_string(),
            connector_type: "youtube".to_string(),
            auth: AuthConfig::Token {
                token: "tok".to_string(),
            },
            namespace: "test".to_string(),
            settings: {
                let mut s = HashMap::new();
                s.insert("video_ids".to_string(), serde_json::json!(["abc"]));
                s
            },
            enabled: true,
        };
        let err = connector.validate_config(&config);
        assert!(err.is_err());
        assert!(err.unwrap_err().to_string().contains("ApiKey"));
    }

    #[test]
    fn test_validate_config_no_source() {
        let connector = YouTubeConnector::new();
        let config = make_config("key", HashMap::new());
        let err = connector.validate_config(&config);
        assert!(err.is_err());
        assert!(err.unwrap_err().to_string().contains("video_ids"));
    }

    #[test]
    fn test_get_video_ids() {
        let config = make_video_config("key", vec!["abc123", "def456"]);
        let ids = YouTubeConnector::get_video_ids(&config);
        assert_eq!(ids, vec!["abc123", "def456"]);
    }

    #[test]
    fn test_get_video_ids_empty() {
        let config = make_config("key", HashMap::new());
        let ids = YouTubeConnector::get_video_ids(&config);
        assert!(ids.is_empty());
    }

    #[test]
    fn test_get_channel_id() {
        let config = make_channel_config("key", "UC123");
        assert_eq!(
            YouTubeConnector::get_channel_id(&config),
            Some("UC123".to_string())
        );
    }

    #[test]
    fn test_get_language_default() {
        let config = make_config("key", HashMap::new());
        assert_eq!(YouTubeConnector::get_language(&config), "en");
    }

    #[test]
    fn test_get_language_custom() {
        let mut settings = HashMap::new();
        settings.insert("language".to_string(), serde_json::json!("es"));
        let config = make_config("key", settings);
        assert_eq!(YouTubeConnector::get_language(&config), "es");
    }

    #[test]
    fn test_include_auto_captions_default() {
        let config = make_config("key", HashMap::new());
        assert!(YouTubeConnector::include_auto_captions(&config));
    }

    #[test]
    fn test_include_auto_captions_disabled() {
        let mut settings = HashMap::new();
        settings.insert(
            "include_auto_captions".to_string(),
            serde_json::json!(false),
        );
        let config = make_config("key", settings);
        assert!(!YouTubeConnector::include_auto_captions(&config));
    }

    #[test]
    fn test_get_max_results_default() {
        let config = make_config("key", HashMap::new());
        assert_eq!(YouTubeConnector::get_max_results(&config), 50);
    }

    #[test]
    fn test_get_max_results_capped() {
        let mut settings = HashMap::new();
        settings.insert("max_results".to_string(), serde_json::json!(100));
        let config = make_config("key", settings);
        assert_eq!(YouTubeConnector::get_max_results(&config), 50);
    }

    #[test]
    fn test_parse_srv3_captions_basic() {
        let xml = r#"<?xml version="1.0" encoding="utf-8"?>
<transcript>
<text start="0" dur="5.2">Hello everyone</text>
<text start="5.2" dur="3.1">Welcome to the video</text>
<text start="8.3" dur="4.0">Today we will learn Rust</text>
</transcript>"#;

        let result = YouTubeConnector::parse_srv3_captions(xml);
        assert_eq!(
            result,
            "Hello everyone Welcome to the video Today we will learn Rust"
        );
    }

    #[test]
    fn test_parse_srv3_captions_with_entities() {
        let xml = r#"<transcript>
<text start="0" dur="3">Hello &amp; welcome</text>
<text start="3" dur="2">Use &lt;script&gt; tags</text>
</transcript>"#;

        let result = YouTubeConnector::parse_srv3_captions(xml);
        assert_eq!(result, "Hello & welcome Use <script> tags");
    }

    #[test]
    fn test_parse_srv3_captions_empty() {
        let xml = "<transcript></transcript>";
        let result = YouTubeConnector::parse_srv3_captions(xml);
        assert_eq!(result, "");
    }

    #[test]
    fn test_parse_srv3_captions_whitespace_only() {
        let xml = r#"<transcript>
<text start="0" dur="1">  </text>
<text start="1" dur="1">Hello</text>
</transcript>"#;

        let result = YouTubeConnector::parse_srv3_captions(xml);
        assert_eq!(result, "Hello");
    }

    #[test]
    fn test_decode_xml_entities() {
        assert_eq!(
            YouTubeConnector::decode_xml_entities("&amp; &lt; &gt; &quot; &apos; &#39;"),
            "& < > \" ' '"
        );
    }

    #[test]
    fn test_video_to_content_item_with_transcript() {
        let video = VideoDetails {
            id: "dQw4w9WgXcQ".to_string(),
            snippet: VideoSnippet {
                title: "Test Video".to_string(),
                description: "A test video description".to_string(),
                published_at: "2024-01-15T10:30:00Z".to_string(),
                channel_id: Some("UC123".to_string()),
                channel_title: Some("Test Channel".to_string()),
            },
            content_details: Some(VideoContentDetails {
                duration: Some("PT5M30S".to_string()),
            }),
        };

        let item = YouTubeConnector::video_to_content_item(
            &video,
            Some("Hello everyone welcome to the video"),
            "conn-1",
        );

        assert!(item.content.contains("Title: Test Video"));
        assert!(item.content.contains("Channel: Test Channel"));
        assert!(item
            .content
            .contains("Description: A test video description"));
        assert!(item
            .content
            .contains("Transcript: Hello everyone welcome to the video"));
        assert_eq!(item.source.connector_type, "youtube");
        assert_eq!(item.source.connector_id, "conn-1");
        assert_eq!(item.source.source_id, "dQw4w9WgXcQ");
        assert_eq!(
            item.source.source_url.as_deref(),
            Some("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
        );
        assert_eq!(item.source.author.as_deref(), Some("Test Channel"));
        assert!(item.source.created_at.is_some());
        assert_eq!(
            item.source
                .extra
                .get("has_transcript")
                .and_then(|v| v.as_bool()),
            Some(true)
        );
        assert_eq!(
            item.source.extra.get("duration").and_then(|v| v.as_str()),
            Some("PT5M30S")
        );
    }

    #[test]
    fn test_video_to_content_item_without_transcript() {
        let video = VideoDetails {
            id: "abc123".to_string(),
            snippet: VideoSnippet {
                title: "No Captions Video".to_string(),
                description: "".to_string(),
                published_at: "2024-06-01T00:00:00Z".to_string(),
                channel_id: None,
                channel_title: None,
            },
            content_details: None,
        };

        let item = YouTubeConnector::video_to_content_item(&video, None, "conn-2");

        assert!(item.content.contains("Title: No Captions Video"));
        assert!(!item.content.contains("Transcript:"));
        assert!(!item.content.contains("Description:"));
        assert_eq!(
            item.source
                .extra
                .get("has_transcript")
                .and_then(|v| v.as_bool()),
            Some(false)
        );
    }

    #[test]
    fn test_video_to_content_item_long_description_truncated() {
        let long_desc = "A".repeat(600);
        let video = VideoDetails {
            id: "xyz".to_string(),
            snippet: VideoSnippet {
                title: "Long Desc".to_string(),
                description: long_desc,
                published_at: "2024-01-01T00:00:00Z".to_string(),
                channel_id: None,
                channel_title: None,
            },
            content_details: None,
        };

        let item = YouTubeConnector::video_to_content_item(&video, None, "conn");
        // Description should be truncated at 500 chars + "..."
        assert!(item.content.contains("..."));
        assert!(item.content.len() < 600);
    }

    #[test]
    fn test_parse_iso8601_timestamp() {
        // 2024-01-15T10:30:00Z
        let ts = parse_iso8601_timestamp("2024-01-15T10:30:00Z");
        assert!(ts.is_some());
        let secs = ts.unwrap();
        // Approximate check: should be somewhere around 1705312200
        assert!(secs > 1705000000);
        assert!(secs < 1706000000);
    }

    #[test]
    fn test_parse_iso8601_timestamp_epoch() {
        let ts = parse_iso8601_timestamp("1970-01-01T00:00:00Z");
        assert_eq!(ts, Some(0));
    }

    #[test]
    fn test_parse_iso8601_timestamp_invalid() {
        assert!(parse_iso8601_timestamp("not a date").is_none());
        assert!(parse_iso8601_timestamp("2024-01-15").is_none());
    }

    #[test]
    fn test_extract_video_id_from_atom() {
        let xml = r#"<feed xmlns:yt="http://www.youtube.com/xml/schemas/2015">
  <entry>
    <yt:videoId>dQw4w9WgXcQ</yt:videoId>
    <title>Test Video</title>
  </entry>
</feed>"#;

        let video_id = extract_video_id_from_atom(xml);
        assert_eq!(video_id, Some("dQw4w9WgXcQ".to_string()));
    }

    #[test]
    fn test_extract_video_id_from_atom_missing() {
        let xml = "<feed><entry><title>No video ID</title></entry></feed>";
        assert!(extract_video_id_from_atom(xml).is_none());
    }

    #[test]
    fn test_extract_video_id_from_atom_empty() {
        let xml = "<feed><entry><yt:videoId></yt:videoId></entry></feed>";
        assert!(extract_video_id_from_atom(xml).is_none());
    }

    #[tokio::test]
    async fn test_handle_webhook_hub_verification() {
        let connector = YouTubeConnector::new();
        let config = make_video_config("key", vec!["abc"]);
        let mut headers = HashMap::new();
        headers.insert("hub.challenge".to_string(), "challenge123".to_string());

        let payload = WebhookPayload {
            body: Vec::new(),
            headers,
            content_type: None,
        };

        let items = connector.handle_webhook(&config, payload).await.unwrap();
        assert!(items.is_empty());
    }

    #[test]
    fn test_default_constructor() {
        let connector = YouTubeConnector::default();
        assert_eq!(connector.id(), "youtube");
    }

    #[test]
    fn test_is_leap_year() {
        assert!(is_leap_year(2000));
        assert!(is_leap_year(2024));
        assert!(!is_leap_year(2023));
        assert!(!is_leap_year(1900));
    }
}
