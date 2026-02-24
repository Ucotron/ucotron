//! Google Drive connector — fetches images, videos, and documents from Google Drive.
//!
//! Uses OAuth2 access tokens to authenticate with the Google Drive API v3.
//! Supports full sync (all files from configured folder IDs or file IDs)
//! and incremental sync via `modifiedTime` filtering.
//! Maps files to ContentItem with media metadata (MIME type, filename, size).

use std::collections::HashMap;

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};

use crate::connector::{
    AuthConfig, Connector, ConnectorConfig, ContentItem, MediaAttachment, SourceMetadata,
    SyncCursor, SyncResult, WebhookPayload,
};

const DRIVE_API_BASE: &str = "https://www.googleapis.com/drive/v3";

/// Maximum file size to download for media attachments (20 MB).
const MAX_MEDIA_SIZE: u64 = 20 * 1024 * 1024;

/// Maximum number of files to fetch per request (Drive API max is 1000).
const MAX_PAGE_SIZE: u32 = 100;

/// MIME types considered as images.
const IMAGE_MIMES: &[&str] = &[
    "image/png",
    "image/jpeg",
    "image/gif",
    "image/webp",
    "image/svg+xml",
    "image/bmp",
    "image/tiff",
];

/// MIME types considered as videos.
const VIDEO_MIMES: &[&str] = &[
    "video/mp4",
    "video/quicktime",
    "video/x-msvideo",
    "video/webm",
    "video/mpeg",
    "video/x-matroska",
];

/// MIME types considered as documents (text-based, content extracted as text).
const DOCUMENT_MIMES: &[&str] = &[
    "application/pdf",
    "text/plain",
    "text/csv",
    "text/html",
    "text/markdown",
    "application/rtf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation",
];

/// Google Workspace MIME types that can be exported.
const GOOGLE_DOC_MIME: &str = "application/vnd.google-apps.document";
const GOOGLE_SHEET_MIME: &str = "application/vnd.google-apps.spreadsheet";
const GOOGLE_SLIDES_MIME: &str = "application/vnd.google-apps.presentation";

/// Google Drive connector for fetching media files (images, videos, documents).
///
/// Requires an OAuth2 access token with the following scopes:
/// - `https://www.googleapis.com/auth/drive.readonly`
///
/// Configure folder IDs to discover files, or provide explicit file IDs.
/// Supports filtering by MIME type categories (images, videos, documents).
pub struct GDriveConnector {
    client: reqwest::Client,
}

impl GDriveConnector {
    /// Creates a new GDriveConnector with a default HTTP client.
    pub fn new() -> Self {
        Self {
            client: reqwest::Client::builder()
                .user_agent("ucotron-connector/0.1")
                .build()
                .unwrap_or_else(|_| reqwest::Client::new()),
        }
    }

    /// Creates a new GDriveConnector with a custom HTTP client.
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
            AuthConfig::OAuth2 { .. } => {
                bail!("Google Drive connector requires OAuth2 with a valid access_token")
            }
            _ => bail!("Google Drive connector requires OAuth2 authentication"),
        }
    }

    /// Extracts configured folder IDs from settings.
    fn get_folder_ids(config: &ConnectorConfig) -> Vec<String> {
        config
            .settings
            .get("folder_ids")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str().map(String::from))
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Extracts configured file IDs from settings.
    fn get_file_ids(config: &ConnectorConfig) -> Vec<String> {
        config
            .settings
            .get("file_ids")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str().map(String::from))
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Whether to include images (default: true).
    fn include_images(config: &ConnectorConfig) -> bool {
        config
            .settings
            .get("include_images")
            .and_then(|v| v.as_bool())
            .unwrap_or(true)
    }

    /// Whether to include videos (default: true).
    fn include_videos(config: &ConnectorConfig) -> bool {
        config
            .settings
            .get("include_videos")
            .and_then(|v| v.as_bool())
            .unwrap_or(true)
    }

    /// Whether to include documents (default: true).
    fn include_documents(config: &ConnectorConfig) -> bool {
        config
            .settings
            .get("include_documents")
            .and_then(|v| v.as_bool())
            .unwrap_or(true)
    }

    /// Whether to download media content as attachments (default: false).
    /// When false, only metadata is returned (saves bandwidth).
    fn download_media(config: &ConnectorConfig) -> bool {
        config
            .settings
            .get("download_media")
            .and_then(|v| v.as_bool())
            .unwrap_or(false)
    }

    /// Maximum file size for media downloads in bytes (default: 20 MB).
    fn max_file_size(config: &ConnectorConfig) -> u64 {
        config
            .settings
            .get("max_file_size")
            .and_then(|v| v.as_u64())
            .unwrap_or(MAX_MEDIA_SIZE)
    }

    /// Whether to recurse into subfolders (default: false).
    fn recurse_folders(config: &ConnectorConfig) -> bool {
        config
            .settings
            .get("recurse_folders")
            .and_then(|v| v.as_bool())
            .unwrap_or(false)
    }

    /// Builds the MIME type filter query for the Drive API.
    fn build_mime_filter(config: &ConnectorConfig) -> String {
        let mut mime_conditions = Vec::new();

        if Self::include_images(config) {
            for mime in IMAGE_MIMES {
                mime_conditions.push(format!("mimeType = '{}'", mime));
            }
        }

        if Self::include_videos(config) {
            for mime in VIDEO_MIMES {
                mime_conditions.push(format!("mimeType = '{}'", mime));
            }
        }

        if Self::include_documents(config) {
            for mime in DOCUMENT_MIMES {
                mime_conditions.push(format!("mimeType = '{}'", mime));
            }
            // Include Google Workspace types
            mime_conditions.push(format!("mimeType = '{}'", GOOGLE_DOC_MIME));
            mime_conditions.push(format!("mimeType = '{}'", GOOGLE_SHEET_MIME));
            mime_conditions.push(format!("mimeType = '{}'", GOOGLE_SLIDES_MIME));
        }

        if mime_conditions.is_empty() {
            // If nothing is selected, return a filter that matches nothing
            return "mimeType = 'application/x-ucotron-none'".to_string();
        }

        format!("({})", mime_conditions.join(" or "))
    }

    /// Lists files in a Google Drive folder.
    async fn list_folder_files(
        &self,
        token: &str,
        folder_id: &str,
        mime_filter: &str,
        modified_after: Option<&str>,
        recurse: bool,
    ) -> Result<Vec<DriveFile>> {
        let mut all_files = Vec::new();
        let mut folders_to_process = vec![folder_id.to_string()];

        while let Some(current_folder) = folders_to_process.pop() {
            let mut page_token: Option<String> = None;

            loop {
                let mut query = format!(
                    "'{}' in parents and trashed = false and {}",
                    current_folder, mime_filter
                );
                if let Some(after) = modified_after {
                    query.push_str(&format!(" and modifiedTime > '{}'", after));
                }

                let fields = "nextPageToken,files(id,name,mimeType,size,modifiedTime,createdTime,owners,webViewLink,webContentLink,thumbnailLink,description)";
                let page_size = MAX_PAGE_SIZE.to_string();
                let mut params: Vec<(&str, &str)> =
                    vec![("q", &query), ("fields", fields), ("pageSize", &page_size)];
                let page_token_val;
                if let Some(ref pt) = page_token {
                    page_token_val = pt.clone();
                    params.push(("pageToken", &page_token_val));
                }

                let resp = self
                    .client
                    .get(format!("{}/files", DRIVE_API_BASE))
                    .bearer_auth(token)
                    .query(&params)
                    .send()
                    .await
                    .context("Failed to list Google Drive folder files")?;

                if !resp.status().is_success() {
                    let status = resp.status();
                    let body = resp.text().await.unwrap_or_default();
                    bail!(
                        "Google Drive API error listing folder {}: {} - {}",
                        current_folder,
                        status,
                        body
                    );
                }

                let data: DriveListResponse = resp
                    .json()
                    .await
                    .context("Failed to parse Drive folder listing")?;

                all_files.extend(data.files);

                match data.next_page_token {
                    Some(t) => page_token = Some(t),
                    None => break,
                }
            }

            // If recursing, also list subfolders
            if recurse {
                let subfolder_files = self.list_subfolders(token, &current_folder).await?;
                for sf in subfolder_files {
                    folders_to_process.push(sf.id);
                }
            }
        }

        Ok(all_files)
    }

    /// Lists subfolders within a given folder.
    async fn list_subfolders(&self, token: &str, folder_id: &str) -> Result<Vec<DriveFile>> {
        let mut subfolders = Vec::new();
        let mut page_token: Option<String> = None;

        loop {
            let query = format!(
                "'{}' in parents and mimeType = 'application/vnd.google-apps.folder' and trashed = false",
                folder_id
            );
            let page_size = MAX_PAGE_SIZE.to_string();
            let mut params: Vec<(&str, &str)> = vec![
                ("q", &query),
                ("fields", "nextPageToken,files(id,name)"),
                ("pageSize", &page_size),
            ];
            let page_token_val;
            if let Some(ref pt) = page_token {
                page_token_val = pt.clone();
                params.push(("pageToken", &page_token_val));
            }

            let resp = self
                .client
                .get(format!("{}/files", DRIVE_API_BASE))
                .bearer_auth(token)
                .query(&params)
                .send()
                .await
                .context("Failed to list subfolders")?;

            if !resp.status().is_success() {
                break;
            }

            let data: DriveListResponse = resp
                .json()
                .await
                .context("Failed to parse subfolder listing")?;

            subfolders.extend(data.files);

            match data.next_page_token {
                Some(t) => page_token = Some(t),
                None => break,
            }
        }

        Ok(subfolders)
    }

    /// Fetches a single file's metadata by its ID.
    async fn get_file_metadata(&self, token: &str, file_id: &str) -> Result<DriveFile> {
        let fields = "id,name,mimeType,size,modifiedTime,createdTime,owners,webViewLink,webContentLink,thumbnailLink,description";
        let resp = self
            .client
            .get(format!("{}/files/{}", DRIVE_API_BASE, file_id))
            .bearer_auth(token)
            .query(&[("fields", fields)])
            .send()
            .await
            .context("Failed to fetch file metadata")?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            bail!(
                "Google Drive API error fetching file {}: {} - {}",
                file_id,
                status,
                body
            );
        }

        resp.json()
            .await
            .context("Failed to parse file metadata response")
    }

    /// Downloads file content bytes.
    async fn download_file(&self, token: &str, file_id: &str, mime_type: &str) -> Result<Vec<u8>> {
        let url = if is_google_workspace_type(mime_type) {
            // Export Google Workspace files to a downloadable format
            let export_mime = google_export_mime(mime_type);
            format!(
                "{}/files/{}/export?mimeType={}",
                DRIVE_API_BASE, file_id, export_mime
            )
        } else {
            format!("{}/files/{}?alt=media", DRIVE_API_BASE, file_id)
        };

        let resp = self
            .client
            .get(&url)
            .bearer_auth(token)
            .send()
            .await
            .context("Failed to download file")?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            bail!(
                "Google Drive download error for {}: {} - {}",
                file_id,
                status,
                body
            );
        }

        let bytes = resp.bytes().await.context("Failed to read file bytes")?;
        Ok(bytes.to_vec())
    }

    /// Converts a Drive file into a ContentItem.
    fn file_to_content_item(
        &self,
        file: &DriveFile,
        connector_id: &str,
        media_data: Option<Vec<u8>>,
    ) -> ContentItem {
        let mime = file
            .mime_type
            .as_deref()
            .unwrap_or("application/octet-stream");
        let name = file.name.as_deref().unwrap_or("Untitled");
        let file_size = file.size.as_deref().and_then(|s| s.parse::<u64>().ok());

        // Build descriptive content from file metadata
        let mut parts = Vec::new();
        parts.push(name.to_string());

        if let Some(ref desc) = file.description {
            if !desc.is_empty() {
                parts.push(desc.clone());
            }
        }

        // Add file type info
        let category = categorize_mime(mime);
        parts.push(format!("Type: {} ({})", category, mime));

        if let Some(size) = file_size {
            parts.push(format!("Size: {}", format_file_size(size)));
        }

        let content = parts.join("\n");

        // Build extra metadata
        let mut extra = HashMap::new();
        extra.insert(
            "file_id".to_string(),
            serde_json::Value::String(file.id.clone()),
        );
        extra.insert(
            "mime_type".to_string(),
            serde_json::Value::String(mime.to_string()),
        );
        extra.insert(
            "category".to_string(),
            serde_json::Value::String(category.to_string()),
        );
        if let Some(ref name) = file.name {
            extra.insert(
                "filename".to_string(),
                serde_json::Value::String(name.clone()),
            );
        }
        if let Some(size) = file_size {
            extra.insert("file_size".to_string(), serde_json::json!(size));
        }
        if let Some(ref thumb) = file.thumbnail_link {
            extra.insert(
                "thumbnail_url".to_string(),
                serde_json::Value::String(thumb.clone()),
            );
        }
        if let Some(ref web_link) = file.web_content_link {
            extra.insert(
                "download_url".to_string(),
                serde_json::Value::String(web_link.clone()),
            );
        }

        // Source URL
        let source_url = file
            .web_view_link
            .clone()
            .or_else(|| Some(format!("https://drive.google.com/file/d/{}/view", file.id)));

        // Timestamps
        let created_at = file
            .modified_time
            .as_deref()
            .or(file.created_time.as_deref())
            .and_then(parse_rfc3339_timestamp);

        // Owner
        let author = file
            .owners
            .as_ref()
            .and_then(|owners| owners.first())
            .and_then(|o| o.display_name.clone());

        // Build media attachment if download was requested
        let media = media_data.map(|data| MediaAttachment {
            mime_type: mime.to_string(),
            data,
            filename: file.name.clone(),
        });

        ContentItem {
            content,
            source: SourceMetadata {
                connector_type: "google_drive".to_string(),
                connector_id: connector_id.to_string(),
                source_id: file.id.clone(),
                source_url,
                author,
                created_at,
                extra,
            },
            media,
        }
    }
}

impl Default for GDriveConnector {
    fn default() -> Self {
        Self::new()
    }
}

impl Connector for GDriveConnector {
    fn id(&self) -> &str {
        "google_drive"
    }

    fn name(&self) -> &str {
        "Google Drive"
    }

    fn config_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "auth": {
                    "type": "object",
                    "description": "OAuth2 credentials for Google Drive API",
                    "properties": {
                        "client_id": { "type": "string", "description": "OAuth2 client ID" },
                        "client_secret": { "type": "string", "description": "OAuth2 client secret" },
                        "access_token": { "type": "string", "description": "OAuth2 access token" },
                        "refresh_token": { "type": "string", "description": "OAuth2 refresh token" }
                    },
                    "required": ["access_token"]
                },
                "settings": {
                    "type": "object",
                    "properties": {
                        "folder_ids": {
                            "type": "array",
                            "items": { "type": "string" },
                            "description": "Google Drive folder IDs to fetch files from"
                        },
                        "file_ids": {
                            "type": "array",
                            "items": { "type": "string" },
                            "description": "Specific file IDs to fetch"
                        },
                        "include_images": {
                            "type": "boolean",
                            "description": "Include image files (default: true)"
                        },
                        "include_videos": {
                            "type": "boolean",
                            "description": "Include video files (default: true)"
                        },
                        "include_documents": {
                            "type": "boolean",
                            "description": "Include document files (default: true)"
                        },
                        "download_media": {
                            "type": "boolean",
                            "description": "Download media content as attachments (default: false)"
                        },
                        "max_file_size": {
                            "type": "integer",
                            "description": "Maximum file size for downloads in bytes (default: 20MB)"
                        },
                        "recurse_folders": {
                            "type": "boolean",
                            "description": "Recurse into subfolders (default: false)"
                        }
                    }
                }
            },
            "required": ["auth"]
        })
    }

    fn validate_config(&self, config: &ConnectorConfig) -> Result<()> {
        if config.connector_type != "google_drive" {
            bail!(
                "Invalid connector type '{}', expected 'google_drive'",
                config.connector_type
            );
        }
        Self::get_access_token(config)?;

        let folder_ids = Self::get_folder_ids(config);
        let file_ids = Self::get_file_ids(config);

        if folder_ids.is_empty() && file_ids.is_empty() {
            bail!("Google Drive connector requires either folder_ids or file_ids in settings");
        }

        // Validate that at least one content type is included
        if !Self::include_images(config)
            && !Self::include_videos(config)
            && !Self::include_documents(config)
        {
            bail!("At least one content type (images, videos, documents) must be enabled");
        }

        Ok(())
    }

    async fn fetch(&self, config: &ConnectorConfig) -> Result<Vec<ContentItem>> {
        let token = Self::get_access_token(config)?;
        let do_download = Self::download_media(config);
        let max_size = Self::max_file_size(config);
        let recurse = Self::recurse_folders(config);
        let mime_filter = Self::build_mime_filter(config);
        let mut items = Vec::new();
        let mut seen_ids = std::collections::HashSet::new();

        // Fetch from folders
        for folder_id in &Self::get_folder_ids(config) {
            let files = self
                .list_folder_files(token, folder_id, &mime_filter, None, recurse)
                .await?;

            for file in &files {
                if !seen_ids.insert(file.id.clone()) {
                    continue; // Skip duplicates
                }

                let media_data = if do_download {
                    self.try_download_file(token, file, max_size).await
                } else {
                    None
                };

                items.push(self.file_to_content_item(file, &config.id, media_data));
            }
        }

        // Fetch explicit file IDs
        for file_id in &Self::get_file_ids(config) {
            if !seen_ids.insert(file_id.clone()) {
                continue;
            }

            match self.get_file_metadata(token, file_id).await {
                Ok(file) => {
                    let media_data = if do_download {
                        self.try_download_file(token, &file, max_size).await
                    } else {
                        None
                    };
                    items.push(self.file_to_content_item(&file, &config.id, media_data));
                }
                Err(e) => {
                    eprintln!("Warning: failed to fetch Drive file {}: {}", file_id, e);
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
        let token = Self::get_access_token(config)?;
        let do_download = Self::download_media(config);
        let max_size = Self::max_file_size(config);
        let recurse = Self::recurse_folders(config);
        let mime_filter = Self::build_mime_filter(config);

        let modified_after = cursor.value.as_deref();
        let mut items = Vec::new();
        let mut latest_modified: Option<String> = cursor.value.clone();
        let mut seen_ids = std::collections::HashSet::new();

        // Fetch from folders with modifiedTime filter
        for folder_id in &Self::get_folder_ids(config) {
            let files = self
                .list_folder_files(token, folder_id, &mime_filter, modified_after, recurse)
                .await?;

            for file in &files {
                if !seen_ids.insert(file.id.clone()) {
                    continue;
                }

                if let Some(ref mod_time) = file.modified_time {
                    if latest_modified
                        .as_ref()
                        .is_none_or(|current| mod_time > current)
                    {
                        latest_modified = Some(mod_time.clone());
                    }
                }

                let media_data = if do_download {
                    self.try_download_file(token, file, max_size).await
                } else {
                    None
                };

                items.push(self.file_to_content_item(file, &config.id, media_data));
            }
        }

        // For explicit file IDs, re-fetch all (no modifiedTime filter on get)
        for file_id in &Self::get_file_ids(config) {
            if !seen_ids.insert(file_id.clone()) {
                continue;
            }

            match self.get_file_metadata(token, file_id).await {
                Ok(file) => {
                    // Check if this file was modified after cursor
                    if let Some(after) = modified_after {
                        if let Some(ref mod_time) = file.modified_time {
                            if mod_time.as_str() <= after {
                                continue; // Skip unmodified files
                            }
                        }
                    }

                    if let Some(ref mod_time) = file.modified_time {
                        if latest_modified
                            .as_ref()
                            .is_none_or(|current| mod_time > current)
                        {
                            latest_modified = Some(mod_time.clone());
                        }
                    }

                    let media_data = if do_download {
                        self.try_download_file(token, &file, max_size).await
                    } else {
                        None
                    };
                    items.push(self.file_to_content_item(&file, &config.id, media_data));
                }
                Err(e) => {
                    eprintln!("Warning: failed to fetch Drive file {}: {}", file_id, e);
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
                value: latest_modified,
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
        // Google Drive push notifications (via changes API) send file change events.
        // The resource ID can come from the JSON body or from X-Goog-Resource-Id header.

        let file_id = if let Ok(body) = serde_json::from_slice::<serde_json::Value>(&payload.body) {
            body.get("fileId")
                .or_else(|| body.get("resourceId"))
                .or_else(|| body.get("file_id"))
                .and_then(|v| v.as_str())
                .map(String::from)
        } else {
            None
        };

        let file_id = file_id.or_else(|| payload.headers.get("x-goog-resource-id").cloned());

        let Some(file_id) = file_id else {
            return Ok(Vec::new());
        };

        let token = Self::get_access_token(config)?;
        let do_download = Self::download_media(config);
        let max_size = Self::max_file_size(config);

        match self.get_file_metadata(token, &file_id).await {
            Ok(file) => {
                let media_data = if do_download {
                    self.try_download_file(token, &file, max_size).await
                } else {
                    None
                };
                Ok(vec![
                    self.file_to_content_item(&file, &config.id, media_data)
                ])
            }
            Err(e) => {
                bail!("Failed to fetch file {} from webhook: {}", file_id, e);
            }
        }
    }
}

impl GDriveConnector {
    /// Attempts to download a file, returning None if it's too large or fails.
    async fn try_download_file(
        &self,
        token: &str,
        file: &DriveFile,
        max_size: u64,
    ) -> Option<Vec<u8>> {
        let mime = file.mime_type.as_deref().unwrap_or("");
        let file_size = file
            .size
            .as_deref()
            .and_then(|s| s.parse::<u64>().ok())
            .unwrap_or(0);

        // Skip files that are too large (Google Workspace files have size=0, always download)
        if file_size > max_size && !is_google_workspace_type(mime) {
            return None;
        }

        match self.download_file(token, &file.id, mime).await {
            Ok(data) => {
                if data.len() as u64 > max_size {
                    None
                } else {
                    Some(data)
                }
            }
            Err(e) => {
                eprintln!("Warning: failed to download file {}: {}", file.id, e);
                None
            }
        }
    }
}

// --- Helper functions ---

/// Determines if a MIME type is a Google Workspace type (requires export).
fn is_google_workspace_type(mime: &str) -> bool {
    mime.starts_with("application/vnd.google-apps.")
}

/// Returns the export MIME type for a Google Workspace file.
fn google_export_mime(mime: &str) -> &str {
    match mime {
        GOOGLE_DOC_MIME => "text/plain",
        GOOGLE_SHEET_MIME => "text/csv",
        GOOGLE_SLIDES_MIME => "text/plain",
        _ => "text/plain",
    }
}

/// Categorizes a MIME type into a human-readable category.
fn categorize_mime(mime: &str) -> &str {
    if IMAGE_MIMES.contains(&mime) {
        "Image"
    } else if VIDEO_MIMES.contains(&mime) {
        "Video"
    } else if DOCUMENT_MIMES.contains(&mime) {
        "Document"
    } else if mime == GOOGLE_DOC_MIME {
        "Google Doc"
    } else if mime == GOOGLE_SHEET_MIME {
        "Google Sheet"
    } else if mime == GOOGLE_SLIDES_MIME {
        "Google Slides"
    } else if mime.starts_with("image/") {
        "Image"
    } else if mime.starts_with("video/") {
        "Video"
    } else if mime.starts_with("audio/") {
        "Audio"
    } else {
        "File"
    }
}

/// Formats a file size in human-readable form.
fn format_file_size(bytes: u64) -> String {
    if bytes < 1024 {
        format!("{} B", bytes)
    } else if bytes < 1024 * 1024 {
        format!("{:.1} KB", bytes as f64 / 1024.0)
    } else if bytes < 1024 * 1024 * 1024 {
        format!("{:.1} MB", bytes as f64 / (1024.0 * 1024.0))
    } else {
        format!("{:.2} GB", bytes as f64 / (1024.0 * 1024.0 * 1024.0))
    }
}

/// Parses an RFC 3339 timestamp (e.g., "2024-01-15T10:30:00.000Z") to Unix seconds.
fn parse_rfc3339_timestamp(ts: &str) -> Option<u64> {
    let ts = ts.trim_end_matches('Z');
    let ts = if let Some(dot_pos) = ts.rfind('.') {
        if ts[..dot_pos].contains('T') {
            &ts[..dot_pos]
        } else {
            ts
        }
    } else {
        ts
    };

    let parts: Vec<&str> = ts.split('T').collect();
    if parts.is_empty() {
        return None;
    }
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

// --- Google Drive API response types ---

#[derive(Debug, Clone, Deserialize, Serialize)]
struct DriveListResponse {
    #[serde(default)]
    files: Vec<DriveFile>,
    #[serde(default, rename = "nextPageToken")]
    next_page_token: Option<String>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
struct DriveFile {
    id: String,
    #[serde(default)]
    name: Option<String>,
    #[serde(default, rename = "mimeType")]
    mime_type: Option<String>,
    #[serde(default)]
    size: Option<String>,
    #[serde(default, rename = "modifiedTime")]
    modified_time: Option<String>,
    #[serde(default, rename = "createdTime")]
    created_time: Option<String>,
    #[serde(default)]
    owners: Option<Vec<DriveOwner>>,
    #[serde(default, rename = "webViewLink")]
    web_view_link: Option<String>,
    #[serde(default, rename = "webContentLink")]
    web_content_link: Option<String>,
    #[serde(default, rename = "thumbnailLink")]
    thumbnail_link: Option<String>,
    #[serde(default)]
    description: Option<String>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
struct DriveOwner {
    #[serde(default, rename = "displayName")]
    display_name: Option<String>,
    #[serde(default, rename = "emailAddress")]
    email_address: Option<String>,
}

// --- Tests ---

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn make_config(access_token: &str, folder_ids: Vec<&str>) -> ConnectorConfig {
        let mut settings = HashMap::new();
        settings.insert("folder_ids".to_string(), serde_json::json!(folder_ids));
        ConnectorConfig {
            id: "gdrive-test".to_string(),
            name: "Test Google Drive".to_string(),
            connector_type: "google_drive".to_string(),
            auth: AuthConfig::OAuth2 {
                client_id: "client-123".to_string(),
                client_secret: "secret-456".to_string(),
                access_token: Some(access_token.to_string()),
                refresh_token: None,
            },
            namespace: "test".to_string(),
            settings,
            enabled: true,
        }
    }

    fn make_config_with_files(access_token: &str, file_ids: Vec<&str>) -> ConnectorConfig {
        let mut settings = HashMap::new();
        settings.insert("file_ids".to_string(), serde_json::json!(file_ids));
        ConnectorConfig {
            id: "gdrive-files-test".to_string(),
            name: "Test Drive Files".to_string(),
            connector_type: "google_drive".to_string(),
            auth: AuthConfig::OAuth2 {
                client_id: "client-123".to_string(),
                client_secret: "secret-456".to_string(),
                access_token: Some(access_token.to_string()),
                refresh_token: None,
            },
            namespace: "test".to_string(),
            settings,
            enabled: true,
        }
    }

    fn make_drive_file(id: &str, name: &str, mime: &str, size: u64) -> DriveFile {
        DriveFile {
            id: id.to_string(),
            name: Some(name.to_string()),
            mime_type: Some(mime.to_string()),
            size: Some(size.to_string()),
            modified_time: Some("2024-06-15T12:00:00.000Z".to_string()),
            created_time: Some("2024-06-10T08:00:00.000Z".to_string()),
            owners: Some(vec![DriveOwner {
                display_name: Some("Alice Smith".to_string()),
                email_address: Some("alice@example.com".to_string()),
            }]),
            web_view_link: Some(format!("https://drive.google.com/file/d/{}/view", id)),
            web_content_link: Some(format!("https://drive.google.com/uc?id={}", id)),
            thumbnail_link: Some(format!("https://lh3.googleusercontent.com/{}", id)),
            description: Some("A test file".to_string()),
        }
    }

    fn make_image_file() -> DriveFile {
        make_drive_file("img-001", "photo.jpg", "image/jpeg", 2048000)
    }

    fn make_video_file() -> DriveFile {
        make_drive_file("vid-001", "video.mp4", "video/mp4", 50000000)
    }

    fn make_doc_file() -> DriveFile {
        make_drive_file("doc-001", "report.pdf", "application/pdf", 512000)
    }

    fn make_google_doc_file() -> DriveFile {
        DriveFile {
            id: "gdoc-001".to_string(),
            name: Some("My Google Doc".to_string()),
            mime_type: Some(GOOGLE_DOC_MIME.to_string()),
            size: None, // Google Workspace files have no size
            modified_time: Some("2024-07-20T15:30:00.000Z".to_string()),
            created_time: Some("2024-07-01T09:00:00.000Z".to_string()),
            owners: Some(vec![DriveOwner {
                display_name: Some("Bob Jones".to_string()),
                email_address: Some("bob@example.com".to_string()),
            }]),
            web_view_link: Some("https://docs.google.com/document/d/gdoc-001/edit".to_string()),
            web_content_link: None,
            thumbnail_link: None,
            description: None,
        }
    }

    #[test]
    fn test_connector_id_and_name() {
        let connector = GDriveConnector::new();
        assert_eq!(connector.id(), "google_drive");
        assert_eq!(connector.name(), "Google Drive");
    }

    #[test]
    fn test_default_constructor() {
        let connector = GDriveConnector::default();
        assert_eq!(connector.id(), "google_drive");
    }

    #[test]
    fn test_config_schema() {
        let connector = GDriveConnector::new();
        let schema = connector.config_schema();
        assert_eq!(schema["type"], "object");
        assert!(schema["properties"]["auth"].is_object());
        assert!(schema["properties"]["settings"]["properties"]["folder_ids"].is_object());
        assert!(schema["properties"]["settings"]["properties"]["file_ids"].is_object());
        assert!(schema["properties"]["settings"]["properties"]["include_images"].is_object());
        assert!(schema["properties"]["settings"]["properties"]["include_videos"].is_object());
        assert!(schema["properties"]["settings"]["properties"]["include_documents"].is_object());
        assert!(schema["properties"]["settings"]["properties"]["download_media"].is_object());
        assert!(schema["properties"]["settings"]["properties"]["max_file_size"].is_object());
        assert!(schema["properties"]["settings"]["properties"]["recurse_folders"].is_object());
    }

    #[test]
    fn test_validate_config_valid_with_folders() {
        let connector = GDriveConnector::new();
        let config = make_config("ya29.test-token", vec!["folder-abc"]);
        assert!(connector.validate_config(&config).is_ok());
    }

    #[test]
    fn test_validate_config_valid_with_files() {
        let connector = GDriveConnector::new();
        let config = make_config_with_files("ya29.test-token", vec!["file-xyz"]);
        assert!(connector.validate_config(&config).is_ok());
    }

    #[test]
    fn test_validate_config_wrong_type() {
        let connector = GDriveConnector::new();
        let mut config = make_config("ya29.test-token", vec!["folder-1"]);
        config.connector_type = "slack".to_string();
        let err = connector.validate_config(&config);
        assert!(err.is_err());
        assert!(err.unwrap_err().to_string().contains("google_drive"));
    }

    #[test]
    fn test_validate_config_wrong_auth_type() {
        let connector = GDriveConnector::new();
        let config = ConnectorConfig {
            id: "gdrive-test".to_string(),
            name: "Test".to_string(),
            connector_type: "google_drive".to_string(),
            auth: AuthConfig::Token {
                token: "not-oauth".to_string(),
            },
            namespace: "test".to_string(),
            settings: {
                let mut s = HashMap::new();
                s.insert("folder_ids".to_string(), serde_json::json!(["folder-1"]));
                s
            },
            enabled: true,
        };
        let err = connector.validate_config(&config);
        assert!(err.is_err());
        assert!(err.unwrap_err().to_string().contains("OAuth2"));
    }

    #[test]
    fn test_validate_config_missing_access_token() {
        let connector = GDriveConnector::new();
        let config = ConnectorConfig {
            id: "gdrive-test".to_string(),
            name: "Test".to_string(),
            connector_type: "google_drive".to_string(),
            auth: AuthConfig::OAuth2 {
                client_id: "c".to_string(),
                client_secret: "s".to_string(),
                access_token: None,
                refresh_token: None,
            },
            namespace: "test".to_string(),
            settings: {
                let mut s = HashMap::new();
                s.insert("folder_ids".to_string(), serde_json::json!(["folder-1"]));
                s
            },
            enabled: true,
        };
        let err = connector.validate_config(&config);
        assert!(err.is_err());
        assert!(err.unwrap_err().to_string().contains("access_token"));
    }

    #[test]
    fn test_validate_config_no_folders_or_files() {
        let connector = GDriveConnector::new();
        let config = ConnectorConfig {
            id: "gdrive-test".to_string(),
            name: "Test".to_string(),
            connector_type: "google_drive".to_string(),
            auth: AuthConfig::OAuth2 {
                client_id: "c".to_string(),
                client_secret: "s".to_string(),
                access_token: Some("token".to_string()),
                refresh_token: None,
            },
            namespace: "test".to_string(),
            settings: HashMap::new(),
            enabled: true,
        };
        let err = connector.validate_config(&config);
        assert!(err.is_err());
        assert!(err.unwrap_err().to_string().contains("folder_ids"));
    }

    #[test]
    fn test_validate_config_no_content_types() {
        let connector = GDriveConnector::new();
        let mut settings = HashMap::new();
        settings.insert("folder_ids".to_string(), serde_json::json!(["folder-1"]));
        settings.insert("include_images".to_string(), serde_json::json!(false));
        settings.insert("include_videos".to_string(), serde_json::json!(false));
        settings.insert("include_documents".to_string(), serde_json::json!(false));
        let config = ConnectorConfig {
            id: "gdrive-test".to_string(),
            name: "Test".to_string(),
            connector_type: "google_drive".to_string(),
            auth: AuthConfig::OAuth2 {
                client_id: "c".to_string(),
                client_secret: "s".to_string(),
                access_token: Some("token".to_string()),
                refresh_token: None,
            },
            namespace: "test".to_string(),
            settings,
            enabled: true,
        };
        let err = connector.validate_config(&config);
        assert!(err.is_err());
        assert!(err.unwrap_err().to_string().contains("content type"));
    }

    #[test]
    fn test_file_to_content_item_image() {
        let connector = GDriveConnector::new();
        let file = make_image_file();

        let item = connector.file_to_content_item(&file, "conn-1", None);

        assert!(item.content.contains("photo.jpg"));
        assert!(item.content.contains("Image"));
        assert!(item.content.contains("image/jpeg"));
        assert_eq!(item.source.connector_type, "google_drive");
        assert_eq!(item.source.connector_id, "conn-1");
        assert_eq!(item.source.source_id, "img-001");
        assert!(item.source.source_url.is_some());
        assert_eq!(item.source.author.as_deref(), Some("Alice Smith"));
        assert!(item.source.created_at.is_some());
        assert_eq!(
            item.source.extra.get("file_id").unwrap(),
            &serde_json::Value::String("img-001".to_string())
        );
        assert_eq!(
            item.source.extra.get("mime_type").unwrap(),
            &serde_json::Value::String("image/jpeg".to_string())
        );
        assert_eq!(
            item.source.extra.get("category").unwrap(),
            &serde_json::Value::String("Image".to_string())
        );
        assert!(item.source.extra.contains_key("thumbnail_url"));
        assert!(item.media.is_none()); // No download requested
    }

    #[test]
    fn test_file_to_content_item_video() {
        let connector = GDriveConnector::new();
        let file = make_video_file();

        let item = connector.file_to_content_item(&file, "conn-1", None);

        assert!(item.content.contains("video.mp4"));
        assert!(item.content.contains("Video"));
        assert_eq!(
            item.source.extra.get("category").unwrap(),
            &serde_json::Value::String("Video".to_string())
        );
    }

    #[test]
    fn test_file_to_content_item_document() {
        let connector = GDriveConnector::new();
        let file = make_doc_file();

        let item = connector.file_to_content_item(&file, "conn-1", None);

        assert!(item.content.contains("report.pdf"));
        assert!(item.content.contains("Document"));
        assert!(item.content.contains("500.0 KB"));
    }

    #[test]
    fn test_file_to_content_item_google_doc() {
        let connector = GDriveConnector::new();
        let file = make_google_doc_file();

        let item = connector.file_to_content_item(&file, "conn-1", None);

        assert!(item.content.contains("My Google Doc"));
        assert!(item.content.contains("Google Doc"));
        assert_eq!(item.source.author.as_deref(), Some("Bob Jones"));
        assert!(item.source.source_url.is_some());
    }

    #[test]
    fn test_file_to_content_item_with_media_data() {
        let connector = GDriveConnector::new();
        let file = make_image_file();
        let data = vec![0xFF, 0xD8, 0xFF, 0xE0]; // JPEG magic bytes

        let item = connector.file_to_content_item(&file, "conn-1", Some(data.clone()));

        assert!(item.media.is_some());
        let media = item.media.unwrap();
        assert_eq!(media.mime_type, "image/jpeg");
        assert_eq!(media.data, data);
        assert_eq!(media.filename.as_deref(), Some("photo.jpg"));
    }

    #[test]
    fn test_file_to_content_item_with_description() {
        let connector = GDriveConnector::new();
        let file = make_drive_file("f-001", "notes.txt", "text/plain", 256);

        let item = connector.file_to_content_item(&file, "conn-1", None);
        assert!(item.content.contains("A test file")); // description
    }

    #[test]
    fn test_file_to_content_item_no_metadata() {
        let connector = GDriveConnector::new();
        let file = DriveFile {
            id: "bare-file".to_string(),
            name: None,
            mime_type: None,
            size: None,
            modified_time: None,
            created_time: None,
            owners: None,
            web_view_link: None,
            web_content_link: None,
            thumbnail_link: None,
            description: None,
        };

        let item = connector.file_to_content_item(&file, "conn-1", None);
        assert!(item.content.contains("Untitled"));
        assert!(item.content.contains("application/octet-stream"));
        assert!(item.source.author.is_none());
        assert!(item.source.created_at.is_none());
        // Should still have a fallback source_url
        assert!(item.source.source_url.is_some());
    }

    #[test]
    fn test_get_folder_ids() {
        let config = make_config("token", vec!["folder-1", "folder-2"]);
        let ids = GDriveConnector::get_folder_ids(&config);
        assert_eq!(ids, vec!["folder-1", "folder-2"]);
    }

    #[test]
    fn test_get_folder_ids_empty() {
        let mut config = make_config("token", vec![]);
        config.settings.clear();
        let ids = GDriveConnector::get_folder_ids(&config);
        assert!(ids.is_empty());
    }

    #[test]
    fn test_get_file_ids() {
        let config = make_config_with_files("token", vec!["file-1", "file-2"]);
        let ids = GDriveConnector::get_file_ids(&config);
        assert_eq!(ids, vec!["file-1", "file-2"]);
    }

    #[test]
    fn test_include_images_default() {
        let config = make_config("token", vec!["f"]);
        assert!(GDriveConnector::include_images(&config));
    }

    #[test]
    fn test_include_images_disabled() {
        let mut config = make_config("token", vec!["f"]);
        config
            .settings
            .insert("include_images".to_string(), serde_json::json!(false));
        assert!(!GDriveConnector::include_images(&config));
    }

    #[test]
    fn test_include_videos_default() {
        let config = make_config("token", vec!["f"]);
        assert!(GDriveConnector::include_videos(&config));
    }

    #[test]
    fn test_include_documents_default() {
        let config = make_config("token", vec!["f"]);
        assert!(GDriveConnector::include_documents(&config));
    }

    #[test]
    fn test_download_media_default_false() {
        let config = make_config("token", vec!["f"]);
        assert!(!GDriveConnector::download_media(&config));
    }

    #[test]
    fn test_download_media_enabled() {
        let mut config = make_config("token", vec!["f"]);
        config
            .settings
            .insert("download_media".to_string(), serde_json::json!(true));
        assert!(GDriveConnector::download_media(&config));
    }

    #[test]
    fn test_max_file_size_default() {
        let config = make_config("token", vec!["f"]);
        assert_eq!(GDriveConnector::max_file_size(&config), MAX_MEDIA_SIZE);
    }

    #[test]
    fn test_max_file_size_custom() {
        let mut config = make_config("token", vec!["f"]);
        config
            .settings
            .insert("max_file_size".to_string(), serde_json::json!(5_000_000));
        assert_eq!(GDriveConnector::max_file_size(&config), 5_000_000);
    }

    #[test]
    fn test_recurse_folders_default() {
        let config = make_config("token", vec!["f"]);
        assert!(!GDriveConnector::recurse_folders(&config));
    }

    #[test]
    fn test_build_mime_filter_all() {
        let config = make_config("token", vec!["f"]);
        let filter = GDriveConnector::build_mime_filter(&config);
        assert!(filter.contains("image/png"));
        assert!(filter.contains("video/mp4"));
        assert!(filter.contains("application/pdf"));
        assert!(filter.contains(GOOGLE_DOC_MIME));
    }

    #[test]
    fn test_build_mime_filter_images_only() {
        let mut config = make_config("token", vec!["f"]);
        config
            .settings
            .insert("include_videos".to_string(), serde_json::json!(false));
        config
            .settings
            .insert("include_documents".to_string(), serde_json::json!(false));
        let filter = GDriveConnector::build_mime_filter(&config);
        assert!(filter.contains("image/png"));
        assert!(filter.contains("image/jpeg"));
        assert!(!filter.contains("video/mp4"));
        assert!(!filter.contains("application/pdf"));
    }

    #[test]
    fn test_build_mime_filter_none() {
        let mut config = make_config("token", vec!["f"]);
        config
            .settings
            .insert("include_images".to_string(), serde_json::json!(false));
        config
            .settings
            .insert("include_videos".to_string(), serde_json::json!(false));
        config
            .settings
            .insert("include_documents".to_string(), serde_json::json!(false));
        let filter = GDriveConnector::build_mime_filter(&config);
        assert!(filter.contains("ucotron-none"));
    }

    #[test]
    fn test_categorize_mime() {
        assert_eq!(categorize_mime("image/jpeg"), "Image");
        assert_eq!(categorize_mime("image/png"), "Image");
        assert_eq!(categorize_mime("video/mp4"), "Video");
        assert_eq!(categorize_mime("video/webm"), "Video");
        assert_eq!(categorize_mime("application/pdf"), "Document");
        assert_eq!(categorize_mime("text/plain"), "Document");
        assert_eq!(categorize_mime(GOOGLE_DOC_MIME), "Google Doc");
        assert_eq!(categorize_mime(GOOGLE_SHEET_MIME), "Google Sheet");
        assert_eq!(categorize_mime(GOOGLE_SLIDES_MIME), "Google Slides");
        assert_eq!(categorize_mime("image/heic"), "Image"); // fallback prefix
        assert_eq!(categorize_mime("video/x-flv"), "Video"); // fallback prefix
        assert_eq!(categorize_mime("audio/mpeg"), "Audio"); // fallback prefix
        assert_eq!(categorize_mime("application/zip"), "File"); // unknown
    }

    #[test]
    fn test_format_file_size() {
        assert_eq!(format_file_size(0), "0 B");
        assert_eq!(format_file_size(512), "512 B");
        assert_eq!(format_file_size(1024), "1.0 KB");
        assert_eq!(format_file_size(1536), "1.5 KB");
        assert_eq!(format_file_size(1048576), "1.0 MB");
        assert_eq!(format_file_size(1073741824), "1.00 GB");
    }

    #[test]
    fn test_is_google_workspace_type() {
        assert!(is_google_workspace_type(GOOGLE_DOC_MIME));
        assert!(is_google_workspace_type(GOOGLE_SHEET_MIME));
        assert!(is_google_workspace_type(GOOGLE_SLIDES_MIME));
        assert!(!is_google_workspace_type("image/png"));
        assert!(!is_google_workspace_type("application/pdf"));
    }

    #[test]
    fn test_google_export_mime() {
        assert_eq!(google_export_mime(GOOGLE_DOC_MIME), "text/plain");
        assert_eq!(google_export_mime(GOOGLE_SHEET_MIME), "text/csv");
        assert_eq!(google_export_mime(GOOGLE_SLIDES_MIME), "text/plain");
        assert_eq!(google_export_mime("unknown"), "text/plain");
    }

    #[test]
    fn test_parse_rfc3339_timestamp_full() {
        let ts = parse_rfc3339_timestamp("2024-01-01T00:00:00.000Z");
        assert_eq!(ts, Some(1704067200));
    }

    #[test]
    fn test_parse_rfc3339_timestamp_no_millis() {
        let ts = parse_rfc3339_timestamp("2024-01-01T00:00:00Z");
        assert_eq!(ts, Some(1704067200));
    }

    #[test]
    fn test_parse_rfc3339_timestamp_invalid() {
        assert!(parse_rfc3339_timestamp("not-a-date").is_none());
        assert!(parse_rfc3339_timestamp("").is_none());
    }

    #[test]
    fn test_drive_file_deserialization() {
        let json = serde_json::json!({
            "id": "1abc",
            "name": "Test Image",
            "mimeType": "image/png",
            "size": "2048",
            "modifiedTime": "2024-06-15T12:00:00.000Z",
            "createdTime": "2024-06-10T08:00:00.000Z",
            "owners": [{"displayName": "Alice", "emailAddress": "alice@example.com"}],
            "webViewLink": "https://drive.google.com/file/d/1abc/view",
            "webContentLink": "https://drive.google.com/uc?id=1abc",
            "thumbnailLink": "https://lh3.googleusercontent.com/1abc",
            "description": "My photo"
        });
        let file: DriveFile = serde_json::from_value(json).unwrap();
        assert_eq!(file.id, "1abc");
        assert_eq!(file.name.as_deref(), Some("Test Image"));
        assert_eq!(file.mime_type.as_deref(), Some("image/png"));
        assert_eq!(file.size.as_deref(), Some("2048"));
        assert_eq!(file.description.as_deref(), Some("My photo"));
        assert!(file.owners.unwrap()[0]
            .display_name
            .as_deref()
            .unwrap()
            .contains("Alice"));
    }

    #[test]
    fn test_drive_file_deserialization_minimal() {
        let json = serde_json::json!({ "id": "minimal" });
        let file: DriveFile = serde_json::from_value(json).unwrap();
        assert_eq!(file.id, "minimal");
        assert!(file.name.is_none());
        assert!(file.mime_type.is_none());
        assert!(file.size.is_none());
    }

    #[test]
    fn test_drive_list_response_deserialization() {
        let json = serde_json::json!({
            "files": [
                { "id": "f1", "name": "File 1" },
                { "id": "f2", "name": "File 2" }
            ],
            "nextPageToken": "token-abc"
        });
        let resp: DriveListResponse = serde_json::from_value(json).unwrap();
        assert_eq!(resp.files.len(), 2);
        assert_eq!(resp.next_page_token.as_deref(), Some("token-abc"));
    }

    #[test]
    fn test_drive_list_response_empty() {
        let json = serde_json::json!({});
        let resp: DriveListResponse = serde_json::from_value(json).unwrap();
        assert!(resp.files.is_empty());
        assert!(resp.next_page_token.is_none());
    }

    #[tokio::test]
    async fn test_handle_webhook_no_file_id() {
        let connector = GDriveConnector::new();
        let config = make_config("token", vec!["folder-1"]);

        let payload = WebhookPayload {
            body: serde_json::to_vec(&serde_json::json!({
                "event": "unknown"
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
        let connector = GDriveConnector::new();
        let config = make_config("token", vec!["folder-1"]);

        let payload = WebhookPayload {
            body: b"not json".to_vec(),
            headers: HashMap::new(),
            content_type: None,
        };

        // Invalid JSON body with no headers should return empty (not error)
        let items = connector.handle_webhook(&config, payload).await.unwrap();
        assert!(items.is_empty());
    }
}
