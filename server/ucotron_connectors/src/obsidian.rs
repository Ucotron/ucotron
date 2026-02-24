//! Obsidian connector — reads Markdown files from a local Obsidian vault.
//!
//! This connector reads `.md` files from a local directory, parses Obsidian-specific
//! features like `[[wikilinks]]` and `#tags`, and supports incremental sync
//! via file modification times.
//!
//! # Settings
//!
//! - `vault_path` (required): absolute path to the Obsidian vault directory
//! - `include_patterns`: array of glob patterns to include (default: `["**/*.md"]`)
//! - `exclude_patterns`: array of glob patterns to exclude (default: `[".obsidian/**", ".trash/**"]`)
//! - `max_content_length`: maximum content length per file (default: 10000)
//! - `parse_wikilinks`: whether to extract wikilinks as metadata (default: true)
//! - `parse_tags`: whether to extract tags as metadata (default: true)
//! - `parse_frontmatter`: whether to parse YAML frontmatter (default: true)

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};

use crate::connector::{
    AuthConfig, Connector, ConnectorConfig, ContentItem, SourceMetadata, SyncCursor, SyncResult,
    WebhookPayload,
};

/// Obsidian vault connector for reading local Markdown files.
///
/// Reads `.md` files from a local Obsidian vault directory, parsing
/// wikilinks (`[[Page]]`, `[[Page|alias]]`), tags (`#tag`), and
/// optional YAML frontmatter.
///
/// Uses `AuthConfig::None` since no authentication is needed for local files.
///
/// # Settings
///
/// - `vault_path`: path to the vault root directory (required)
/// - `include_patterns`: glob patterns for files to include (default: `["**/*.md"]`)
/// - `exclude_patterns`: glob patterns for files to exclude (default: `[".obsidian/**", ".trash/**"]`)
/// - `max_content_length`: max chars per file (default: 10000)
/// - `parse_wikilinks`: extract `[[wikilinks]]` to metadata (default: true)
/// - `parse_tags`: extract `#tags` to metadata (default: true)
/// - `parse_frontmatter`: parse YAML frontmatter (default: true)
pub struct ObsidianConnector;

impl ObsidianConnector {
    /// Creates a new ObsidianConnector.
    pub fn new() -> Self {
        Self
    }

    /// Extracts the vault path from connector settings.
    fn get_vault_path(config: &ConnectorConfig) -> Result<PathBuf> {
        let path_str = config
            .settings
            .get("vault_path")
            .and_then(|v| v.as_str())
            .ok_or_else(|| {
                anyhow::anyhow!("Missing 'vault_path' in Obsidian connector settings")
            })?;
        let path = PathBuf::from(path_str);
        if !path.is_absolute() {
            bail!("vault_path must be an absolute path, got: {}", path_str);
        }
        Ok(path)
    }

    /// Returns exclude patterns from settings (defaults to .obsidian/ and .trash/).
    fn get_exclude_patterns(config: &ConnectorConfig) -> Vec<String> {
        config
            .settings
            .get("exclude_patterns")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str().map(String::from))
                    .collect()
            })
            .unwrap_or_else(|| vec![".obsidian/**".to_string(), ".trash/**".to_string()])
    }

    /// Returns include patterns from settings (defaults to all .md files).
    fn get_include_patterns(config: &ConnectorConfig) -> Vec<String> {
        config
            .settings
            .get("include_patterns")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str().map(String::from))
                    .collect()
            })
            .unwrap_or_else(|| vec!["**/*.md".to_string()])
    }

    /// Maximum content length per file (default: 10000).
    fn max_content_length(config: &ConnectorConfig) -> usize {
        config
            .settings
            .get("max_content_length")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .unwrap_or(10000)
    }

    /// Whether to parse wikilinks (default: true).
    fn parse_wikilinks(config: &ConnectorConfig) -> bool {
        config
            .settings
            .get("parse_wikilinks")
            .and_then(|v| v.as_bool())
            .unwrap_or(true)
    }

    /// Whether to parse tags (default: true).
    fn parse_tags(config: &ConnectorConfig) -> bool {
        config
            .settings
            .get("parse_tags")
            .and_then(|v| v.as_bool())
            .unwrap_or(true)
    }

    /// Whether to parse YAML frontmatter (default: true).
    fn parse_frontmatter(config: &ConnectorConfig) -> bool {
        config
            .settings
            .get("parse_frontmatter")
            .and_then(|v| v.as_bool())
            .unwrap_or(true)
    }

    /// Collects all Markdown files in the vault that match include/exclude patterns.
    fn collect_files(vault_path: &Path, config: &ConnectorConfig) -> Result<Vec<PathBuf>> {
        let include = Self::get_include_patterns(config);
        let exclude = Self::get_exclude_patterns(config);

        let mut files = Vec::new();
        Self::walk_dir(vault_path, vault_path, &include, &exclude, &mut files)?;
        files.sort();
        Ok(files)
    }

    /// Recursively walks a directory collecting matching files.
    fn walk_dir(
        root: &Path,
        dir: &Path,
        include: &[String],
        exclude: &[String],
        files: &mut Vec<PathBuf>,
    ) -> Result<()> {
        let entries = std::fs::read_dir(dir)
            .with_context(|| format!("Failed to read directory: {}", dir.display()))?;

        for entry in entries {
            let entry = entry?;
            let path = entry.path();
            let relative = path.strip_prefix(root).unwrap_or(&path);
            let rel_str = relative.to_string_lossy();

            // Check exclusion first
            if exclude.iter().any(|pat| Self::glob_match(pat, &rel_str)) {
                continue;
            }

            if path.is_dir() {
                // Check if directory itself is excluded
                let dir_rel = format!("{}/", rel_str);
                if exclude.iter().any(|pat| Self::glob_match(pat, &dir_rel)) {
                    continue;
                }
                Self::walk_dir(root, &path, include, exclude, files)?;
            } else if include.iter().any(|pat| Self::glob_match(pat, &rel_str)) {
                files.push(path);
            }
        }

        Ok(())
    }

    /// Simple glob matching supporting `*` (single segment) and `**` (any depth).
    fn glob_match(pattern: &str, path: &str) -> bool {
        let pat_parts: Vec<&str> = pattern.split('/').collect();
        let path_parts: Vec<&str> = path.split('/').collect();
        Self::glob_match_parts(&pat_parts, &path_parts)
    }

    fn glob_match_parts(pat: &[&str], path: &[&str]) -> bool {
        if pat.is_empty() && path.is_empty() {
            return true;
        }
        if pat.is_empty() {
            return false;
        }

        if pat[0] == "**" {
            // ** matches zero or more path segments
            if pat.len() == 1 {
                return true; // ** at end matches everything
            }
            // Try matching remaining pattern against each suffix of path
            for i in 0..=path.len() {
                if Self::glob_match_parts(&pat[1..], &path[i..]) {
                    return true;
                }
            }
            return false;
        }

        if path.is_empty() {
            return false;
        }

        // Match current segment with wildcard support
        if Self::segment_match(pat[0], path[0]) {
            Self::glob_match_parts(&pat[1..], &path[1..])
        } else {
            false
        }
    }

    /// Matches a single path segment against a pattern segment with `*` wildcard.
    fn segment_match(pattern: &str, segment: &str) -> bool {
        if pattern == "*" {
            return true;
        }
        if !pattern.contains('*') {
            return pattern == segment;
        }
        // Simple prefix*suffix matching
        let parts: Vec<&str> = pattern.splitn(2, '*').collect();
        if parts.len() == 2 {
            segment.starts_with(parts[0]) && segment.ends_with(parts[1])
        } else {
            pattern == segment
        }
    }

    /// Reads a file and converts it to a ContentItem.
    fn file_to_content_item(
        vault_path: &Path,
        file_path: &Path,
        connector_id: &str,
        config: &ConnectorConfig,
    ) -> Result<ContentItem> {
        let raw = std::fs::read_to_string(file_path)
            .with_context(|| format!("Failed to read file: {}", file_path.display()))?;

        let relative = file_path.strip_prefix(vault_path).unwrap_or(file_path);
        let rel_str = relative.to_string_lossy().to_string();

        // Derive note title from filename (without .md extension)
        let title = file_path
            .file_stem()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_default();

        let max_len = Self::max_content_length(config);
        let do_parse_wikilinks = Self::parse_wikilinks(config);
        let do_parse_tags = Self::parse_tags(config);
        let do_parse_frontmatter = Self::parse_frontmatter(config);

        // Parse frontmatter
        let (frontmatter, body) = if do_parse_frontmatter {
            Self::extract_frontmatter(&raw)
        } else {
            (HashMap::new(), raw.as_str())
        };

        // Truncate body if needed
        let content = if body.len() > max_len {
            &body[..body
                .char_indices()
                .take_while(|(i, _)| *i < max_len)
                .last()
                .map(|(i, c)| i + c.len_utf8())
                .unwrap_or(0)]
        } else {
            body
        };

        // Build metadata
        let mut extra: HashMap<String, serde_json::Value> = HashMap::new();
        extra.insert(
            "path".to_string(),
            serde_json::Value::String(rel_str.clone()),
        );
        extra.insert("title".to_string(), serde_json::Value::String(title));

        if do_parse_wikilinks {
            let wikilinks = Self::extract_wikilinks(body);
            if !wikilinks.is_empty() {
                extra.insert("wikilinks".to_string(), serde_json::json!(wikilinks));
            }
        }

        if do_parse_tags {
            let tags = Self::extract_tags(body);
            if !tags.is_empty() {
                extra.insert("tags".to_string(), serde_json::json!(tags));
            }
        }

        if !frontmatter.is_empty() {
            extra.insert("frontmatter".to_string(), serde_json::json!(frontmatter));
        }

        // Get file modification time for created_at
        let modified = std::fs::metadata(file_path)
            .ok()
            .and_then(|m| m.modified().ok())
            .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
            .map(|d| d.as_secs());

        Ok(ContentItem {
            content: content.to_string(),
            source: SourceMetadata {
                connector_type: "obsidian".to_string(),
                connector_id: connector_id.to_string(),
                source_id: rel_str,
                source_url: None,
                author: None,
                created_at: modified,
                extra,
            },
            media: None,
        })
    }

    /// Extracts YAML frontmatter from content delimited by `---`.
    ///
    /// Returns a map of key-value pairs and the remaining body content.
    fn extract_frontmatter(content: &str) -> (HashMap<String, String>, &str) {
        let trimmed = content.trim_start();
        if !trimmed.starts_with("---") {
            return (HashMap::new(), content);
        }

        // Find closing ---
        let after_first = &trimmed[3..];
        let after_first = after_first.trim_start_matches(['\r', '\n']);
        if let Some(end_pos) = after_first.find("\n---") {
            let frontmatter_str = &after_first[..end_pos];
            let body_start = end_pos + 4; // skip \n---
            let body = after_first[body_start..].trim_start_matches(['\r', '\n']);

            // Simple YAML key: value parser
            let mut map = HashMap::new();
            for line in frontmatter_str.lines() {
                let line = line.trim();
                if line.is_empty() || line.starts_with('#') {
                    continue;
                }
                if let Some((key, value)) = line.split_once(':') {
                    let key = key.trim().to_string();
                    let value = value.trim().to_string();
                    if !key.is_empty() {
                        map.insert(key, value);
                    }
                }
            }

            // Calculate body offset relative to original content
            let body_offset = content.len() - body.len();
            (map, &content[body_offset..])
        } else {
            (HashMap::new(), content)
        }
    }

    /// Extracts `[[wikilinks]]` and `[[target|alias]]` from content.
    ///
    /// Returns deduplicated list of link targets (without aliases).
    fn extract_wikilinks(content: &str) -> Vec<String> {
        let mut links = Vec::new();
        let mut seen = std::collections::HashSet::new();

        let bytes = content.as_bytes();
        let len = bytes.len();
        let mut i = 0;

        while i + 1 < len {
            if bytes[i] == b'[' && bytes[i + 1] == b'[' {
                // Found opening [[
                let start = i + 2;
                // Find closing ]], but stop if we hit another [[ first
                let mut found_end = None;
                let mut j = start;
                while j + 1 < len {
                    if bytes[j] == b']' && bytes[j + 1] == b']' {
                        found_end = Some(j);
                        break;
                    }
                    if bytes[j] == b'[' && bytes[j + 1] == b'[' {
                        // Nested/unmatched [[ — skip this link attempt
                        break;
                    }
                    j += 1;
                }
                if let Some(end) = found_end {
                    let inner = &content[start..end];
                    // Handle [[target|alias]] — take the target part
                    let target = inner.split('|').next().unwrap_or(inner).trim();
                    // Also handle [[target#heading]] — take just the page name
                    let target = target.split('#').next().unwrap_or(target).trim();
                    if !target.is_empty() && seen.insert(target.to_string()) {
                        links.push(target.to_string());
                    }
                    i = end + 2;
                } else {
                    i += 2;
                }
            } else {
                i += 1;
            }
        }

        links
    }

    /// Extracts `#tags` from content (Obsidian-style).
    ///
    /// Tags must start with `#` followed by alphanumeric chars, hyphens, or underscores.
    /// Ignores tags inside code blocks and headings (lines starting with `#`).
    fn extract_tags(content: &str) -> Vec<String> {
        let mut tags = Vec::new();
        let mut seen = std::collections::HashSet::new();
        let mut in_code_block = false;

        for line in content.lines() {
            let trimmed = line.trim();

            // Toggle code block state
            if trimmed.starts_with("```") {
                in_code_block = !in_code_block;
                continue;
            }
            if in_code_block {
                continue;
            }

            // Skip heading lines (# Heading)
            if trimmed.starts_with("# ")
                || trimmed.starts_with("## ")
                || trimmed.starts_with("### ")
            {
                // Still extract tags from headings if they contain them inline
                // Actually, skip lines that ARE headings to avoid false positives with # prefix
                // But we should still parse inline tags in the heading text
            }

            // Find #tag patterns
            let bytes = line.as_bytes();
            let len = bytes.len();
            let mut i = 0;

            while i < len {
                if bytes[i] == b'#' {
                    // Check this is a tag, not a heading or anchor
                    // Must be preceded by whitespace, start of line, or certain punctuation
                    let is_start_of_word =
                        i == 0 || matches!(bytes[i - 1], b' ' | b'\t' | b'(' | b'[' | b',');

                    if is_start_of_word
                        && i + 1 < len
                        && (bytes[i + 1].is_ascii_alphanumeric() || bytes[i + 1] == b'_')
                    {
                        // Collect tag chars
                        let tag_start = i + 1;
                        let mut j = tag_start;
                        while j < len
                            && (bytes[j].is_ascii_alphanumeric()
                                || bytes[j] == b'-'
                                || bytes[j] == b'_'
                                || bytes[j] == b'/')
                        {
                            j += 1;
                        }
                        let tag = &line[tag_start..j];
                        if !tag.is_empty() && seen.insert(tag.to_string()) {
                            tags.push(tag.to_string());
                        }
                        i = j;
                    } else {
                        i += 1;
                    }
                } else {
                    i += 1;
                }
            }
        }

        tags
    }
}

impl Default for ObsidianConnector {
    fn default() -> Self {
        Self::new()
    }
}

impl Connector for ObsidianConnector {
    fn id(&self) -> &str {
        "obsidian"
    }

    fn name(&self) -> &str {
        "Obsidian"
    }

    fn config_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "auth": {
                    "type": "object",
                    "description": "No authentication required (local files)",
                    "properties": {
                        "type": { "const": "None" }
                    }
                },
                "settings": {
                    "type": "object",
                    "properties": {
                        "vault_path": {
                            "type": "string",
                            "description": "Absolute path to the Obsidian vault directory"
                        },
                        "include_patterns": {
                            "type": "array",
                            "items": { "type": "string" },
                            "description": "Glob patterns for files to include (default: ['**/*.md'])"
                        },
                        "exclude_patterns": {
                            "type": "array",
                            "items": { "type": "string" },
                            "description": "Glob patterns for files to exclude (default: ['.obsidian/**', '.trash/**'])"
                        },
                        "max_content_length": {
                            "type": "integer",
                            "description": "Maximum content length per file (default: 10000)"
                        },
                        "parse_wikilinks": {
                            "type": "boolean",
                            "description": "Extract [[wikilinks]] as metadata (default: true)"
                        },
                        "parse_tags": {
                            "type": "boolean",
                            "description": "Extract #tags as metadata (default: true)"
                        },
                        "parse_frontmatter": {
                            "type": "boolean",
                            "description": "Parse YAML frontmatter (default: true)"
                        }
                    },
                    "required": ["vault_path"]
                }
            },
            "required": ["settings"]
        })
    }

    fn validate_config(&self, config: &ConnectorConfig) -> Result<()> {
        if config.connector_type != "obsidian" {
            bail!(
                "Invalid connector type '{}', expected 'obsidian'",
                config.connector_type
            );
        }
        // Auth must be None
        match &config.auth {
            AuthConfig::None => {}
            _ => bail!("Obsidian connector requires AuthConfig::None (no authentication needed)"),
        }
        let vault_path = Self::get_vault_path(config)?;
        if !vault_path.exists() {
            bail!("Vault path does not exist: {}", vault_path.display());
        }
        if !vault_path.is_dir() {
            bail!("Vault path is not a directory: {}", vault_path.display());
        }
        Ok(())
    }

    async fn fetch(&self, config: &ConnectorConfig) -> Result<Vec<ContentItem>> {
        let vault_path = Self::get_vault_path(config)?;

        let files =
            Self::collect_files(&vault_path, config).context("Failed to collect vault files")?;

        let mut items = Vec::new();
        for file in &files {
            match Self::file_to_content_item(&vault_path, file, &config.id, config) {
                Ok(item) => items.push(item),
                Err(e) => {
                    eprintln!(
                        "Warning: failed to read vault file {}: {}",
                        file.display(),
                        e
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
        let vault_path = Self::get_vault_path(config)?;
        let last_sync_ts = cursor.last_sync.unwrap_or(0);

        let files =
            Self::collect_files(&vault_path, config).context("Failed to collect vault files")?;

        let mut items = Vec::new();
        let mut skipped = 0;
        let mut latest_mtime: u64 = last_sync_ts;

        for file in &files {
            // Check modification time
            let mtime = std::fs::metadata(file)
                .ok()
                .and_then(|m| m.modified().ok())
                .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
                .map(|d| d.as_secs())
                .unwrap_or(0);

            if mtime <= last_sync_ts {
                skipped += 1;
                continue;
            }

            if mtime > latest_mtime {
                latest_mtime = mtime;
            }

            match Self::file_to_content_item(&vault_path, file, &config.id, config) {
                Ok(item) => items.push(item),
                Err(e) => {
                    eprintln!(
                        "Warning: failed to read vault file {}: {}",
                        file.display(),
                        e
                    );
                    skipped += 1;
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
                value: Some(latest_mtime.to_string()),
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
        // Obsidian is a local vault — no webhook support.
        // Could be extended with file system watchers in the future.
        bail!("Obsidian connector does not support webhooks (local vault only)")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use std::io::Write;

    fn make_config(vault_path: &str) -> ConnectorConfig {
        let mut settings = HashMap::new();
        settings.insert(
            "vault_path".to_string(),
            serde_json::Value::String(vault_path.to_string()),
        );
        ConnectorConfig {
            id: "obsidian-test".to_string(),
            name: "Test Vault".to_string(),
            connector_type: "obsidian".to_string(),
            auth: AuthConfig::None,
            namespace: "test".to_string(),
            settings,
            enabled: true,
        }
    }

    fn create_temp_vault() -> tempfile::TempDir {
        let dir = tempfile::tempdir().unwrap();

        // Create vault structure
        std::fs::create_dir_all(dir.path().join(".obsidian")).unwrap();
        std::fs::create_dir_all(dir.path().join("notes")).unwrap();
        std::fs::create_dir_all(dir.path().join(".trash")).unwrap();

        // .obsidian config (should be excluded)
        std::fs::write(
            dir.path().join(".obsidian/app.json"),
            r#"{"theme": "dark"}"#,
        )
        .unwrap();

        // Regular notes
        std::fs::write(
            dir.path().join("README.md"),
            "# My Vault\n\nWelcome to my vault.\n",
        )
        .unwrap();

        std::fs::write(
            dir.path().join("notes/daily.md"),
            "---\ndate: 2024-01-15\ntags: journal, daily\n---\n\n# Daily Note\n\nToday I worked on [[Project Alpha]] and met with [[Alice]].\n\n#journal #productivity\n",
        )
        .unwrap();

        std::fs::write(
            dir.path().join("notes/project.md"),
            "# Project Alpha\n\nThis is the main project page.\n\n## Links\n- [[Alice|Team Lead]]\n- [[Bob]]\n- [[notes/daily|Daily Notes]]\n\n## Tags\n#project #alpha #work\n",
        )
        .unwrap();

        // Trash (should be excluded)
        std::fs::write(
            dir.path().join(".trash/deleted.md"),
            "This file was deleted.\n",
        )
        .unwrap();

        dir
    }

    #[test]
    fn test_obsidian_connector_id_and_name() {
        let connector = ObsidianConnector::new();
        assert_eq!(connector.id(), "obsidian");
        assert_eq!(connector.name(), "Obsidian");
    }

    #[test]
    fn test_obsidian_config_schema() {
        let connector = ObsidianConnector::new();
        let schema = connector.config_schema();
        assert_eq!(schema["type"], "object");
        assert!(schema["properties"]["settings"]["properties"]["vault_path"].is_object());
        assert!(schema["properties"]["settings"]["properties"]["parse_wikilinks"].is_object());
        assert!(schema["properties"]["settings"]["properties"]["parse_tags"].is_object());
        assert!(schema["properties"]["settings"]["properties"]["parse_frontmatter"].is_object());
    }

    #[test]
    fn test_validate_config_valid() {
        let vault = create_temp_vault();
        let connector = ObsidianConnector::new();
        let config = make_config(vault.path().to_str().unwrap());
        assert!(connector.validate_config(&config).is_ok());
    }

    #[test]
    fn test_validate_config_wrong_type() {
        let vault = create_temp_vault();
        let connector = ObsidianConnector::new();
        let mut config = make_config(vault.path().to_str().unwrap());
        config.connector_type = "slack".to_string();
        assert!(connector.validate_config(&config).is_err());
    }

    #[test]
    fn test_validate_config_wrong_auth() {
        let vault = create_temp_vault();
        let connector = ObsidianConnector::new();
        let mut config = make_config(vault.path().to_str().unwrap());
        config.auth = AuthConfig::Token {
            token: "should_not_be_here".to_string(),
        };
        let err = connector.validate_config(&config);
        assert!(err.is_err());
        assert!(err.unwrap_err().to_string().contains("None"));
    }

    #[test]
    fn test_validate_config_nonexistent_path() {
        let connector = ObsidianConnector::new();
        let config = make_config("/nonexistent/vault/path/12345");
        let err = connector.validate_config(&config);
        assert!(err.is_err());
        assert!(err.unwrap_err().to_string().contains("does not exist"));
    }

    #[test]
    fn test_validate_config_missing_vault_path() {
        let connector = ObsidianConnector::new();
        let config = ConnectorConfig {
            id: "test".to_string(),
            name: "Test".to_string(),
            connector_type: "obsidian".to_string(),
            auth: AuthConfig::None,
            namespace: "test".to_string(),
            settings: HashMap::new(),
            enabled: true,
        };
        let err = connector.validate_config(&config);
        assert!(err.is_err());
        assert!(err.unwrap_err().to_string().contains("vault_path"));
    }

    #[test]
    fn test_validate_config_relative_path() {
        let connector = ObsidianConnector::new();
        let config = make_config("relative/path/to/vault");
        let err = connector.validate_config(&config);
        assert!(err.is_err());
        assert!(err.unwrap_err().to_string().contains("absolute"));
    }

    #[test]
    fn test_extract_wikilinks() {
        let content = "Check [[Page One]] and [[Page Two|alias]]. Also see [[Page One#section]].";
        let links = ObsidianConnector::extract_wikilinks(content);
        assert_eq!(links, vec!["Page One", "Page Two"]);
    }

    #[test]
    fn test_extract_wikilinks_empty() {
        let content = "No links here, just plain text.";
        let links = ObsidianConnector::extract_wikilinks(content);
        assert!(links.is_empty());
    }

    #[test]
    fn test_extract_wikilinks_nested_brackets() {
        let content = "A [[Link]] followed by unmatched [[ and then [[Another]].";
        let links = ObsidianConnector::extract_wikilinks(content);
        assert_eq!(links, vec!["Link", "Another"]);
    }

    #[test]
    fn test_extract_tags() {
        let content = "Some text #rust and #programming-tips here.\nAlso #web_dev.";
        let tags = ObsidianConnector::extract_tags(content);
        assert_eq!(tags, vec!["rust", "programming-tips", "web_dev"]);
    }

    #[test]
    fn test_extract_tags_ignores_code_blocks() {
        let content = "Before #visible\n```\n#inside_code\n```\nAfter #also_visible";
        let tags = ObsidianConnector::extract_tags(content);
        assert_eq!(tags, vec!["visible", "also_visible"]);
    }

    #[test]
    fn test_extract_tags_deduplicates() {
        let content = "#tag1 #tag2 #tag1 again #tag2 again";
        let tags = ObsidianConnector::extract_tags(content);
        assert_eq!(tags, vec!["tag1", "tag2"]);
    }

    #[test]
    fn test_extract_tags_nested() {
        let content = "A nested #tag/subtag here.";
        let tags = ObsidianConnector::extract_tags(content);
        assert_eq!(tags, vec!["tag/subtag"]);
    }

    #[test]
    fn test_extract_frontmatter() {
        let content = "---\ntitle: My Note\ndate: 2024-01-15\ntags: journal, daily\n---\n\n# My Note\n\nContent here.";
        let (fm, body) = ObsidianConnector::extract_frontmatter(content);
        assert_eq!(fm.get("title").unwrap(), "My Note");
        assert_eq!(fm.get("date").unwrap(), "2024-01-15");
        assert_eq!(fm.get("tags").unwrap(), "journal, daily");
        assert!(body.starts_with("# My Note"));
    }

    #[test]
    fn test_extract_frontmatter_none() {
        let content = "# No Frontmatter\n\nJust content.";
        let (fm, body) = ObsidianConnector::extract_frontmatter(content);
        assert!(fm.is_empty());
        assert_eq!(body, content);
    }

    #[test]
    fn test_extract_frontmatter_empty() {
        let content = "---\n---\n\nContent after empty frontmatter.";
        let (fm, body) = ObsidianConnector::extract_frontmatter(content);
        assert!(fm.is_empty());
        assert!(body.contains("Content after empty frontmatter"));
    }

    #[test]
    fn test_glob_match_simple() {
        assert!(ObsidianConnector::glob_match("*.md", "README.md"));
        assert!(!ObsidianConnector::glob_match("*.md", "README.txt"));
        assert!(!ObsidianConnector::glob_match("*.md", "notes/README.md"));
    }

    #[test]
    fn test_glob_match_recursive() {
        assert!(ObsidianConnector::glob_match("**/*.md", "README.md"));
        assert!(ObsidianConnector::glob_match("**/*.md", "notes/README.md"));
        assert!(ObsidianConnector::glob_match("**/*.md", "a/b/c/d.md"));
        assert!(!ObsidianConnector::glob_match("**/*.md", "file.txt"));
    }

    #[test]
    fn test_glob_match_directory() {
        assert!(ObsidianConnector::glob_match(
            ".obsidian/**",
            ".obsidian/app.json"
        ));
        assert!(ObsidianConnector::glob_match(
            ".obsidian/**",
            ".obsidian/plugins/x.json"
        ));
        assert!(!ObsidianConnector::glob_match(
            ".obsidian/**",
            "notes/file.md"
        ));
    }

    #[test]
    fn test_glob_match_exact() {
        assert!(ObsidianConnector::glob_match("README.md", "README.md"));
        assert!(!ObsidianConnector::glob_match("README.md", "OTHER.md"));
    }

    #[test]
    fn test_collect_files_excludes_obsidian_and_trash() {
        let vault = create_temp_vault();
        let config = make_config(vault.path().to_str().unwrap());

        let files = ObsidianConnector::collect_files(vault.path(), &config).unwrap();
        let rel_paths: Vec<String> = files
            .iter()
            .map(|f| {
                f.strip_prefix(vault.path())
                    .unwrap()
                    .to_string_lossy()
                    .to_string()
            })
            .collect();

        // Should include vault .md files
        assert!(rel_paths.iter().any(|p| p == "README.md"));
        assert!(rel_paths.iter().any(|p| p.contains("daily.md")));
        assert!(rel_paths.iter().any(|p| p.contains("project.md")));

        // Should exclude .obsidian and .trash
        assert!(!rel_paths.iter().any(|p| p.contains(".obsidian")));
        assert!(!rel_paths.iter().any(|p| p.contains(".trash")));
    }

    #[tokio::test]
    async fn test_fetch_full_sync() {
        let vault = create_temp_vault();
        let connector = ObsidianConnector::new();
        let config = make_config(vault.path().to_str().unwrap());

        let items = connector.fetch(&config).await.unwrap();

        assert_eq!(items.len(), 3); // README.md, notes/daily.md, notes/project.md

        // All items should have obsidian connector type
        for item in &items {
            assert_eq!(item.source.connector_type, "obsidian");
            assert_eq!(item.source.connector_id, "obsidian-test");
            assert!(item.source.source_url.is_none());
            assert!(item.source.created_at.is_some());
            assert!(item.media.is_none());
        }

        // Check daily note has wikilinks and tags
        let daily = items
            .iter()
            .find(|i| i.source.source_id.contains("daily.md"))
            .unwrap();
        let wikilinks = daily
            .source
            .extra
            .get("wikilinks")
            .unwrap()
            .as_array()
            .unwrap();
        assert!(wikilinks
            .iter()
            .any(|v| v.as_str() == Some("Project Alpha")));
        assert!(wikilinks.iter().any(|v| v.as_str() == Some("Alice")));

        let tags = daily.source.extra.get("tags").unwrap().as_array().unwrap();
        assert!(tags.iter().any(|v| v.as_str() == Some("journal")));
        assert!(tags.iter().any(|v| v.as_str() == Some("productivity")));

        // Check frontmatter was parsed
        let fm = daily
            .source
            .extra
            .get("frontmatter")
            .unwrap()
            .as_object()
            .unwrap();
        assert_eq!(fm.get("date").unwrap().as_str().unwrap(), "2024-01-15");
    }

    #[tokio::test]
    async fn test_fetch_with_wikilinks_disabled() {
        let vault = create_temp_vault();
        let connector = ObsidianConnector::new();
        let mut config = make_config(vault.path().to_str().unwrap());
        config
            .settings
            .insert("parse_wikilinks".to_string(), serde_json::json!(false));

        let items = connector.fetch(&config).await.unwrap();
        let daily = items
            .iter()
            .find(|i| i.source.source_id.contains("daily.md"))
            .unwrap();

        // Wikilinks should not be extracted
        assert!(!daily.source.extra.contains_key("wikilinks"));
        // But tags should still be present
        assert!(daily.source.extra.contains_key("tags"));
    }

    #[tokio::test]
    async fn test_fetch_with_tags_disabled() {
        let vault = create_temp_vault();
        let connector = ObsidianConnector::new();
        let mut config = make_config(vault.path().to_str().unwrap());
        config
            .settings
            .insert("parse_tags".to_string(), serde_json::json!(false));

        let items = connector.fetch(&config).await.unwrap();
        let daily = items
            .iter()
            .find(|i| i.source.source_id.contains("daily.md"))
            .unwrap();

        // Tags should not be extracted
        assert!(!daily.source.extra.contains_key("tags"));
        // But wikilinks should still be present
        assert!(daily.source.extra.contains_key("wikilinks"));
    }

    #[tokio::test]
    async fn test_fetch_with_custom_exclude() {
        let vault = create_temp_vault();
        let connector = ObsidianConnector::new();
        let mut config = make_config(vault.path().to_str().unwrap());
        // Exclude everything under notes/
        config.settings.insert(
            "exclude_patterns".to_string(),
            serde_json::json!([".obsidian/**", ".trash/**", "notes/**"]),
        );

        let items = connector.fetch(&config).await.unwrap();
        assert_eq!(items.len(), 1); // Only README.md
        assert!(items[0].source.source_id.contains("README.md"));
    }

    #[tokio::test]
    async fn test_sync_incremental() {
        let vault = create_temp_vault();
        let connector = ObsidianConnector::new();
        let config = make_config(vault.path().to_str().unwrap());

        // First sync — get all files
        let result = connector
            .sync_incremental(&config, &SyncCursor::default())
            .await
            .unwrap();
        assert_eq!(result.items.len(), 3);
        assert!(result.cursor.last_sync.is_some());

        // Use the max mtime from cursor as the sync point
        // Wait > 1 second since filesystem mtime has 1-second resolution
        std::thread::sleep(std::time::Duration::from_secs(2));

        let new_file = vault.path().join("new_note.md");
        let mut f = std::fs::File::create(&new_file).unwrap();
        writeln!(
            f,
            "# New Note\n\nThis is a new note with [[links]] and #tags."
        )
        .unwrap();
        drop(f);

        // Build cursor from the max mtime value returned by first sync
        let cursor = SyncCursor {
            value: result.cursor.value.clone(),
            last_sync: result
                .cursor
                .value
                .as_ref()
                .and_then(|v| v.parse::<u64>().ok()),
        };

        // Second incremental sync — should only get the new file
        let result2 = connector.sync_incremental(&config, &cursor).await.unwrap();
        assert_eq!(result2.items.len(), 1);
        assert!(result2.items[0].source.source_id.contains("new_note.md"));
        assert!(result2.skipped > 0);
    }

    #[tokio::test]
    async fn test_handle_webhook_returns_error() {
        let connector = ObsidianConnector::new();
        let vault = create_temp_vault();
        let config = make_config(vault.path().to_str().unwrap());
        let payload = WebhookPayload {
            body: vec![],
            headers: HashMap::new(),
            content_type: None,
        };

        let result = connector.handle_webhook(&config, payload).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("webhook"));
    }

    #[test]
    fn test_max_content_length_default() {
        let config = make_config("/tmp/vault");
        assert_eq!(ObsidianConnector::max_content_length(&config), 10000);
    }

    #[test]
    fn test_max_content_length_custom() {
        let mut config = make_config("/tmp/vault");
        config
            .settings
            .insert("max_content_length".to_string(), serde_json::json!(500));
        assert_eq!(ObsidianConnector::max_content_length(&config), 500);
    }

    #[tokio::test]
    async fn test_fetch_max_content_length() {
        let vault = create_temp_vault();
        let connector = ObsidianConnector::new();
        let mut config = make_config(vault.path().to_str().unwrap());

        // Write a very long file
        let long_content = "x".repeat(20000);
        std::fs::write(vault.path().join("long.md"), &long_content).unwrap();

        // Set max length to 100
        config
            .settings
            .insert("max_content_length".to_string(), serde_json::json!(100));

        let items = connector.fetch(&config).await.unwrap();
        let long_item = items
            .iter()
            .find(|i| i.source.source_id.contains("long.md"))
            .unwrap();
        assert!(long_item.content.len() <= 100);
    }

    #[test]
    fn test_default_constructor() {
        let connector = ObsidianConnector;
        assert_eq!(connector.id(), "obsidian");
    }

    #[test]
    fn test_settings_defaults() {
        let config = make_config("/tmp/vault");
        assert_eq!(
            ObsidianConnector::get_include_patterns(&config),
            vec!["**/*.md"]
        );
        assert_eq!(
            ObsidianConnector::get_exclude_patterns(&config),
            vec![".obsidian/**", ".trash/**"]
        );
        assert!(ObsidianConnector::parse_wikilinks(&config));
        assert!(ObsidianConnector::parse_tags(&config));
        assert!(ObsidianConnector::parse_frontmatter(&config));
    }

    #[test]
    fn test_file_to_content_item_with_frontmatter() {
        let vault = create_temp_vault();
        let config = make_config(vault.path().to_str().unwrap());

        let file_path = vault.path().join("notes/daily.md");
        let item =
            ObsidianConnector::file_to_content_item(vault.path(), &file_path, "test-id", &config)
                .unwrap();

        assert_eq!(item.source.connector_type, "obsidian");
        assert_eq!(item.source.connector_id, "test-id");
        assert!(item.source.extra.contains_key("title"));
        assert_eq!(
            item.source.extra.get("title").unwrap().as_str().unwrap(),
            "daily"
        );
        // Should have frontmatter parsed
        assert!(item.source.extra.contains_key("frontmatter"));
        // Content should NOT include frontmatter
        assert!(!item.content.contains("---"));
    }
}
