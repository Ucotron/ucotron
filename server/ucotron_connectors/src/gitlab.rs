//! GitLab connector — fetches issues, merge requests, and snippets from GitLab projects via the REST API.
//!
//! Uses Personal Access Tokens (PAT) to authenticate with the GitLab API.
//! Supports full sync (all issues/MRs from configured projects) and
//! incremental sync via `updated_after` timestamp-based pagination.

use std::collections::HashMap;

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};

use crate::connector::{
    AuthConfig, Connector, ConnectorConfig, ContentItem, SourceMetadata, SyncCursor, SyncResult,
    WebhookPayload,
};

const GITLAB_API_DEFAULT: &str = "https://gitlab.com/api/v4";

/// GitLab connector for fetching issues, merge requests, and snippets.
///
/// Requires a Personal Access Token with the following scopes:
/// - `read_api` — read access to projects, issues, MRs, and snippets
/// - For self-hosted GitLab instances, set `base_url` in settings
pub struct GitLabConnector {
    client: reqwest::Client,
}

impl GitLabConnector {
    /// Creates a new GitLabConnector with a default HTTP client.
    pub fn new() -> Self {
        Self {
            client: reqwest::Client::builder()
                .user_agent("ucotron-connector/0.1")
                .build()
                .unwrap_or_else(|_| reqwest::Client::new()),
        }
    }

    /// Creates a new GitLabConnector with a custom HTTP client.
    pub fn with_client(client: reqwest::Client) -> Self {
        Self { client }
    }

    /// Extracts the PAT from the connector config.
    fn get_token(config: &ConnectorConfig) -> Result<&str> {
        match &config.auth {
            AuthConfig::Token { token } => Ok(token.as_str()),
            _ => bail!("GitLab connector requires Token authentication (Personal Access Token)"),
        }
    }

    /// Returns the GitLab API base URL (supports self-hosted instances).
    fn get_base_url(config: &ConnectorConfig) -> String {
        config
            .settings
            .get("base_url")
            .and_then(|v| v.as_str())
            .map(|s| s.trim_end_matches('/').to_string())
            .unwrap_or_else(|| GITLAB_API_DEFAULT.to_string())
    }

    /// Extracts configured project paths from settings (e.g., ["group/project"]).
    fn get_projects(config: &ConnectorConfig) -> Vec<String> {
        config
            .settings
            .get("projects")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str().map(String::from))
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Checks if issues should be fetched (default: true).
    fn include_issues(config: &ConnectorConfig) -> bool {
        config
            .settings
            .get("include_issues")
            .and_then(|v| v.as_bool())
            .unwrap_or(true)
    }

    /// Checks if merge requests should be fetched (default: true).
    fn include_mrs(config: &ConnectorConfig) -> bool {
        config
            .settings
            .get("include_mrs")
            .and_then(|v| v.as_bool())
            .unwrap_or(true)
    }

    /// Checks if snippets should be fetched (default: false).
    fn include_snippets(config: &ConnectorConfig) -> bool {
        config
            .settings
            .get("include_snippets")
            .and_then(|v| v.as_bool())
            .unwrap_or(false)
    }

    /// URL-encodes a project path for the GitLab API (e.g., "group/project" → "group%2Fproject").
    fn encode_project_path(path: &str) -> String {
        path.replace('/', "%2F")
    }

    /// Fetches issues from a project with pagination.
    async fn fetch_issues(
        &self,
        token: &str,
        base_url: &str,
        project_path: &str,
        updated_after: Option<&str>,
    ) -> Result<Vec<GitLabIssue>> {
        let encoded = Self::encode_project_path(project_path);
        let mut all_issues = Vec::new();
        let mut page = 1u32;

        loop {
            let mut url = format!(
                "{}/projects/{}/issues?per_page=100&page={}&order_by=updated_at&sort=desc&state=all",
                base_url, encoded, page
            );
            if let Some(since) = updated_after {
                url.push_str(&format!("&updated_after={}", since));
            }

            let resp = self
                .client
                .get(&url)
                .header("PRIVATE-TOKEN", token)
                .send()
                .await
                .context("Failed to fetch GitLab issues")?;

            if !resp.status().is_success() {
                let status = resp.status();
                let body = resp.text().await.unwrap_or_default();
                bail!("GitLab API error fetching issues: {} - {}", status, body);
            }

            let issues: Vec<GitLabIssue> = resp
                .json()
                .await
                .context("Failed to parse GitLab issues response")?;

            let count = issues.len();
            all_issues.extend(issues);

            if count < 100 {
                break;
            }
            page += 1;
        }

        Ok(all_issues)
    }

    /// Fetches merge requests from a project with pagination.
    async fn fetch_merge_requests(
        &self,
        token: &str,
        base_url: &str,
        project_path: &str,
        updated_after: Option<&str>,
    ) -> Result<Vec<GitLabMergeRequest>> {
        let encoded = Self::encode_project_path(project_path);
        let mut all_mrs = Vec::new();
        let mut page = 1u32;

        loop {
            let mut url = format!(
                "{}/projects/{}/merge_requests?per_page=100&page={}&order_by=updated_at&sort=desc&state=all",
                base_url, encoded, page
            );
            if let Some(since) = updated_after {
                url.push_str(&format!("&updated_after={}", since));
            }

            let resp = self
                .client
                .get(&url)
                .header("PRIVATE-TOKEN", token)
                .send()
                .await
                .context("Failed to fetch GitLab merge requests")?;

            if !resp.status().is_success() {
                let status = resp.status();
                let body = resp.text().await.unwrap_or_default();
                bail!("GitLab API error fetching MRs: {} - {}", status, body);
            }

            let mrs: Vec<GitLabMergeRequest> = resp
                .json()
                .await
                .context("Failed to parse GitLab MRs response")?;

            let count = mrs.len();
            all_mrs.extend(mrs);

            if count < 100 {
                break;
            }
            page += 1;
        }

        Ok(all_mrs)
    }

    /// Fetches project snippets with pagination.
    async fn fetch_snippets(
        &self,
        token: &str,
        base_url: &str,
        project_path: &str,
    ) -> Result<Vec<GitLabSnippet>> {
        let encoded = Self::encode_project_path(project_path);
        let mut all_snippets = Vec::new();
        let mut page = 1u32;

        loop {
            let url = format!(
                "{}/projects/{}/snippets?per_page=100&page={}",
                base_url, encoded, page
            );

            let resp = self
                .client
                .get(&url)
                .header("PRIVATE-TOKEN", token)
                .send()
                .await
                .context("Failed to fetch GitLab snippets")?;

            if !resp.status().is_success() {
                let status = resp.status();
                let body = resp.text().await.unwrap_or_default();
                bail!("GitLab API error fetching snippets: {} - {}", status, body);
            }

            let snippets: Vec<GitLabSnippet> = resp
                .json()
                .await
                .context("Failed to parse GitLab snippets response")?;

            let count = snippets.len();
            all_snippets.extend(snippets);

            if count < 100 {
                break;
            }
            page += 1;
        }

        Ok(all_snippets)
    }

    /// Converts a GitLab issue to a ContentItem.
    fn issue_to_content_item(
        &self,
        issue: &GitLabIssue,
        project_path: &str,
        connector_id: &str,
    ) -> ContentItem {
        let labels_str: String = issue.labels.join(", ");

        let body_preview = issue
            .description
            .as_deref()
            .unwrap_or("")
            .chars()
            .take(2000)
            .collect::<String>();

        let content = if body_preview.is_empty() {
            format!(
                "[{}] Issue #{}: {}",
                project_path, issue.iid, issue.title
            )
        } else {
            format!(
                "[{}] Issue #{}: {}\n\n{}",
                project_path, issue.iid, issue.title, body_preview
            )
        };

        let created_at = parse_gitlab_timestamp(&issue.created_at);

        let mut extra = HashMap::new();
        extra.insert(
            "project".to_string(),
            serde_json::Value::String(project_path.to_string()),
        );
        extra.insert(
            "iid".to_string(),
            serde_json::Value::Number(issue.iid.into()),
        );
        extra.insert(
            "state".to_string(),
            serde_json::Value::String(issue.state.clone()),
        );
        extra.insert(
            "kind".to_string(),
            serde_json::Value::String("Issue".to_string()),
        );
        if !labels_str.is_empty() {
            extra.insert(
                "labels".to_string(),
                serde_json::Value::String(labels_str),
            );
        }
        if let Some(ref assignee) = issue.assignee {
            extra.insert(
                "assignee".to_string(),
                serde_json::Value::String(assignee.username.clone()),
            );
        }

        ContentItem {
            content,
            source: SourceMetadata {
                connector_type: "gitlab".to_string(),
                connector_id: connector_id.to_string(),
                source_id: format!("{}#{}", project_path, issue.iid),
                source_url: Some(issue.web_url.clone()),
                author: Some(issue.author.username.clone()),
                created_at,
                extra,
            },
            media: None,
        }
    }

    /// Converts a GitLab merge request to a ContentItem.
    fn mr_to_content_item(
        &self,
        mr: &GitLabMergeRequest,
        project_path: &str,
        connector_id: &str,
    ) -> ContentItem {
        let labels_str: String = mr.labels.join(", ");

        let body_preview = mr
            .description
            .as_deref()
            .unwrap_or("")
            .chars()
            .take(2000)
            .collect::<String>();

        let content = if body_preview.is_empty() {
            format!(
                "[{}] MR !{}: {}",
                project_path, mr.iid, mr.title
            )
        } else {
            format!(
                "[{}] MR !{}: {}\n\n{}",
                project_path, mr.iid, mr.title, body_preview
            )
        };

        let created_at = parse_gitlab_timestamp(&mr.created_at);

        let mut extra = HashMap::new();
        extra.insert(
            "project".to_string(),
            serde_json::Value::String(project_path.to_string()),
        );
        extra.insert(
            "iid".to_string(),
            serde_json::Value::Number(mr.iid.into()),
        );
        extra.insert(
            "state".to_string(),
            serde_json::Value::String(mr.state.clone()),
        );
        extra.insert(
            "kind".to_string(),
            serde_json::Value::String("MR".to_string()),
        );
        extra.insert(
            "source_branch".to_string(),
            serde_json::Value::String(mr.source_branch.clone()),
        );
        extra.insert(
            "target_branch".to_string(),
            serde_json::Value::String(mr.target_branch.clone()),
        );
        if !labels_str.is_empty() {
            extra.insert(
                "labels".to_string(),
                serde_json::Value::String(labels_str),
            );
        }
        if let Some(ref assignee) = mr.assignee {
            extra.insert(
                "assignee".to_string(),
                serde_json::Value::String(assignee.username.clone()),
            );
        }

        ContentItem {
            content,
            source: SourceMetadata {
                connector_type: "gitlab".to_string(),
                connector_id: connector_id.to_string(),
                source_id: format!("{}!{}", project_path, mr.iid),
                source_url: Some(mr.web_url.clone()),
                author: Some(mr.author.username.clone()),
                created_at,
                extra,
            },
            media: None,
        }
    }

    /// Converts a GitLab snippet to a ContentItem.
    fn snippet_to_content_item(
        &self,
        snippet: &GitLabSnippet,
        project_path: &str,
        connector_id: &str,
    ) -> ContentItem {
        let body_preview = snippet
            .description
            .as_deref()
            .unwrap_or("")
            .chars()
            .take(2000)
            .collect::<String>();

        let content = if body_preview.is_empty() {
            format!(
                "[{}] Snippet #{}: {}",
                project_path, snippet.id, snippet.title
            )
        } else {
            format!(
                "[{}] Snippet #{}: {}\n\n{}",
                project_path, snippet.id, snippet.title, body_preview
            )
        };

        let created_at = parse_gitlab_timestamp(&snippet.created_at);

        let mut extra = HashMap::new();
        extra.insert(
            "project".to_string(),
            serde_json::Value::String(project_path.to_string()),
        );
        extra.insert(
            "kind".to_string(),
            serde_json::Value::String("Snippet".to_string()),
        );
        if let Some(ref file_name) = snippet.file_name {
            extra.insert(
                "file_name".to_string(),
                serde_json::Value::String(file_name.clone()),
            );
        }

        ContentItem {
            content,
            source: SourceMetadata {
                connector_type: "gitlab".to_string(),
                connector_id: connector_id.to_string(),
                source_id: format!("{}~{}", project_path, snippet.id),
                source_url: Some(snippet.web_url.clone()),
                author: Some(snippet.author.username.clone()),
                created_at,
                extra,
            },
            media: None,
        }
    }
}

impl Default for GitLabConnector {
    fn default() -> Self {
        Self::new()
    }
}

impl Connector for GitLabConnector {
    fn id(&self) -> &str {
        "gitlab"
    }

    fn name(&self) -> &str {
        "GitLab"
    }

    fn config_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "auth": {
                    "type": "object",
                    "description": "Personal Access Token credentials",
                    "properties": {
                        "token": { "type": "string", "description": "GitLab PAT (glpat-...)" }
                    },
                    "required": ["token"]
                },
                "settings": {
                    "type": "object",
                    "properties": {
                        "projects": {
                            "type": "array",
                            "items": { "type": "string" },
                            "description": "Projects to sync in 'group/project' format"
                        },
                        "base_url": {
                            "type": "string",
                            "description": "GitLab API base URL (default: https://gitlab.com/api/v4)"
                        },
                        "include_issues": {
                            "type": "boolean",
                            "description": "Whether to fetch issues (default: true)"
                        },
                        "include_mrs": {
                            "type": "boolean",
                            "description": "Whether to fetch merge requests (default: true)"
                        },
                        "include_snippets": {
                            "type": "boolean",
                            "description": "Whether to fetch project snippets (default: false)"
                        }
                    }
                }
            },
            "required": ["auth", "settings"]
        })
    }

    fn validate_config(&self, config: &ConnectorConfig) -> Result<()> {
        if config.connector_type != "gitlab" {
            bail!(
                "Invalid connector type '{}', expected 'gitlab'",
                config.connector_type
            );
        }
        Self::get_token(config)?;
        let projects = Self::get_projects(config);
        if projects.is_empty() {
            bail!("GitLab connector requires at least one project in settings.projects");
        }
        for project in &projects {
            if !project.contains('/') || project.split('/').count() < 2 {
                bail!(
                    "Invalid project format '{}', expected 'group/project' or 'group/subgroup/project'",
                    project
                );
            }
        }
        Ok(())
    }

    async fn fetch(&self, config: &ConnectorConfig) -> Result<Vec<ContentItem>> {
        let token = Self::get_token(config)?;
        let base_url = Self::get_base_url(config);
        let projects = Self::get_projects(config);
        let fetch_issues = Self::include_issues(config);
        let fetch_mrs = Self::include_mrs(config);
        let fetch_snippets = Self::include_snippets(config);

        let mut items = Vec::new();

        for project_path in &projects {
            if fetch_issues {
                let issues = self
                    .fetch_issues(token, &base_url, project_path, None)
                    .await?;
                for issue in &issues {
                    items.push(self.issue_to_content_item(issue, project_path, &config.id));
                }
            }

            if fetch_mrs {
                let mrs = self
                    .fetch_merge_requests(token, &base_url, project_path, None)
                    .await?;
                for mr in &mrs {
                    items.push(self.mr_to_content_item(mr, project_path, &config.id));
                }
            }

            if fetch_snippets {
                let snippets = self
                    .fetch_snippets(token, &base_url, project_path)
                    .await?;
                for snippet in &snippets {
                    items.push(self.snippet_to_content_item(snippet, project_path, &config.id));
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
        let base_url = Self::get_base_url(config);
        let projects = Self::get_projects(config);
        let fetch_issues = Self::include_issues(config);
        let fetch_mrs = Self::include_mrs(config);

        // Use cursor value as ISO 8601 timestamp for `updated_after` parameter
        let updated_after = cursor.value.as_deref();

        let mut items = Vec::new();
        let mut latest_updated: Option<String> = cursor.value.clone();

        for project_path in &projects {
            if fetch_issues {
                let issues = self
                    .fetch_issues(token, &base_url, project_path, updated_after)
                    .await?;
                for issue in &issues {
                    if latest_updated
                        .as_ref()
                        .map_or(true, |current| &issue.updated_at > current)
                    {
                        latest_updated = Some(issue.updated_at.clone());
                    }
                    items.push(self.issue_to_content_item(issue, project_path, &config.id));
                }
            }

            if fetch_mrs {
                let mrs = self
                    .fetch_merge_requests(token, &base_url, project_path, updated_after)
                    .await?;
                for mr in &mrs {
                    if latest_updated
                        .as_ref()
                        .map_or(true, |current| &mr.updated_at > current)
                    {
                        latest_updated = Some(mr.updated_at.clone());
                    }
                    items.push(self.mr_to_content_item(mr, project_path, &config.id));
                }
            }
            // Snippets don't support updated_after, skip in incremental sync
        }

        let now_secs = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        Ok(SyncResult {
            items,
            cursor: SyncCursor {
                value: latest_updated,
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

        // GitLab webhooks use object_kind to identify the event type
        let object_kind = body
            .get("object_kind")
            .and_then(|v| v.as_str())
            .unwrap_or("unknown");

        let project_path = body
            .get("project")
            .and_then(|p| p.get("path_with_namespace"))
            .and_then(|v| v.as_str())
            .unwrap_or("unknown/unknown");

        match object_kind {
            "issue" => {
                let attrs = body
                    .get("object_attributes")
                    .context("Missing object_attributes in issue webhook")?;

                let issue: GitLabIssue = serde_json::from_value(attrs.clone())
                    .context("Failed to parse issue from webhook")?;

                Ok(vec![
                    self.issue_to_content_item(&issue, project_path, &config.id)
                ])
            }
            "merge_request" => {
                let attrs = body
                    .get("object_attributes")
                    .context("Missing object_attributes in MR webhook")?;

                let mr: GitLabMergeRequest = serde_json::from_value(attrs.clone())
                    .context("Failed to parse MR from webhook")?;

                Ok(vec![
                    self.mr_to_content_item(&mr, project_path, &config.id)
                ])
            }
            _ => Ok(Vec::new()),
        }
    }
}

/// Parses a GitLab ISO 8601 timestamp (e.g., "2024-01-15T10:30:00.000Z") to Unix seconds.
fn parse_gitlab_timestamp(ts: &str) -> Option<u64> {
    // GitLab timestamps may include milliseconds: "YYYY-MM-DDTHH:MM:SS.mmmZ"
    let ts = ts.trim_end_matches('Z');
    let parts: Vec<&str> = ts.split('T').collect();
    if parts.len() != 2 {
        return None;
    }

    let date_parts: Vec<u64> = parts[0].split('-').filter_map(|s| s.parse().ok()).collect();
    // Strip fractional seconds if present
    let time_str = parts[1].split('.').next().unwrap_or(parts[1]);
    let time_parts: Vec<u64> = time_str
        .split(':')
        .filter_map(|s| s.parse().ok())
        .collect();

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
    (year % 4 == 0 && year % 100 != 0) || (year % 400 == 0)
}

// --- GitLab API response types ---

#[derive(Debug, Clone, Deserialize, Serialize)]
struct GitLabUser {
    username: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
struct GitLabIssue {
    /// Project-local issue number.
    iid: u64,
    title: String,
    description: Option<String>,
    state: String,
    web_url: String,
    created_at: String,
    updated_at: String,
    author: GitLabUser,
    #[serde(default)]
    labels: Vec<String>,
    assignee: Option<GitLabUser>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
struct GitLabMergeRequest {
    /// Project-local MR number.
    iid: u64,
    title: String,
    description: Option<String>,
    state: String,
    web_url: String,
    created_at: String,
    updated_at: String,
    author: GitLabUser,
    source_branch: String,
    target_branch: String,
    #[serde(default)]
    labels: Vec<String>,
    assignee: Option<GitLabUser>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
struct GitLabSnippet {
    id: u64,
    title: String,
    description: Option<String>,
    web_url: String,
    created_at: String,
    author: GitLabUser,
    file_name: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn make_config(token: &str, projects: Vec<&str>) -> ConnectorConfig {
        let mut settings = HashMap::new();
        settings.insert("projects".to_string(), serde_json::json!(projects));
        ConnectorConfig {
            id: "gitlab-test".to_string(),
            name: "Test GitLab".to_string(),
            connector_type: "gitlab".to_string(),
            auth: AuthConfig::Token {
                token: token.to_string(),
            },
            namespace: "test".to_string(),
            settings,
            enabled: true,
        }
    }

    fn make_issue(iid: u64) -> GitLabIssue {
        GitLabIssue {
            iid,
            title: format!("Test issue #{}", iid),
            description: Some("This is the description of the issue.".to_string()),
            state: "opened".to_string(),
            web_url: format!("https://gitlab.com/group/project/-/issues/{}", iid),
            created_at: "2024-06-15T10:30:00.000Z".to_string(),
            updated_at: "2024-06-16T12:00:00.000Z".to_string(),
            author: GitLabUser {
                username: "testuser".to_string(),
            },
            labels: vec!["bug".to_string(), "priority::high".to_string()],
            assignee: Some(GitLabUser {
                username: "assignee1".to_string(),
            }),
        }
    }

    fn make_mr(iid: u64) -> GitLabMergeRequest {
        GitLabMergeRequest {
            iid,
            title: format!("Feature branch #{}", iid),
            description: Some("Implements feature X.".to_string()),
            state: "opened".to_string(),
            web_url: format!("https://gitlab.com/group/project/-/merge_requests/{}", iid),
            created_at: "2024-06-15T10:30:00.000Z".to_string(),
            updated_at: "2024-06-16T12:00:00.000Z".to_string(),
            author: GitLabUser {
                username: "developer".to_string(),
            },
            source_branch: "feature-x".to_string(),
            target_branch: "main".to_string(),
            labels: vec!["enhancement".to_string()],
            assignee: Some(GitLabUser {
                username: "reviewer1".to_string(),
            }),
        }
    }

    fn make_snippet(id: u64) -> GitLabSnippet {
        GitLabSnippet {
            id,
            title: format!("Snippet #{}", id),
            description: Some("A code snippet for reference.".to_string()),
            web_url: format!("https://gitlab.com/group/project/-/snippets/{}", id),
            created_at: "2024-06-15T10:30:00.000Z".to_string(),
            author: GitLabUser {
                username: "snippeter".to_string(),
            },
            file_name: Some("example.py".to_string()),
        }
    }

    #[test]
    fn test_gitlab_connector_id_and_name() {
        let connector = GitLabConnector::new();
        assert_eq!(connector.id(), "gitlab");
        assert_eq!(connector.name(), "GitLab");
    }

    #[test]
    fn test_gitlab_config_schema() {
        let connector = GitLabConnector::new();
        let schema = connector.config_schema();
        assert_eq!(schema["type"], "object");
        assert!(schema["properties"]["auth"].is_object());
        assert!(schema["properties"]["settings"]["properties"]["projects"].is_object());
        assert!(schema["properties"]["settings"]["properties"]["base_url"].is_object());
        assert!(schema["properties"]["settings"]["properties"]["include_issues"].is_object());
        assert!(schema["properties"]["settings"]["properties"]["include_mrs"].is_object());
        assert!(schema["properties"]["settings"]["properties"]["include_snippets"].is_object());
    }

    #[test]
    fn test_validate_config_valid() {
        let connector = GitLabConnector::new();
        let config = make_config("glpat-test123", vec!["group/project"]);
        assert!(connector.validate_config(&config).is_ok());
    }

    #[test]
    fn test_validate_config_subgroup() {
        let connector = GitLabConnector::new();
        let config = make_config("glpat-test123", vec!["group/subgroup/project"]);
        assert!(connector.validate_config(&config).is_ok());
    }

    #[test]
    fn test_validate_config_wrong_type() {
        let connector = GitLabConnector::new();
        let mut config = make_config("glpat-test123", vec!["group/project"]);
        config.connector_type = "github".to_string();
        assert!(connector.validate_config(&config).is_err());
    }

    #[test]
    fn test_validate_config_wrong_auth_type() {
        let connector = GitLabConnector::new();
        let config = ConnectorConfig {
            id: "gl-test".to_string(),
            name: "Test".to_string(),
            connector_type: "gitlab".to_string(),
            auth: AuthConfig::ApiKey {
                key: "key".to_string(),
            },
            namespace: "test".to_string(),
            settings: {
                let mut s = HashMap::new();
                s.insert("projects".to_string(), serde_json::json!(["group/project"]));
                s
            },
            enabled: true,
        };
        let err = connector.validate_config(&config);
        assert!(err.is_err());
        assert!(err.unwrap_err().to_string().contains("Token"));
    }

    #[test]
    fn test_validate_config_no_projects() {
        let connector = GitLabConnector::new();
        let config = make_config("glpat-test123", vec![]);
        let err = connector.validate_config(&config);
        assert!(err.is_err());
        assert!(err.unwrap_err().to_string().contains("project"));
    }

    #[test]
    fn test_validate_config_invalid_project_format() {
        let connector = GitLabConnector::new();
        let config = make_config("glpat-test123", vec!["just-a-name"]);
        let err = connector.validate_config(&config);
        assert!(err.is_err());
        assert!(err.unwrap_err().to_string().contains("group/project"));
    }

    #[test]
    fn test_issue_to_content_item() {
        let connector = GitLabConnector::new();
        let issue = make_issue(42);

        let item = connector.issue_to_content_item(&issue, "group/project", "conn-1");

        assert!(item.content.contains("[group/project] Issue #42"));
        assert!(item.content.contains("Test issue #42"));
        assert!(item.content.contains("description of the issue"));
        assert_eq!(item.source.connector_type, "gitlab");
        assert_eq!(item.source.connector_id, "conn-1");
        assert_eq!(item.source.source_id, "group/project#42");
        assert_eq!(item.source.author.as_deref(), Some("testuser"));
        assert!(item.source.source_url.is_some());
        assert_eq!(
            item.source.extra.get("kind").unwrap(),
            &serde_json::Value::String("Issue".to_string())
        );
        assert_eq!(
            item.source.extra.get("state").unwrap(),
            &serde_json::Value::String("opened".to_string())
        );
        assert!(item
            .source
            .extra
            .get("labels")
            .unwrap()
            .as_str()
            .unwrap()
            .contains("bug"));
        assert_eq!(
            item.source.extra.get("assignee").unwrap(),
            &serde_json::Value::String("assignee1".to_string())
        );
        assert!(item.media.is_none());
    }

    #[test]
    fn test_issue_to_content_item_no_description() {
        let connector = GitLabConnector::new();
        let mut issue = make_issue(1);
        issue.description = None;

        let item = connector.issue_to_content_item(&issue, "group/project", "conn-1");

        assert!(!item.content.contains("\n\n"));
        assert!(item.content.contains("[group/project] Issue #1"));
    }

    #[test]
    fn test_mr_to_content_item() {
        let connector = GitLabConnector::new();
        let mr = make_mr(10);

        let item = connector.mr_to_content_item(&mr, "group/project", "conn-1");

        assert!(item.content.contains("[group/project] MR !10"));
        assert!(item.content.contains("Feature branch #10"));
        assert!(item.content.contains("Implements feature X"));
        assert_eq!(item.source.connector_type, "gitlab");
        assert_eq!(item.source.source_id, "group/project!10");
        assert_eq!(item.source.author.as_deref(), Some("developer"));
        assert_eq!(
            item.source.extra.get("kind").unwrap(),
            &serde_json::Value::String("MR".to_string())
        );
        assert_eq!(
            item.source.extra.get("source_branch").unwrap(),
            &serde_json::Value::String("feature-x".to_string())
        );
        assert_eq!(
            item.source.extra.get("target_branch").unwrap(),
            &serde_json::Value::String("main".to_string())
        );
        assert_eq!(
            item.source.extra.get("assignee").unwrap(),
            &serde_json::Value::String("reviewer1".to_string())
        );
    }

    #[test]
    fn test_mr_to_content_item_no_description() {
        let connector = GitLabConnector::new();
        let mut mr = make_mr(5);
        mr.description = None;

        let item = connector.mr_to_content_item(&mr, "group/project", "conn-1");

        assert!(!item.content.contains("\n\n"));
        assert!(item.content.contains("[group/project] MR !5"));
    }

    #[test]
    fn test_snippet_to_content_item() {
        let connector = GitLabConnector::new();
        let snippet = make_snippet(99);

        let item = connector.snippet_to_content_item(&snippet, "group/project", "conn-1");

        assert!(item.content.contains("[group/project] Snippet #99"));
        assert!(item.content.contains("code snippet for reference"));
        assert_eq!(item.source.connector_type, "gitlab");
        assert_eq!(item.source.source_id, "group/project~99");
        assert_eq!(item.source.author.as_deref(), Some("snippeter"));
        assert_eq!(
            item.source.extra.get("kind").unwrap(),
            &serde_json::Value::String("Snippet".to_string())
        );
        assert_eq!(
            item.source.extra.get("file_name").unwrap(),
            &serde_json::Value::String("example.py".to_string())
        );
    }

    #[test]
    fn test_parse_gitlab_timestamp() {
        let ts = parse_gitlab_timestamp("2024-01-01T00:00:00.000Z");
        assert!(ts.is_some());
        assert_eq!(ts.unwrap(), 1704067200);
    }

    #[test]
    fn test_parse_gitlab_timestamp_no_millis() {
        let ts = parse_gitlab_timestamp("2024-01-01T00:00:00Z");
        assert!(ts.is_some());
        assert_eq!(ts.unwrap(), 1704067200);
    }

    #[test]
    fn test_parse_gitlab_timestamp_with_time() {
        let ts = parse_gitlab_timestamp("2024-06-15T10:30:00.000Z");
        assert!(ts.is_some());
        assert!(ts.unwrap() > 1704067200);
    }

    #[test]
    fn test_parse_gitlab_timestamp_invalid() {
        assert!(parse_gitlab_timestamp("not-a-date").is_none());
        assert!(parse_gitlab_timestamp("").is_none());
    }

    #[test]
    fn test_get_projects_from_settings() {
        let config = make_config("token", vec!["group/project1", "org/repo2"]);
        let projects = GitLabConnector::get_projects(&config);
        assert_eq!(projects, vec!["group/project1", "org/repo2"]);
    }

    #[test]
    fn test_get_projects_empty_settings() {
        let mut config = make_config("token", vec![]);
        config.settings.clear();
        let projects = GitLabConnector::get_projects(&config);
        assert!(projects.is_empty());
    }

    #[test]
    fn test_include_flags_defaults() {
        let config = make_config("token", vec!["g/p"]);
        assert!(GitLabConnector::include_issues(&config));
        assert!(GitLabConnector::include_mrs(&config));
        assert!(!GitLabConnector::include_snippets(&config)); // Default is false
    }

    #[test]
    fn test_include_flags_custom() {
        let mut config = make_config("token", vec!["g/p"]);
        config
            .settings
            .insert("include_issues".to_string(), serde_json::json!(false));
        config
            .settings
            .insert("include_mrs".to_string(), serde_json::json!(false));
        config
            .settings
            .insert("include_snippets".to_string(), serde_json::json!(true));

        assert!(!GitLabConnector::include_issues(&config));
        assert!(!GitLabConnector::include_mrs(&config));
        assert!(GitLabConnector::include_snippets(&config));
    }

    #[test]
    fn test_default_constructor() {
        let connector = GitLabConnector::default();
        assert_eq!(connector.id(), "gitlab");
    }

    #[test]
    fn test_validate_config_multiple_projects() {
        let connector = GitLabConnector::new();
        let config = make_config("glpat-test", vec!["group/p1", "org/p2", "user/p3"]);
        assert!(connector.validate_config(&config).is_ok());
    }

    #[test]
    fn test_encode_project_path() {
        assert_eq!(GitLabConnector::encode_project_path("group/project"), "group%2Fproject");
        assert_eq!(
            GitLabConnector::encode_project_path("group/subgroup/project"),
            "group%2Fsubgroup%2Fproject"
        );
    }

    #[test]
    fn test_get_base_url_default() {
        let config = make_config("token", vec!["g/p"]);
        let url = GitLabConnector::get_base_url(&config);
        assert_eq!(url, GITLAB_API_DEFAULT);
    }

    #[test]
    fn test_get_base_url_custom() {
        let mut config = make_config("token", vec!["g/p"]);
        config.settings.insert(
            "base_url".to_string(),
            serde_json::json!("https://gitlab.example.com/api/v4/"),
        );
        let url = GitLabConnector::get_base_url(&config);
        assert_eq!(url, "https://gitlab.example.com/api/v4");
    }

    #[tokio::test]
    async fn test_handle_webhook_issue() {
        let connector = GitLabConnector::new();
        let config = make_config("token", vec!["group/project"]);
        let payload = WebhookPayload {
            body: serde_json::to_vec(&serde_json::json!({
                "object_kind": "issue",
                "project": {
                    "path_with_namespace": "group/project"
                },
                "object_attributes": {
                    "iid": 42,
                    "title": "New bug report",
                    "description": "Something is broken",
                    "state": "opened",
                    "web_url": "https://gitlab.com/group/project/-/issues/42",
                    "created_at": "2024-06-15T10:00:00.000Z",
                    "updated_at": "2024-06-15T10:00:00.000Z",
                    "author": { "username": "reporter" },
                    "labels": ["bug"],
                    "assignee": null
                }
            }))
            .unwrap(),
            headers: HashMap::new(),
            content_type: Some("application/json".to_string()),
        };

        let items = connector.handle_webhook(&config, payload).await.unwrap();
        assert_eq!(items.len(), 1);
        assert!(items[0].content.contains("[group/project] Issue #42"));
        assert!(items[0].content.contains("New bug report"));
        assert_eq!(items[0].source.author.as_deref(), Some("reporter"));
    }

    #[tokio::test]
    async fn test_handle_webhook_merge_request() {
        let connector = GitLabConnector::new();
        let config = make_config("token", vec!["group/project"]);
        let payload = WebhookPayload {
            body: serde_json::to_vec(&serde_json::json!({
                "object_kind": "merge_request",
                "project": {
                    "path_with_namespace": "group/project"
                },
                "object_attributes": {
                    "iid": 10,
                    "title": "Add feature X",
                    "description": "Implements feature X",
                    "state": "opened",
                    "web_url": "https://gitlab.com/group/project/-/merge_requests/10",
                    "created_at": "2024-06-15T10:00:00.000Z",
                    "updated_at": "2024-06-15T10:00:00.000Z",
                    "author": { "username": "developer" },
                    "source_branch": "feature-x",
                    "target_branch": "main",
                    "labels": [],
                    "assignee": null
                }
            }))
            .unwrap(),
            headers: HashMap::new(),
            content_type: Some("application/json".to_string()),
        };

        let items = connector.handle_webhook(&config, payload).await.unwrap();
        assert_eq!(items.len(), 1);
        assert!(items[0].content.contains("MR !10"));
        assert!(items[0].content.contains("Add feature X"));
    }

    #[tokio::test]
    async fn test_handle_webhook_unknown_event() {
        let connector = GitLabConnector::new();
        let config = make_config("token", vec!["group/project"]);
        let payload = WebhookPayload {
            body: serde_json::to_vec(&serde_json::json!({
                "object_kind": "pipeline",
                "project": { "path_with_namespace": "group/project" }
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
        let connector = GitLabConnector::new();
        let config = make_config("token", vec!["group/project"]);
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
