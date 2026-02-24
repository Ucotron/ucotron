//! GitHub connector — fetches issues and PRs from GitHub repositories via the REST API.
//!
//! Uses Personal Access Tokens (PAT) to authenticate with the GitHub API.
//! Supports full sync (all issues/PRs from configured repos) and
//! incremental sync via `since` timestamp-based pagination.

use std::collections::HashMap;

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};

use crate::connector::{
    AuthConfig, Connector, ConnectorConfig, ContentItem, SourceMetadata, SyncCursor, SyncResult,
    WebhookPayload,
};

const GITHUB_API_BASE: &str = "https://api.github.com";

/// GitHub connector for fetching issues and pull requests.
///
/// Requires a Personal Access Token with the following scopes:
/// - `repo` — read access to private repositories
/// - For public repos only, the `public_repo` scope is sufficient
pub struct GitHubConnector {
    client: reqwest::Client,
}

impl GitHubConnector {
    /// Creates a new GitHubConnector with a default HTTP client.
    pub fn new() -> Self {
        Self {
            client: reqwest::Client::builder()
                .user_agent("ucotron-connector/0.1")
                .build()
                .unwrap_or_else(|_| reqwest::Client::new()),
        }
    }

    /// Creates a new GitHubConnector with a custom HTTP client.
    pub fn with_client(client: reqwest::Client) -> Self {
        Self { client }
    }

    /// Extracts the PAT from the connector config.
    fn get_token(config: &ConnectorConfig) -> Result<&str> {
        match &config.auth {
            AuthConfig::Token { token } => Ok(token.as_str()),
            _ => bail!("GitHub connector requires Token authentication (Personal Access Token)"),
        }
    }

    /// Extracts configured repository names from settings (e.g., ["owner/repo"]).
    fn get_repos(config: &ConnectorConfig) -> Vec<String> {
        config
            .settings
            .get("repos")
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

    /// Checks if PRs should be fetched (default: true).
    fn include_prs(config: &ConnectorConfig) -> bool {
        config
            .settings
            .get("include_prs")
            .and_then(|v| v.as_bool())
            .unwrap_or(true)
    }

    /// Checks if comments should be fetched (default: true).
    fn include_comments(config: &ConnectorConfig) -> bool {
        config
            .settings
            .get("include_comments")
            .and_then(|v| v.as_bool())
            .unwrap_or(true)
    }

    /// Fetches issues from a repository with pagination.
    /// When `since` is provided, only returns issues updated after that timestamp.
    async fn fetch_issues(
        &self,
        token: &str,
        owner: &str,
        repo: &str,
        since: Option<&str>,
    ) -> Result<Vec<GitHubIssue>> {
        let mut all_issues = Vec::new();
        let mut page = 1u32;

        loop {
            let mut url = format!(
                "{}/repos/{}/{}/issues?state=all&per_page=100&page={}&sort=updated&direction=desc",
                GITHUB_API_BASE, owner, repo, page
            );
            if let Some(since_ts) = since {
                url.push_str(&format!("&since={}", since_ts));
            }

            let resp = self
                .client
                .get(&url)
                .bearer_auth(token)
                .header("Accept", "application/vnd.github+json")
                .header("X-GitHub-Api-Version", "2022-11-28")
                .send()
                .await
                .context("Failed to fetch GitHub issues")?;

            if !resp.status().is_success() {
                let status = resp.status();
                let body = resp.text().await.unwrap_or_default();
                bail!("GitHub API error fetching issues: {} - {}", status, body);
            }

            let issues: Vec<GitHubIssue> = resp
                .json()
                .await
                .context("Failed to parse GitHub issues response")?;

            let count = issues.len();
            all_issues.extend(issues);

            if count < 100 {
                break;
            }
            page += 1;
        }

        Ok(all_issues)
    }

    /// Fetches comments for an issue or PR.
    async fn fetch_comments(
        &self,
        token: &str,
        owner: &str,
        repo: &str,
        issue_number: u64,
        since: Option<&str>,
    ) -> Result<Vec<GitHubComment>> {
        let mut all_comments = Vec::new();
        let mut page = 1u32;

        loop {
            let mut url = format!(
                "{}/repos/{}/{}/issues/{}/comments?per_page=100&page={}",
                GITHUB_API_BASE, owner, repo, issue_number, page
            );
            if let Some(since_ts) = since {
                url.push_str(&format!("&since={}", since_ts));
            }

            let resp = self
                .client
                .get(&url)
                .bearer_auth(token)
                .header("Accept", "application/vnd.github+json")
                .header("X-GitHub-Api-Version", "2022-11-28")
                .send()
                .await
                .context("Failed to fetch GitHub comments")?;

            if !resp.status().is_success() {
                let status = resp.status();
                let body = resp.text().await.unwrap_or_default();
                bail!(
                    "GitHub API error fetching comments for #{}: {} - {}",
                    issue_number,
                    status,
                    body
                );
            }

            let comments: Vec<GitHubComment> = resp
                .json()
                .await
                .context("Failed to parse GitHub comments response")?;

            let count = comments.len();
            all_comments.extend(comments);

            if count < 100 {
                break;
            }
            page += 1;
        }

        Ok(all_comments)
    }

    /// Converts a GitHub issue/PR to a ContentItem.
    fn issue_to_content_item(
        &self,
        issue: &GitHubIssue,
        owner: &str,
        repo: &str,
        connector_id: &str,
    ) -> ContentItem {
        let kind = if issue.pull_request.is_some() {
            "PR"
        } else {
            "Issue"
        };

        let labels_str: String = issue
            .labels
            .iter()
            .map(|l| l.name.as_str())
            .collect::<Vec<_>>()
            .join(", ");

        // Build rich content with title, body, labels, and state
        let body_preview = issue
            .body
            .as_deref()
            .unwrap_or("")
            .chars()
            .take(2000)
            .collect::<String>();

        let content = if body_preview.is_empty() {
            format!(
                "[{}/{}] {} #{}: {}",
                owner, repo, kind, issue.number, issue.title
            )
        } else {
            format!(
                "[{}/{}] {} #{}: {}\n\n{}",
                owner, repo, kind, issue.number, issue.title, body_preview
            )
        };

        let created_at = parse_github_timestamp(&issue.created_at);

        let mut extra = HashMap::new();
        extra.insert(
            "repo".to_string(),
            serde_json::Value::String(format!("{}/{}", owner, repo)),
        );
        extra.insert(
            "number".to_string(),
            serde_json::Value::Number(issue.number.into()),
        );
        extra.insert(
            "state".to_string(),
            serde_json::Value::String(issue.state.clone()),
        );
        extra.insert(
            "kind".to_string(),
            serde_json::Value::String(kind.to_string()),
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
                serde_json::Value::String(assignee.login.clone()),
            );
        }

        ContentItem {
            content,
            source: SourceMetadata {
                connector_type: "github".to_string(),
                connector_id: connector_id.to_string(),
                source_id: format!("{}/{}#{}", owner, repo, issue.number),
                source_url: Some(issue.html_url.clone()),
                author: Some(issue.user.login.clone()),
                created_at,
                extra,
            },
            media: None,
        }
    }

    /// Converts a GitHub comment to a ContentItem.
    fn comment_to_content_item(
        &self,
        comment: &GitHubComment,
        issue_number: u64,
        issue_title: &str,
        kind: &str,
        owner: &str,
        repo: &str,
        connector_id: &str,
    ) -> ContentItem {
        let body_preview = comment.body.chars().take(2000).collect::<String>();

        let content = format!(
            "[{}/{}] Comment on {} #{} ({}): {}",
            owner, repo, kind, issue_number, issue_title, body_preview
        );

        let created_at = parse_github_timestamp(&comment.created_at);

        let mut extra = HashMap::new();
        extra.insert(
            "repo".to_string(),
            serde_json::Value::String(format!("{}/{}", owner, repo)),
        );
        extra.insert(
            "issue_number".to_string(),
            serde_json::Value::Number(issue_number.into()),
        );
        extra.insert(
            "kind".to_string(),
            serde_json::Value::String(format!("{}_comment", kind.to_lowercase())),
        );

        ContentItem {
            content,
            source: SourceMetadata {
                connector_type: "github".to_string(),
                connector_id: connector_id.to_string(),
                source_id: format!("{}/{}#{}c{}", owner, repo, issue_number, comment.id),
                source_url: Some(comment.html_url.clone()),
                author: Some(comment.user.login.clone()),
                created_at,
                extra,
            },
            media: None,
        }
    }
}

impl Default for GitHubConnector {
    fn default() -> Self {
        Self::new()
    }
}

impl Connector for GitHubConnector {
    fn id(&self) -> &str {
        "github"
    }

    fn name(&self) -> &str {
        "GitHub"
    }

    fn config_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "auth": {
                    "type": "object",
                    "description": "Personal Access Token credentials",
                    "properties": {
                        "token": { "type": "string", "description": "GitHub PAT (ghp_...)" }
                    },
                    "required": ["token"]
                },
                "settings": {
                    "type": "object",
                    "properties": {
                        "repos": {
                            "type": "array",
                            "items": { "type": "string" },
                            "description": "Repositories to sync in 'owner/repo' format"
                        },
                        "include_issues": {
                            "type": "boolean",
                            "description": "Whether to fetch issues (default: true)"
                        },
                        "include_prs": {
                            "type": "boolean",
                            "description": "Whether to fetch pull requests (default: true)"
                        },
                        "include_comments": {
                            "type": "boolean",
                            "description": "Whether to fetch issue/PR comments (default: true)"
                        }
                    }
                }
            },
            "required": ["auth", "settings"]
        })
    }

    fn validate_config(&self, config: &ConnectorConfig) -> Result<()> {
        if config.connector_type != "github" {
            bail!(
                "Invalid connector type '{}', expected 'github'",
                config.connector_type
            );
        }
        Self::get_token(config)?;
        let repos = Self::get_repos(config);
        if repos.is_empty() {
            bail!("GitHub connector requires at least one repository in settings.repos");
        }
        for repo in &repos {
            if !repo.contains('/') || repo.split('/').count() != 2 {
                bail!(
                    "Invalid repository format '{}', expected 'owner/repo'",
                    repo
                );
            }
        }
        Ok(())
    }

    async fn fetch(&self, config: &ConnectorConfig) -> Result<Vec<ContentItem>> {
        let token = Self::get_token(config)?;
        let repos = Self::get_repos(config);
        let fetch_issues = Self::include_issues(config);
        let fetch_prs = Self::include_prs(config);
        let fetch_comments = Self::include_comments(config);

        let mut items = Vec::new();

        for repo_full in &repos {
            let parts: Vec<&str> = repo_full.split('/').collect();
            let (owner, repo) = (parts[0], parts[1]);

            let issues = self.fetch_issues(token, owner, repo, None).await?;

            for issue in &issues {
                let is_pr = issue.pull_request.is_some();

                // Skip based on config
                if is_pr && !fetch_prs {
                    continue;
                }
                if !is_pr && !fetch_issues {
                    continue;
                }

                items.push(self.issue_to_content_item(issue, owner, repo, &config.id));

                // Fetch comments if enabled and there are comments
                if fetch_comments && issue.comments > 0 {
                    let kind = if is_pr { "PR" } else { "Issue" };
                    match self
                        .fetch_comments(token, owner, repo, issue.number, None)
                        .await
                    {
                        Ok(comments) => {
                            for comment in &comments {
                                items.push(self.comment_to_content_item(
                                    comment,
                                    issue.number,
                                    &issue.title,
                                    kind,
                                    owner,
                                    repo,
                                    &config.id,
                                ));
                            }
                        }
                        Err(e) => {
                            // Log but don't fail — some comments may be inaccessible
                            eprintln!(
                                "Warning: failed to fetch comments for {}/{}#{}: {}",
                                owner, repo, issue.number, e
                            );
                        }
                    }
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
        let repos = Self::get_repos(config);
        let fetch_issues = Self::include_issues(config);
        let fetch_prs = Self::include_prs(config);
        let fetch_comments = Self::include_comments(config);

        // Use cursor value as ISO 8601 timestamp for `since` parameter
        let since = cursor.value.as_deref();

        let mut items = Vec::new();
        let mut latest_updated: Option<String> = cursor.value.clone();

        for repo_full in &repos {
            let parts: Vec<&str> = repo_full.split('/').collect();
            let (owner, repo) = (parts[0], parts[1]);

            let issues = self.fetch_issues(token, owner, repo, since).await?;

            for issue in &issues {
                let is_pr = issue.pull_request.is_some();

                if is_pr && !fetch_prs {
                    continue;
                }
                if !is_pr && !fetch_issues {
                    continue;
                }

                // Track the most recent updated_at across all items
                if latest_updated
                    .as_ref()
                    .map_or(true, |current| &issue.updated_at > current)
                {
                    latest_updated = Some(issue.updated_at.clone());
                }

                items.push(self.issue_to_content_item(issue, owner, repo, &config.id));

                if fetch_comments && issue.comments > 0 {
                    let kind = if is_pr { "PR" } else { "Issue" };
                    if let Ok(comments) = self
                        .fetch_comments(token, owner, repo, issue.number, since)
                        .await
                    {
                        for comment in &comments {
                            items.push(self.comment_to_content_item(
                                comment,
                                issue.number,
                                &issue.title,
                                kind,
                                owner,
                                repo,
                                &config.id,
                            ));
                        }
                    }
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

        // Determine the event type from the X-GitHub-Event header
        let event_type = payload
            .headers
            .get("x-github-event")
            .or_else(|| payload.headers.get("X-GitHub-Event"))
            .map(|s| s.as_str())
            .unwrap_or("unknown");

        match event_type {
            "issues" | "pull_request" => {
                let action = body
                    .get("action")
                    .and_then(|v| v.as_str())
                    .unwrap_or_default();

                // Only process opened, edited, closed, reopened
                if !["opened", "edited", "closed", "reopened"].contains(&action) {
                    return Ok(Vec::new());
                }

                let item_key = if event_type == "pull_request" {
                    "pull_request"
                } else {
                    "issue"
                };

                let issue_obj = body
                    .get(item_key)
                    .context("Missing issue/pull_request in webhook body")?;

                let issue: GitHubIssue = serde_json::from_value(issue_obj.clone())
                    .context("Failed to parse issue from webhook")?;

                let repo_full = body
                    .get("repository")
                    .and_then(|r| r.get("full_name"))
                    .and_then(|v| v.as_str())
                    .unwrap_or("unknown/unknown");

                let parts: Vec<&str> = repo_full.split('/').collect();
                let (owner, repo) = if parts.len() == 2 {
                    (parts[0], parts[1])
                } else {
                    ("unknown", "unknown")
                };

                Ok(vec![
                    self.issue_to_content_item(&issue, owner, repo, &config.id)
                ])
            }
            "issue_comment" => {
                let action = body
                    .get("action")
                    .and_then(|v| v.as_str())
                    .unwrap_or_default();

                if action != "created" {
                    return Ok(Vec::new());
                }

                let comment_obj = body
                    .get("comment")
                    .context("Missing comment in webhook body")?;
                let comment: GitHubComment = serde_json::from_value(comment_obj.clone())
                    .context("Failed to parse comment from webhook")?;

                let issue_obj = body.get("issue").context("Missing issue in webhook body")?;

                let issue_number = issue_obj
                    .get("number")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(0);
                let issue_title = issue_obj
                    .get("title")
                    .and_then(|v| v.as_str())
                    .unwrap_or("Unknown");
                let is_pr = issue_obj
                    .get("pull_request")
                    .is_some_and(|v| !v.is_null());
                let kind = if is_pr { "PR" } else { "Issue" };

                let repo_full = body
                    .get("repository")
                    .and_then(|r| r.get("full_name"))
                    .and_then(|v| v.as_str())
                    .unwrap_or("unknown/unknown");

                let parts: Vec<&str> = repo_full.split('/').collect();
                let (owner, repo) = if parts.len() == 2 {
                    (parts[0], parts[1])
                } else {
                    ("unknown", "unknown")
                };

                Ok(vec![self.comment_to_content_item(
                    &comment,
                    issue_number,
                    issue_title,
                    kind,
                    owner,
                    repo,
                    &config.id,
                )])
            }
            _ => Ok(Vec::new()),
        }
    }
}

/// Parses a GitHub ISO 8601 timestamp (e.g., "2024-01-15T10:30:00Z") to Unix seconds.
fn parse_github_timestamp(ts: &str) -> Option<u64> {
    // Simple ISO 8601 parser: "YYYY-MM-DDTHH:MM:SSZ"
    // We parse just enough to get Unix seconds without pulling in chrono
    let ts = ts.trim_end_matches('Z');
    let parts: Vec<&str> = ts.split('T').collect();
    if parts.len() != 2 {
        return None;
    }

    let date_parts: Vec<u64> = parts[0].split('-').filter_map(|s| s.parse().ok()).collect();
    let time_parts: Vec<u64> = parts[1]
        .split(':')
        .filter_map(|s| s.parse().ok())
        .collect();

    if date_parts.len() != 3 || time_parts.len() != 3 {
        return None;
    }

    let (year, month, day) = (date_parts[0], date_parts[1], date_parts[2]);
    let (hour, min, sec) = (time_parts[0], time_parts[1], time_parts[2]);

    // Convert to Unix timestamp using a simple formula
    // Days from epoch (1970-01-01) to the given date
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

// --- GitHub API response types ---

#[derive(Debug, Clone, Deserialize, Serialize)]
struct GitHubIssue {
    number: u64,
    title: String,
    body: Option<String>,
    state: String,
    html_url: String,
    created_at: String,
    updated_at: String,
    user: GitHubUser,
    labels: Vec<GitHubLabel>,
    assignee: Option<GitHubUser>,
    /// Present only for pull requests (GitHub issues API includes PRs).
    pull_request: Option<serde_json::Value>,
    #[serde(default)]
    comments: u64,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
struct GitHubUser {
    login: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
struct GitHubLabel {
    name: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
struct GitHubComment {
    id: u64,
    body: String,
    html_url: String,
    created_at: String,
    user: GitHubUser,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn make_config(token: &str, repos: Vec<&str>) -> ConnectorConfig {
        let mut settings = HashMap::new();
        settings.insert("repos".to_string(), serde_json::json!(repos));
        ConnectorConfig {
            id: "github-test".to_string(),
            name: "Test GitHub".to_string(),
            connector_type: "github".to_string(),
            auth: AuthConfig::Token {
                token: token.to_string(),
            },
            namespace: "test".to_string(),
            settings,
            enabled: true,
        }
    }

    fn make_issue(number: u64, is_pr: bool) -> GitHubIssue {
        GitHubIssue {
            number,
            title: format!("Test issue #{}", number),
            body: Some("This is the body of the issue.".to_string()),
            state: "open".to_string(),
            html_url: format!("https://github.com/owner/repo/issues/{}", number),
            created_at: "2024-06-15T10:30:00Z".to_string(),
            updated_at: "2024-06-16T12:00:00Z".to_string(),
            user: GitHubUser {
                login: "testuser".to_string(),
            },
            labels: vec![
                GitHubLabel {
                    name: "bug".to_string(),
                },
                GitHubLabel {
                    name: "priority:high".to_string(),
                },
            ],
            assignee: Some(GitHubUser {
                login: "assignee1".to_string(),
            }),
            pull_request: if is_pr {
                Some(serde_json::json!({}))
            } else {
                None
            },
            comments: 2,
        }
    }

    fn make_comment(id: u64, issue_number: u64) -> GitHubComment {
        GitHubComment {
            id,
            body: format!("Comment {} on issue #{}", id, issue_number),
            html_url: format!(
                "https://github.com/owner/repo/issues/{}#issuecomment-{}",
                issue_number, id
            ),
            created_at: "2024-06-15T11:00:00Z".to_string(),
            user: GitHubUser {
                login: "commenter".to_string(),
            },
        }
    }

    #[test]
    fn test_github_connector_id_and_name() {
        let connector = GitHubConnector::new();
        assert_eq!(connector.id(), "github");
        assert_eq!(connector.name(), "GitHub");
    }

    #[test]
    fn test_github_config_schema() {
        let connector = GitHubConnector::new();
        let schema = connector.config_schema();
        assert_eq!(schema["type"], "object");
        assert!(schema["properties"]["auth"].is_object());
        assert!(schema["properties"]["settings"]["properties"]["repos"].is_object());
        assert!(schema["properties"]["settings"]["properties"]["include_issues"].is_object());
        assert!(schema["properties"]["settings"]["properties"]["include_prs"].is_object());
    }

    #[test]
    fn test_validate_config_valid() {
        let connector = GitHubConnector::new();
        let config = make_config("ghp_test123", vec!["owner/repo"]);
        assert!(connector.validate_config(&config).is_ok());
    }

    #[test]
    fn test_validate_config_wrong_type() {
        let connector = GitHubConnector::new();
        let mut config = make_config("ghp_test123", vec!["owner/repo"]);
        config.connector_type = "slack".to_string();
        assert!(connector.validate_config(&config).is_err());
    }

    #[test]
    fn test_validate_config_wrong_auth_type() {
        let connector = GitHubConnector::new();
        let config = ConnectorConfig {
            id: "gh-test".to_string(),
            name: "Test".to_string(),
            connector_type: "github".to_string(),
            auth: AuthConfig::OAuth2 {
                client_id: "cid".to_string(),
                client_secret: "cs".to_string(),
                access_token: Some("token".to_string()),
                refresh_token: None,
            },
            namespace: "test".to_string(),
            settings: {
                let mut s = HashMap::new();
                s.insert("repos".to_string(), serde_json::json!(["owner/repo"]));
                s
            },
            enabled: true,
        };
        let err = connector.validate_config(&config);
        assert!(err.is_err());
        assert!(err.unwrap_err().to_string().contains("Token"));
    }

    #[test]
    fn test_validate_config_no_repos() {
        let connector = GitHubConnector::new();
        let config = make_config("ghp_test123", vec![]);
        let err = connector.validate_config(&config);
        assert!(err.is_err());
        assert!(err.unwrap_err().to_string().contains("repository"));
    }

    #[test]
    fn test_validate_config_invalid_repo_format() {
        let connector = GitHubConnector::new();
        let config = make_config("ghp_test123", vec!["just-a-name"]);
        let err = connector.validate_config(&config);
        assert!(err.is_err());
        assert!(err.unwrap_err().to_string().contains("owner/repo"));
    }

    #[test]
    fn test_issue_to_content_item_issue() {
        let connector = GitHubConnector::new();
        let issue = make_issue(42, false);

        let item = connector.issue_to_content_item(&issue, "owner", "repo", "conn-1");

        assert!(item.content.contains("[owner/repo] Issue #42"));
        assert!(item.content.contains("Test issue #42"));
        assert!(item.content.contains("body of the issue"));
        assert_eq!(item.source.connector_type, "github");
        assert_eq!(item.source.connector_id, "conn-1");
        assert_eq!(item.source.source_id, "owner/repo#42");
        assert_eq!(item.source.author.as_deref(), Some("testuser"));
        assert!(item.source.source_url.is_some());
        assert_eq!(
            item.source.extra.get("kind").unwrap(),
            &serde_json::Value::String("Issue".to_string())
        );
        assert_eq!(
            item.source.extra.get("state").unwrap(),
            &serde_json::Value::String("open".to_string())
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
    fn test_issue_to_content_item_pr() {
        let connector = GitHubConnector::new();
        let issue = make_issue(99, true);

        let item = connector.issue_to_content_item(&issue, "owner", "repo", "conn-1");

        assert!(item.content.contains("[owner/repo] PR #99"));
        assert_eq!(
            item.source.extra.get("kind").unwrap(),
            &serde_json::Value::String("PR".to_string())
        );
    }

    #[test]
    fn test_issue_to_content_item_no_body() {
        let connector = GitHubConnector::new();
        let mut issue = make_issue(1, false);
        issue.body = None;

        let item = connector.issue_to_content_item(&issue, "owner", "repo", "conn-1");

        // Should not contain double newlines from empty body
        assert!(!item.content.contains("\n\n"));
        assert!(item.content.contains("[owner/repo] Issue #1"));
    }

    #[test]
    fn test_comment_to_content_item() {
        let connector = GitHubConnector::new();
        let comment = make_comment(100, 42);

        let item = connector.comment_to_content_item(
            &comment,
            42,
            "Test issue #42",
            "Issue",
            "owner",
            "repo",
            "conn-1",
        );

        assert!(item.content.contains("[owner/repo] Comment on Issue #42"));
        assert!(item.content.contains("Comment 100 on issue #42"));
        assert_eq!(item.source.connector_type, "github");
        assert_eq!(item.source.source_id, "owner/repo#42c100");
        assert_eq!(item.source.author.as_deref(), Some("commenter"));
        assert_eq!(
            item.source.extra.get("kind").unwrap(),
            &serde_json::Value::String("issue_comment".to_string())
        );
    }

    #[test]
    fn test_parse_github_timestamp() {
        let ts = parse_github_timestamp("2024-01-01T00:00:00Z");
        assert!(ts.is_some());
        // 2024-01-01 00:00:00 UTC = 1704067200
        assert_eq!(ts.unwrap(), 1704067200);
    }

    #[test]
    fn test_parse_github_timestamp_with_time() {
        let ts = parse_github_timestamp("2024-06-15T10:30:00Z");
        assert!(ts.is_some());
        // Verify it's a reasonable value (after 2024-01-01)
        assert!(ts.unwrap() > 1704067200);
    }

    #[test]
    fn test_parse_github_timestamp_invalid() {
        assert!(parse_github_timestamp("not-a-date").is_none());
        assert!(parse_github_timestamp("").is_none());
    }

    #[test]
    fn test_get_repos_from_settings() {
        let config = make_config("token", vec!["owner/repo1", "org/repo2"]);
        let repos = GitHubConnector::get_repos(&config);
        assert_eq!(repos, vec!["owner/repo1", "org/repo2"]);
    }

    #[test]
    fn test_get_repos_empty_settings() {
        let mut config = make_config("token", vec![]);
        config.settings.clear();
        let repos = GitHubConnector::get_repos(&config);
        assert!(repos.is_empty());
    }

    #[test]
    fn test_include_flags_defaults() {
        let config = make_config("token", vec!["o/r"]);
        assert!(GitHubConnector::include_issues(&config));
        assert!(GitHubConnector::include_prs(&config));
        assert!(GitHubConnector::include_comments(&config));
    }

    #[test]
    fn test_include_flags_custom() {
        let mut config = make_config("token", vec!["o/r"]);
        config
            .settings
            .insert("include_issues".to_string(), serde_json::json!(false));
        config
            .settings
            .insert("include_prs".to_string(), serde_json::json!(false));
        config
            .settings
            .insert("include_comments".to_string(), serde_json::json!(false));

        assert!(!GitHubConnector::include_issues(&config));
        assert!(!GitHubConnector::include_prs(&config));
        assert!(!GitHubConnector::include_comments(&config));
    }

    #[test]
    fn test_default_constructor() {
        let connector = GitHubConnector::default();
        assert_eq!(connector.id(), "github");
    }

    #[test]
    fn test_validate_config_multiple_repos() {
        let connector = GitHubConnector::new();
        let config = make_config("ghp_test", vec!["owner/repo1", "org/repo2", "user/repo3"]);
        assert!(connector.validate_config(&config).is_ok());
    }

    #[tokio::test]
    async fn test_handle_webhook_issue_opened() {
        let connector = GitHubConnector::new();
        let config = make_config("token", vec!["owner/repo"]);
        let payload = WebhookPayload {
            body: serde_json::to_vec(&serde_json::json!({
                "action": "opened",
                "issue": {
                    "number": 42,
                    "title": "New bug report",
                    "body": "Something is broken",
                    "state": "open",
                    "html_url": "https://github.com/owner/repo/issues/42",
                    "created_at": "2024-06-15T10:00:00Z",
                    "updated_at": "2024-06-15T10:00:00Z",
                    "user": { "login": "reporter" },
                    "labels": [{ "name": "bug" }],
                    "assignee": null,
                    "comments": 0
                },
                "repository": { "full_name": "owner/repo" }
            }))
            .unwrap(),
            headers: {
                let mut h = HashMap::new();
                h.insert("x-github-event".to_string(), "issues".to_string());
                h
            },
            content_type: Some("application/json".to_string()),
        };

        let items = connector.handle_webhook(&config, payload).await.unwrap();
        assert_eq!(items.len(), 1);
        assert!(items[0].content.contains("[owner/repo] Issue #42"));
        assert!(items[0].content.contains("New bug report"));
        assert_eq!(items[0].source.author.as_deref(), Some("reporter"));
    }

    #[tokio::test]
    async fn test_handle_webhook_pr_opened() {
        let connector = GitHubConnector::new();
        let config = make_config("token", vec!["owner/repo"]);
        let payload = WebhookPayload {
            body: serde_json::to_vec(&serde_json::json!({
                "action": "opened",
                "pull_request": {
                    "number": 10,
                    "title": "Add feature X",
                    "body": "Implements feature X",
                    "state": "open",
                    "html_url": "https://github.com/owner/repo/pull/10",
                    "created_at": "2024-06-15T10:00:00Z",
                    "updated_at": "2024-06-15T10:00:00Z",
                    "user": { "login": "developer" },
                    "labels": [],
                    "assignee": null,
                    "pull_request": {},
                    "comments": 0
                },
                "repository": { "full_name": "owner/repo" }
            }))
            .unwrap(),
            headers: {
                let mut h = HashMap::new();
                h.insert("x-github-event".to_string(), "pull_request".to_string());
                h
            },
            content_type: Some("application/json".to_string()),
        };

        let items = connector.handle_webhook(&config, payload).await.unwrap();
        assert_eq!(items.len(), 1);
        assert!(items[0].content.contains("PR #10"));
        assert!(items[0].content.contains("Add feature X"));
    }

    #[tokio::test]
    async fn test_handle_webhook_comment_created() {
        let connector = GitHubConnector::new();
        let config = make_config("token", vec!["owner/repo"]);
        let payload = WebhookPayload {
            body: serde_json::to_vec(&serde_json::json!({
                "action": "created",
                "comment": {
                    "id": 555,
                    "body": "LGTM! Ship it.",
                    "html_url": "https://github.com/owner/repo/issues/42#issuecomment-555",
                    "created_at": "2024-06-15T11:00:00Z",
                    "user": { "login": "reviewer" }
                },
                "issue": {
                    "number": 42,
                    "title": "Fix auth bug",
                    "pull_request": null
                },
                "repository": { "full_name": "owner/repo" }
            }))
            .unwrap(),
            headers: {
                let mut h = HashMap::new();
                h.insert("x-github-event".to_string(), "issue_comment".to_string());
                h
            },
            content_type: Some("application/json".to_string()),
        };

        let items = connector.handle_webhook(&config, payload).await.unwrap();
        assert_eq!(items.len(), 1);
        assert!(items[0].content.contains("Comment on Issue #42"));
        assert!(items[0].content.contains("LGTM! Ship it."));
        assert_eq!(items[0].source.author.as_deref(), Some("reviewer"));
    }

    #[tokio::test]
    async fn test_handle_webhook_unknown_event() {
        let connector = GitHubConnector::new();
        let config = make_config("token", vec!["owner/repo"]);
        let payload = WebhookPayload {
            body: serde_json::to_vec(&serde_json::json!({ "action": "completed" })).unwrap(),
            headers: {
                let mut h = HashMap::new();
                h.insert("x-github-event".to_string(), "check_run".to_string());
                h
            },
            content_type: Some("application/json".to_string()),
        };

        let items = connector.handle_webhook(&config, payload).await.unwrap();
        assert!(items.is_empty());
    }

    #[tokio::test]
    async fn test_handle_webhook_issue_labeled_skipped() {
        let connector = GitHubConnector::new();
        let config = make_config("token", vec!["owner/repo"]);
        let payload = WebhookPayload {
            body: serde_json::to_vec(&serde_json::json!({
                "action": "labeled",
                "issue": {
                    "number": 1,
                    "title": "x",
                    "body": null,
                    "state": "open",
                    "html_url": "https://github.com/o/r/issues/1",
                    "created_at": "2024-01-01T00:00:00Z",
                    "updated_at": "2024-01-01T00:00:00Z",
                    "user": { "login": "u" },
                    "labels": [],
                    "assignee": null,
                    "comments": 0
                },
                "repository": { "full_name": "owner/repo" }
            }))
            .unwrap(),
            headers: {
                let mut h = HashMap::new();
                h.insert("x-github-event".to_string(), "issues".to_string());
                h
            },
            content_type: Some("application/json".to_string()),
        };

        let items = connector.handle_webhook(&config, payload).await.unwrap();
        assert!(items.is_empty());
    }

    #[tokio::test]
    async fn test_handle_webhook_invalid_json() {
        let connector = GitHubConnector::new();
        let config = make_config("token", vec!["owner/repo"]);
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
