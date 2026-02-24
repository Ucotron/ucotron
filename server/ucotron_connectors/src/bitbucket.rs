//! Bitbucket connector — fetches pull requests, issues, and comments from Bitbucket Cloud repositories.
//!
//! Uses App Passwords to authenticate with the Bitbucket API 2.0.
//! Supports full sync (all PRs/issues from configured repositories) and
//! incremental sync via `updated_on` timestamp-based pagination.

use std::collections::HashMap;

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};

use crate::connector::{
    AuthConfig, Connector, ConnectorConfig, ContentItem, SourceMetadata, SyncCursor, SyncResult,
    WebhookPayload,
};

const BITBUCKET_API_BASE: &str = "https://api.bitbucket.org/2.0";

/// Bitbucket connector for fetching pull requests, issues, and PR comments.
///
/// Requires an App Password with the following permissions:
/// - `repository:read` — read access to repositories
/// - `pullrequest:read` — read access to pull requests
/// - `issue:read` — read access to issues (if issue tracker is enabled)
///
/// Configuration settings:
/// - `repos`: list of "workspace/repo-slug" strings (required)
/// - `username`: Bitbucket username for Basic Auth (required)
/// - `include_prs`: whether to fetch pull requests (default: true)
/// - `include_issues`: whether to fetch issues (default: true)
/// - `include_comments`: whether to fetch PR comments (default: true)
pub struct BitbucketConnector {
    client: reqwest::Client,
}

impl BitbucketConnector {
    pub fn new() -> Self {
        Self {
            client: reqwest::Client::builder()
                .user_agent("ucotron-connector/0.1")
                .build()
                .unwrap_or_else(|_| reqwest::Client::new()),
        }
    }

    pub fn with_client(client: reqwest::Client) -> Self {
        Self { client }
    }

    /// Extracts the app password from the connector config.
    fn get_token(config: &ConnectorConfig) -> Result<&str> {
        match &config.auth {
            AuthConfig::Token { token } => Ok(token.as_str()),
            _ => bail!("Bitbucket connector requires Token authentication (App Password)"),
        }
    }

    /// Extracts the username from settings (required for Basic Auth).
    fn get_username(config: &ConnectorConfig) -> Result<String> {
        config
            .settings
            .get("username")
            .and_then(|v| v.as_str())
            .map(String::from)
            .ok_or_else(|| anyhow::anyhow!("Bitbucket connector requires 'username' in settings"))
    }

    /// Extracts configured repository slugs from settings (e.g., ["workspace/repo-slug"]).
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

    fn include_prs(config: &ConnectorConfig) -> bool {
        config
            .settings
            .get("include_prs")
            .and_then(|v| v.as_bool())
            .unwrap_or(true)
    }

    fn include_issues(config: &ConnectorConfig) -> bool {
        config
            .settings
            .get("include_issues")
            .and_then(|v| v.as_bool())
            .unwrap_or(true)
    }

    fn include_comments(config: &ConnectorConfig) -> bool {
        config
            .settings
            .get("include_comments")
            .and_then(|v| v.as_bool())
            .unwrap_or(true)
    }

    /// Fetches pull requests from a repository with pagination.
    async fn fetch_pull_requests(
        &self,
        username: &str,
        app_password: &str,
        repo_full_name: &str,
        updated_after: Option<&str>,
    ) -> Result<Vec<BitbucketPullRequest>> {
        let mut all_prs = Vec::new();
        let mut next_url: Option<String> = None;

        // Build initial URL with optional q filter for incremental sync
        let initial_url = if let Some(since) = updated_after {
            format!(
                "{}/repositories/{}/pullrequests?pagelen=50&state=OPEN&state=MERGED&state=DECLINED&state=SUPERSEDED&q=updated_on>\"{}\"",
                BITBUCKET_API_BASE, repo_full_name, since
            )
        } else {
            format!(
                "{}/repositories/{}/pullrequests?pagelen=50&state=OPEN&state=MERGED&state=DECLINED&state=SUPERSEDED",
                BITBUCKET_API_BASE, repo_full_name
            )
        };

        let mut current_url = initial_url;

        loop {
            let url = next_url.take().unwrap_or_else(|| current_url.clone());

            let resp = self
                .client
                .get(&url)
                .basic_auth(username, Some(app_password))
                .send()
                .await
                .context("Failed to fetch Bitbucket pull requests")?;

            if !resp.status().is_success() {
                let status = resp.status();
                let body = resp.text().await.unwrap_or_default();
                bail!("Bitbucket API error fetching PRs: {} - {}", status, body);
            }

            let page: BitbucketPage<BitbucketPullRequest> = resp
                .json()
                .await
                .context("Failed to parse Bitbucket PRs response")?;

            all_prs.extend(page.values);

            match page.next {
                Some(next) => {
                    current_url = next;
                }
                None => break,
            }
        }

        Ok(all_prs)
    }

    /// Fetches issues from a repository with pagination.
    async fn fetch_issues(
        &self,
        username: &str,
        app_password: &str,
        repo_full_name: &str,
        updated_after: Option<&str>,
    ) -> Result<Vec<BitbucketIssue>> {
        let mut all_issues = Vec::new();
        let mut next_url: Option<String> = None;

        let initial_url = if let Some(since) = updated_after {
            format!(
                "{}/repositories/{}/issues?pagelen=50&q=updated_on>\"{}\"",
                BITBUCKET_API_BASE, repo_full_name, since
            )
        } else {
            format!(
                "{}/repositories/{}/issues?pagelen=50",
                BITBUCKET_API_BASE, repo_full_name
            )
        };

        let mut current_url = initial_url;

        loop {
            let url = next_url.take().unwrap_or_else(|| current_url.clone());

            let resp = self
                .client
                .get(&url)
                .basic_auth(username, Some(app_password))
                .send()
                .await
                .context("Failed to fetch Bitbucket issues")?;

            if !resp.status().is_success() {
                let status = resp.status();
                let body = resp.text().await.unwrap_or_default();
                // Issue tracker might be disabled for this repo — treat 404 as empty
                if status == reqwest::StatusCode::NOT_FOUND {
                    return Ok(Vec::new());
                }
                bail!("Bitbucket API error fetching issues: {} - {}", status, body);
            }

            let page: BitbucketPage<BitbucketIssue> = resp
                .json()
                .await
                .context("Failed to parse Bitbucket issues response")?;

            all_issues.extend(page.values);

            match page.next {
                Some(next) => {
                    current_url = next;
                }
                None => break,
            }
        }

        Ok(all_issues)
    }

    /// Fetches comments for a pull request with pagination.
    async fn fetch_pr_comments(
        &self,
        username: &str,
        app_password: &str,
        repo_full_name: &str,
        pr_id: u64,
    ) -> Result<Vec<BitbucketComment>> {
        let mut all_comments = Vec::new();
        let mut next_url: Option<String> = None;

        let initial_url = format!(
            "{}/repositories/{}/pullrequests/{}/comments?pagelen=50",
            BITBUCKET_API_BASE, repo_full_name, pr_id
        );

        let mut current_url = initial_url;

        loop {
            let url = next_url.take().unwrap_or_else(|| current_url.clone());

            let resp = self
                .client
                .get(&url)
                .basic_auth(username, Some(app_password))
                .send()
                .await
                .context("Failed to fetch Bitbucket PR comments")?;

            if !resp.status().is_success() {
                let status = resp.status();
                let body = resp.text().await.unwrap_or_default();
                bail!(
                    "Bitbucket API error fetching PR #{} comments: {} - {}",
                    pr_id,
                    status,
                    body
                );
            }

            let page: BitbucketPage<BitbucketComment> = resp
                .json()
                .await
                .context("Failed to parse Bitbucket comments response")?;

            all_comments.extend(page.values);

            match page.next {
                Some(next) => {
                    current_url = next;
                }
                None => break,
            }
        }

        Ok(all_comments)
    }

    /// Converts a Bitbucket pull request to a ContentItem.
    fn pr_to_content_item(
        &self,
        pr: &BitbucketPullRequest,
        repo_full_name: &str,
        connector_id: &str,
    ) -> ContentItem {
        let body_preview = pr
            .description
            .as_deref()
            .unwrap_or("")
            .chars()
            .take(2000)
            .collect::<String>();

        let content = if body_preview.is_empty() {
            format!("[{}] PR #{}: {}", repo_full_name, pr.id, pr.title)
        } else {
            format!(
                "[{}] PR #{}: {}\n\n{}",
                repo_full_name, pr.id, pr.title, body_preview
            )
        };

        let created_at = parse_bitbucket_timestamp(&pr.created_on);

        let mut extra = HashMap::new();
        extra.insert(
            "repo".to_string(),
            serde_json::Value::String(repo_full_name.to_string()),
        );
        extra.insert("pr_id".to_string(), serde_json::Value::Number(pr.id.into()));
        extra.insert(
            "state".to_string(),
            serde_json::Value::String(pr.state.clone()),
        );
        extra.insert(
            "kind".to_string(),
            serde_json::Value::String("PR".to_string()),
        );
        if let Some(ref src) = pr.source {
            if let Some(ref branch) = src.branch {
                extra.insert(
                    "source_branch".to_string(),
                    serde_json::Value::String(branch.name.clone()),
                );
            }
        }
        if let Some(ref dst) = pr.destination {
            if let Some(ref branch) = dst.branch {
                extra.insert(
                    "target_branch".to_string(),
                    serde_json::Value::String(branch.name.clone()),
                );
            }
        }

        let author_name = pr
            .author
            .as_ref()
            .map(|a| a.display_name.clone())
            .unwrap_or_default();

        let source_url = pr
            .links
            .as_ref()
            .and_then(|l| l.html.as_ref())
            .map(|h| h.href.clone());

        ContentItem {
            content,
            source: SourceMetadata {
                connector_type: "bitbucket".to_string(),
                connector_id: connector_id.to_string(),
                source_id: format!("{}!{}", repo_full_name, pr.id),
                source_url,
                author: if author_name.is_empty() {
                    None
                } else {
                    Some(author_name)
                },
                created_at,
                extra,
            },
            media: None,
        }
    }

    /// Converts a Bitbucket issue to a ContentItem.
    fn issue_to_content_item(
        &self,
        issue: &BitbucketIssue,
        repo_full_name: &str,
        connector_id: &str,
    ) -> ContentItem {
        let body_preview = issue
            .content
            .as_ref()
            .and_then(|c| c.raw.as_deref())
            .unwrap_or("")
            .chars()
            .take(2000)
            .collect::<String>();

        let content = if body_preview.is_empty() {
            format!("[{}] Issue #{}: {}", repo_full_name, issue.id, issue.title)
        } else {
            format!(
                "[{}] Issue #{}: {}\n\n{}",
                repo_full_name, issue.id, issue.title, body_preview
            )
        };

        let created_at = parse_bitbucket_timestamp(&issue.created_on);

        let mut extra = HashMap::new();
        extra.insert(
            "repo".to_string(),
            serde_json::Value::String(repo_full_name.to_string()),
        );
        extra.insert(
            "issue_id".to_string(),
            serde_json::Value::Number(issue.id.into()),
        );
        extra.insert(
            "state".to_string(),
            serde_json::Value::String(issue.state.clone()),
        );
        extra.insert(
            "kind".to_string(),
            serde_json::Value::String("Issue".to_string()),
        );
        extra.insert(
            "priority".to_string(),
            serde_json::Value::String(issue.priority.clone()),
        );
        extra.insert(
            "issue_kind".to_string(),
            serde_json::Value::String(issue.kind.clone()),
        );

        let author_name = issue
            .reporter
            .as_ref()
            .map(|r| r.display_name.clone())
            .unwrap_or_default();

        let source_url = issue
            .links
            .as_ref()
            .and_then(|l| l.html.as_ref())
            .map(|h| h.href.clone());

        ContentItem {
            content,
            source: SourceMetadata {
                connector_type: "bitbucket".to_string(),
                connector_id: connector_id.to_string(),
                source_id: format!("{}#{}", repo_full_name, issue.id),
                source_url,
                author: if author_name.is_empty() {
                    None
                } else {
                    Some(author_name)
                },
                created_at,
                extra,
            },
            media: None,
        }
    }

    /// Converts a Bitbucket PR comment to a ContentItem.
    fn comment_to_content_item(
        &self,
        comment: &BitbucketComment,
        pr_id: u64,
        repo_full_name: &str,
        connector_id: &str,
    ) -> ContentItem {
        let body_text = comment
            .content
            .as_ref()
            .and_then(|c| c.raw.as_deref())
            .unwrap_or("")
            .chars()
            .take(2000)
            .collect::<String>();

        let content = format!(
            "[{}] Comment on PR #{}: {}",
            repo_full_name, pr_id, body_text
        );

        let created_at = parse_bitbucket_timestamp(&comment.created_on);

        let mut extra = HashMap::new();
        extra.insert(
            "repo".to_string(),
            serde_json::Value::String(repo_full_name.to_string()),
        );
        extra.insert("pr_id".to_string(), serde_json::Value::Number(pr_id.into()));
        extra.insert(
            "comment_id".to_string(),
            serde_json::Value::Number(comment.id.into()),
        );
        extra.insert(
            "kind".to_string(),
            serde_json::Value::String("Comment".to_string()),
        );

        let author_name = comment
            .user
            .as_ref()
            .map(|u| u.display_name.clone())
            .unwrap_or_default();

        let source_url = comment
            .links
            .as_ref()
            .and_then(|l| l.html.as_ref())
            .map(|h| h.href.clone());

        ContentItem {
            content,
            source: SourceMetadata {
                connector_type: "bitbucket".to_string(),
                connector_id: connector_id.to_string(),
                source_id: format!("{}!{}~{}", repo_full_name, pr_id, comment.id),
                source_url,
                author: if author_name.is_empty() {
                    None
                } else {
                    Some(author_name)
                },
                created_at,
                extra,
            },
            media: None,
        }
    }
}

impl Default for BitbucketConnector {
    fn default() -> Self {
        Self::new()
    }
}

impl Connector for BitbucketConnector {
    fn id(&self) -> &str {
        "bitbucket"
    }

    fn name(&self) -> &str {
        "Bitbucket"
    }

    fn config_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "auth": {
                    "type": "object",
                    "description": "App Password credentials",
                    "properties": {
                        "token": { "type": "string", "description": "Bitbucket App Password" }
                    },
                    "required": ["token"]
                },
                "settings": {
                    "type": "object",
                    "properties": {
                        "username": {
                            "type": "string",
                            "description": "Bitbucket username for Basic Auth"
                        },
                        "repos": {
                            "type": "array",
                            "items": { "type": "string" },
                            "description": "Repositories in 'workspace/repo-slug' format"
                        },
                        "include_prs": {
                            "type": "boolean",
                            "description": "Whether to fetch pull requests (default: true)"
                        },
                        "include_issues": {
                            "type": "boolean",
                            "description": "Whether to fetch issues (default: true)"
                        },
                        "include_comments": {
                            "type": "boolean",
                            "description": "Whether to fetch PR comments (default: true)"
                        }
                    },
                    "required": ["username", "repos"]
                }
            },
            "required": ["auth", "settings"]
        })
    }

    fn validate_config(&self, config: &ConnectorConfig) -> Result<()> {
        if config.connector_type != "bitbucket" {
            bail!(
                "Invalid connector type '{}', expected 'bitbucket'",
                config.connector_type
            );
        }
        Self::get_token(config)?;
        Self::get_username(config)?;
        let repos = Self::get_repos(config);
        if repos.is_empty() {
            bail!("Bitbucket connector requires at least one repository in settings.repos");
        }
        for repo in &repos {
            let parts: Vec<&str> = repo.split('/').collect();
            if parts.len() != 2 || parts[0].is_empty() || parts[1].is_empty() {
                bail!(
                    "Invalid repository format '{}', expected 'workspace/repo-slug'",
                    repo
                );
            }
        }
        Ok(())
    }

    async fn fetch(&self, config: &ConnectorConfig) -> Result<Vec<ContentItem>> {
        let app_password = Self::get_token(config)?;
        let username = Self::get_username(config)?;
        let repos = Self::get_repos(config);
        let fetch_prs = Self::include_prs(config);
        let fetch_issues = Self::include_issues(config);
        let fetch_comments = Self::include_comments(config);

        let mut items = Vec::new();

        for repo_full_name in &repos {
            if fetch_prs {
                let prs = self
                    .fetch_pull_requests(&username, app_password, repo_full_name, None)
                    .await?;

                for pr in &prs {
                    items.push(self.pr_to_content_item(pr, repo_full_name, &config.id));

                    if fetch_comments {
                        let comments = self
                            .fetch_pr_comments(&username, app_password, repo_full_name, pr.id)
                            .await?;
                        for comment in &comments {
                            items.push(self.comment_to_content_item(
                                comment,
                                pr.id,
                                repo_full_name,
                                &config.id,
                            ));
                        }
                    }
                }
            }

            if fetch_issues {
                let issues = self
                    .fetch_issues(&username, app_password, repo_full_name, None)
                    .await?;
                for issue in &issues {
                    items.push(self.issue_to_content_item(issue, repo_full_name, &config.id));
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
        let app_password = Self::get_token(config)?;
        let username = Self::get_username(config)?;
        let repos = Self::get_repos(config);
        let fetch_prs = Self::include_prs(config);
        let fetch_issues = Self::include_issues(config);
        let fetch_comments = Self::include_comments(config);

        let updated_after = cursor.value.as_deref();

        let mut items = Vec::new();
        let mut latest_updated: Option<String> = cursor.value.clone();

        for repo_full_name in &repos {
            if fetch_prs {
                let prs = self
                    .fetch_pull_requests(&username, app_password, repo_full_name, updated_after)
                    .await?;

                for pr in &prs {
                    if latest_updated
                        .as_ref()
                        .is_none_or(|current| &pr.updated_on > current)
                    {
                        latest_updated = Some(pr.updated_on.clone());
                    }
                    items.push(self.pr_to_content_item(pr, repo_full_name, &config.id));

                    if fetch_comments {
                        let comments = self
                            .fetch_pr_comments(&username, app_password, repo_full_name, pr.id)
                            .await?;
                        for comment in &comments {
                            items.push(self.comment_to_content_item(
                                comment,
                                pr.id,
                                repo_full_name,
                                &config.id,
                            ));
                        }
                    }
                }
            }

            if fetch_issues {
                let issues = self
                    .fetch_issues(&username, app_password, repo_full_name, updated_after)
                    .await?;
                for issue in &issues {
                    if latest_updated
                        .as_ref()
                        .is_none_or(|current| &issue.updated_on > current)
                    {
                        latest_updated = Some(issue.updated_on.clone());
                    }
                    items.push(self.issue_to_content_item(issue, repo_full_name, &config.id));
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

        // Bitbucket webhooks include event type in X-Event-Key header
        let event_key = payload
            .headers
            .get("x-event-key")
            .or_else(|| payload.headers.get("X-Event-Key"))
            .map(|s| s.as_str())
            .unwrap_or("unknown");

        let repo_full_name = body
            .get("repository")
            .and_then(|r| r.get("full_name"))
            .and_then(|v| v.as_str())
            .unwrap_or("unknown/unknown");

        match event_key {
            "pullrequest:created"
            | "pullrequest:updated"
            | "pullrequest:fulfilled"
            | "pullrequest:rejected" => {
                let pr_data = body
                    .get("pullrequest")
                    .context("Missing pullrequest in webhook")?;

                let pr: BitbucketPullRequest = serde_json::from_value(pr_data.clone())
                    .context("Failed to parse PR from webhook")?;

                Ok(vec![self.pr_to_content_item(
                    &pr,
                    repo_full_name,
                    &config.id,
                )])
            }
            "pullrequest:comment_created" | "pullrequest:comment_updated" => {
                let comment_data = body.get("comment").context("Missing comment in webhook")?;
                let pr_data = body
                    .get("pullrequest")
                    .context("Missing pullrequest in comment webhook")?;

                let comment: BitbucketComment = serde_json::from_value(comment_data.clone())
                    .context("Failed to parse comment from webhook")?;
                let pr_id = pr_data.get("id").and_then(|v| v.as_u64()).unwrap_or(0);

                Ok(vec![self.comment_to_content_item(
                    &comment,
                    pr_id,
                    repo_full_name,
                    &config.id,
                )])
            }
            "issue:created" | "issue:updated" => {
                let issue_data = body.get("issue").context("Missing issue in webhook")?;

                let issue: BitbucketIssue = serde_json::from_value(issue_data.clone())
                    .context("Failed to parse issue from webhook")?;

                Ok(vec![self.issue_to_content_item(
                    &issue,
                    repo_full_name,
                    &config.id,
                )])
            }
            _ => Ok(Vec::new()),
        }
    }
}

/// Parses a Bitbucket ISO 8601 timestamp (e.g., "2024-01-15T10:30:00.000000+00:00") to Unix seconds.
fn parse_bitbucket_timestamp(ts: &str) -> Option<u64> {
    // Bitbucket timestamps use format: "YYYY-MM-DDTHH:MM:SS.ffffff+00:00"
    // Strip timezone suffix and fractional seconds
    let ts = ts.split('+').next().unwrap_or(ts);
    let ts = ts.trim_end_matches('Z');
    let parts: Vec<&str> = ts.split('T').collect();
    if parts.len() != 2 {
        return None;
    }

    let date_parts: Vec<u64> = parts[0].split('-').filter_map(|s| s.parse().ok()).collect();
    let time_str = parts[1].split('.').next().unwrap_or(parts[1]);
    let time_parts: Vec<u64> = time_str.split(':').filter_map(|s| s.parse().ok()).collect();

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

// --- Bitbucket API response types ---

/// Generic paginated response from Bitbucket API 2.0.
#[derive(Debug, Clone, Deserialize, Serialize)]
struct BitbucketPage<T> {
    values: Vec<T>,
    next: Option<String>,
    #[allow(dead_code)]
    #[serde(default)]
    size: Option<u64>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
struct BitbucketUser {
    display_name: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
struct BitbucketBranch {
    name: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
struct BitbucketRef {
    branch: Option<BitbucketBranch>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
struct BitbucketLink {
    href: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
struct BitbucketLinks {
    html: Option<BitbucketLink>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
struct BitbucketRendered {
    raw: Option<String>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
struct BitbucketPullRequest {
    id: u64,
    title: String,
    description: Option<String>,
    state: String,
    created_on: String,
    updated_on: String,
    author: Option<BitbucketUser>,
    source: Option<BitbucketRef>,
    destination: Option<BitbucketRef>,
    links: Option<BitbucketLinks>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
struct BitbucketIssue {
    id: u64,
    title: String,
    content: Option<BitbucketRendered>,
    state: String,
    priority: String,
    kind: String,
    created_on: String,
    updated_on: String,
    reporter: Option<BitbucketUser>,
    links: Option<BitbucketLinks>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
struct BitbucketComment {
    id: u64,
    content: Option<BitbucketRendered>,
    created_on: String,
    user: Option<BitbucketUser>,
    links: Option<BitbucketLinks>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn make_config(token: &str, repos: Vec<&str>) -> ConnectorConfig {
        let mut settings = HashMap::new();
        settings.insert("repos".to_string(), serde_json::json!(repos));
        settings.insert("username".to_string(), serde_json::json!("testuser"));
        ConnectorConfig {
            id: "bb-test".to_string(),
            name: "Test Bitbucket".to_string(),
            connector_type: "bitbucket".to_string(),
            auth: AuthConfig::Token {
                token: token.to_string(),
            },
            namespace: "test".to_string(),
            settings,
            enabled: true,
        }
    }

    fn make_pr(id: u64) -> BitbucketPullRequest {
        BitbucketPullRequest {
            id,
            title: format!("Fix bug #{}", id),
            description: Some("This PR fixes the bug described in issue.".to_string()),
            state: "OPEN".to_string(),
            created_on: "2024-06-15T10:30:00.000000+00:00".to_string(),
            updated_on: "2024-06-16T12:00:00.000000+00:00".to_string(),
            author: Some(BitbucketUser {
                display_name: "Alice Dev".to_string(),
            }),
            source: Some(BitbucketRef {
                branch: Some(BitbucketBranch {
                    name: "fix/bug-42".to_string(),
                }),
            }),
            destination: Some(BitbucketRef {
                branch: Some(BitbucketBranch {
                    name: "main".to_string(),
                }),
            }),
            links: Some(BitbucketLinks {
                html: Some(BitbucketLink {
                    href: format!("https://bitbucket.org/workspace/repo/pull-requests/{}", id),
                }),
            }),
        }
    }

    fn make_issue(id: u64) -> BitbucketIssue {
        BitbucketIssue {
            id,
            title: format!("Bug report #{}", id),
            content: Some(BitbucketRendered {
                raw: Some("Steps to reproduce the problem.".to_string()),
            }),
            state: "open".to_string(),
            priority: "major".to_string(),
            kind: "bug".to_string(),
            created_on: "2024-06-15T10:30:00.000000+00:00".to_string(),
            updated_on: "2024-06-16T12:00:00.000000+00:00".to_string(),
            reporter: Some(BitbucketUser {
                display_name: "Bob Reporter".to_string(),
            }),
            links: Some(BitbucketLinks {
                html: Some(BitbucketLink {
                    href: format!("https://bitbucket.org/workspace/repo/issues/{}", id),
                }),
            }),
        }
    }

    fn make_comment(id: u64) -> BitbucketComment {
        BitbucketComment {
            id,
            content: Some(BitbucketRendered {
                raw: Some("Looks good to me!".to_string()),
            }),
            created_on: "2024-06-16T14:00:00.000000+00:00".to_string(),
            user: Some(BitbucketUser {
                display_name: "Charlie Reviewer".to_string(),
            }),
            links: Some(BitbucketLinks {
                html: Some(BitbucketLink {
                    href: format!(
                        "https://bitbucket.org/workspace/repo/pull-requests/1#comment-{}",
                        id
                    ),
                }),
            }),
        }
    }

    #[test]
    fn test_connector_id_and_name() {
        let connector = BitbucketConnector::new();
        assert_eq!(connector.id(), "bitbucket");
        assert_eq!(connector.name(), "Bitbucket");
    }

    #[test]
    fn test_config_schema() {
        let connector = BitbucketConnector::new();
        let schema = connector.config_schema();
        assert_eq!(schema["type"], "object");
        assert!(schema["properties"]["auth"].is_object());
        assert!(schema["properties"]["settings"]["properties"]["username"].is_object());
        assert!(schema["properties"]["settings"]["properties"]["repos"].is_object());
        assert!(schema["properties"]["settings"]["properties"]["include_prs"].is_object());
        assert!(schema["properties"]["settings"]["properties"]["include_issues"].is_object());
        assert!(schema["properties"]["settings"]["properties"]["include_comments"].is_object());
    }

    #[test]
    fn test_validate_config_valid() {
        let connector = BitbucketConnector::new();
        let config = make_config("app-pass-123", vec!["workspace/repo"]);
        assert!(connector.validate_config(&config).is_ok());
    }

    #[test]
    fn test_validate_config_wrong_type() {
        let connector = BitbucketConnector::new();
        let mut config = make_config("app-pass-123", vec!["workspace/repo"]);
        config.connector_type = "github".to_string();
        assert!(connector.validate_config(&config).is_err());
    }

    #[test]
    fn test_validate_config_wrong_auth() {
        let connector = BitbucketConnector::new();
        let config = ConnectorConfig {
            id: "bb-test".to_string(),
            name: "Test".to_string(),
            connector_type: "bitbucket".to_string(),
            auth: AuthConfig::ApiKey {
                key: "key".to_string(),
            },
            namespace: "test".to_string(),
            settings: {
                let mut s = HashMap::new();
                s.insert("repos".to_string(), serde_json::json!(["ws/repo"]));
                s.insert("username".to_string(), serde_json::json!("user"));
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
        let connector = BitbucketConnector::new();
        let config = make_config("app-pass-123", vec![]);
        let err = connector.validate_config(&config);
        assert!(err.is_err());
        assert!(err.unwrap_err().to_string().contains("repository"));
    }

    #[test]
    fn test_validate_config_invalid_repo_format() {
        let connector = BitbucketConnector::new();
        let config = make_config("app-pass-123", vec!["just-a-name"]);
        let err = connector.validate_config(&config);
        assert!(err.is_err());
        assert!(err.unwrap_err().to_string().contains("workspace/repo-slug"));
    }

    #[test]
    fn test_validate_config_no_username() {
        let connector = BitbucketConnector::new();
        let mut config = make_config("app-pass-123", vec!["ws/repo"]);
        config.settings.remove("username");
        let err = connector.validate_config(&config);
        assert!(err.is_err());
        assert!(err.unwrap_err().to_string().contains("username"));
    }

    #[test]
    fn test_pr_to_content_item() {
        let connector = BitbucketConnector::new();
        let pr = make_pr(42);

        let item = connector.pr_to_content_item(&pr, "workspace/repo", "conn-1");

        assert!(item.content.contains("[workspace/repo] PR #42"));
        assert!(item.content.contains("Fix bug #42"));
        assert!(item.content.contains("fixes the bug"));
        assert_eq!(item.source.connector_type, "bitbucket");
        assert_eq!(item.source.connector_id, "conn-1");
        assert_eq!(item.source.source_id, "workspace/repo!42");
        assert_eq!(item.source.author.as_deref(), Some("Alice Dev"));
        assert!(item.source.source_url.is_some());
        assert_eq!(
            item.source.extra.get("kind").unwrap(),
            &serde_json::Value::String("PR".to_string())
        );
        assert_eq!(
            item.source.extra.get("state").unwrap(),
            &serde_json::Value::String("OPEN".to_string())
        );
        assert_eq!(
            item.source.extra.get("source_branch").unwrap(),
            &serde_json::Value::String("fix/bug-42".to_string())
        );
        assert_eq!(
            item.source.extra.get("target_branch").unwrap(),
            &serde_json::Value::String("main".to_string())
        );
        assert!(item.media.is_none());
    }

    #[test]
    fn test_pr_to_content_item_no_description() {
        let connector = BitbucketConnector::new();
        let mut pr = make_pr(1);
        pr.description = None;

        let item = connector.pr_to_content_item(&pr, "workspace/repo", "conn-1");

        assert!(!item.content.contains("\n\n"));
        assert!(item.content.contains("[workspace/repo] PR #1"));
    }

    #[test]
    fn test_issue_to_content_item() {
        let connector = BitbucketConnector::new();
        let issue = make_issue(10);

        let item = connector.issue_to_content_item(&issue, "workspace/repo", "conn-1");

        assert!(item.content.contains("[workspace/repo] Issue #10"));
        assert!(item.content.contains("Bug report #10"));
        assert!(item.content.contains("Steps to reproduce"));
        assert_eq!(item.source.connector_type, "bitbucket");
        assert_eq!(item.source.source_id, "workspace/repo#10");
        assert_eq!(item.source.author.as_deref(), Some("Bob Reporter"));
        assert_eq!(
            item.source.extra.get("kind").unwrap(),
            &serde_json::Value::String("Issue".to_string())
        );
        assert_eq!(
            item.source.extra.get("priority").unwrap(),
            &serde_json::Value::String("major".to_string())
        );
        assert_eq!(
            item.source.extra.get("issue_kind").unwrap(),
            &serde_json::Value::String("bug".to_string())
        );
    }

    #[test]
    fn test_issue_to_content_item_no_content() {
        let connector = BitbucketConnector::new();
        let mut issue = make_issue(1);
        issue.content = None;

        let item = connector.issue_to_content_item(&issue, "workspace/repo", "conn-1");

        assert!(!item.content.contains("\n\n"));
        assert!(item.content.contains("[workspace/repo] Issue #1"));
    }

    #[test]
    fn test_comment_to_content_item() {
        let connector = BitbucketConnector::new();
        let comment = make_comment(55);

        let item = connector.comment_to_content_item(&comment, 42, "workspace/repo", "conn-1");

        assert!(item.content.contains("[workspace/repo] Comment on PR #42"));
        assert!(item.content.contains("Looks good to me!"));
        assert_eq!(item.source.connector_type, "bitbucket");
        assert_eq!(item.source.source_id, "workspace/repo!42~55");
        assert_eq!(item.source.author.as_deref(), Some("Charlie Reviewer"));
        assert_eq!(
            item.source.extra.get("kind").unwrap(),
            &serde_json::Value::String("Comment".to_string())
        );
        assert_eq!(
            item.source.extra.get("pr_id").unwrap(),
            &serde_json::json!(42)
        );
    }

    #[test]
    fn test_parse_bitbucket_timestamp() {
        let ts = parse_bitbucket_timestamp("2024-01-01T00:00:00.000000+00:00");
        assert!(ts.is_some());
        assert_eq!(ts.unwrap(), 1704067200);
    }

    #[test]
    fn test_parse_bitbucket_timestamp_z_suffix() {
        let ts = parse_bitbucket_timestamp("2024-01-01T00:00:00Z");
        assert!(ts.is_some());
        assert_eq!(ts.unwrap(), 1704067200);
    }

    #[test]
    fn test_parse_bitbucket_timestamp_with_time() {
        let ts = parse_bitbucket_timestamp("2024-06-15T10:30:00.123456+00:00");
        assert!(ts.is_some());
        assert!(ts.unwrap() > 1704067200);
    }

    #[test]
    fn test_parse_bitbucket_timestamp_invalid() {
        assert!(parse_bitbucket_timestamp("not-a-date").is_none());
        assert!(parse_bitbucket_timestamp("").is_none());
    }

    #[test]
    fn test_get_repos_from_settings() {
        let config = make_config("token", vec!["ws/repo1", "org/repo2"]);
        let repos = BitbucketConnector::get_repos(&config);
        assert_eq!(repos, vec!["ws/repo1", "org/repo2"]);
    }

    #[test]
    fn test_get_repos_empty_settings() {
        let mut config = make_config("token", vec![]);
        config.settings.clear();
        let repos = BitbucketConnector::get_repos(&config);
        assert!(repos.is_empty());
    }

    #[test]
    fn test_include_flags_defaults() {
        let config = make_config("token", vec!["ws/repo"]);
        assert!(BitbucketConnector::include_prs(&config));
        assert!(BitbucketConnector::include_issues(&config));
        assert!(BitbucketConnector::include_comments(&config));
    }

    #[test]
    fn test_include_flags_custom() {
        let mut config = make_config("token", vec!["ws/repo"]);
        config
            .settings
            .insert("include_prs".to_string(), serde_json::json!(false));
        config
            .settings
            .insert("include_issues".to_string(), serde_json::json!(false));
        config
            .settings
            .insert("include_comments".to_string(), serde_json::json!(false));

        assert!(!BitbucketConnector::include_prs(&config));
        assert!(!BitbucketConnector::include_issues(&config));
        assert!(!BitbucketConnector::include_comments(&config));
    }

    #[test]
    fn test_default_constructor() {
        let connector = BitbucketConnector::default();
        assert_eq!(connector.id(), "bitbucket");
    }

    #[test]
    fn test_validate_config_multiple_repos() {
        let connector = BitbucketConnector::new();
        let config = make_config("token", vec!["ws/r1", "ws/r2", "org/r3"]);
        assert!(connector.validate_config(&config).is_ok());
    }

    #[tokio::test]
    async fn test_handle_webhook_pr_created() {
        let connector = BitbucketConnector::new();
        let config = make_config("token", vec!["workspace/repo"]);
        let mut headers = HashMap::new();
        headers.insert("x-event-key".to_string(), "pullrequest:created".to_string());
        let payload = WebhookPayload {
            body: serde_json::to_vec(&serde_json::json!({
                "repository": { "full_name": "workspace/repo" },
                "pullrequest": {
                    "id": 42,
                    "title": "New feature",
                    "description": "Adds feature X",
                    "state": "OPEN",
                    "created_on": "2024-06-15T10:00:00.000000+00:00",
                    "updated_on": "2024-06-15T10:00:00.000000+00:00",
                    "author": { "display_name": "Alice" },
                    "source": { "branch": { "name": "feature-x" } },
                    "destination": { "branch": { "name": "main" } },
                    "links": { "html": { "href": "https://bitbucket.org/workspace/repo/pull-requests/42" } }
                }
            }))
            .unwrap(),
            headers,
            content_type: Some("application/json".to_string()),
        };

        let items = connector.handle_webhook(&config, payload).await.unwrap();
        assert_eq!(items.len(), 1);
        assert!(items[0].content.contains("[workspace/repo] PR #42"));
        assert!(items[0].content.contains("New feature"));
        assert_eq!(items[0].source.author.as_deref(), Some("Alice"));
    }

    #[tokio::test]
    async fn test_handle_webhook_issue_created() {
        let connector = BitbucketConnector::new();
        let config = make_config("token", vec!["workspace/repo"]);
        let mut headers = HashMap::new();
        headers.insert("x-event-key".to_string(), "issue:created".to_string());
        let payload = WebhookPayload {
            body: serde_json::to_vec(&serde_json::json!({
                "repository": { "full_name": "workspace/repo" },
                "issue": {
                    "id": 5,
                    "title": "Bug found",
                    "content": { "raw": "Something is broken" },
                    "state": "open",
                    "priority": "critical",
                    "kind": "bug",
                    "created_on": "2024-06-15T10:00:00.000000+00:00",
                    "updated_on": "2024-06-15T10:00:00.000000+00:00",
                    "reporter": { "display_name": "Bob" },
                    "links": { "html": { "href": "https://bitbucket.org/workspace/repo/issues/5" } }
                }
            }))
            .unwrap(),
            headers,
            content_type: Some("application/json".to_string()),
        };

        let items = connector.handle_webhook(&config, payload).await.unwrap();
        assert_eq!(items.len(), 1);
        assert!(items[0].content.contains("[workspace/repo] Issue #5"));
        assert!(items[0].content.contains("Bug found"));
    }

    #[tokio::test]
    async fn test_handle_webhook_unknown_event() {
        let connector = BitbucketConnector::new();
        let config = make_config("token", vec!["workspace/repo"]);
        let mut headers = HashMap::new();
        headers.insert("x-event-key".to_string(), "repo:push".to_string());
        let payload = WebhookPayload {
            body: serde_json::to_vec(&serde_json::json!({
                "repository": { "full_name": "workspace/repo" }
            }))
            .unwrap(),
            headers,
            content_type: Some("application/json".to_string()),
        };

        let items = connector.handle_webhook(&config, payload).await.unwrap();
        assert!(items.is_empty());
    }

    #[tokio::test]
    async fn test_handle_webhook_invalid_json() {
        let connector = BitbucketConnector::new();
        let config = make_config("token", vec!["workspace/repo"]);
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

    #[test]
    fn test_get_username() {
        let config = make_config("token", vec!["ws/repo"]);
        let username = BitbucketConnector::get_username(&config).unwrap();
        assert_eq!(username, "testuser");
    }

    #[test]
    fn test_get_username_missing() {
        let mut config = make_config("token", vec!["ws/repo"]);
        config.settings.remove("username");
        assert!(BitbucketConnector::get_username(&config).is_err());
    }
}
