//! Immutable audit logging system for compliance and debugging.
//!
//! Provides an append-only audit trail recording all API operations.
//! Entries are stored in an in-memory `Vec` protected by `RwLock` and
//! optionally persisted as graph nodes for durability.
//!
//! Design principles:
//! - **Append-only**: entries can never be modified or deleted through the API.
//! - **Timestamped**: each entry records the exact time of the operation.
//! - **Queryable**: supports filtering by time range, action, and user/key.
//! - **Exportable**: full log can be exported as JSON for compliance audits.

use std::sync::{Arc, RwLock};
use std::time::{SystemTime, UNIX_EPOCH};

use axum::extract::State;
use axum::http::Request;
use axum::middleware::Next;
use axum::response::Response;
use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

use crate::auth::AuthContext;
use crate::state::AppState;

/// A single audit log entry.
#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct AuditEntry {
    /// Unix timestamp (seconds) when the operation occurred.
    pub timestamp: u64,
    /// HTTP method (GET, POST, PUT, DELETE).
    pub method: String,
    /// Request path (e.g., "/api/v1/memories").
    pub path: String,
    /// Action category derived from the path (e.g., "memories.create", "search", "auth.keys.create").
    pub action: String,
    /// HTTP status code of the response.
    pub status: u16,
    /// Request duration in microseconds.
    pub duration_us: u64,
    /// API key name used for this request (None if auth disabled).
    pub user: Option<String>,
    /// Role of the caller.
    pub role: String,
    /// Namespace targeted by this request.
    pub namespace: Option<String>,
    /// Target resource ID (if applicable, e.g., memory ID).
    pub resource_id: Option<String>,
}

/// Thread-safe, append-only audit log.
pub struct AuditLog {
    entries: RwLock<Vec<AuditEntry>>,
    max_entries: usize,
    retention_secs: u64,
}

impl AuditLog {
    /// Create a new audit log with the given capacity and retention limits.
    pub fn new(max_entries: usize, retention_secs: u64) -> Self {
        Self {
            entries: RwLock::new(Vec::with_capacity(1024)),
            max_entries,
            retention_secs,
        }
    }

    /// Append an entry to the audit log.
    ///
    /// If the log exceeds `max_entries`, the oldest entries are evicted.
    pub fn append(&self, entry: AuditEntry) {
        let mut entries = self.entries.write().unwrap();
        entries.push(entry);

        // Evict oldest entries if over capacity.
        if self.max_entries > 0 && entries.len() > self.max_entries {
            let excess = entries.len() - self.max_entries;
            entries.drain(..excess);
        }
    }

    /// Query audit entries with optional filters.
    ///
    /// All filters are AND-ed together.
    pub fn query(&self, filter: &AuditFilter) -> Vec<AuditEntry> {
        let entries = self.entries.read().unwrap();
        entries
            .iter()
            .filter(|e| {
                if let Some(from) = filter.from {
                    if e.timestamp < from {
                        return false;
                    }
                }
                if let Some(to) = filter.to {
                    if e.timestamp > to {
                        return false;
                    }
                }
                if let Some(ref action) = filter.action {
                    if !e.action.contains(action.as_str()) {
                        return false;
                    }
                }
                if let Some(ref user) = filter.user {
                    match &e.user {
                        Some(u) => {
                            if u != user {
                                return false;
                            }
                        }
                        None => return false,
                    }
                }
                if let Some(ref method) = filter.method {
                    if e.method != method.to_uppercase() {
                        return false;
                    }
                }
                if let Some(status) = filter.status {
                    if e.status != status {
                        return false;
                    }
                }
                true
            })
            .cloned()
            .collect()
    }

    /// Return all entries (for export).
    pub fn export_all(&self) -> Vec<AuditEntry> {
        let entries = self.entries.read().unwrap();
        entries.clone()
    }

    /// Return the total number of entries.
    pub fn len(&self) -> usize {
        let entries = self.entries.read().unwrap();
        entries.len()
    }

    /// Check if the audit log is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Prune entries older than the retention period.
    /// Returns the number of entries pruned.
    pub fn prune_expired(&self) -> usize {
        if self.retention_secs == 0 {
            return 0; // No retention limit.
        }
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        let cutoff = now.saturating_sub(self.retention_secs);

        let mut entries = self.entries.write().unwrap();
        let before = entries.len();
        entries.retain(|e| e.timestamp >= cutoff);
        before - entries.len()
    }
}

/// Filter criteria for audit log queries.
#[derive(Debug, Default, Deserialize, utoipa::IntoParams)]
pub struct AuditFilter {
    /// Start of time range (Unix timestamp, inclusive).
    pub from: Option<u64>,
    /// End of time range (Unix timestamp, inclusive).
    pub to: Option<u64>,
    /// Filter by action (substring match, e.g., "memories", "search").
    pub action: Option<String>,
    /// Filter by API key name / user.
    pub user: Option<String>,
    /// Filter by HTTP method (GET, POST, PUT, DELETE).
    pub method: Option<String>,
    /// Filter by HTTP status code.
    pub status: Option<u16>,
}

/// Derive an action string from HTTP method and path.
pub fn derive_action(method: &str, path: &str) -> String {
    // Strip "/api/v1/" prefix for cleaner action names.
    let clean = path
        .strip_prefix("/api/v1/")
        .unwrap_or(path)
        .trim_end_matches('/');

    match (method, clean) {
        ("POST", "memories") => "memories.create".into(),
        ("GET", "memories") => "memories.list".into(),
        ("POST", "memories/search") => "search".into(),
        ("GET", p) if p.starts_with("memories/") => "memories.get".into(),
        ("PUT", p) if p.starts_with("memories/") => "memories.update".into(),
        ("DELETE", p) if p.starts_with("memories/") => "memories.delete".into(),
        ("GET", "entities") => "entities.list".into(),
        ("GET", p) if p.starts_with("entities/") => "entities.get".into(),
        ("POST", "augment") => "augment".into(),
        ("POST", "learn") => "learn".into(),
        ("GET", "export") => "export".into(),
        ("POST", "import") => "import".into(),
        ("POST", p) if p.starts_with("import/") => format!("import.{}", &p[7..]),
        ("GET", "health") => "health".into(),
        ("GET", "metrics") => "metrics".into(),
        ("POST", "transcribe") => "transcribe".into(),
        ("POST", "images") => "images.index".into(),
        ("POST", "images/search") => "images.search".into(),
        ("POST", "ocr") => "ocr".into(),
        ("GET", p) if p.starts_with("admin/") => format!("admin.{}", p[6..].replace('/', ".")),
        ("POST", p) if p.starts_with("admin/") => {
            format!("admin.{}.create", p[6..].replace('/', "."))
        }
        ("DELETE", p) if p.starts_with("admin/") => {
            format!("admin.{}.delete", p[6..].replace('/', "."))
        }
        ("DELETE", p) if p.starts_with("gdpr/") => format!("gdpr.{}", p[5..].replace('/', ".")),
        ("GET", p) if p.starts_with("gdpr/") => format!("gdpr.{}", p[5..].replace('/', ".")),
        ("POST", p) if p.starts_with("gdpr/") => format!("gdpr.{}", p[5..].replace('/', ".")),
        ("GET", "auth/whoami") => "auth.whoami".into(),
        ("GET", "auth/keys") => "auth.keys.list".into(),
        ("POST", "auth/keys") => "auth.keys.create".into(),
        ("DELETE", p) if p.starts_with("auth/keys/") => "auth.keys.revoke".into(),
        ("GET", "audit") => "audit.query".into(),
        ("GET", "audit/export") => "audit.export".into(),
        ("GET", "graph") => "graph".into(),
        _ => format!("{}.{}", method.to_lowercase(), clean.replace('/', ".")),
    }
}

/// Extract a resource ID from a path like "/api/v1/memories/42".
pub fn extract_resource_id(path: &str) -> Option<String> {
    let clean = path.strip_prefix("/api/v1/").unwrap_or(path);
    let parts: Vec<&str> = clean.split('/').collect();
    // Patterns: memories/{id}, entities/{id}, admin/namespaces/{name}, auth/keys/{name}
    match parts.as_slice() {
        [_, id] if !id.is_empty() => Some(id.to_string()),
        [_, _, id] if !id.is_empty() => Some(id.to_string()),
        _ => None,
    }
}

/// Middleware that records every request in the immutable audit log.
///
/// Captures method, path, auth context, response status, and duration.
/// Runs after auth middleware (so AuthContext is available) and captures
/// the response status code.
///
/// Namespace is extracted from the `X-Ucotron-Namespace` request header,
/// falling back to the API key's `namespace_scope` if the header is absent.
pub async fn audit_middleware(
    State(state): State<Arc<AppState>>,
    request: Request<axum::body::Body>,
    next: Next,
) -> Response {
    if !state.config.audit.enabled {
        return next.run(request).await;
    }

    let method = request.method().to_string();
    let path = request.uri().path().to_string();

    // Skip noisy health/metrics endpoints from audit log.
    if path == "/api/v1/health" || path == "/metrics" {
        return next.run(request).await;
    }

    // Extract auth context (set by auth middleware that runs before us).
    let auth_ctx = request.extensions().get::<AuthContext>().cloned();

    let user = auth_ctx.as_ref().and_then(|c| c.key_name.clone());
    let role = auth_ctx
        .as_ref()
        .map(|c| c.role.as_str().to_string())
        .unwrap_or_else(|| "unknown".to_string());

    // BUG-5 fix: Extract namespace from X-Ucotron-Namespace header (the actual
    // namespace targeted by the request), falling back to the API key's
    // namespace_scope. Previously only the key scope was used, which is None
    // for most keys, causing audit entries to have namespace=null.
    let header_namespace = request
        .headers()
        .get("X-Ucotron-Namespace")
        .and_then(|v| v.to_str().ok())
        .map(|s| s.to_string());
    let namespace =
        header_namespace.or_else(|| auth_ctx.as_ref().and_then(|c| c.namespace_scope.clone()));

    let action = derive_action(&method, &path);
    let resource_id = extract_resource_id(&path);

    let start = std::time::Instant::now();
    let response = next.run(request).await;
    let duration_us = start.elapsed().as_micros() as u64;

    let status = response.status().as_u16();
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    state.audit_log.append(AuditEntry {
        timestamp,
        method,
        path,
        action,
        status,
        duration_us,
        user,
        role,
        namespace,
        resource_id,
    });

    response
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_audit_log_append_and_query() {
        let log = AuditLog::new(100, 0);
        log.append(AuditEntry {
            timestamp: 1000,
            method: "POST".into(),
            path: "/api/v1/memories".into(),
            action: "memories.create".into(),
            status: 200,
            duration_us: 500,
            user: Some("admin-key".into()),
            role: "admin".into(),
            namespace: Some("default".into()),
            resource_id: None,
        });
        log.append(AuditEntry {
            timestamp: 2000,
            method: "GET".into(),
            path: "/api/v1/memories".into(),
            action: "memories.list".into(),
            status: 200,
            duration_us: 100,
            user: Some("reader-key".into()),
            role: "reader".into(),
            namespace: Some("default".into()),
            resource_id: None,
        });

        assert_eq!(log.len(), 2);

        // Query all
        let all = log.query(&AuditFilter::default());
        assert_eq!(all.len(), 2);

        // Filter by user
        let admin_only = log.query(&AuditFilter {
            user: Some("admin-key".into()),
            ..Default::default()
        });
        assert_eq!(admin_only.len(), 1);
        assert_eq!(admin_only[0].action, "memories.create");

        // Filter by action
        let list_only = log.query(&AuditFilter {
            action: Some("list".into()),
            ..Default::default()
        });
        assert_eq!(list_only.len(), 1);
        assert_eq!(list_only[0].action, "memories.list");

        // Filter by time range
        let time_filter = log.query(&AuditFilter {
            from: Some(1500),
            to: Some(2500),
            ..Default::default()
        });
        assert_eq!(time_filter.len(), 1);
        assert_eq!(time_filter[0].timestamp, 2000);
    }

    #[test]
    fn test_audit_log_max_entries_eviction() {
        let log = AuditLog::new(3, 0);
        for i in 0..5 {
            log.append(AuditEntry {
                timestamp: i as u64,
                method: "GET".into(),
                path: "/api/v1/health".into(),
                action: "health".into(),
                status: 200,
                duration_us: 10,
                user: None,
                role: "viewer".into(),
                namespace: None,
                resource_id: None,
            });
        }
        // Should only keep the last 3
        assert_eq!(log.len(), 3);
        let all = log.export_all();
        assert_eq!(all[0].timestamp, 2);
        assert_eq!(all[1].timestamp, 3);
        assert_eq!(all[2].timestamp, 4);
    }

    #[test]
    fn test_audit_log_prune_expired() {
        let log = AuditLog::new(100, 3600); // 1 hour retention

        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();

        // Old entry (2 hours ago)
        log.append(AuditEntry {
            timestamp: now - 7200,
            method: "GET".into(),
            path: "/api/v1/health".into(),
            action: "health".into(),
            status: 200,
            duration_us: 10,
            user: None,
            role: "viewer".into(),
            namespace: None,
            resource_id: None,
        });
        // Recent entry
        log.append(AuditEntry {
            timestamp: now,
            method: "POST".into(),
            path: "/api/v1/memories".into(),
            action: "memories.create".into(),
            status: 200,
            duration_us: 500,
            user: Some("admin-key".into()),
            role: "admin".into(),
            namespace: Some("default".into()),
            resource_id: None,
        });

        assert_eq!(log.len(), 2);
        let pruned = log.prune_expired();
        assert_eq!(pruned, 1);
        assert_eq!(log.len(), 1);
        let remaining = log.export_all();
        assert_eq!(remaining[0].action, "memories.create");
    }

    #[test]
    fn test_derive_action() {
        assert_eq!(derive_action("POST", "/api/v1/memories"), "memories.create");
        assert_eq!(derive_action("GET", "/api/v1/memories"), "memories.list");
        assert_eq!(derive_action("GET", "/api/v1/memories/42"), "memories.get");
        assert_eq!(
            derive_action("PUT", "/api/v1/memories/42"),
            "memories.update"
        );
        assert_eq!(
            derive_action("DELETE", "/api/v1/memories/42"),
            "memories.delete"
        );
        assert_eq!(derive_action("POST", "/api/v1/memories/search"), "search");
        assert_eq!(derive_action("GET", "/api/v1/entities"), "entities.list");
        assert_eq!(derive_action("POST", "/api/v1/augment"), "augment");
        assert_eq!(derive_action("POST", "/api/v1/learn"), "learn");
        assert_eq!(
            derive_action("POST", "/api/v1/auth/keys"),
            "auth.keys.create"
        );
        assert_eq!(
            derive_action("DELETE", "/api/v1/auth/keys/mykey"),
            "auth.keys.revoke"
        );
        assert_eq!(derive_action("GET", "/api/v1/audit"), "audit.query");
        assert_eq!(derive_action("GET", "/api/v1/audit/export"), "audit.export");
    }

    #[test]
    fn test_extract_resource_id() {
        assert_eq!(
            extract_resource_id("/api/v1/memories/42"),
            Some("42".into())
        );
        assert_eq!(extract_resource_id("/api/v1/entities/7"), Some("7".into()));
        assert_eq!(
            extract_resource_id("/api/v1/auth/keys/mykey"),
            Some("mykey".into())
        );
        assert_eq!(extract_resource_id("/api/v1/memories"), None);
        assert_eq!(extract_resource_id("/api/v1/health"), None);
    }

    #[test]
    fn test_query_by_method_and_status() {
        let log = AuditLog::new(100, 0);
        log.append(AuditEntry {
            timestamp: 1000,
            method: "POST".into(),
            path: "/api/v1/memories".into(),
            action: "memories.create".into(),
            status: 200,
            duration_us: 500,
            user: None,
            role: "admin".into(),
            namespace: None,
            resource_id: None,
        });
        log.append(AuditEntry {
            timestamp: 2000,
            method: "GET".into(),
            path: "/api/v1/memories".into(),
            action: "memories.list".into(),
            status: 403,
            duration_us: 50,
            user: None,
            role: "viewer".into(),
            namespace: None,
            resource_id: None,
        });

        // Filter by method
        let posts = log.query(&AuditFilter {
            method: Some("POST".into()),
            ..Default::default()
        });
        assert_eq!(posts.len(), 1);
        assert_eq!(posts[0].action, "memories.create");

        // Filter by status
        let forbidden = log.query(&AuditFilter {
            status: Some(403),
            ..Default::default()
        });
        assert_eq!(forbidden.len(), 1);
        assert_eq!(forbidden[0].status, 403);
    }

    #[test]
    fn test_prune_disabled_when_zero_retention() {
        let log = AuditLog::new(100, 0); // retention_secs = 0 → keep forever
        log.append(AuditEntry {
            timestamp: 1, // very old
            method: "GET".into(),
            path: "/api/v1/health".into(),
            action: "health".into(),
            status: 200,
            duration_us: 10,
            user: None,
            role: "viewer".into(),
            namespace: None,
            resource_id: None,
        });
        let pruned = log.prune_expired();
        assert_eq!(pruned, 0);
        assert_eq!(log.len(), 1);
    }
}
