//! RBAC authorization middleware and helpers.
//!
//! When `auth.enabled` is true in config, every request must include
//! `Authorization: Bearer <api-key>`. The middleware resolves the key to an
//! [`AuthRole`] and optional namespace scope, then stores the result as a
//! request extension for handlers to inspect.

use std::sync::Arc;

use axum::extract::State;
use axum::http::{HeaderMap, Request, StatusCode};
use axum::middleware::Next;
use axum::response::{IntoResponse, Response};

use ucotron_config::AuthRole;

use crate::error::AppError;
use crate::state::AppState;

/// Resolved identity attached to the request as an extension.
#[derive(Debug, Clone)]
pub struct AuthContext {
    /// The role of the authenticated caller.
    pub role: AuthRole,
    /// If the API key is scoped to a specific namespace, this is set.
    /// Handlers should restrict results to this namespace.
    pub namespace_scope: Option<String>,
    /// Name of the API key used (for audit logging).
    pub key_name: Option<String>,
}

impl AuthContext {
    /// Check if this context allows access to the given namespace.
    pub fn can_access_namespace(&self, namespace: &str) -> bool {
        match &self.namespace_scope {
            Some(scope) => scope == namespace,
            None => true, // No scope restriction
        }
    }
}

/// Authorization middleware.
///
/// When auth is disabled, injects a default Admin context.
/// When auth is enabled, validates the Bearer token and injects the resolved context.
/// Health and metrics endpoints are always allowed (no auth required).
pub async fn auth_middleware(
    State(state): State<Arc<AppState>>,
    mut request: Request<axum::body::Body>,
    next: Next,
) -> Response {
    let path = request.uri().path().to_string();

    // Always allow health, metrics, swagger, OpenAPI, and webhook endpoints without auth.
    // Webhooks come from external services (GitHub, Slack, etc.) that cannot include
    // our Bearer token. Webhook validation is handled per-connector (e.g., HMAC signatures).
    // Frame embed endpoint is also allowed without auth for iframe embedding scenarios.
    if path == "/api/v1/health"
        || path == "/metrics"
        || path.starts_with("/swagger-ui")
        || path == "/api/v1/openapi.json"
        || path.starts_with("/api/v1/webhooks/")
        || path.contains("/embed")
    {
        // Inject a Viewer context for unauthenticated health/metrics access.
        request.extensions_mut().insert(AuthContext {
            role: AuthRole::Viewer,
            namespace_scope: None,
            key_name: None,
        });
        return next.run(request).await;
    }

    if !state.config.auth.enabled {
        // Auth disabled — everyone is admin.
        request.extensions_mut().insert(AuthContext {
            role: AuthRole::Admin,
            namespace_scope: None,
            key_name: None,
        });
        return next.run(request).await;
    }

    // Extract Bearer token.
    let headers = request.headers();
    let token = extract_bearer_token(headers);

    match token {
        None => {
            let err = AppError {
                status: StatusCode::UNAUTHORIZED,
                code: "UNAUTHORIZED".into(),
                message: "Missing or invalid Authorization header. Use: Authorization: Bearer <api-key>".into(),
            };
            err.into_response()
        }
        Some(token) => {
            // Authenticate against runtime-mutable api_keys + legacy config key.
            let auth_result = {
                let keys = state.api_keys.read().unwrap();
                authenticate_token(token, &keys, &state.config.auth.api_key)
            };
            match auth_result {
                None => {
                    let err = AppError {
                        status: StatusCode::UNAUTHORIZED,
                        code: "INVALID_API_KEY".into(),
                        message: "Invalid or revoked API key.".into(),
                    };
                    err.into_response()
                }
                Some((role, namespace_scope, key_name)) => {
                    request.extensions_mut().insert(AuthContext {
                        role,
                        namespace_scope,
                        key_name,
                    });
                    next.run(request).await
                }
            }
        }
    }
}

/// Extract the Bearer token from the Authorization header.
fn extract_bearer_token(headers: &HeaderMap) -> Option<&str> {
    headers
        .get("authorization")
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.strip_prefix("Bearer "))
}

/// Authenticate a bearer token against the runtime API keys list + legacy key.
/// Returns (role, namespace_scope, key_name) on success.
fn authenticate_token(
    token: &str,
    api_keys: &[ucotron_config::ApiKeyEntry],
    legacy_key: &Option<String>,
) -> Option<(AuthRole, Option<String>, Option<String>)> {
    // Check named API keys first.
    for entry in api_keys {
        if entry.active && entry.key == token {
            if let Some(role) = AuthRole::parse_role(&entry.role) {
                return Some((role, entry.namespace.clone(), Some(entry.name.clone())));
            }
        }
    }
    // Fall back to legacy single API key (grants admin, no name).
    if let Some(ref lk) = legacy_key {
        if lk == token {
            return Some((AuthRole::Admin, None, None));
        }
    }
    None
}

/// Require the caller to have at least the given role.
/// Returns 403 Forbidden if insufficient.
pub fn require_role(ctx: &AuthContext, required: AuthRole) -> Result<(), AppError> {
    if ctx.role.has_privilege(required) {
        Ok(())
    } else {
        Err(AppError {
            status: StatusCode::FORBIDDEN,
            code: "FORBIDDEN".into(),
            message: format!(
                "Insufficient permissions. Required role: {}, your role: {}.",
                required.as_str(),
                ctx.role.as_str()
            ),
        })
    }
}

/// Require the caller can access the given namespace.
/// Returns 403 if the key is scoped to a different namespace.
pub fn require_namespace_access(ctx: &AuthContext, namespace: &str) -> Result<(), AppError> {
    if ctx.can_access_namespace(namespace) {
        Ok(())
    } else {
        Err(AppError {
            status: StatusCode::FORBIDDEN,
            code: "NAMESPACE_FORBIDDEN".into(),
            message: format!(
                "API key is scoped to namespace '{}', cannot access '{}'.",
                ctx.namespace_scope.as_deref().unwrap_or("?"),
                namespace
            ),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_auth_role_ordering() {
        assert!(AuthRole::Admin.has_privilege(AuthRole::Admin));
        assert!(AuthRole::Admin.has_privilege(AuthRole::Writer));
        assert!(AuthRole::Admin.has_privilege(AuthRole::Reader));
        assert!(AuthRole::Admin.has_privilege(AuthRole::Viewer));
        assert!(AuthRole::Writer.has_privilege(AuthRole::Writer));
        assert!(AuthRole::Writer.has_privilege(AuthRole::Reader));
        assert!(!AuthRole::Reader.has_privilege(AuthRole::Writer));
        assert!(!AuthRole::Viewer.has_privilege(AuthRole::Reader));
    }

    #[test]
    fn test_namespace_scope() {
        let ctx = AuthContext {
            role: AuthRole::Reader,
            namespace_scope: Some("tenant-a".to_string()),
            key_name: None,
        };
        assert!(ctx.can_access_namespace("tenant-a"));
        assert!(!ctx.can_access_namespace("tenant-b"));

        let unscoped = AuthContext {
            role: AuthRole::Admin,
            namespace_scope: None,
            key_name: None,
        };
        assert!(unscoped.can_access_namespace("anything"));
    }
}
