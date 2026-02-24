//! Structured error types for the Ucotron REST API.

use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use serde::Serialize;

/// Structured API error response.
#[derive(Debug, Serialize)]
pub struct ApiError {
    pub code: String,
    pub message: String,
}

/// Application-level error that converts into an HTTP response.
#[derive(Debug)]
pub struct AppError {
    pub status: StatusCode,
    pub code: String,
    pub message: String,
}

impl AppError {
    pub fn not_found(msg: impl Into<String>) -> Self {
        Self {
            status: StatusCode::NOT_FOUND,
            code: "NOT_FOUND".into(),
            message: msg.into(),
        }
    }

    pub fn bad_request(msg: impl Into<String>) -> Self {
        Self {
            status: StatusCode::BAD_REQUEST,
            code: "BAD_REQUEST".into(),
            message: msg.into(),
        }
    }

    pub fn internal(msg: impl Into<String>) -> Self {
        Self {
            status: StatusCode::INTERNAL_SERVER_ERROR,
            code: "INTERNAL_ERROR".into(),
            message: msg.into(),
        }
    }

    pub fn not_implemented(msg: impl Into<String>) -> Self {
        Self {
            status: StatusCode::NOT_IMPLEMENTED,
            code: "NOT_IMPLEMENTED".into(),
            message: msg.into(),
        }
    }

    pub fn read_only() -> Self {
        Self {
            status: StatusCode::FORBIDDEN,
            code: "READ_ONLY_INSTANCE".into(),
            message: "This instance is configured as read-only (role=reader). Write operations must be sent to a writer instance.".into(),
        }
    }

    pub fn forbidden(msg: impl Into<String>) -> Self {
        Self {
            status: StatusCode::FORBIDDEN,
            code: "FORBIDDEN".into(),
            message: msg.into(),
        }
    }
}

impl IntoResponse for AppError {
    fn into_response(self) -> Response {
        let body = axum::Json(ApiError {
            code: self.code,
            message: self.message,
        });
        (self.status, body).into_response()
    }
}

impl From<anyhow::Error> for AppError {
    fn from(err: anyhow::Error) -> Self {
        AppError::internal(err.to_string())
    }
}
