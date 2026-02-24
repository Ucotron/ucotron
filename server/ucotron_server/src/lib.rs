//! # Ucotron Server Library
//!
//! Shared types, application state, and route handlers for the Ucotron REST API
//! and MCP (Model Context Protocol) server.
//!
//! Separated from `main.rs` so that handlers can be unit-tested without starting
//! a real TCP listener.

pub mod audit;
pub mod auth;
pub mod error;
pub mod handlers;
pub mod llm;
pub mod mcp;
pub mod metrics;
pub mod openapi;
pub mod state;
pub mod telemetry;
pub mod types;
pub mod writer_lock;
