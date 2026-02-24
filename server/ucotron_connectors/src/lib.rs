//! Ucotron Connectors — private crate for data source integrations.
//!
//! This crate provides the [`Connector`](connector::Connector) trait and supporting
//! types for integrating external data sources (Slack, GitHub, Notion, etc.)
//! into the Ucotron memory graph.
//!
//! # Modules
//!
//! - [`connector`]: Core trait, content items, and authentication types
//! - [`filters`]: Content filtering for selective sync
//! - [`scheduler`]: Sync scheduling and history tracking

pub mod bitbucket;
pub mod connector;
pub mod discord;
pub mod filters;
pub mod github;
pub mod gitlab;
pub mod google_docs;
pub mod google_drive;
pub mod mongodb;
pub mod notion;
pub mod obsidian;
pub mod postgres;
pub mod scheduler;
pub mod slack;
pub mod spotify;
pub mod telegram;
pub mod youtube;

// Re-export primary types for convenience
pub use bitbucket::BitbucketConnector;
pub use connector::{
    AuthConfig, Connector, ConnectorConfig, ConnectorId, ContentItem, MediaAttachment,
    SourceMetadata, SyncCursor, SyncResult, WebhookPayload,
};
pub use discord::DiscordConnector;
pub use filters::{ContentFilter, SourceFilter};
pub use github::GitHubConnector;
pub use gitlab::GitLabConnector;
pub use google_docs::GoogleDocsConnector;
pub use google_drive::GDriveConnector;
pub use mongodb::MongoConnector;
pub use notion::NotionConnector;
pub use obsidian::ObsidianConnector;
pub use postgres::PostgresConnector;
pub use scheduler::{
    next_fire_time, validate_cron_expression, CronScheduler, CronSchedulerConfig, Scheduler,
    SyncFn, SyncRecord, SyncSchedule, SyncStatus, WebhookFn,
};
pub use slack::SlackConnector;
pub use spotify::SpotifyConnector;
pub use telegram::TelegramConnector;
pub use youtube::YouTubeConnector;
