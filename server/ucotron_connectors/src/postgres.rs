//! Postgres/Neon connector — fetches rows from PostgreSQL tables for ingestion.
//!
//! Uses `sqlx` with a connection string (e.g., Neon serverless Postgres or standard Postgres).
//! Supports full sync (all rows from configured tables) and incremental sync
//! via a timestamp or serial column cursor.
//!
//! # Settings
//!
//! - `tables` (required): array of table names to fetch from
//! - `content_columns` (optional): map of table → array of column names to concatenate as content
//!   (default: all text/varchar columns)
//! - `id_column` (optional): column name used as source_id (default: first primary key or "id")
//! - `cursor_column` (optional): column name for incremental sync ordering (e.g., "updated_at", "id")
//! - `limit` (optional): maximum rows per table per fetch (default: 10000)
//! - `schema` (optional): schema name (default: "public")
//! - `extra_columns` (optional): array of column names to include as extra metadata

use std::collections::HashMap;

use anyhow::{bail, Context, Result};
use sqlx::postgres::{PgPool, PgPoolOptions, PgRow};
use sqlx::Row;

use crate::connector::{
    AuthConfig, Connector, ConnectorConfig, ContentItem, SourceMetadata, SyncCursor, SyncResult,
    WebhookPayload,
};

/// Postgres/Neon connector for fetching database rows as content items.
///
/// Connects to a PostgreSQL database (including Neon serverless) using a
/// connection string provided via `AuthConfig::ConnectionString`. Each
/// configured table is queried, and rows are mapped to [`ContentItem`]s
/// with text columns concatenated as content and metadata columns preserved.
///
/// # Authentication
///
/// Uses `AuthConfig::ConnectionString { uri }` where `uri` is a standard
/// PostgreSQL connection string like:
/// ```text
/// postgres://user:password@host:5432/dbname?sslmode=require
/// ```
///
/// # Example Configuration
///
/// ```json
/// {
///     "id": "pg-main",
///     "name": "Main Database",
///     "connector_type": "postgres",
///     "auth": { "type": "ConnectionString", "uri": "postgres://..." },
///     "namespace": "prod",
///     "settings": {
///         "tables": ["articles", "comments"],
///         "content_columns": { "articles": ["title", "body"], "comments": ["text"] },
///         "id_column": "id",
///         "cursor_column": "updated_at",
///         "limit": 5000,
///         "schema": "public"
///     }
/// }
/// ```
pub struct PostgresConnector;

impl PostgresConnector {
    /// Creates a new PostgresConnector.
    pub fn new() -> Self {
        Self
    }

    /// Extracts the connection URI from the connector config.
    fn get_connection_uri(config: &ConnectorConfig) -> Result<&str> {
        match &config.auth {
            AuthConfig::ConnectionString { uri } => Ok(uri.as_str()),
            _ => bail!(
                "Postgres connector requires ConnectionString authentication \
                 (e.g., postgres://user:pass@host:5432/db)"
            ),
        }
    }

    /// Returns the list of tables to fetch from settings.
    fn get_tables(config: &ConnectorConfig) -> Result<Vec<String>> {
        let tables = config
            .settings
            .get("tables")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str().map(String::from))
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();

        if tables.is_empty() {
            bail!("Postgres connector requires at least one table in 'tables' setting");
        }
        Ok(tables)
    }

    /// Returns content columns for a specific table (or None to use all text columns).
    fn get_content_columns(config: &ConnectorConfig, table: &str) -> Option<Vec<String>> {
        config
            .settings
            .get("content_columns")
            .and_then(|v| v.as_object())
            .and_then(|map| map.get(table))
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str().map(String::from))
                    .collect()
            })
    }

    /// Returns the ID column name (default: "id").
    fn get_id_column(config: &ConnectorConfig) -> String {
        config
            .settings
            .get("id_column")
            .and_then(|v| v.as_str())
            .unwrap_or("id")
            .to_string()
    }

    /// Returns the cursor column name for incremental sync (if configured).
    fn get_cursor_column(config: &ConnectorConfig) -> Option<String> {
        config
            .settings
            .get("cursor_column")
            .and_then(|v| v.as_str())
            .map(String::from)
    }

    /// Returns the maximum rows per table (default: 10000).
    fn get_limit(config: &ConnectorConfig) -> i64 {
        config
            .settings
            .get("limit")
            .and_then(|v| v.as_i64())
            .unwrap_or(10000)
    }

    /// Returns the schema name (default: "public").
    fn get_schema(config: &ConnectorConfig) -> String {
        config
            .settings
            .get("schema")
            .and_then(|v| v.as_str())
            .unwrap_or("public")
            .to_string()
    }

    /// Returns extra columns to include as metadata.
    fn get_extra_columns(config: &ConnectorConfig) -> Vec<String> {
        config
            .settings
            .get("extra_columns")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str().map(String::from))
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Creates a connection pool from the config.
    async fn create_pool(config: &ConnectorConfig) -> Result<PgPool> {
        let uri = Self::get_connection_uri(config)?;
        let pool = PgPoolOptions::new()
            .max_connections(5)
            .connect(uri)
            .await
            .context("Failed to connect to PostgreSQL database")?;
        Ok(pool)
    }

    /// Validates that a table/column name is safe for inclusion in SQL.
    /// Only allows alphanumeric chars, underscores, and dots (for schema.table).
    fn validate_identifier(name: &str) -> Result<()> {
        if name.is_empty() {
            bail!("Empty identifier");
        }
        if !name
            .chars()
            .all(|c| c.is_alphanumeric() || c == '_' || c == '.')
        {
            bail!(
                "Invalid identifier '{}': only alphanumeric, underscore, and dot allowed",
                name
            );
        }
        Ok(())
    }

    /// Discovers text-like columns for a table from information_schema.
    async fn discover_text_columns(
        pool: &PgPool,
        schema: &str,
        table: &str,
    ) -> Result<Vec<String>> {
        let rows = sqlx::query(
            "SELECT column_name FROM information_schema.columns \
             WHERE table_schema = $1 AND table_name = $2 \
             AND data_type IN ('text', 'character varying', 'character', 'varchar', 'name') \
             ORDER BY ordinal_position",
        )
        .bind(schema)
        .bind(table)
        .fetch_all(pool)
        .await
        .context("Failed to discover text columns")?;

        let columns: Vec<String> = rows
            .iter()
            .map(|row| row.get::<String, _>("column_name"))
            .collect();

        Ok(columns)
    }

    /// Fetches rows from a single table and converts them to ContentItems.
    async fn fetch_table(
        pool: &PgPool,
        config: &ConnectorConfig,
        table: &str,
        cursor_value: Option<&str>,
    ) -> Result<Vec<ContentItem>> {
        let schema = Self::get_schema(config);
        let id_col = Self::get_id_column(config);
        let cursor_col = Self::get_cursor_column(config);
        let limit = Self::get_limit(config);
        let extra_cols = Self::get_extra_columns(config);

        Self::validate_identifier(&schema)?;
        Self::validate_identifier(table)?;
        Self::validate_identifier(&id_col)?;
        if let Some(ref cc) = cursor_col {
            Self::validate_identifier(cc)?;
        }
        for ec in &extra_cols {
            Self::validate_identifier(ec)?;
        }

        // Determine which columns to use as content
        let content_cols = match Self::get_content_columns(config, table) {
            Some(cols) => {
                for c in &cols {
                    Self::validate_identifier(c)?;
                }
                cols
            }
            None => Self::discover_text_columns(pool, &schema, table).await?,
        };

        if content_cols.is_empty() {
            return Ok(Vec::new());
        }

        // Build SELECT columns: id_col, content columns, cursor column, extra columns
        let mut select_cols = vec![format!("\"{}\"", id_col)];
        for col in &content_cols {
            select_cols.push(format!("\"{}\"::text", col));
        }
        if let Some(ref cc) = cursor_col {
            select_cols.push(format!("\"{}\"::text AS __cursor_val", cc));
        }
        for ec in &extra_cols {
            select_cols.push(format!("\"{}\"::text AS \"{}\"", ec, ec));
        }

        let select_clause = select_cols.join(", ");

        // Build WHERE clause for incremental sync
        let where_clause = match (&cursor_col, cursor_value) {
            (Some(cc), Some(_val)) => {
                // Use parameterized comparison — cursor values are passed as strings
                format!(" WHERE \"{}\"::text > $1", cc)
            }
            _ => String::new(),
        };

        // Build ORDER BY for consistent pagination
        let order_clause = match &cursor_col {
            Some(cc) => format!(" ORDER BY \"{}\" ASC", cc),
            None => format!(" ORDER BY \"{}\" ASC", id_col),
        };

        let query = format!(
            "SELECT {} FROM \"{}\".\"{}\"{}{}  LIMIT {}",
            select_clause, schema, table, where_clause, order_clause, limit
        );

        let rows: Vec<PgRow> = if let (Some(_), Some(val)) = (&cursor_col, cursor_value) {
            sqlx::query(&query)
                .bind(val)
                .fetch_all(pool)
                .await
                .with_context(|| format!("Failed to fetch rows from {}.{}", schema, table))?
        } else {
            sqlx::query(&query)
                .fetch_all(pool)
                .await
                .with_context(|| format!("Failed to fetch rows from {}.{}", schema, table))?
        };

        let mut items = Vec::with_capacity(rows.len());

        for row in &rows {
            // Extract the row ID as a string
            let row_id: String = row
                .try_get::<String, _>(id_col.as_str())
                .or_else(|_| {
                    row.try_get::<i32, _>(id_col.as_str())
                        .map(|v| v.to_string())
                })
                .or_else(|_| {
                    row.try_get::<i64, _>(id_col.as_str())
                        .map(|v| v.to_string())
                })
                .unwrap_or_else(|_| "unknown".to_string());

            // Concatenate content columns
            let content_parts: Vec<String> = content_cols
                .iter()
                .filter_map(|col| row.try_get::<String, _>(col.as_str()).ok())
                .filter(|s| !s.is_empty())
                .collect();

            let content = content_parts.join("\n\n");
            if content.is_empty() {
                continue;
            }

            // Build extra metadata from extra_columns
            let mut extra = HashMap::new();
            extra.insert(
                "table".to_string(),
                serde_json::Value::String(table.to_string()),
            );
            extra.insert(
                "schema".to_string(),
                serde_json::Value::String(schema.clone()),
            );
            for ec in &extra_cols {
                if let Ok(val) = row.try_get::<String, _>(ec.as_str()) {
                    extra.insert(ec.clone(), serde_json::Value::String(val));
                }
            }

            items.push(ContentItem {
                content,
                source: SourceMetadata {
                    connector_type: "postgres".to_string(),
                    connector_id: config.id.clone(),
                    source_id: format!("{}.{}.{}", schema, table, row_id),
                    source_url: None,
                    author: None,
                    created_at: None,
                    extra,
                },
                media: None,
            });
        }

        Ok(items)
    }
}

impl Default for PostgresConnector {
    fn default() -> Self {
        Self::new()
    }
}

impl Connector for PostgresConnector {
    fn id(&self) -> &str {
        "postgres"
    }

    fn name(&self) -> &str {
        "PostgreSQL / Neon"
    }

    fn config_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "auth": {
                    "type": "object",
                    "properties": {
                        "type": { "const": "ConnectionString" },
                        "uri": {
                            "type": "string",
                            "description": "PostgreSQL connection string (e.g., postgres://user:pass@host:5432/db)"
                        }
                    },
                    "required": ["uri"]
                },
                "settings": {
                    "type": "object",
                    "properties": {
                        "tables": {
                            "type": "array",
                            "items": { "type": "string" },
                            "description": "Table names to fetch from"
                        },
                        "content_columns": {
                            "type": "object",
                            "description": "Map of table name to array of column names for content",
                            "additionalProperties": {
                                "type": "array",
                                "items": { "type": "string" }
                            }
                        },
                        "id_column": {
                            "type": "string",
                            "description": "Column to use as row identifier (default: 'id')"
                        },
                        "cursor_column": {
                            "type": "string",
                            "description": "Column for incremental sync ordering (e.g., 'updated_at')"
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Max rows per table per fetch (default: 10000)"
                        },
                        "schema": {
                            "type": "string",
                            "description": "Database schema (default: 'public')"
                        },
                        "extra_columns": {
                            "type": "array",
                            "items": { "type": "string" },
                            "description": "Additional columns to include as metadata"
                        }
                    },
                    "required": ["tables"]
                }
            },
            "required": ["auth", "settings"]
        })
    }

    fn validate_config(&self, config: &ConnectorConfig) -> Result<()> {
        // Validate auth type
        Self::get_connection_uri(config)?;

        // Validate tables
        let tables = Self::get_tables(config)?;
        for table in &tables {
            Self::validate_identifier(table)?;
        }

        // Validate id_column
        let id_col = Self::get_id_column(config);
        Self::validate_identifier(&id_col)?;

        // Validate cursor_column if present
        if let Some(cursor_col) = Self::get_cursor_column(config) {
            Self::validate_identifier(&cursor_col)?;
        }

        // Validate schema
        let schema = Self::get_schema(config);
        Self::validate_identifier(&schema)?;

        // Validate content_columns if present
        if let Some(cc_map) = config
            .settings
            .get("content_columns")
            .and_then(|v| v.as_object())
        {
            for (table_name, cols) in cc_map {
                Self::validate_identifier(table_name)?;
                if let Some(arr) = cols.as_array() {
                    for col in arr {
                        if let Some(col_name) = col.as_str() {
                            Self::validate_identifier(col_name)?;
                        }
                    }
                }
            }
        }

        // Validate extra_columns
        for ec in Self::get_extra_columns(config) {
            Self::validate_identifier(&ec)?;
        }

        // Validate limit is positive
        let limit = Self::get_limit(config);
        if limit <= 0 {
            bail!("'limit' must be a positive integer, got: {}", limit);
        }

        Ok(())
    }

    async fn fetch(&self, config: &ConnectorConfig) -> Result<Vec<ContentItem>> {
        let pool = Self::create_pool(config).await?;
        let tables = Self::get_tables(config)?;

        let mut all_items = Vec::new();

        for table in &tables {
            let items = Self::fetch_table(&pool, config, table, None).await?;
            all_items.extend(items);
        }

        pool.close().await;
        Ok(all_items)
    }

    async fn sync_incremental(
        &self,
        config: &ConnectorConfig,
        cursor: &SyncCursor,
    ) -> Result<SyncResult> {
        let pool = Self::create_pool(config).await?;
        let tables = Self::get_tables(config)?;
        let cursor_col = Self::get_cursor_column(config);

        let mut all_items = Vec::new();
        #[allow(unused_mut)]
        let mut max_cursor: Option<String> = cursor.value.clone();

        for table in &tables {
            let cursor_val = cursor.value.as_deref();
            let items = if cursor_col.is_some() {
                Self::fetch_table(&pool, config, table, cursor_val).await?
            } else {
                // Without cursor column, full fetch every time
                Self::fetch_table(&pool, config, table, None).await?
            };
            all_items.extend(items);
        }

        // Update cursor to the latest value seen
        let new_cursor = SyncCursor {
            value: max_cursor.or_else(|| cursor.value.clone()),
            last_sync: Some(
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs(),
            ),
        };

        pool.close().await;
        Ok(SyncResult {
            items: all_items,
            cursor: new_cursor,
            skipped: 0,
        })
    }

    async fn handle_webhook(
        &self,
        _config: &ConnectorConfig,
        _payload: WebhookPayload,
    ) -> Result<Vec<ContentItem>> {
        bail!(
            "PostgreSQL connector does not support webhooks. \
             Use cron-based scheduling (US-26.18) for periodic sync instead."
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_config(uri: &str) -> ConnectorConfig {
        let mut settings = HashMap::new();
        settings.insert(
            "tables".to_string(),
            serde_json::json!(["articles", "comments"]),
        );

        ConnectorConfig {
            id: "pg-test".to_string(),
            name: "Test Postgres".to_string(),
            connector_type: "postgres".to_string(),
            auth: AuthConfig::ConnectionString {
                uri: uri.to_string(),
            },
            namespace: "test".to_string(),
            settings,
            enabled: true,
        }
    }

    #[test]
    fn test_connector_id_and_name() {
        let connector = PostgresConnector::new();
        assert_eq!(connector.id(), "postgres");
        assert_eq!(connector.name(), "PostgreSQL / Neon");
    }

    #[test]
    fn test_default_constructor() {
        let connector = PostgresConnector;
        assert_eq!(connector.id(), "postgres");
    }

    #[test]
    fn test_config_schema_has_required_fields() {
        let connector = PostgresConnector::new();
        let schema = connector.config_schema();

        assert_eq!(schema["type"], "object");
        let required = schema["required"].as_array().unwrap();
        assert!(required.contains(&serde_json::json!("auth")));
        assert!(required.contains(&serde_json::json!("settings")));

        // Settings should require tables
        let settings_required = schema["properties"]["settings"]["required"]
            .as_array()
            .unwrap();
        assert!(settings_required.contains(&serde_json::json!("tables")));
    }

    #[test]
    fn test_validate_config_valid() {
        let connector = PostgresConnector::new();
        let config = make_config("postgres://user:pass@localhost:5432/test");
        assert!(connector.validate_config(&config).is_ok());
    }

    #[test]
    fn test_validate_config_wrong_auth_type() {
        let connector = PostgresConnector::new();
        let mut config = make_config("postgres://localhost/db");
        config.auth = AuthConfig::Token {
            token: "bad".to_string(),
        };
        let result = connector.validate_config(&config);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("ConnectionString"));
    }

    #[test]
    fn test_validate_config_no_tables() {
        let connector = PostgresConnector::new();
        let mut config = make_config("postgres://localhost/db");
        config.settings.remove("tables");
        let result = connector.validate_config(&config);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("table"));
    }

    #[test]
    fn test_validate_config_empty_tables() {
        let connector = PostgresConnector::new();
        let mut config = make_config("postgres://localhost/db");
        config
            .settings
            .insert("tables".to_string(), serde_json::json!([]));
        let result = connector.validate_config(&config);
        assert!(result.is_err());
    }

    #[test]
    fn test_validate_config_invalid_table_name() {
        let connector = PostgresConnector::new();
        let mut config = make_config("postgres://localhost/db");
        config.settings.insert(
            "tables".to_string(),
            serde_json::json!(["valid_table", "invalid;table"]),
        );
        let result = connector.validate_config(&config);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("Invalid identifier"));
    }

    #[test]
    fn test_validate_config_invalid_id_column() {
        let connector = PostgresConnector::new();
        let mut config = make_config("postgres://localhost/db");
        config.settings.insert(
            "id_column".to_string(),
            serde_json::json!("bad; DROP TABLE"),
        );
        let result = connector.validate_config(&config);
        assert!(result.is_err());
    }

    #[test]
    fn test_validate_config_invalid_schema() {
        let connector = PostgresConnector::new();
        let mut config = make_config("postgres://localhost/db");
        config
            .settings
            .insert("schema".to_string(), serde_json::json!("bad schema!"));
        let result = connector.validate_config(&config);
        assert!(result.is_err());
    }

    #[test]
    fn test_validate_config_invalid_cursor_column() {
        let connector = PostgresConnector::new();
        let mut config = make_config("postgres://localhost/db");
        config
            .settings
            .insert("cursor_column".to_string(), serde_json::json!("col; DROP"));
        let result = connector.validate_config(&config);
        assert!(result.is_err());
    }

    #[test]
    fn test_validate_config_invalid_extra_columns() {
        let connector = PostgresConnector::new();
        let mut config = make_config("postgres://localhost/db");
        config.settings.insert(
            "extra_columns".to_string(),
            serde_json::json!(["good_col", "bad col!"]),
        );
        let result = connector.validate_config(&config);
        assert!(result.is_err());
    }

    #[test]
    fn test_validate_config_invalid_content_columns() {
        let connector = PostgresConnector::new();
        let mut config = make_config("postgres://localhost/db");
        config.settings.insert(
            "content_columns".to_string(),
            serde_json::json!({"articles": ["title", "bad; col"]}),
        );
        let result = connector.validate_config(&config);
        assert!(result.is_err());
    }

    #[test]
    fn test_validate_config_zero_limit() {
        let connector = PostgresConnector::new();
        let mut config = make_config("postgres://localhost/db");
        config
            .settings
            .insert("limit".to_string(), serde_json::json!(0));
        let result = connector.validate_config(&config);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("positive"));
    }

    #[test]
    fn test_validate_config_negative_limit() {
        let connector = PostgresConnector::new();
        let mut config = make_config("postgres://localhost/db");
        config
            .settings
            .insert("limit".to_string(), serde_json::json!(-5));
        let result = connector.validate_config(&config);
        assert!(result.is_err());
    }

    #[test]
    fn test_get_tables() {
        let config = make_config("postgres://localhost/db");
        let tables = PostgresConnector::get_tables(&config).unwrap();
        assert_eq!(tables, vec!["articles", "comments"]);
    }

    #[test]
    fn test_get_id_column_default() {
        let config = make_config("postgres://localhost/db");
        assert_eq!(PostgresConnector::get_id_column(&config), "id");
    }

    #[test]
    fn test_get_id_column_custom() {
        let mut config = make_config("postgres://localhost/db");
        config
            .settings
            .insert("id_column".to_string(), serde_json::json!("article_id"));
        assert_eq!(PostgresConnector::get_id_column(&config), "article_id");
    }

    #[test]
    fn test_get_schema_default() {
        let config = make_config("postgres://localhost/db");
        assert_eq!(PostgresConnector::get_schema(&config), "public");
    }

    #[test]
    fn test_get_schema_custom() {
        let mut config = make_config("postgres://localhost/db");
        config
            .settings
            .insert("schema".to_string(), serde_json::json!("myschema"));
        assert_eq!(PostgresConnector::get_schema(&config), "myschema");
    }

    #[test]
    fn test_get_limit_default() {
        let config = make_config("postgres://localhost/db");
        assert_eq!(PostgresConnector::get_limit(&config), 10000);
    }

    #[test]
    fn test_get_limit_custom() {
        let mut config = make_config("postgres://localhost/db");
        config
            .settings
            .insert("limit".to_string(), serde_json::json!(500));
        assert_eq!(PostgresConnector::get_limit(&config), 500);
    }

    #[test]
    fn test_get_cursor_column_none() {
        let config = make_config("postgres://localhost/db");
        assert!(PostgresConnector::get_cursor_column(&config).is_none());
    }

    #[test]
    fn test_get_cursor_column_custom() {
        let mut config = make_config("postgres://localhost/db");
        config
            .settings
            .insert("cursor_column".to_string(), serde_json::json!("updated_at"));
        assert_eq!(
            PostgresConnector::get_cursor_column(&config).unwrap(),
            "updated_at"
        );
    }

    #[test]
    fn test_get_content_columns_none() {
        let config = make_config("postgres://localhost/db");
        assert!(PostgresConnector::get_content_columns(&config, "articles").is_none());
    }

    #[test]
    fn test_get_content_columns_configured() {
        let mut config = make_config("postgres://localhost/db");
        config.settings.insert(
            "content_columns".to_string(),
            serde_json::json!({"articles": ["title", "body"]}),
        );
        let cols = PostgresConnector::get_content_columns(&config, "articles").unwrap();
        assert_eq!(cols, vec!["title", "body"]);
    }

    #[test]
    fn test_get_content_columns_table_not_in_map() {
        let mut config = make_config("postgres://localhost/db");
        config.settings.insert(
            "content_columns".to_string(),
            serde_json::json!({"articles": ["title", "body"]}),
        );
        assert!(PostgresConnector::get_content_columns(&config, "comments").is_none());
    }

    #[test]
    fn test_get_extra_columns_default() {
        let config = make_config("postgres://localhost/db");
        assert!(PostgresConnector::get_extra_columns(&config).is_empty());
    }

    #[test]
    fn test_get_extra_columns_configured() {
        let mut config = make_config("postgres://localhost/db");
        config.settings.insert(
            "extra_columns".to_string(),
            serde_json::json!(["author", "status"]),
        );
        assert_eq!(
            PostgresConnector::get_extra_columns(&config),
            vec!["author", "status"]
        );
    }

    #[test]
    fn test_validate_identifier_valid() {
        assert!(PostgresConnector::validate_identifier("my_table").is_ok());
        assert!(PostgresConnector::validate_identifier("Table123").is_ok());
        assert!(PostgresConnector::validate_identifier("schema.table").is_ok());
        assert!(PostgresConnector::validate_identifier("id").is_ok());
    }

    #[test]
    fn test_validate_identifier_invalid() {
        assert!(PostgresConnector::validate_identifier("").is_err());
        assert!(PostgresConnector::validate_identifier("bad;table").is_err());
        assert!(PostgresConnector::validate_identifier("bad table").is_err());
        assert!(PostgresConnector::validate_identifier("bad'table").is_err());
        assert!(PostgresConnector::validate_identifier("bad\"table").is_err());
        assert!(PostgresConnector::validate_identifier("bad--table").is_err());
        assert!(PostgresConnector::validate_identifier("DROP TABLE").is_err());
    }

    #[test]
    fn test_connection_uri_extraction() {
        let config = make_config("postgres://user:pass@localhost:5432/testdb");
        let uri = PostgresConnector::get_connection_uri(&config).unwrap();
        assert_eq!(uri, "postgres://user:pass@localhost:5432/testdb");
    }

    #[test]
    fn test_connection_uri_wrong_auth() {
        let mut config = make_config("postgres://localhost/db");
        config.auth = AuthConfig::None;
        let result = PostgresConnector::get_connection_uri(&config);
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_handle_webhook_returns_error() {
        let connector = PostgresConnector::new();
        let config = make_config("postgres://localhost/db");
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
    fn test_validate_config_with_all_settings() {
        let connector = PostgresConnector::new();
        let mut settings = HashMap::new();
        settings.insert("tables".to_string(), serde_json::json!(["users"]));
        settings.insert("id_column".to_string(), serde_json::json!("user_id"));
        settings.insert("cursor_column".to_string(), serde_json::json!("updated_at"));
        settings.insert("limit".to_string(), serde_json::json!(5000));
        settings.insert("schema".to_string(), serde_json::json!("app"));
        settings.insert(
            "content_columns".to_string(),
            serde_json::json!({"users": ["name", "bio"]}),
        );
        settings.insert(
            "extra_columns".to_string(),
            serde_json::json!(["email", "role"]),
        );

        let config = ConnectorConfig {
            id: "pg-full".to_string(),
            name: "Full Config Test".to_string(),
            connector_type: "postgres".to_string(),
            auth: AuthConfig::ConnectionString {
                uri: "postgres://u:p@host/db".to_string(),
            },
            namespace: "test".to_string(),
            settings,
            enabled: true,
        };

        assert!(connector.validate_config(&config).is_ok());
    }

    #[tokio::test]
    async fn test_create_pool_invalid_uri_fails() {
        let config = make_config("postgres://invalid:pass@nonexistent.host.example:5432/db");
        let result = PostgresConnector::create_pool(&config).await;
        // This should fail because the host doesn't exist
        assert!(result.is_err());
    }

    #[test]
    fn test_validate_config_content_columns_with_invalid_table_key() {
        let connector = PostgresConnector::new();
        let mut config = make_config("postgres://localhost/db");
        config.settings.insert(
            "content_columns".to_string(),
            serde_json::json!({"bad;table": ["col"]}),
        );
        let result = connector.validate_config(&config);
        assert!(result.is_err());
    }
}
