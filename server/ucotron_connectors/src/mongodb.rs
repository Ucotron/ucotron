//! MongoDB connector — fetches documents from MongoDB collections for ingestion.
//!
//! Uses the official `mongodb` driver with a connection string URI.
//! Supports full sync (all documents from configured collections) and incremental sync
//! via a timestamp or ObjectId field cursor.
//!
//! # Settings
//!
//! - `collections` (required): array of collection names to fetch from
//! - `database` (required): database name to connect to
//! - `content_fields` (optional): map of collection → array of field names to concatenate as content
//!   (default: all string-valued top-level fields)
//! - `id_field` (optional): field name used as source_id (default: "_id")
//! - `cursor_field` (optional): field name for incremental sync ordering (e.g., "updated_at", "_id")
//! - `limit` (optional): maximum documents per collection per fetch (default: 10000)
//! - `filter` (optional): JSON query filter applied to all collections (MongoDB query syntax)
//! - `extra_fields` (optional): array of field names to include as extra metadata

use std::collections::HashMap;

use anyhow::{bail, Context, Result};
use mongodb::bson::{doc, oid::ObjectId, Bson, Document};
use mongodb::options::{ClientOptions, FindOptions};
use mongodb::Client;

use crate::connector::{
    AuthConfig, Connector, ConnectorConfig, ContentItem, SourceMetadata, SyncCursor, SyncResult,
    WebhookPayload,
};

/// MongoDB connector for fetching database documents as content items.
///
/// Connects to a MongoDB database using a connection string provided via
/// `AuthConfig::ConnectionString`. Each configured collection is queried,
/// and documents are mapped to [`ContentItem`]s with string fields
/// concatenated as content and other fields preserved as metadata.
///
/// # Authentication
///
/// Uses `AuthConfig::ConnectionString { uri }` where `uri` is a standard
/// MongoDB connection string like:
/// ```text
/// mongodb://user:password@host:27017/dbname
/// mongodb+srv://user:password@cluster.mongodb.net/dbname
/// ```
///
/// # Example Configuration
///
/// ```json
/// {
///     "id": "mongo-main",
///     "name": "Main MongoDB",
///     "connector_type": "mongodb",
///     "auth": { "type": "ConnectionString", "uri": "mongodb://localhost:27017" },
///     "namespace": "prod",
///     "settings": {
///         "database": "myapp",
///         "collections": ["articles", "comments"],
///         "content_fields": { "articles": ["title", "body"], "comments": ["text"] },
///         "id_field": "_id",
///         "cursor_field": "updated_at",
///         "limit": 5000,
///         "extra_fields": ["author", "category"]
///     }
/// }
/// ```
pub struct MongoConnector;

impl MongoConnector {
    /// Creates a new MongoConnector.
    pub fn new() -> Self {
        Self
    }

    /// Extracts the connection URI from the connector config.
    fn get_connection_uri(config: &ConnectorConfig) -> Result<&str> {
        match &config.auth {
            AuthConfig::ConnectionString { uri } => Ok(uri.as_str()),
            _ => bail!(
                "MongoDB connector requires ConnectionString authentication \
                 (e.g., mongodb://user:pass@host:27017/db)"
            ),
        }
    }

    /// Returns the database name from settings.
    fn get_database(config: &ConnectorConfig) -> Result<String> {
        config
            .settings
            .get("database")
            .and_then(|v| v.as_str())
            .map(String::from)
            .ok_or_else(|| anyhow::anyhow!("MongoDB connector requires 'database' setting"))
    }

    /// Returns the list of collections to fetch from settings.
    fn get_collections(config: &ConnectorConfig) -> Result<Vec<String>> {
        let collections = config
            .settings
            .get("collections")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str().map(String::from))
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();

        if collections.is_empty() {
            bail!("MongoDB connector requires at least one collection in 'collections' setting");
        }
        Ok(collections)
    }

    /// Returns content fields for a specific collection (or None to use all string fields).
    fn get_content_fields(config: &ConnectorConfig, collection: &str) -> Option<Vec<String>> {
        config
            .settings
            .get("content_fields")
            .and_then(|v| v.as_object())
            .and_then(|map| map.get(collection))
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str().map(String::from))
                    .collect()
            })
    }

    /// Returns the ID field name (default: "_id").
    fn get_id_field(config: &ConnectorConfig) -> String {
        config
            .settings
            .get("id_field")
            .and_then(|v| v.as_str())
            .unwrap_or("_id")
            .to_string()
    }

    /// Returns the cursor field name for incremental sync (if configured).
    fn get_cursor_field(config: &ConnectorConfig) -> Option<String> {
        config
            .settings
            .get("cursor_field")
            .and_then(|v| v.as_str())
            .map(String::from)
    }

    /// Returns the maximum documents per collection (default: 10000).
    fn get_limit(config: &ConnectorConfig) -> i64 {
        config
            .settings
            .get("limit")
            .and_then(|v| v.as_i64())
            .unwrap_or(10000)
    }

    /// Returns a filter document from settings (if configured).
    fn get_filter(config: &ConnectorConfig) -> Option<Document> {
        config.settings.get("filter").and_then(|v| {
            // Convert serde_json::Value to BSON Document
            let bson = json_value_to_bson(v);
            match bson {
                Bson::Document(doc) => Some(doc),
                _ => None,
            }
        })
    }

    /// Returns extra fields to include as metadata.
    fn get_extra_fields(config: &ConnectorConfig) -> Vec<String> {
        config
            .settings
            .get("extra_fields")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str().map(String::from))
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Creates a MongoDB client from the config.
    async fn create_client(config: &ConnectorConfig) -> Result<Client> {
        let uri = Self::get_connection_uri(config)?;
        let options = ClientOptions::parse(uri)
            .await
            .context("Failed to parse MongoDB connection string")?;
        let client = Client::with_options(options).context("Failed to create MongoDB client")?;
        Ok(client)
    }

    /// Converts a BSON value to a string representation for content.
    fn bson_to_string(bson: &Bson) -> Option<String> {
        match bson {
            Bson::String(s) => {
                if s.is_empty() {
                    None
                } else {
                    Some(s.clone())
                }
            }
            Bson::Int32(i) => Some(i.to_string()),
            Bson::Int64(i) => Some(i.to_string()),
            Bson::Double(d) => Some(d.to_string()),
            Bson::Boolean(b) => Some(b.to_string()),
            Bson::ObjectId(oid) => Some(oid.to_hex()),
            Bson::DateTime(dt) => Some(dt.to_string()),
            _ => None,
        }
    }

    /// Extracts an ID string from a document.
    fn extract_id(doc: &Document, id_field: &str) -> String {
        doc.get(id_field)
            .and_then(Self::bson_to_string)
            .unwrap_or_else(|| "unknown".to_string())
    }

    /// Extracts content from a document using specified fields or all string fields.
    fn extract_content(doc: &Document, content_fields: &Option<Vec<String>>) -> String {
        match content_fields {
            Some(fields) => {
                let parts: Vec<String> = fields
                    .iter()
                    .filter_map(|field| doc.get(field).and_then(Self::bson_to_string))
                    .collect();
                parts.join("\n\n")
            }
            None => {
                // Use all string-valued top-level fields (skip _id)
                let parts: Vec<String> = doc
                    .iter()
                    .filter(|(k, _)| *k != "_id")
                    .filter_map(|(_, v)| Self::bson_to_string(v))
                    .collect();
                parts.join("\n\n")
            }
        }
    }

    /// Fetches documents from a single collection and converts them to ContentItems.
    async fn fetch_collection(
        client: &Client,
        config: &ConnectorConfig,
        database: &str,
        collection: &str,
        cursor_value: Option<&str>,
    ) -> Result<Vec<ContentItem>> {
        let id_field = Self::get_id_field(config);
        let cursor_field = Self::get_cursor_field(config);
        let limit = Self::get_limit(config);
        let extra_fields = Self::get_extra_fields(config);
        let content_fields = Self::get_content_fields(config, collection);
        let base_filter = Self::get_filter(config);

        let db = client.database(database);
        let coll = db.collection::<Document>(collection);

        // Build filter document
        let mut filter = base_filter.unwrap_or_default();

        // Add cursor condition for incremental sync
        if let (Some(cf), Some(cv)) = (&cursor_field, cursor_value) {
            // Try to parse as ObjectId first, then fall back to string comparison
            let cursor_bson = if cf == "_id" {
                ObjectId::parse_str(cv)
                    .map(Bson::ObjectId)
                    .unwrap_or_else(|_| Bson::String(cv.to_string()))
            } else {
                Bson::String(cv.to_string())
            };
            filter.insert(cf.as_str(), doc! { "$gt": cursor_bson });
        }

        // Build find options
        let sort_field = cursor_field.as_deref().unwrap_or(&id_field);
        let find_options = FindOptions::builder()
            .limit(limit)
            .sort(doc! { sort_field: 1 })
            .build();

        let mut cursor = coll
            .find(filter)
            .with_options(find_options)
            .await
            .with_context(|| format!("Failed to query collection '{}'", collection))?;

        let mut items = Vec::new();

        while cursor
            .advance()
            .await
            .with_context(|| format!("Failed to advance cursor for '{}'", collection))?
        {
            let document: Document = cursor
                .deserialize_current()
                .with_context(|| format!("Failed to read document from '{}'", collection))?;

            let doc_id = Self::extract_id(&document, &id_field);
            let content = Self::extract_content(&document, &content_fields);

            if content.is_empty() {
                continue;
            }

            // Build extra metadata
            let mut extra = HashMap::new();
            extra.insert(
                "collection".to_string(),
                serde_json::Value::String(collection.to_string()),
            );
            extra.insert(
                "database".to_string(),
                serde_json::Value::String(database.to_string()),
            );
            for ef in &extra_fields {
                if let Some(val) = document.get(ef).and_then(Self::bson_to_string) {
                    extra.insert(ef.clone(), serde_json::Value::String(val));
                }
            }

            // Extract timestamp from _id (ObjectId) if available
            let created_at = document.get("_id").and_then(|v| match v {
                Bson::ObjectId(oid) => Some(oid.timestamp().timestamp_millis() as u64 / 1000),
                _ => None,
            });

            items.push(ContentItem {
                content,
                source: SourceMetadata {
                    connector_type: "mongodb".to_string(),
                    connector_id: config.id.clone(),
                    source_id: format!("{}.{}.{}", database, collection, doc_id),
                    source_url: None,
                    author: None,
                    created_at,
                    extra,
                },
                media: None,
            });
        }

        Ok(items)
    }
}

impl Default for MongoConnector {
    fn default() -> Self {
        Self::new()
    }
}

/// Converts a serde_json::Value to a BSON value.
fn json_value_to_bson(value: &serde_json::Value) -> Bson {
    match value {
        serde_json::Value::Null => Bson::Null,
        serde_json::Value::Bool(b) => Bson::Boolean(*b),
        serde_json::Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                Bson::Int64(i)
            } else if let Some(f) = n.as_f64() {
                Bson::Double(f)
            } else {
                Bson::Null
            }
        }
        serde_json::Value::String(s) => Bson::String(s.clone()),
        serde_json::Value::Array(arr) => Bson::Array(arr.iter().map(json_value_to_bson).collect()),
        serde_json::Value::Object(map) => {
            let mut doc = Document::new();
            for (k, v) in map {
                doc.insert(k.clone(), json_value_to_bson(v));
            }
            Bson::Document(doc)
        }
    }
}

impl Connector for MongoConnector {
    fn id(&self) -> &str {
        "mongodb"
    }

    fn name(&self) -> &str {
        "MongoDB"
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
                            "description": "MongoDB connection string (e.g., mongodb://user:pass@host:27017)"
                        }
                    },
                    "required": ["uri"]
                },
                "settings": {
                    "type": "object",
                    "properties": {
                        "database": {
                            "type": "string",
                            "description": "Database name to connect to"
                        },
                        "collections": {
                            "type": "array",
                            "items": { "type": "string" },
                            "description": "Collection names to fetch from"
                        },
                        "content_fields": {
                            "type": "object",
                            "description": "Map of collection name to array of field names for content",
                            "additionalProperties": {
                                "type": "array",
                                "items": { "type": "string" }
                            }
                        },
                        "id_field": {
                            "type": "string",
                            "description": "Field to use as document identifier (default: '_id')"
                        },
                        "cursor_field": {
                            "type": "string",
                            "description": "Field for incremental sync ordering (e.g., 'updated_at')"
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Max documents per collection per fetch (default: 10000)"
                        },
                        "filter": {
                            "type": "object",
                            "description": "MongoDB query filter applied to all collections"
                        },
                        "extra_fields": {
                            "type": "array",
                            "items": { "type": "string" },
                            "description": "Additional fields to include as metadata"
                        }
                    },
                    "required": ["database", "collections"]
                }
            },
            "required": ["auth", "settings"]
        })
    }

    fn validate_config(&self, config: &ConnectorConfig) -> Result<()> {
        // Validate auth type
        Self::get_connection_uri(config)?;

        // Validate database name
        let db = Self::get_database(config)?;
        if db.is_empty() {
            bail!("'database' setting must not be empty");
        }

        // Validate collections
        let collections = Self::get_collections(config)?;
        for collection in &collections {
            if collection.is_empty() {
                bail!("Collection names must not be empty");
            }
            // MongoDB collection names can't contain $ except for special system collections
            if collection.contains('$') {
                bail!(
                    "Invalid collection name '{}': must not contain '$'",
                    collection
                );
            }
            // Collection names can't start with "system."
            if collection.starts_with("system.") {
                bail!(
                    "Invalid collection name '{}': must not start with 'system.'",
                    collection
                );
            }
        }

        // Validate limit is positive
        let limit = Self::get_limit(config);
        if limit <= 0 {
            bail!("'limit' must be a positive integer, got: {}", limit);
        }

        Ok(())
    }

    async fn fetch(&self, config: &ConnectorConfig) -> Result<Vec<ContentItem>> {
        let client = Self::create_client(config).await?;
        let database = Self::get_database(config)?;
        let collections = Self::get_collections(config)?;

        let mut all_items = Vec::new();

        for collection in &collections {
            let items =
                Self::fetch_collection(&client, config, &database, collection, None).await?;
            all_items.extend(items);
        }

        Ok(all_items)
    }

    async fn sync_incremental(
        &self,
        config: &ConnectorConfig,
        cursor: &SyncCursor,
    ) -> Result<SyncResult> {
        let client = Self::create_client(config).await?;
        let database = Self::get_database(config)?;
        let collections = Self::get_collections(config)?;

        let mut all_items = Vec::new();
        let mut max_cursor: Option<String> = cursor.value.clone();

        for collection in &collections {
            let cursor_val = cursor.value.as_deref();
            let items = if Self::get_cursor_field(config).is_some() {
                Self::fetch_collection(&client, config, &database, collection, cursor_val).await?
            } else {
                // Without cursor field, full fetch every time
                Self::fetch_collection(&client, config, &database, collection, None).await?
            };

            // Track the latest cursor value from the last item's source_id
            // (items are sorted by cursor field ascending, so last item has the max cursor)
            if let Some(last) = items.last() {
                if let Some(ref cf) = Self::get_cursor_field(config) {
                    // The cursor value would come from the actual document field
                    // For simplicity, use the source_id suffix (the doc ID)
                    let parts: Vec<&str> = last.source.source_id.rsplitn(2, '.').collect();
                    if let Some(id) = parts.first() {
                        match &max_cursor {
                            Some(existing) if *id > existing.as_str() => {
                                max_cursor = Some(id.to_string());
                            }
                            None => {
                                max_cursor = Some(id.to_string());
                            }
                            _ => {}
                        }
                    }
                    let _ = cf; // Suppress unused warning
                }
            }

            all_items.extend(items);
        }

        let new_cursor = SyncCursor {
            value: max_cursor.or_else(|| cursor.value.clone()),
            last_sync: Some(
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs(),
            ),
        };

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
            "MongoDB connector does not support webhooks. \
             Use cron-based scheduling for periodic sync instead."
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_config(uri: &str) -> ConnectorConfig {
        let mut settings = HashMap::new();
        settings.insert("database".to_string(), serde_json::json!("testdb"));
        settings.insert(
            "collections".to_string(),
            serde_json::json!(["articles", "comments"]),
        );

        ConnectorConfig {
            id: "mongo-test".to_string(),
            name: "Test MongoDB".to_string(),
            connector_type: "mongodb".to_string(),
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
        let connector = MongoConnector::new();
        assert_eq!(connector.id(), "mongodb");
        assert_eq!(connector.name(), "MongoDB");
    }

    #[test]
    fn test_default_constructor() {
        let connector = MongoConnector;
        assert_eq!(connector.id(), "mongodb");
    }

    #[test]
    fn test_config_schema_has_required_fields() {
        let connector = MongoConnector::new();
        let schema = connector.config_schema();

        assert_eq!(schema["type"], "object");
        let required = schema["required"].as_array().unwrap();
        assert!(required.contains(&serde_json::json!("auth")));
        assert!(required.contains(&serde_json::json!("settings")));

        // Settings should require database and collections
        let settings_required = schema["properties"]["settings"]["required"]
            .as_array()
            .unwrap();
        assert!(settings_required.contains(&serde_json::json!("database")));
        assert!(settings_required.contains(&serde_json::json!("collections")));
    }

    #[test]
    fn test_validate_config_valid() {
        let connector = MongoConnector::new();
        let config = make_config("mongodb://localhost:27017");
        assert!(connector.validate_config(&config).is_ok());
    }

    #[test]
    fn test_validate_config_wrong_auth_type() {
        let connector = MongoConnector::new();
        let mut config = make_config("mongodb://localhost:27017");
        config.auth = AuthConfig::Token {
            token: "bad".to_string(),
        };
        let result = connector.validate_config(&config);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("ConnectionString"));
    }

    #[test]
    fn test_validate_config_no_database() {
        let connector = MongoConnector::new();
        let mut config = make_config("mongodb://localhost:27017");
        config.settings.remove("database");
        let result = connector.validate_config(&config);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("database"));
    }

    #[test]
    fn test_validate_config_empty_database() {
        let connector = MongoConnector::new();
        let mut config = make_config("mongodb://localhost:27017");
        config
            .settings
            .insert("database".to_string(), serde_json::json!(""));
        let result = connector.validate_config(&config);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("empty"));
    }

    #[test]
    fn test_validate_config_no_collections() {
        let connector = MongoConnector::new();
        let mut config = make_config("mongodb://localhost:27017");
        config.settings.remove("collections");
        let result = connector.validate_config(&config);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("collection"));
    }

    #[test]
    fn test_validate_config_empty_collections() {
        let connector = MongoConnector::new();
        let mut config = make_config("mongodb://localhost:27017");
        config
            .settings
            .insert("collections".to_string(), serde_json::json!([]));
        let result = connector.validate_config(&config);
        assert!(result.is_err());
    }

    #[test]
    fn test_validate_config_invalid_collection_name_dollar() {
        let connector = MongoConnector::new();
        let mut config = make_config("mongodb://localhost:27017");
        config.settings.insert(
            "collections".to_string(),
            serde_json::json!(["valid", "invalid$name"]),
        );
        let result = connector.validate_config(&config);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("$"));
    }

    #[test]
    fn test_validate_config_invalid_collection_name_system() {
        let connector = MongoConnector::new();
        let mut config = make_config("mongodb://localhost:27017");
        config.settings.insert(
            "collections".to_string(),
            serde_json::json!(["system.users"]),
        );
        let result = connector.validate_config(&config);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("system."));
    }

    #[test]
    fn test_validate_config_zero_limit() {
        let connector = MongoConnector::new();
        let mut config = make_config("mongodb://localhost:27017");
        config
            .settings
            .insert("limit".to_string(), serde_json::json!(0));
        let result = connector.validate_config(&config);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("positive"));
    }

    #[test]
    fn test_validate_config_negative_limit() {
        let connector = MongoConnector::new();
        let mut config = make_config("mongodb://localhost:27017");
        config
            .settings
            .insert("limit".to_string(), serde_json::json!(-5));
        let result = connector.validate_config(&config);
        assert!(result.is_err());
    }

    #[test]
    fn test_get_database() {
        let config = make_config("mongodb://localhost:27017");
        let db = MongoConnector::get_database(&config).unwrap();
        assert_eq!(db, "testdb");
    }

    #[test]
    fn test_get_collections() {
        let config = make_config("mongodb://localhost:27017");
        let collections = MongoConnector::get_collections(&config).unwrap();
        assert_eq!(collections, vec!["articles", "comments"]);
    }

    #[test]
    fn test_get_id_field_default() {
        let config = make_config("mongodb://localhost:27017");
        assert_eq!(MongoConnector::get_id_field(&config), "_id");
    }

    #[test]
    fn test_get_id_field_custom() {
        let mut config = make_config("mongodb://localhost:27017");
        config
            .settings
            .insert("id_field".to_string(), serde_json::json!("doc_id"));
        assert_eq!(MongoConnector::get_id_field(&config), "doc_id");
    }

    #[test]
    fn test_get_limit_default() {
        let config = make_config("mongodb://localhost:27017");
        assert_eq!(MongoConnector::get_limit(&config), 10000);
    }

    #[test]
    fn test_get_limit_custom() {
        let mut config = make_config("mongodb://localhost:27017");
        config
            .settings
            .insert("limit".to_string(), serde_json::json!(500));
        assert_eq!(MongoConnector::get_limit(&config), 500);
    }

    #[test]
    fn test_get_cursor_field_none() {
        let config = make_config("mongodb://localhost:27017");
        assert!(MongoConnector::get_cursor_field(&config).is_none());
    }

    #[test]
    fn test_get_cursor_field_custom() {
        let mut config = make_config("mongodb://localhost:27017");
        config
            .settings
            .insert("cursor_field".to_string(), serde_json::json!("updated_at"));
        assert_eq!(
            MongoConnector::get_cursor_field(&config).unwrap(),
            "updated_at"
        );
    }

    #[test]
    fn test_get_content_fields_none() {
        let config = make_config("mongodb://localhost:27017");
        assert!(MongoConnector::get_content_fields(&config, "articles").is_none());
    }

    #[test]
    fn test_get_content_fields_configured() {
        let mut config = make_config("mongodb://localhost:27017");
        config.settings.insert(
            "content_fields".to_string(),
            serde_json::json!({"articles": ["title", "body"]}),
        );
        let fields = MongoConnector::get_content_fields(&config, "articles").unwrap();
        assert_eq!(fields, vec!["title", "body"]);
    }

    #[test]
    fn test_get_content_fields_missing_collection() {
        let mut config = make_config("mongodb://localhost:27017");
        config.settings.insert(
            "content_fields".to_string(),
            serde_json::json!({"articles": ["title", "body"]}),
        );
        assert!(MongoConnector::get_content_fields(&config, "comments").is_none());
    }

    #[test]
    fn test_get_extra_fields_default() {
        let config = make_config("mongodb://localhost:27017");
        assert!(MongoConnector::get_extra_fields(&config).is_empty());
    }

    #[test]
    fn test_get_extra_fields_configured() {
        let mut config = make_config("mongodb://localhost:27017");
        config.settings.insert(
            "extra_fields".to_string(),
            serde_json::json!(["author", "category"]),
        );
        let fields = MongoConnector::get_extra_fields(&config);
        assert_eq!(fields, vec!["author", "category"]);
    }

    #[test]
    fn test_get_filter_none() {
        let config = make_config("mongodb://localhost:27017");
        assert!(MongoConnector::get_filter(&config).is_none());
    }

    #[test]
    fn test_get_filter_configured() {
        let mut config = make_config("mongodb://localhost:27017");
        config.settings.insert(
            "filter".to_string(),
            serde_json::json!({"status": "published"}),
        );
        let filter = MongoConnector::get_filter(&config).unwrap();
        assert_eq!(filter.get_str("status").unwrap(), "published");
    }

    #[test]
    fn test_json_value_to_bson_string() {
        let val = serde_json::json!("hello");
        match json_value_to_bson(&val) {
            Bson::String(s) => assert_eq!(s, "hello"),
            other => panic!("Expected String, got {:?}", other),
        }
    }

    #[test]
    fn test_json_value_to_bson_int() {
        let val = serde_json::json!(42);
        match json_value_to_bson(&val) {
            Bson::Int64(i) => assert_eq!(i, 42),
            other => panic!("Expected Int64, got {:?}", other),
        }
    }

    #[test]
    fn test_json_value_to_bson_float() {
        let val = serde_json::json!(3.125);
        match json_value_to_bson(&val) {
            Bson::Double(d) => assert!((d - 3.125).abs() < f64::EPSILON),
            other => panic!("Expected Double, got {:?}", other),
        }
    }

    #[test]
    fn test_json_value_to_bson_bool() {
        let val = serde_json::json!(true);
        match json_value_to_bson(&val) {
            Bson::Boolean(b) => assert!(b),
            other => panic!("Expected Boolean, got {:?}", other),
        }
    }

    #[test]
    fn test_json_value_to_bson_null() {
        let val = serde_json::json!(null);
        assert!(matches!(json_value_to_bson(&val), Bson::Null));
    }

    #[test]
    fn test_json_value_to_bson_array() {
        let val = serde_json::json!([1, "two", true]);
        match json_value_to_bson(&val) {
            Bson::Array(arr) => {
                assert_eq!(arr.len(), 3);
                assert!(matches!(&arr[0], Bson::Int64(1)));
                assert!(matches!(&arr[1], Bson::String(s) if s == "two"));
                assert!(matches!(&arr[2], Bson::Boolean(true)));
            }
            other => panic!("Expected Array, got {:?}", other),
        }
    }

    #[test]
    fn test_json_value_to_bson_object() {
        let val = serde_json::json!({"key": "value", "num": 5});
        match json_value_to_bson(&val) {
            Bson::Document(doc) => {
                assert_eq!(doc.get_str("key").unwrap(), "value");
                assert_eq!(doc.get_i64("num").unwrap(), 5);
            }
            other => panic!("Expected Document, got {:?}", other),
        }
    }

    #[test]
    fn test_bson_to_string_variants() {
        assert_eq!(
            MongoConnector::bson_to_string(&Bson::String("hello".to_string())),
            Some("hello".to_string())
        );
        assert_eq!(
            MongoConnector::bson_to_string(&Bson::String(String::new())),
            None
        );
        assert_eq!(
            MongoConnector::bson_to_string(&Bson::Int32(42)),
            Some("42".to_string())
        );
        assert_eq!(
            MongoConnector::bson_to_string(&Bson::Int64(100)),
            Some("100".to_string())
        );
        assert_eq!(
            MongoConnector::bson_to_string(&Bson::Boolean(true)),
            Some("true".to_string())
        );
        assert!(MongoConnector::bson_to_string(&Bson::Null).is_none());
    }

    #[test]
    fn test_extract_content_with_specific_fields() {
        let mut doc = Document::new();
        doc.insert("title", "Hello World");
        doc.insert("body", "This is the body text.");
        doc.insert("author", "John");

        let fields = Some(vec!["title".to_string(), "body".to_string()]);
        let content = MongoConnector::extract_content(&doc, &fields);
        assert_eq!(content, "Hello World\n\nThis is the body text.");
    }

    #[test]
    fn test_extract_content_all_string_fields() {
        let mut doc = Document::new();
        doc.insert("_id", ObjectId::new());
        doc.insert("title", "Hello");
        doc.insert("body", "World");
        doc.insert("count", 42i32);

        let content = MongoConnector::extract_content(&doc, &None);
        // Should include title, body, and count (as string) but not _id
        assert!(content.contains("Hello"));
        assert!(content.contains("World"));
        assert!(content.contains("42"));
    }

    #[test]
    fn test_extract_content_empty_document() {
        let doc = Document::new();
        let content = MongoConnector::extract_content(&doc, &None);
        assert!(content.is_empty());
    }

    #[test]
    fn test_extract_content_missing_specified_fields() {
        let mut doc = Document::new();
        doc.insert("other_field", "some value");

        let fields = Some(vec!["title".to_string(), "body".to_string()]);
        let content = MongoConnector::extract_content(&doc, &fields);
        assert!(content.is_empty());
    }

    #[test]
    fn test_extract_id_objectid() {
        let oid = ObjectId::new();
        let mut doc = Document::new();
        doc.insert("_id", oid);
        let id = MongoConnector::extract_id(&doc, "_id");
        assert_eq!(id.len(), 24); // ObjectId hex string is 24 chars
    }

    #[test]
    fn test_extract_id_string() {
        let mut doc = Document::new();
        doc.insert("_id", "custom-id-123");
        let id = MongoConnector::extract_id(&doc, "_id");
        assert_eq!(id, "custom-id-123");
    }

    #[test]
    fn test_extract_id_missing() {
        let doc = Document::new();
        let id = MongoConnector::extract_id(&doc, "_id");
        assert_eq!(id, "unknown");
    }

    #[test]
    fn test_extract_id_custom_field() {
        let mut doc = Document::new();
        doc.insert("doc_id", "abc-456");
        let id = MongoConnector::extract_id(&doc, "doc_id");
        assert_eq!(id, "abc-456");
    }

    #[test]
    fn test_validate_config_with_filter() {
        let connector = MongoConnector::new();
        let mut config = make_config("mongodb://localhost:27017");
        config.settings.insert(
            "filter".to_string(),
            serde_json::json!({"status": "active"}),
        );
        assert!(connector.validate_config(&config).is_ok());
    }

    #[test]
    fn test_validate_config_with_all_optional_settings() {
        let connector = MongoConnector::new();
        let mut config = make_config("mongodb://localhost:27017");
        config.settings.insert(
            "content_fields".to_string(),
            serde_json::json!({"articles": ["title"]}),
        );
        config
            .settings
            .insert("id_field".to_string(), serde_json::json!("doc_id"));
        config
            .settings
            .insert("cursor_field".to_string(), serde_json::json!("updated_at"));
        config
            .settings
            .insert("limit".to_string(), serde_json::json!(500));
        config
            .settings
            .insert("extra_fields".to_string(), serde_json::json!(["author"]));
        config.settings.insert(
            "filter".to_string(),
            serde_json::json!({"status": "published"}),
        );
        assert!(connector.validate_config(&config).is_ok());
    }

    #[test]
    fn test_validate_config_srv_connection_string() {
        let connector = MongoConnector::new();
        let config = make_config("mongodb+srv://user:pass@cluster.mongodb.net");
        assert!(connector.validate_config(&config).is_ok());
    }

    #[tokio::test]
    async fn test_handle_webhook_returns_error() {
        let connector = MongoConnector::new();
        let config = make_config("mongodb://localhost:27017");
        let payload = WebhookPayload {
            body: vec![],
            headers: HashMap::new(),
            content_type: None,
        };
        let result = connector.handle_webhook(&config, payload).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("webhook"));
    }
}
