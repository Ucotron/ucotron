//! LLM Provider Registry
//!
//! Manages LLM provider configurations with CRUD operations.
//! API keys are encrypted at rest using AES-256-GCM.
//! Uses LMDB for persistent storage when available, with in-memory fallback.

use std::sync::Arc;

use aes_gcm::{
    aead::{Aead, KeyInit},
    Aes256Gcm, Nonce,
};
use axum::{
    extract::{Path, State},
    http::HeaderMap,
    routing::{delete, get, post, put},
    Extension, Json, Router,
};
use heed::{types::SerdeBincode, Database};
use rand::Rng;
use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

use crate::auth::{require_namespace_access, require_role, AuthContext};
use crate::error::AppError;
use crate::state::AppState;
use crate::types::ApiErrorResponse;

const NONCE_SIZE: usize = 12;
const ENCRYPTION_KEY_SIZE: usize = 32;
const LLM_PROVIDERS_DB_NAME: &str = "llm_providers";

type ProviderDb = Database<SerdeBincode<String>, SerdeBincode<LLMProvider>>;

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
#[serde(rename_all = "lowercase")]
pub enum ProviderType {
    OpenAI,
    Anthropic,
    Fireworks,
    Custom,
}

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct Pricing {
    pub input_per_1k_tokens: f64,
    pub output_per_1k_tokens: f64,
}

impl Default for Pricing {
    fn default() -> Self {
        Self {
            input_per_1k_tokens: 0.0,
            output_per_1k_tokens: 0.0,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct LLMProvider {
    pub id: String,
    pub name: String,
    pub provider_type: ProviderType,
    pub api_base_url: String,
    #[serde(skip_serializing)]
    pub api_key: Vec<u8>,
    pub default_model: String,
    pub supported_models: Vec<String>,
    pub pricing: Pricing,
}

impl LLMProvider {
    pub fn masked_api_key(&self) -> String {
        if self.api_key.is_empty() {
            "".to_string()
        } else {
            "******".to_string()
        }
    }
}

#[derive(Debug, Deserialize, ToSchema)]
pub struct CreateProviderRequest {
    pub id: String,
    pub name: String,
    pub provider_type: ProviderType,
    pub api_base_url: String,
    pub api_key: String,
    pub default_model: String,
    pub supported_models: Vec<String>,
    pub pricing: Pricing,
}

#[derive(Debug, Deserialize, ToSchema)]
pub struct UpdateProviderRequest {
    pub name: Option<String>,
    pub api_base_url: Option<String>,
    pub api_key: Option<String>,
    pub default_model: Option<String>,
    pub supported_models: Option<Vec<String>>,
    pub pricing: Option<Pricing>,
}

#[derive(Debug, Serialize, ToSchema)]
pub struct ProviderResponse {
    pub id: String,
    pub name: String,
    pub provider_type: ProviderType,
    pub api_base_url: String,
    pub api_key_masked: String,
    pub default_model: String,
    pub supported_models: Vec<String>,
    pub pricing: Pricing,
}

impl From<&LLMProvider> for ProviderResponse {
    fn from(p: &LLMProvider) -> Self {
        Self {
            id: p.id.clone(),
            name: p.name.clone(),
            provider_type: p.provider_type.clone(),
            api_base_url: p.api_base_url.clone(),
            api_key_masked: p.masked_api_key(),
            default_model: p.default_model.clone(),
            supported_models: p.supported_models.clone(),
            pricing: p.pricing.clone(),
        }
    }
}

#[derive(Debug, Serialize, ToSchema)]
pub struct ProviderListResponse {
    pub providers: Vec<ProviderResponse>,
}

pub fn encrypt_api_key(key: &str, encryption_key: &[u8; ENCRYPTION_KEY_SIZE]) -> Vec<u8> {
    let cipher = Aes256Gcm::new_from_slice(encryption_key).expect("Invalid key length");
    let mut nonce_bytes = [0u8; NONCE_SIZE];
    rand::thread_rng().fill(&mut nonce_bytes);
    let nonce = Nonce::from_slice(&nonce_bytes);

    let ciphertext = cipher
        .encrypt(nonce, key.as_bytes())
        .expect("Encryption failed");

    let mut result = nonce_bytes.to_vec();
    result.extend(ciphertext);
    result
}

pub fn decrypt_api_key(
    encrypted: &[u8],
    encryption_key: &[u8; ENCRYPTION_KEY_SIZE],
) -> Option<String> {
    if encrypted.len() < NONCE_SIZE {
        return None;
    }

    let cipher = Aes256Gcm::new_from_slice(encryption_key).ok()?;
    let nonce = Nonce::from_slice(&encrypted[..NONCE_SIZE]);
    let ciphertext = &encrypted[NONCE_SIZE..];

    let plaintext = cipher.decrypt(nonce, ciphertext).ok()?;
    String::from_utf8(plaintext).ok()
}

pub(crate) fn get_encryption_key(
    config: &ucotron_config::UcotronConfig,
) -> [u8; ENCRYPTION_KEY_SIZE] {
    let mut key = [0u8; ENCRYPTION_KEY_SIZE];
    let key_source = config
        .auth
        .api_key
        .as_ref()
        .cloned()
        .unwrap_or_else(|| "default-ucotron-encryption-key-change-me".to_string());
    let key_bytes = key_source.as_bytes();
    for (i, byte) in key_bytes.iter().enumerate() {
        key[i % ENCRYPTION_KEY_SIZE] ^= byte;
    }
    key
}

fn extract_namespace(headers: &HeaderMap) -> String {
    headers
        .get("X-Ucotron-Namespace")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("default")
        .to_string()
}

fn get_provider_db(state: &AppState) -> Option<ProviderDb> {
    let env = state.llm_providers_env.as_ref()?;
    let read_txn = env.read_txn().ok()?;
    match env.open_database(&read_txn, Some(LLM_PROVIDERS_DB_NAME)) {
        Ok(Some(db)) => Some(db),
        _ => {
            drop(read_txn);
            let mut write_txn = env.write_txn().ok()?;
            match env.create_database(&mut write_txn, Some(LLM_PROVIDERS_DB_NAME)) {
                Ok(db) => {
                    write_txn.commit().ok()?;
                    Some(db)
                }
                Err(_) => None,
            }
        }
    }
}

#[utoipa::path(
    post,
    path = "/api/v1/llm/providers",
    tag = "LLM Providers",
    request_body = CreateProviderRequest,
    params(
        ("X-Ucotron-Namespace" = Option<String>, Header, description = "Namespace for multi-tenancy isolation")
    ),
    responses(
        (status = 201, description = "Provider created", body = ProviderResponse),
        (status = 400, description = "Invalid request", body = ApiErrorResponse),
        (status = 403, description = "Forbidden", body = ApiErrorResponse),
        (status = 500, description = "Internal error", body = ApiErrorResponse)
    )
)]
pub async fn create_provider_handler(
    State(state): State<Arc<AppState>>,
    Extension(auth): Extension<AuthContext>,
    headers: HeaderMap,
    Json(body): Json<CreateProviderRequest>,
) -> Result<(axum::http::StatusCode, Json<ProviderResponse>), AppError> {
    require_role(&auth, ucotron_config::AuthRole::Admin)?;
    let namespace = extract_namespace(&headers);
    require_namespace_access(&auth, &namespace)?;

    if body.id.is_empty() {
        return Err(AppError::bad_request("provider id must not be empty"));
    }
    if body.api_key.is_empty() {
        return Err(AppError::bad_request("api_key must not be empty"));
    }

    let encryption_key = get_encryption_key(&state.config);
    let encrypted_key = encrypt_api_key(&body.api_key, &encryption_key);

    let provider = LLMProvider {
        id: body.id.clone(),
        name: body.name.clone(),
        provider_type: body.provider_type,
        api_base_url: body.api_base_url.clone(),
        api_key: encrypted_key,
        default_model: body.default_model.clone(),
        supported_models: body.supported_models.clone(),
        pricing: body.pricing,
    };

    // Try LMDB first, fall back to in-memory
    let db_key = format!("{}:{}", namespace, provider.id);

    if let Some(db) = get_provider_db(&state) {
        let env = state.llm_providers_env.as_ref().unwrap();
        let mut write_txn = env
            .write_txn()
            .map_err(|e| AppError::internal(format!("Failed to start LMDB transaction: {}", e)))?;

        if db
            .get(&write_txn, &db_key)
            .map_err(|e| AppError::internal(format!("LMDB read error: {}", e)))?
            .is_some()
        {
            return Err(AppError::bad_request(format!(
                "provider with id '{}' already exists",
                provider.id
            )));
        }

        db.put(&mut write_txn, &db_key, &provider)
            .map_err(|e| AppError::internal(format!("LMDB write error: {}", e)))?;
        write_txn
            .commit()
            .map_err(|e| AppError::internal(format!("LMDB commit error: {}", e)))?;

        let response = ProviderResponse::from(&provider);
        return Ok((axum::http::StatusCode::CREATED, Json(response)));
    }

    // Fall back to in-memory storage
    let mut providers = state
        .llm_providers
        .write()
        .map_err(|e| AppError::internal(format!("Failed to acquire lock on providers: {}", e)))?;

    if providers.iter().any(|p| p.id == provider.id) {
        return Err(AppError::bad_request(format!(
            "provider with id '{}' already exists",
            provider.id
        )));
    }

    providers.push(provider);

    let response = ProviderResponse::from(
        providers
            .iter()
            .find(|p| p.id == body.id)
            .expect("just inserted"),
    );

    Ok((axum::http::StatusCode::CREATED, Json(response)))
}

#[utoipa::path(
    get,
    path = "/api/v1/llm/providers",
    tag = "LLM Providers",
    params(
        ("X-Ucotron-Namespace" = Option<String>, Header, description = "Namespace for multi-tenancy isolation")
    ),
    responses(
        (status = 200, description = "List of providers", body = ProviderListResponse),
        (status = 403, description = "Forbidden", body = ApiErrorResponse)
    )
)]
pub async fn list_providers_handler(
    State(state): State<Arc<AppState>>,
    Extension(auth): Extension<AuthContext>,
    headers: HeaderMap,
) -> Result<Json<ProviderListResponse>, AppError> {
    require_role(&auth, ucotron_config::AuthRole::Reader)?;
    let namespace = extract_namespace(&headers);
    require_namespace_access(&auth, &namespace)?;

    // Try LMDB first, fall back to in-memory
    if let Some(db) = get_provider_db(&state) {
        let env = state.llm_providers_env.as_ref().unwrap();
        let read_txn = env.read_txn().map_err(|e| {
            AppError::internal(format!("Failed to start LMDB read transaction: {}", e))
        })?;

        let prefix = format!("{}:", namespace);
        let mut providers = Vec::new();
        let iter = db
            .iter(&read_txn)
            .map_err(|e| AppError::internal(format!("LMDB iteration error: {}", e)))?;

        for result in iter {
            let (key, provider) =
                result.map_err(|e| AppError::internal(format!("LMDB read error: {}", e)))?;
            if key.starts_with(&prefix) {
                providers.push(ProviderResponse::from(&provider));
            }
        }

        return Ok(Json(ProviderListResponse { providers }));
    }

    // Fall back to in-memory storage
    let providers = state
        .llm_providers
        .read()
        .map_err(|e| AppError::internal(format!("Failed to acquire lock on providers: {}", e)))?;

    let response = ProviderListResponse {
        providers: providers.iter().map(ProviderResponse::from).collect(),
    };

    Ok(Json(response))
}

#[utoipa::path(
    get,
    path = "/api/v1/llm/providers/{id}",
    tag = "LLM Providers",
    params(
        ("id" = String, Path, description = "Provider ID"),
        ("X-Ucotron-Namespace" = Option<String>, Header, description = "Namespace for multi-tenancy isolation")
    ),
    responses(
        (status = 200, description = "Provider details", body = ProviderResponse),
        (status = 404, description = "Provider not found", body = ApiErrorResponse),
        (status = 403, description = "Forbidden", body = ApiErrorResponse)
    )
)]
pub async fn get_provider_handler(
    State(state): State<Arc<AppState>>,
    Extension(auth): Extension<AuthContext>,
    Path(id): Path<String>,
    headers: HeaderMap,
) -> Result<Json<ProviderResponse>, AppError> {
    require_role(&auth, ucotron_config::AuthRole::Reader)?;
    let namespace = extract_namespace(&headers);
    require_namespace_access(&auth, &namespace)?;

    // Try LMDB first, fall back to in-memory
    let db_key = format!("{}:{}", namespace, id);

    if let Some(db) = get_provider_db(&state) {
        let env = state.llm_providers_env.as_ref().unwrap();
        let read_txn = env.read_txn().map_err(|e| {
            AppError::internal(format!("Failed to start LMDB read transaction: {}", e))
        })?;

        let provider = db
            .get(&read_txn, &db_key)
            .map_err(|e| AppError::internal(format!("LMDB read error: {}", e)))?
            .ok_or_else(|| AppError::not_found(format!("provider '{}' not found", id)))?;

        return Ok(Json(ProviderResponse::from(&provider)));
    }

    // Fall back to in-memory storage
    let providers = state
        .llm_providers
        .read()
        .map_err(|e| AppError::internal(format!("Failed to acquire lock on providers: {}", e)))?;

    let provider = providers
        .iter()
        .find(|p| p.id == id)
        .ok_or_else(|| AppError::not_found(format!("provider '{}' not found", id)))?;

    Ok(Json(ProviderResponse::from(provider)))
}

#[utoipa::path(
    put,
    path = "/api/v1/llm/providers/{id}",
    tag = "LLM Providers",
    request_body = UpdateProviderRequest,
    params(
        ("id" = String, Path, description = "Provider ID"),
        ("X-Ucotron-Namespace" = Option<String>, Header, description = "Namespace for multi-tenancy isolation")
    ),
    responses(
        (status = 200, description = "Provider updated", body = ProviderResponse),
        (status = 400, description = "Invalid request", body = ApiErrorResponse),
        (status = 403, description = "Forbidden", body = ApiErrorResponse),
        (status = 404, description = "Provider not found", body = ApiErrorResponse)
    )
)]
pub async fn update_provider_handler(
    State(state): State<Arc<AppState>>,
    Extension(auth): Extension<AuthContext>,
    Path(id): Path<String>,
    headers: HeaderMap,
    Json(body): Json<UpdateProviderRequest>,
) -> Result<Json<ProviderResponse>, AppError> {
    require_role(&auth, ucotron_config::AuthRole::Admin)?;
    let namespace = extract_namespace(&headers);
    require_namespace_access(&auth, &namespace)?;

    let encryption_key = get_encryption_key(&state.config);
    let db_key = format!("{}:{}", namespace, id);

    // Try LMDB first, fall back to in-memory
    if let Some(db) = get_provider_db(&state) {
        let env = state.llm_providers_env.as_ref().unwrap();
        let mut write_txn = env
            .write_txn()
            .map_err(|e| AppError::internal(format!("Failed to start LMDB transaction: {}", e)))?;

        let mut provider = db
            .get(&write_txn, &db_key)
            .map_err(|e| AppError::internal(format!("LMDB read error: {}", e)))?
            .ok_or_else(|| AppError::not_found(format!("provider '{}' not found", id)))?;

        if let Some(name) = body.name {
            provider.name = name;
        }
        if let Some(api_base_url) = body.api_base_url {
            provider.api_base_url = api_base_url;
        }
        if let Some(api_key) = body.api_key {
            if api_key.is_empty() {
                return Err(AppError::bad_request("api_key must not be empty"));
            }
            provider.api_key = encrypt_api_key(&api_key, &encryption_key);
        }
        if let Some(default_model) = body.default_model {
            provider.default_model = default_model;
        }
        if let Some(supported_models) = body.supported_models {
            provider.supported_models = supported_models;
        }
        if let Some(pricing) = body.pricing {
            provider.pricing = pricing;
        }

        db.put(&mut write_txn, &db_key, &provider)
            .map_err(|e| AppError::internal(format!("LMDB write error: {}", e)))?;
        write_txn
            .commit()
            .map_err(|e| AppError::internal(format!("LMDB commit error: {}", e)))?;

        return Ok(Json(ProviderResponse::from(&provider)));
    }

    // Fall back to in-memory storage
    let mut providers = state
        .llm_providers
        .write()
        .map_err(|e| AppError::internal(format!("Failed to acquire lock on providers: {}", e)))?;

    let provider = providers
        .iter_mut()
        .find(|p| p.id == id)
        .ok_or_else(|| AppError::not_found(format!("provider '{}' not found", id)))?;

    if let Some(name) = body.name {
        provider.name = name;
    }
    if let Some(api_base_url) = body.api_base_url {
        provider.api_base_url = api_base_url;
    }
    if let Some(api_key) = body.api_key {
        if api_key.is_empty() {
            return Err(AppError::bad_request("api_key must not be empty"));
        }
        provider.api_key = encrypt_api_key(&api_key, &encryption_key);
    }
    if let Some(default_model) = body.default_model {
        provider.default_model = default_model;
    }
    if let Some(supported_models) = body.supported_models {
        provider.supported_models = supported_models;
    }
    if let Some(pricing) = body.pricing {
        provider.pricing = pricing;
    }

    Ok(Json(ProviderResponse::from(&*provider)))
}

#[utoipa::path(
    delete,
    path = "/api/v1/llm/providers/{id}",
    tag = "LLM Providers",
    params(
        ("id" = String, Path, description = "Provider ID"),
        ("X-Ucotron-Namespace" = Option<String>, Header, description = "Namespace for multi-tenancy isolation")
    ),
    responses(
        (status = 204, description = "Provider deleted"),
        (status = 403, description = "Forbidden", body = ApiErrorResponse),
        (status = 404, description = "Provider not found", body = ApiErrorResponse)
    )
)]
pub async fn delete_provider_handler(
    State(state): State<Arc<AppState>>,
    Extension(auth): Extension<AuthContext>,
    Path(id): Path<String>,
    headers: HeaderMap,
) -> Result<axum::http::StatusCode, AppError> {
    require_role(&auth, ucotron_config::AuthRole::Admin)?;
    let namespace = extract_namespace(&headers);
    require_namespace_access(&auth, &namespace)?;

    let db_key = format!("{}:{}", namespace, id);

    // Try LMDB first, fall back to in-memory
    if let Some(db) = get_provider_db(&state) {
        let env = state.llm_providers_env.as_ref().unwrap();
        let mut write_txn = env
            .write_txn()
            .map_err(|e| AppError::internal(format!("Failed to start LMDB transaction: {}", e)))?;

        let existed = db
            .get(&write_txn, &db_key)
            .map_err(|e| AppError::internal(format!("LMDB read error: {}", e)))?
            .is_some();

        if !existed {
            return Err(AppError::not_found(format!("provider '{}' not found", id)));
        }

        db.delete(&mut write_txn, &db_key)
            .map_err(|e| AppError::internal(format!("LMDB delete error: {}", e)))?;
        write_txn
            .commit()
            .map_err(|e| AppError::internal(format!("LMDB commit error: {}", e)))?;

        return Ok(axum::http::StatusCode::NO_CONTENT);
    }

    // Fall back to in-memory storage
    let mut providers = state
        .llm_providers
        .write()
        .map_err(|e| AppError::internal(format!("Failed to acquire lock on providers: {}", e)))?;

    let initial_len = providers.len();
    providers.retain(|p| p.id != id);

    if providers.len() == initial_len {
        return Err(AppError::not_found(format!("provider '{}' not found", id)));
    }

    Ok(axum::http::StatusCode::NO_CONTENT)
}

pub fn router() -> Router<Arc<AppState>> {
    Router::new()
        .route("/api/v1/llm/providers", post(create_provider_handler))
        .route("/api/v1/llm/providers", get(list_providers_handler))
        .route("/api/v1/llm/providers/{id}", get(get_provider_handler))
        .route("/api/v1/llm/providers/{id}", put(update_provider_handler))
        .route(
            "/api/v1/llm/providers/{id}",
            delete(delete_provider_handler),
        )
}
