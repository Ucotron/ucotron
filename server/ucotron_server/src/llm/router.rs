//! LLM Router
//!
//! Provides intelligent routing of LLM requests to different providers
//! based on configured strategies: cheapest, fastest, quality, round-robin.
//! Tracks latency and success rate for intelligent routing decisions.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

use axum::{
    extract::State,
    http::HeaderMap,
    routing::{get, post},
    Extension, Json, Router,
};
use heed::{Database, types::SerdeBincode};
use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

use crate::auth::{require_namespace_access, require_role, AuthContext};
use crate::error::AppError;
use crate::llm::client::{ChatMessage, CompletionRequest, LLMClient, LLMResponse};
use crate::llm::costs::record_cost;
use crate::llm::registry::{decrypt_api_key, get_encryption_key, ProviderType};
use crate::llm::registry::LLMProvider;
use crate::state::AppState;
use crate::types::ApiErrorResponse;

type ProviderDb = Database<SerdeBincode<String>, SerdeBincode<LLMProvider>>;

const DEFAULT_P50_PERCENTILE: usize = 50;
const MAX_PROVIDER_RETRIES: usize = 3;

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
#[serde(rename_all = "lowercase")]
pub enum RoutingStrategy {
    Cheapest,
    Fastest,
    Quality,
    RoundRobin,
}

impl Default for RoutingStrategy {
    fn default() -> Self {
        Self::Cheapest
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct CompleteRequest {
    pub messages: Vec<ChatMessage>,
    #[serde(default)]
    pub model: String,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default = "default_max_tokens")]
    pub max_tokens: u32,
    #[serde(default)]
    pub strategy: RoutingStrategy,
    #[serde(default)]
    pub provider_id: Option<String>,
}

fn default_max_tokens() -> u32 {
    1024
}

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct CompleteResponse {
    pub response: LLMResponse,
    pub provider_id: String,
    pub provider_name: String,
    pub strategy_used: String,
    pub latency_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct ProviderMetrics {
    pub provider_id: String,
    pub total_requests: u64,
    pub successful_requests: u64,
    pub failed_requests: u64,
    pub success_rate: f64,
    pub avg_latency_ms: f64,
    pub p50_latency_ms: f64,
    pub p95_latency_ms: f64,
    pub p99_latency_ms: f64,
    pub total_cost: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct ProviderMetricsList {
    pub metrics: Vec<ProviderMetrics>,
}

#[derive(Debug, Clone)]
pub struct ProviderStats {
    pub total_requests: u64,
    pub successful_requests: u64,
    pub failed_requests: u64,
    pub latencies_ms: Vec<u64>,
    pub total_cost: f64,
}

impl Default for ProviderStats {
    fn default() -> Self {
        Self {
            total_requests: 0,
            successful_requests: 0,
            failed_requests: 0,
            latencies_ms: Vec::new(),
            total_cost: 0.0,
        }
    }
}

impl ProviderStats {
    pub fn success_rate(&self) -> f64 {
        if self.total_requests == 0 {
            return 1.0;
        }
        self.successful_requests as f64 / self.total_requests as f64
    }

    pub fn avg_latency_ms(&self) -> f64 {
        if self.latencies_ms.is_empty() {
            return 0.0;
        }
        let sum: u64 = self.latencies_ms.iter().sum();
        sum as f64 / self.latencies_ms.len() as f64
    }

    pub fn percentile_latency_ms(&self, percentile: usize) -> f64 {
        if self.latencies_ms.is_empty() {
            return 0.0;
        }
        let mut sorted = self.latencies_ms.clone();
        sorted.sort();
        let idx = (sorted.len() * percentile / 100).min(sorted.len() - 1);
        sorted[idx] as f64
    }

    pub fn record_success(&mut self, latency_ms: u64, cost: f64) {
        self.total_requests += 1;
        self.successful_requests += 1;
        self.latencies_ms.push(latency_ms);
        if self.latencies_ms.len() > 1000 {
            self.latencies_ms.remove(0);
        }
        self.total_cost += cost;
    }

    pub fn record_failure(&mut self, latency_ms: u64) {
        self.total_requests += 1;
        self.failed_requests += 1;
        self.latencies_ms.push(latency_ms);
        if self.latencies_ms.len() > 1000 {
            self.latencies_ms.remove(0);
        }
    }
}

pub fn extract_namespace(headers: &HeaderMap) -> String {
    headers
        .get("X-Ucotron-Namespace")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("default")
        .to_string()
}

fn get_provider_db_key(namespace: &str, provider_id: &str) -> String {
    format!("{}:{}", namespace, provider_id)
}

fn get_providers_from_state(
    state: &AppState,
    namespace: &str,
) -> Result<Vec<LLMProvider>, AppError> {
    let env = state.llm_providers_env.as_ref().ok_or_else(|| {
        AppError::internal("LLM providers environment not initialized")
    })?;

    let read_txn = env
        .read_txn()
        .map_err(|e| AppError::internal(format!("Failed to start LMDB read transaction: {}", e)))?;

    let db_name = "llm_providers";
    let db: ProviderDb = env
        .open_database(&read_txn, Some(db_name))
        .map_err(|e| AppError::internal(format!("Failed to open database: {}", e)))?
        .ok_or_else(|| AppError::internal("LLM providers database not found"))?;

    let prefix = format!("{}:", namespace);
    let mut providers = Vec::new();
    let iter = db
        .iter(&read_txn)
        .map_err(|e| AppError::internal(format!("LMDB iteration error: {}", e)))?;

    for result in iter {
        let (key, provider): (String, LLMProvider) = result.map_err(|e| AppError::internal(format!("LMDB read error: {}", e)))?;
        if key.starts_with(&prefix) {
            providers.push(provider);
        }
    }

    Ok(providers)
}

fn estimate_cost(input_tokens: u32, output_tokens: u32, provider: &LLMProvider) -> f64 {
    let input_cost = (input_tokens as f64 / 1000.0) * provider.pricing.input_per_1k_tokens;
    let output_cost = (output_tokens as f64 / 1000.0) * provider.pricing.output_per_1k_tokens;
    input_cost + output_cost
}

fn create_client_for_provider(
    provider: &LLMProvider,
    encryption_key: &[u8; 32],
) -> Result<Box<dyn LLMClient>, AppError> {
    let api_key = decrypt_api_key(&provider.api_key, encryption_key)
        .ok_or_else(|| AppError::internal("Failed to decrypt API key"))?;

    match provider.provider_type {
        ProviderType::OpenAI => Ok(Box::new(crate::llm::client::OpenAIClient::new(
            api_key,
            provider.api_base_url.clone(),
            provider.default_model.clone(),
        ))),
        ProviderType::Anthropic => Ok(Box::new(crate::llm::client::AnthropicClient::new(
            api_key,
            provider.default_model.clone(),
        ))),
        ProviderType::Fireworks => Ok(Box::new(crate::llm::client::FireworksClient::new(
            api_key,
            provider.api_base_url.clone(),
            provider.default_model.clone(),
        ))),
        ProviderType::Custom => Ok(Box::new(crate::llm::client::CustomClient::new(
            api_key,
            provider.api_base_url.clone(),
            provider.default_model.clone(),
        ))),
    }
}

fn select_provider_by_strategy(
    providers: &[LLMProvider],
    strategy: &RoutingStrategy,
    model: &str,
    stats: &HashMap<String, ProviderStats>,
    round_robin_index: &mut usize,
) -> Option<String> {
    if providers.is_empty() {
        return None;
    }

    match strategy {
        RoutingStrategy::Cheapest => {
            let mut cheapest: Option<(String, f64)> = None;
            for provider in providers {
                if !provider.supported_models.is_empty()
                    && !provider.supported_models.iter().any(|m| m == model)
                    && !model.is_empty()
                {
                    continue;
                }
                let cost = provider.pricing.input_per_1k_tokens
                    + provider.pricing.output_per_1k_tokens;
                match cheapest {
                    None => cheapest = Some((provider.id.clone(), cost)),
                    Some((_, current_cost)) if cost < current_cost => {
                        cheapest = Some((provider.id.clone(), cost))
                    }
                    _ => {}
                }
            }
            cheapest.map(|(id, _)| id)
        }
        RoutingStrategy::Fastest => {
            let mut fastest: Option<String> = None;
            let mut best_latency = f64::MAX;

            for provider in providers {
                if !provider.supported_models.is_empty()
                    && !provider.supported_models.iter().any(|m| m == model)
                    && !model.is_empty()
                {
                    continue;
                }

                if let Some(provider_stats) = stats.get(&provider.id) {
                    let latency = provider_stats.percentile_latency_ms(DEFAULT_P50_PERCENTILE);
                    if latency < best_latency && latency > 0.0 {
                        best_latency = latency;
                        fastest = Some(provider.id.clone());
                    }
                } else {
                    return Some(provider.id.clone());
                }
            }

            fastest.or_else(|| providers.first().map(|p| p.id.clone()))
        }
        RoutingStrategy::Quality => {
            let mut best: Option<String> = None;
            let mut best_score = f64::MIN;

            for provider in providers {
                if !provider.supported_models.is_empty()
                    && !provider.supported_models.iter().any(|m| m == model)
                    && !model.is_empty()
                {
                    continue;
                }

                let provider_stats = stats.get(&provider.id);
                let quality_score = if let Some(ps) = provider_stats {
                    let success_rate = ps.success_rate();
                    let avg_latency = ps.avg_latency_ms();
                    let latency_factor = if avg_latency > 0.0 {
                        1000.0 / avg_latency
                    } else {
                        1.0
                    };
                    success_rate * latency_factor
                } else {
                    1.0
                };

                if quality_score > best_score {
                    best_score = quality_score;
                    best = Some(provider.id.clone());
                }
            }

            best.or_else(|| providers.first().map(|p| p.id.clone()))
        }
        RoutingStrategy::RoundRobin => {
            let idx = *round_robin_index % providers.len();
            *round_robin_index += 1;
            providers.get(idx).map(|p| p.id.clone())
        }
    }
}

#[utoipa::path(
    post,
    path = "/api/v1/llm/complete",
    tag = "LLM Router",
    request_body = CompleteRequest,
    params(
        ("X-Ucotron-Namespace" = Option<String>, Header, description = "Namespace for multi-tenancy isolation")
    ),
    responses(
        (status = 200, description = "LLM completion response", body = CompleteResponse),
        (status = 400, description = "Invalid request", body = ApiErrorResponse),
        (status = 403, description = "Forbidden", body = ApiErrorResponse),
        (status = 500, description = "Internal error", body = ApiErrorResponse)
    )
)]
pub async fn complete_handler(
    State(state): State<Arc<AppState>>,
    Extension(auth): Extension<AuthContext>,
    headers: HeaderMap,
    Json(body): Json<CompleteRequest>,
) -> Result<Json<CompleteResponse>, AppError> {
    require_role(&auth, ucotron_config::AuthRole::Reader)?;
    let namespace = extract_namespace(&headers);
    require_namespace_access(&auth, &namespace)?;

    let providers = get_providers_from_state(state.as_ref(), &namespace)?;
    if providers.is_empty() {
        return Err(AppError::bad_request("No LLM providers configured"));
    }

    let encryption_key = get_encryption_key(&state.config);

    let selected_provider = if let Some(ref provider_id) = body.provider_id {
        providers
            .iter()
            .find(|p| p.id == *provider_id)
            .ok_or_else(|| AppError::not_found(format!("provider '{}' not found", provider_id)))?
    } else {
        let stats_guard = state.llm_provider_stats.read().map_err(|e| {
            AppError::internal(format!("Failed to read provider stats: {}", e))
        })?;
        let stats: HashMap<String, ProviderStats> = stats_guard.clone();
        drop(stats_guard);
        
        let mut round_robin_index = state.llm_round_robin_index.write().map_err(|e| {
            AppError::internal(format!("Failed to write round-robin index: {}", e))
        })?;

        let provider_id = select_provider_by_strategy(
            &providers,
            &body.strategy,
            &body.model,
            &stats,
            &mut round_robin_index,
        )
        .ok_or_else(|| AppError::internal("No suitable provider found"))?;

        providers
            .iter()
            .find(|p| p.id == provider_id)
            .ok_or_else(|| AppError::internal("Selected provider not found"))?
    };

    let client = create_client_for_provider(selected_provider, &encryption_key)?;

    let request = CompletionRequest {
        messages: body.messages,
        model: body.model.clone(),
        temperature: body.temperature,
        max_tokens: Some(body.max_tokens),
        stream: false,
    };

    let start = Instant::now();
    let result = client.complete(request).await;
    let latency = start.elapsed().as_millis() as u64;

    let mut stats = state.llm_provider_stats.write().map_err(|e| {
        AppError::internal(format!("Failed to write provider stats: {}", e))
    })?;

    let provider_stats = stats
        .entry(selected_provider.id.clone())
        .or_insert_with(ProviderStats::default);

    let response = match result {
        Ok(mut response) => {
            let cost = estimate_cost(
                response.input_tokens,
                response.output_tokens,
                selected_provider,
            );
            provider_stats.record_success(latency, cost);
            
            let _ = record_cost(
                state.as_ref(),
                &namespace,
                &selected_provider.id,
                &selected_provider.name,
                &response.model,
                response.input_tokens,
                response.output_tokens,
                cost,
            );
            
            response.provider = selected_provider.name.clone();
            response
        }
        Err(e) => {
            provider_stats.record_failure(latency);
            return Err(e);
        }
    };

    Ok(Json(CompleteResponse {
        response,
        provider_id: selected_provider.id.clone(),
        provider_name: selected_provider.name.clone(),
        strategy_used: format!("{:?}", body.strategy),
        latency_ms: latency,
    }))
}

#[utoipa::path(
    get,
    path = "/api/v1/llm/metrics",
    tag = "LLM Router",
    params(
        ("X-Ucotron-Namespace" = Option<String>, Header, description = "Namespace for multi-tenancy isolation")
    ),
    responses(
        (status = 200, description = "Provider metrics", body = ProviderMetricsList),
        (status = 403, description = "Forbidden", body = ApiErrorResponse)
    )
)]
pub async fn metrics_handler(
    State(state): State<Arc<AppState>>,
    Extension(auth): Extension<AuthContext>,
    headers: HeaderMap,
) -> Result<Json<ProviderMetricsList>, AppError> {
    require_role(&auth, ucotron_config::AuthRole::Reader)?;
    let namespace = extract_namespace(&headers);
    require_namespace_access(&auth, &namespace)?;

    let providers = get_providers_from_state(state.as_ref(), &namespace)?;
    let stats = state.llm_provider_stats.read().map_err(|e| {
        AppError::internal(format!("Failed to read provider stats: {}", e))
    })?;

    let metrics: Vec<ProviderMetrics> = providers
        .iter()
        .map(|p| {
            let provider_stats = stats.get(&p.id);
            let (total_requests, successful_requests, failed_requests, success_rate, avg_latency, p50, p95, p99, total_cost) = 
                if let Some(ps) = provider_stats {
                    (
                        ps.total_requests,
                        ps.successful_requests,
                        ps.failed_requests,
                        ps.success_rate(),
                        ps.avg_latency_ms(),
                        ps.percentile_latency_ms(50),
                        ps.percentile_latency_ms(95),
                        ps.percentile_latency_ms(99),
                        ps.total_cost,
                    )
                } else {
                    (0, 0, 0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0)
                };

            ProviderMetrics {
                provider_id: p.id.clone(),
                total_requests,
                successful_requests,
                failed_requests,
                success_rate,
                avg_latency_ms: avg_latency,
                p50_latency_ms: p50,
                p95_latency_ms: p95,
                p99_latency_ms: p99,
                total_cost,
            }
        })
        .collect();

    Ok(Json(ProviderMetricsList { metrics }))
}

pub async fn reset_metrics_handler(
    State(state): State<Arc<AppState>>,
    Extension(auth): Extension<AuthContext>,
    headers: HeaderMap,
) -> Result<axum::http::StatusCode, AppError> {
    require_role(&auth, ucotron_config::AuthRole::Admin)?;
    let namespace = extract_namespace(&headers);
    require_namespace_access(&auth, &namespace)?;

    let mut stats = state.llm_provider_stats.write().map_err(|e| {
        AppError::internal(format!("Failed to write provider stats: {}", e))
    })?;

    stats.clear();

    let mut index = state.llm_round_robin_index.write().map_err(|e| {
        AppError::internal(format!("Failed to write round-robin index: {}", e))
    })?;

    *index = 0;

    Ok(axum::http::StatusCode::NO_CONTENT)
}

pub fn router() -> Router<Arc<AppState>> {
    Router::new()
        .route("/api/v1/llm/complete", post(complete_handler))
        .route("/api/v1/llm/metrics", get(metrics_handler))
}
