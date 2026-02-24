//! LLM Cost Estimation and Tracking
//!
//! Tracks LLM usage costs per namespace with filtering capabilities.
//! Stores cost records in LMDB for persistent storage.

use std::sync::Arc;

use axum::{
    extract::{Query, State},
    http::HeaderMap,
    routing::get,
    Extension, Json, Router,
};
use chrono::{DateTime, Datelike, NaiveDate, Utc};
use heed::{Database, types::SerdeBincode};
use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

use crate::auth::{require_namespace_access, require_role, AuthContext};
use crate::error::AppError;
use crate::llm::registry::LLMProvider;
use crate::state::AppState;
use crate::types::ApiErrorResponse;

const LLM_COSTS_DB_NAME: &str = "llm_costs";

type CostDb = Database<SerdeBincode<String>, SerdeBincode<CostRecord>>;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CostRecord {
    pub id: String,
    pub namespace: String,
    pub provider_id: String,
    pub provider_name: String,
    pub model: String,
    pub input_tokens: u32,
    pub output_tokens: u32,
    pub estimated_cost: f64,
    pub actual_cost: f64,
    #[serde(with = "chrono::serde::ts_seconds")]
    pub timestamp: DateTime<Utc>,
}

impl CostRecord {
    pub fn to_response(&self) -> CostRecordResponse {
        CostRecordResponse {
            id: self.id.clone(),
            namespace: self.namespace.clone(),
            provider_id: self.provider_id.clone(),
            provider_name: self.provider_name.clone(),
            model: self.model.clone(),
            input_tokens: self.input_tokens,
            output_tokens: self.output_tokens,
            estimated_cost: self.estimated_cost,
            actual_cost: self.actual_cost,
            timestamp: self.timestamp.to_rfc3339(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct CostRecordResponse {
    pub id: String,
    pub namespace: String,
    pub provider_id: String,
    pub provider_name: String,
    pub model: String,
    pub input_tokens: u32,
    pub output_tokens: u32,
    pub estimated_cost: f64,
    pub actual_cost: f64,
    pub timestamp: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct CostListResponse {
    pub costs: Vec<CostRecordResponse>,
    pub total_count: usize,
    pub total_estimated_cost: f64,
    pub total_actual_cost: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct CostSummary {
    pub period: String,
    pub total_cost: f64,
    pub total_input_tokens: u64,
    pub total_output_tokens: u64,
    pub request_count: u64,
    pub by_provider: Vec<ProviderCostSummary>,
}

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct ProviderCostSummary {
    pub provider_id: String,
    pub provider_name: String,
    pub total_cost: f64,
    pub total_input_tokens: u64,
    pub total_output_tokens: u64,
    pub request_count: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema, utoipa::IntoParams)]
#[serde(rename_all = "camelCase")]
pub struct CostFilters {
    #[serde(default)]
    pub provider_id: Option<String>,
    #[serde(default)]
    pub model: Option<String>,
    #[serde(default)]
    pub start_date: Option<String>,
    #[serde(default)]
    pub end_date: Option<String>,
    #[serde(default = "default_limit")]
    pub limit: usize,
    #[serde(default)]
    pub offset: usize,
}

fn default_limit() -> usize {
    100
}

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema, utoipa::IntoParams)]
#[serde(rename_all = "camelCase")]
pub struct SummaryQuery {
    #[serde(default = "default_period")]
    pub period: String,
    #[serde(default)]
    pub year: Option<i32>,
    #[serde(default)]
    pub month: Option<u32>,
}

fn default_period() -> String {
    "monthly".to_string()
}

fn extract_namespace(headers: &HeaderMap) -> String {
    headers
        .get("X-Ucotron-Namespace")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("default")
        .to_string()
}

fn get_cost_db(state: &AppState) -> Option<CostDb> {
    let env = state.llm_costs_env.as_ref()?;
    let read_txn = env.read_txn().ok()?;
    match env.open_database(&read_txn, Some(LLM_COSTS_DB_NAME)) {
        Ok(Some(db)) => Some(db),
        _ => {
            drop(read_txn);
            let mut write_txn = env.write_txn().ok()?;
            match env.create_database(&mut write_txn, Some(LLM_COSTS_DB_NAME)) {
                Ok(db) => {
                    write_txn.commit().ok()?;
                    Some(db)
                }
                Err(_) => None,
            }
        }
    }
}

fn generate_cost_id() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    format!("cost_{}", timestamp)
}

pub fn estimate_cost(input_tokens: u32, output_tokens: u32, provider: &LLMProvider) -> f64 {
    let input_cost = (input_tokens as f64 / 1000.0) * provider.pricing.input_per_1k_tokens;
    let output_cost = (output_tokens as f64 / 1000.0) * provider.pricing.output_per_1k_tokens;
    input_cost + output_cost
}

pub fn record_cost(
    state: &AppState,
    namespace: &str,
    provider_id: &str,
    provider_name: &str,
    model: &str,
    input_tokens: u32,
    output_tokens: u32,
    estimated_cost: f64,
) -> Result<(), AppError> {
    let _actual_cost = if let Some(db) = get_cost_db(state) {
        let env = state.llm_costs_env.as_ref().unwrap();
        
        let provider = get_provider_by_id(state, namespace, provider_id)?;
        let actual = estimate_cost(input_tokens, output_tokens, &provider);
        
        let record = CostRecord {
            id: generate_cost_id(),
            namespace: namespace.to_string(),
            provider_id: provider_id.to_string(),
            provider_name: provider_name.to_string(),
            model: model.to_string(),
            input_tokens,
            output_tokens,
            estimated_cost,
            actual_cost: actual,
            timestamp: Utc::now(),
        };

        let db_key = format!("{}:{}:{}", namespace, provider_id, record.id);
        
        let mut write_txn = env.write_txn().map_err(|e| {
            AppError::internal(format!("Failed to start LMDB transaction: {}", e))
        })?;
        
        db.put(&mut write_txn, &db_key, &record).map_err(|e| {
            AppError::internal(format!("LMDB write error: {}", e))
        })?;
        write_txn.commit().map_err(|e| {
            AppError::internal(format!("LMDB commit error: {}", e))
        })?;
        
        actual
    } else {
        estimated_cost
    };
    
    Ok(())
}

fn get_provider_by_id(
    state: &AppState,
    namespace: &str,
    provider_id: &str,
) -> Result<LLMProvider, AppError> {
    let env = state.llm_providers_env.as_ref().ok_or_else(|| {
        AppError::internal("LLM providers environment not initialized")
    })?;

    let read_txn = env
        .read_txn()
        .map_err(|e| AppError::internal(format!("Failed to start LMDB read transaction: {}", e)))?;

    let db_name = "llm_providers";
    let db: Database<SerdeBincode<String>, SerdeBincode<LLMProvider>> = env
        .open_database(&read_txn, Some(db_name))
        .map_err(|e| AppError::internal(format!("Failed to open database: {}", e)))?
        .ok_or_else(|| AppError::internal("LLM providers database not found"))?;

    let db_key = format!("{}:{}", namespace, provider_id);
    let provider = db
        .get(&read_txn, &db_key)
        .map_err(|e| AppError::internal(format!("LMDB read error: {}", e)))?
        .ok_or_else(|| AppError::not_found(format!("provider '{}' not found", provider_id)))?;

    Ok(provider)
}

fn parse_date(date_str: &str) -> Result<DateTime<Utc>, AppError> {
    let date = NaiveDate::parse_from_str(date_str, "%Y-%m-%d")
        .map_err(|_| AppError::bad_request("Invalid date format. Use YYYY-MM-DD"))?;
    let datetime = date.and_hms_opt(0, 0, 0).unwrap();
    Ok(DateTime::<Utc>::from_naive_utc_and_offset(datetime, Utc))
}

#[utoipa::path(
    get,
    path = "/api/v1/llm/costs",
    tag = "LLM Costs",
    params(
        ("X-Ucotron-Namespace" = Option<String>, Header, description = "Namespace for multi-tenancy isolation"),
        CostFilters
    ),
    responses(
        (status = 200, description = "List of cost records", body = CostListResponse),
        (status = 403, description = "Forbidden", body = ApiErrorResponse)
    )
)]
pub async fn list_costs_handler(
    State(state): State<Arc<AppState>>,
    Extension(auth): Extension<AuthContext>,
    headers: HeaderMap,
    Query(filters): Query<CostFilters>,
) -> Result<Json<CostListResponse>, AppError> {
    require_role(&auth, ucotron_config::AuthRole::Reader)?;
    let namespace = extract_namespace(&headers);
    require_namespace_access(&auth, &namespace)?;

    let start_date = filters
        .start_date
        .as_ref()
        .map(|d| parse_date(d))
        .transpose()?;
    let end_date = filters
        .end_date
        .as_ref()
        .map(|d| parse_date(d))
        .transpose()?;

    let mut costs = Vec::new();
    let mut total_estimated = 0.0;
    let mut total_actual = 0.0;

    if let Some(db) = get_cost_db(&state) {
        let env = state.llm_costs_env.as_ref().unwrap();
        let read_txn = env.read_txn().map_err(|e| {
            AppError::internal(format!("Failed to start LMDB read transaction: {}", e))
        })?;

        let prefix = format!("{}:", namespace);
        let iter = db.iter(&read_txn).map_err(|e| {
            AppError::internal(format!("LMDB iteration error: {}", e))
        })?;

        for result in iter {
            let (key, record) = result.map_err(|e| {
                AppError::internal(format!("LMDB read error: {}", e))
            })?;

            if !key.starts_with(&prefix) {
                continue;
            }

            if let Some(ref provider_id) = filters.provider_id {
                if &record.provider_id != provider_id {
                    continue;
                }
            }

            if let Some(ref model) = filters.model {
                if &record.model != model {
                    continue;
                }
            }

            if let Some(start) = start_date {
                if record.timestamp < start {
                    continue;
                }
            }

            if let Some(end) = end_date {
                if record.timestamp > end {
                    continue;
                }
            }

            total_estimated += record.estimated_cost;
            total_actual += record.actual_cost;
            costs.push(record);
        }
    }

    costs.sort_by(|a, b| b.timestamp.cmp(&a.timestamp));

    let total_count = costs.len();
    let total_estimated_cost = total_estimated;
    let total_actual_cost = total_actual;

    let paginated_costs: Vec<CostRecordResponse> = costs
        .into_iter()
        .skip(filters.offset)
        .take(filters.limit)
        .map(|c| c.to_response())
        .collect();

    Ok(Json(CostListResponse {
        costs: paginated_costs,
        total_count,
        total_estimated_cost,
        total_actual_cost,
    }))
}

#[utoipa::path(
    get,
    path = "/api/v1/llm/costs/summary",
    tag = "LLM Costs",
    params(
        ("X-Ucotron-Namespace" = Option<String>, Header, description = "Namespace for multi-tenancy isolation"),
        SummaryQuery
    ),
    responses(
        (status = 200, description = "Cost summary", body = CostSummary),
        (status = 403, description = "Forbidden", body = ApiErrorResponse)
    )
)]
pub async fn cost_summary_handler(
    State(state): State<Arc<AppState>>,
    Extension(auth): Extension<AuthContext>,
    headers: HeaderMap,
    Query(query): Query<SummaryQuery>,
) -> Result<Json<CostSummary>, AppError> {
    require_role(&auth, ucotron_config::AuthRole::Reader)?;
    let namespace = extract_namespace(&headers);
    require_namespace_access(&auth, &namespace)?;

    let now = Utc::now();
    let current_year = now.year();
    let current_month = now.month();
    let year = query.year.unwrap_or(current_year);
    let month = query.month.unwrap_or(current_month);

    let (period_start, period_end, period_label) = if query.period == "daily" {
        let start = NaiveDate::from_ymd_opt(year, month as u32, 1)
            .ok_or_else(|| AppError::bad_request("Invalid date"))?
            .and_hms_opt(0, 0, 0)
            .unwrap()
            .and_utc();
        let end = if month == 12 {
            NaiveDate::from_ymd_opt(year + 1, 1, 1)
                .ok_or_else(|| AppError::bad_request("Invalid date"))?
                .and_hms_opt(0, 0, 0)
                .unwrap()
                .and_utc()
        } else {
            NaiveDate::from_ymd_opt(year, month + 1, 1)
                .ok_or_else(|| AppError::bad_request("Invalid date"))?
                .and_hms_opt(0, 0, 0)
                .unwrap()
                .and_utc()
        };
        (start, end, format!("{}-{:02}", year, month))
    } else {
        let start = NaiveDate::from_ymd_opt(year, 1, 1)
            .ok_or_else(|| AppError::bad_request("Invalid date"))?
            .and_hms_opt(0, 0, 0)
            .unwrap()
            .and_utc();
        let end = NaiveDate::from_ymd_opt(year + 1, 1, 1)
            .ok_or_else(|| AppError::bad_request("Invalid date"))?
            .and_hms_opt(0, 0, 0)
            .unwrap()
            .and_utc();
        (start, end, format!("{}", year))
    };

    let mut total_cost = 0.0;
    let mut total_input_tokens: u64 = 0;
    let mut total_output_tokens: u64 = 0;
    let mut request_count: u64 = 0;
    let mut provider_costs: std::collections::HashMap<String, (String, f64, u64, u64, u64)> =
        std::collections::HashMap::new();

    if let Some(db) = get_cost_db(&state) {
        let env = state.llm_costs_env.as_ref().unwrap();
        let read_txn = env.read_txn().map_err(|e| {
            AppError::internal(format!("Failed to start LMDB read transaction: {}", e))
        })?;

        let prefix = format!("{}:", namespace);
        let iter = db.iter(&read_txn).map_err(|e| {
            AppError::internal(format!("LMDB iteration error: {}", e))
        })?;

        for result in iter {
            let (key, record) = result.map_err(|e| {
                AppError::internal(format!("LMDB read error: {}", e))
            })?;

            if !key.starts_with(&prefix) {
                continue;
            }

            if record.timestamp < period_start || record.timestamp >= period_end {
                continue;
            }

            total_cost += record.actual_cost;
            total_input_tokens += record.input_tokens as u64;
            total_output_tokens += record.output_tokens as u64;
            request_count += 1;

            let entry = provider_costs
                .entry(record.provider_id.clone())
                .or_insert((
                    record.provider_name.clone(),
                    0.0,
                    0,
                    0,
                    0,
                ));
            entry.1 += record.actual_cost;
            entry.2 += record.input_tokens as u64;
            entry.3 += record.output_tokens as u64;
            entry.4 += 1;
        }
    }

    let by_provider: Vec<ProviderCostSummary> = provider_costs
        .into_iter()
        .map(|(provider_id, (provider_name, cost, input, output, count))| {
            ProviderCostSummary {
                provider_id,
                provider_name,
                total_cost: cost,
                total_input_tokens: input,
                total_output_tokens: output,
                request_count: count,
            }
        })
        .collect();

    Ok(Json(CostSummary {
        period: period_label,
        total_cost,
        total_input_tokens,
        total_output_tokens,
        request_count,
        by_provider,
    }))
}

pub fn router() -> Router<Arc<AppState>> {
    Router::new()
        .route("/api/v1/llm/costs", get(list_costs_handler))
        .route("/api/v1/llm/costs/summary", get(cost_summary_handler))
}
