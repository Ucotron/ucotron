//! # Reranker Pipeline
//!
//! Cross-encoder reranking for search results. Takes (query, document) pairs
//! and returns relevance scores, used as a second-stage refinement after
//! initial vector similarity retrieval.
//!
//! ## Implementations
//!
//! - [`SidecarReranker`] — Delegates to a Python sidecar running a cross-encoder
//!   model (e.g., Qwen3-VL-Reranker-2B).

use anyhow::{Context, Result};

/// Reranker pipeline trait for scoring (query, document) pairs.
///
/// Cross-encoders process the query and document jointly, producing more
/// accurate relevance scores than bi-encoder (embedding) similarity alone.
pub trait RerankerPipeline: Send + Sync {
    /// Score a list of documents against a query.
    ///
    /// Returns a score for each document (higher = more relevant).
    /// Scores are typically in [0, 1] after sigmoid normalization.
    fn rerank(&self, query: &str, documents: &[&str]) -> Result<Vec<f32>>;
}

/// Reranker that delegates to a Python sidecar service via HTTP.
///
/// The sidecar runs a cross-encoder model (e.g., Qwen3-VL-Reranker-2B)
/// and exposes a `/rerank` endpoint.
pub struct SidecarReranker {
    client: reqwest::blocking::Client,
    base_url: String,
    instruction: Option<String>,
}

impl SidecarReranker {
    /// Create a new sidecar reranker.
    ///
    /// # Arguments
    /// * `base_url` - Base URL of the sidecar (e.g., "http://localhost:8421")
    /// * `instruction` - Optional instruction prefix for the reranker
    pub fn new(base_url: &str, instruction: Option<String>) -> Result<Self> {
        let client = reqwest::blocking::Client::builder()
            .timeout(std::time::Duration::from_secs(60))
            .build()
            .context("Failed to create HTTP client")?;

        Ok(Self {
            client,
            base_url: base_url.trim_end_matches('/').to_string(),
            instruction,
        })
    }
}

impl RerankerPipeline for SidecarReranker {
    fn rerank(&self, query: &str, documents: &[&str]) -> Result<Vec<f32>> {
        if documents.is_empty() {
            return Ok(Vec::new());
        }

        let url = format!("{}/rerank", self.base_url);
        let body = serde_json::json!({
            "query": query,
            "documents": documents,
            "instruction": self.instruction,
        });

        let resp = self
            .client
            .post(&url)
            .json(&body)
            .send()
            .context("Sidecar rerank request failed")?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().unwrap_or_default();
            anyhow::bail!("Sidecar rerank returned {}: {}", status, body);
        }

        #[derive(serde::Deserialize)]
        struct RerankResponse {
            scores: Vec<f32>,
        }

        let parsed: RerankResponse = resp
            .json()
            .context("Failed to parse sidecar rerank response")?;
        Ok(parsed.scores)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ─── Unit tests (no sidecar required) ────────────────────────────────

    #[test]
    fn test_reranker_trait_is_object_safe() {
        // Verify the trait can be used as a trait object
        fn _accepts_dyn(_r: &dyn RerankerPipeline) {}
    }

    #[test]
    fn test_sidecar_reranker_creation() {
        let reranker = SidecarReranker::new("http://localhost:8421", None);
        assert!(reranker.is_ok());
    }

    #[test]
    fn test_sidecar_reranker_with_instruction() {
        let reranker = SidecarReranker::new(
            "http://localhost:8421",
            Some("Retrieve relevant passages".to_string()),
        );
        assert!(reranker.is_ok());
        let r = reranker.unwrap();
        assert_eq!(r.instruction.as_deref(), Some("Retrieve relevant passages"));
    }

    #[test]
    fn test_sidecar_reranker_empty_documents() {
        let reranker = SidecarReranker::new("http://localhost:99999", None).unwrap();
        // Empty documents should return empty scores without making a request
        let scores = reranker.rerank("test query", &[]);
        assert!(scores.is_ok());
        assert!(scores.unwrap().is_empty());
    }
}
