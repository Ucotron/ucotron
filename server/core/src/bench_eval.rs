//! # Evaluation Benchmark Harness
//!
//! Framework for evaluating Ucotron memory retrieval against external
//! benchmarks (LongMemEval, LoCoMo, etc.) with standard IR metrics.
//!
//! # Metrics
//!
//! - **Recall@k**: fraction of relevant items found in the top-k results
//! - **MRR** (Mean Reciprocal Rank): average of 1/rank of the first relevant result
//! - **NDCG@k** (Normalized Discounted Cumulative Gain): position-aware relevance
//! - **F1**: harmonic mean of precision and recall
//! - **Latency**: P50, P95, P99 of query execution time
//!
//! # Usage
//!
//! ```rust,no_run
//! use ucotron_core::bench_eval::*;
//!
//! // Load a dataset
//! let dataset = EvalDataset::from_jsonl("path/to/data.jsonl").unwrap();
//!
//! // Configure evaluation
//! let config = EvalConfig {
//!     k_values: vec![1, 5, 10],
//!     ..Default::default()
//! };
//!
//! // Run evaluation (provide a query function)
//! let report = Evaluator::new(config)
//!     .evaluate(&dataset, |query| {
//!         // Return ranked list of retrieved IDs for this query
//!         vec!["doc1".into(), "doc2".into()]
//!     })
//!     .unwrap();
//!
//! // Output results
//! println!("{}", report.to_markdown());
//! println!("{}", report.to_json().unwrap());
//! ```

use std::collections::HashMap;
use std::fs;
use std::io::{BufRead, BufReader};
use std::path::Path;
use std::time::Instant;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Dataset types
// ---------------------------------------------------------------------------

/// A single evaluation query with ground-truth relevant documents.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvalQuery {
    /// Unique query identifier.
    pub id: String,
    /// The query text to search with.
    pub query: String,
    /// IDs of documents/memories that are relevant to this query.
    pub relevant_ids: Vec<String>,
    /// Optional relevance grades per document (for graded relevance / NDCG).
    /// Key = document ID, value = relevance grade (higher = more relevant).
    #[serde(default)]
    pub relevance_grades: HashMap<String, f32>,
    /// Optional category or task type for per-category analysis.
    #[serde(default)]
    pub category: Option<String>,
    /// Optional metadata for extensibility.
    #[serde(default)]
    pub metadata: HashMap<String, serde_json::Value>,
}

/// A document/memory to be ingested before evaluation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvalDocument {
    /// Unique document identifier (used for matching against ground truth).
    pub id: String,
    /// The document text content.
    pub content: String,
    /// Optional metadata (timestamps, source, etc.).
    #[serde(default)]
    pub metadata: HashMap<String, serde_json::Value>,
}

/// Complete evaluation dataset with documents and queries.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvalDataset {
    /// Human-readable name of the dataset.
    pub name: String,
    /// Optional description.
    #[serde(default)]
    pub description: Option<String>,
    /// Documents to ingest before running queries.
    pub documents: Vec<EvalDocument>,
    /// Evaluation queries with ground-truth relevance.
    pub queries: Vec<EvalQuery>,
}

impl EvalDataset {
    /// Load a dataset from a JSON file.
    pub fn from_json<P: AsRef<Path>>(path: P) -> Result<Self> {
        let content = fs::read_to_string(path.as_ref())
            .with_context(|| format!("Failed to read dataset file: {}", path.as_ref().display()))?;
        let dataset: EvalDataset =
            serde_json::from_str(&content).with_context(|| "Failed to parse dataset JSON")?;
        Ok(dataset)
    }

    /// Load a dataset from a JSONL file where each line is either an
    /// `EvalDocument` (has "content" field) or an `EvalQuery` (has "query" field).
    ///
    /// First line may optionally be a metadata header with `{"name": "...", "description": "..."}`.
    pub fn from_jsonl<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file = fs::File::open(path.as_ref())
            .with_context(|| format!("Failed to open JSONL file: {}", path.as_ref().display()))?;
        let reader = BufReader::new(file);

        let mut name = path
            .as_ref()
            .file_stem()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_else(|| "unnamed".to_string());
        let mut description = None;
        let mut documents = Vec::new();
        let mut queries = Vec::new();

        for (line_num, line) in reader.lines().enumerate() {
            let line = line.with_context(|| format!("Failed to read line {}", line_num + 1))?;
            let line = line.trim();
            if line.is_empty() {
                continue;
            }

            let value: serde_json::Value = serde_json::from_str(line)
                .with_context(|| format!("Invalid JSON on line {}", line_num + 1))?;

            if value.get("query").is_some() {
                // This is a query entry
                let q: EvalQuery = serde_json::from_value(value)
                    .with_context(|| format!("Failed to parse query on line {}", line_num + 1))?;
                queries.push(q);
            } else if value.get("content").is_some() {
                // This is a document entry
                let d: EvalDocument = serde_json::from_value(value).with_context(|| {
                    format!("Failed to parse document on line {}", line_num + 1)
                })?;
                documents.push(d);
            } else if value.get("name").is_some() {
                // Metadata header
                if let Some(n) = value.get("name").and_then(|v| v.as_str()) {
                    name = n.to_string();
                }
                if let Some(d) = value.get("description").and_then(|v| v.as_str()) {
                    description = Some(d.to_string());
                }
            }
            // Skip unrecognized lines
        }

        Ok(EvalDataset {
            name,
            description,
            documents,
            queries,
        })
    }

    /// Create a dataset from in-memory documents and queries.
    pub fn new(
        name: impl Into<String>,
        documents: Vec<EvalDocument>,
        queries: Vec<EvalQuery>,
    ) -> Self {
        Self {
            name: name.into(),
            description: None,
            documents,
            queries,
        }
    }
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for the evaluation harness.
#[derive(Debug, Clone)]
pub struct EvalConfig {
    /// k values for Recall@k and NDCG@k (e.g., [1, 5, 10, 20]).
    pub k_values: Vec<usize>,
    /// RNG seed for reproducibility (used if sampling queries).
    pub seed: u64,
    /// Optional maximum number of queries to evaluate (None = all).
    pub max_queries: Option<usize>,
    /// Whether to compute per-category metrics.
    pub per_category: bool,
}

impl Default for EvalConfig {
    fn default() -> Self {
        Self {
            k_values: vec![1, 5, 10, 20],
            seed: 42,
            max_queries: None,
            per_category: true,
        }
    }
}

// ---------------------------------------------------------------------------
// Metrics computation
// ---------------------------------------------------------------------------

/// Result of a single query evaluation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryMetrics {
    /// Query ID.
    pub query_id: String,
    /// Recall@k for each configured k value.
    pub recall_at_k: Vec<(usize, f64)>,
    /// Reciprocal rank (1/rank of first relevant result, 0 if none found).
    pub reciprocal_rank: f64,
    /// NDCG@k for each configured k value.
    pub ndcg_at_k: Vec<(usize, f64)>,
    /// Precision = relevant_retrieved / total_retrieved.
    pub precision: f64,
    /// Recall = relevant_retrieved / total_relevant.
    pub recall: f64,
    /// F1 = 2 * precision * recall / (precision + recall).
    pub f1: f64,
    /// Query execution latency in microseconds.
    pub latency_us: u64,
    /// Number of results returned.
    pub num_results: usize,
    /// Optional category for grouping.
    pub category: Option<String>,
}

/// Aggregated metrics across all queries.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AggregateMetrics {
    /// Mean Recall@k for each k value.
    pub mean_recall_at_k: Vec<(usize, f64)>,
    /// Mean Reciprocal Rank across all queries.
    pub mrr: f64,
    /// Mean NDCG@k for each k value.
    pub mean_ndcg_at_k: Vec<(usize, f64)>,
    /// Mean precision.
    pub mean_precision: f64,
    /// Mean recall.
    pub mean_recall: f64,
    /// Mean F1.
    pub mean_f1: f64,
    /// Latency P50 in microseconds.
    pub latency_p50_us: u64,
    /// Latency P95 in microseconds.
    pub latency_p95_us: u64,
    /// Latency P99 in microseconds.
    pub latency_p99_us: u64,
    /// Mean latency in microseconds.
    pub latency_mean_us: u64,
    /// Total number of queries evaluated.
    pub num_queries: usize,
}

/// Complete evaluation report.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvalReport {
    /// Dataset name.
    pub dataset_name: String,
    /// Configuration used.
    pub k_values: Vec<usize>,
    /// Per-query metrics.
    pub query_metrics: Vec<QueryMetrics>,
    /// Aggregated metrics over all queries.
    pub aggregate: AggregateMetrics,
    /// Per-category aggregated metrics (category name → metrics).
    #[serde(default)]
    pub per_category: HashMap<String, AggregateMetrics>,
    /// Seed used for reproducibility.
    pub seed: u64,
}

impl EvalReport {
    /// Render the report as a Markdown table.
    pub fn to_markdown(&self) -> String {
        let mut md = String::new();

        md.push_str(&format!("# Evaluation Report: {}\n\n", self.dataset_name));
        md.push_str(&format!(
            "**Queries evaluated**: {}\n\n",
            self.aggregate.num_queries
        ));

        // Overall metrics table
        md.push_str("## Overall Metrics\n\n");
        md.push_str("| Metric | Value |\n");
        md.push_str("|--------|-------|\n");
        md.push_str(&format!("| MRR | {:.4} |\n", self.aggregate.mrr));
        md.push_str(&format!(
            "| Mean Precision | {:.4} |\n",
            self.aggregate.mean_precision
        ));
        md.push_str(&format!(
            "| Mean Recall | {:.4} |\n",
            self.aggregate.mean_recall
        ));
        md.push_str(&format!("| Mean F1 | {:.4} |\n", self.aggregate.mean_f1));
        md.push_str(&format!(
            "| Latency P50 | {:.2}ms |\n",
            self.aggregate.latency_p50_us as f64 / 1000.0
        ));
        md.push_str(&format!(
            "| Latency P95 | {:.2}ms |\n",
            self.aggregate.latency_p95_us as f64 / 1000.0
        ));
        md.push_str(&format!(
            "| Latency P99 | {:.2}ms |\n",
            self.aggregate.latency_p99_us as f64 / 1000.0
        ));
        md.push('\n');

        // Recall@k table
        md.push_str("## Recall@k\n\n");
        md.push_str("| k |  Recall  |\n");
        md.push_str("|---|----------|\n");
        for (k, recall) in &self.aggregate.mean_recall_at_k {
            md.push_str(&format!("| {} | {:.4} |\n", k, recall));
        }
        md.push('\n');

        // NDCG@k table
        md.push_str("## NDCG@k\n\n");
        md.push_str("| k |  NDCG  |\n");
        md.push_str("|---|--------|\n");
        for (k, ndcg) in &self.aggregate.mean_ndcg_at_k {
            md.push_str(&format!("| {} | {:.4} |\n", k, ndcg));
        }
        md.push('\n');

        // Per-category metrics
        if !self.per_category.is_empty() {
            md.push_str("## Per-Category Results\n\n");
            md.push_str("| Category | Queries | MRR | Recall@10 | NDCG@10 | F1 | P95 (ms) |\n");
            md.push_str("|----------|---------|-----|-----------|---------|----|---------|\n");

            let mut categories: Vec<_> = self.per_category.iter().collect();
            categories.sort_by(|(a, _), (b, _)| a.cmp(b));

            for (cat, metrics) in &categories {
                let recall_10 = metrics
                    .mean_recall_at_k
                    .iter()
                    .find(|(k, _)| *k == 10)
                    .map(|(_, v)| *v)
                    .unwrap_or(0.0);
                let ndcg_10 = metrics
                    .mean_ndcg_at_k
                    .iter()
                    .find(|(k, _)| *k == 10)
                    .map(|(_, v)| *v)
                    .unwrap_or(0.0);
                md.push_str(&format!(
                    "| {} | {} | {:.4} | {:.4} | {:.4} | {:.4} | {:.2} |\n",
                    cat,
                    metrics.num_queries,
                    metrics.mrr,
                    recall_10,
                    ndcg_10,
                    metrics.mean_f1,
                    metrics.latency_p95_us as f64 / 1000.0,
                ));
            }
            md.push('\n');
        }

        md
    }

    /// Serialize the report to JSON.
    pub fn to_json(&self) -> Result<String> {
        serde_json::to_string_pretty(self)
            .map_err(|e| anyhow::anyhow!("JSON serialization error: {}", e))
    }
}

// ---------------------------------------------------------------------------
// Metric computation functions
// ---------------------------------------------------------------------------

/// Compute Recall@k: fraction of relevant documents found in the top-k results.
pub fn recall_at_k(retrieved: &[String], relevant: &[String], k: usize) -> f64 {
    if relevant.is_empty() {
        return 0.0;
    }
    let top_k: std::collections::HashSet<&str> =
        retrieved.iter().take(k).map(|s| s.as_str()).collect();
    let num_relevant_found = relevant
        .iter()
        .filter(|r| top_k.contains(r.as_str()))
        .count();
    num_relevant_found as f64 / relevant.len() as f64
}

/// Compute Reciprocal Rank: 1/rank of the first relevant result (0 if none).
pub fn reciprocal_rank(retrieved: &[String], relevant: &[String]) -> f64 {
    let relevant_set: std::collections::HashSet<&str> =
        relevant.iter().map(|s| s.as_str()).collect();
    for (i, doc) in retrieved.iter().enumerate() {
        if relevant_set.contains(doc.as_str()) {
            return 1.0 / (i as f64 + 1.0);
        }
    }
    0.0
}

/// Compute NDCG@k (Normalized Discounted Cumulative Gain).
///
/// If `relevance_grades` is provided, uses graded relevance.
/// Otherwise, uses binary relevance (1.0 for relevant, 0.0 otherwise).
pub fn ndcg_at_k(
    retrieved: &[String],
    relevant: &[String],
    relevance_grades: &HashMap<String, f32>,
    k: usize,
) -> f64 {
    let dcg = dcg_at_k(retrieved, relevant, relevance_grades, k);
    let ideal_dcg = ideal_dcg_at_k(relevant, relevance_grades, k);

    if ideal_dcg == 0.0 {
        return 0.0;
    }
    dcg / ideal_dcg
}

/// Compute DCG@k (Discounted Cumulative Gain).
fn dcg_at_k(
    retrieved: &[String],
    relevant: &[String],
    relevance_grades: &HashMap<String, f32>,
    k: usize,
) -> f64 {
    let relevant_set: std::collections::HashSet<&str> =
        relevant.iter().map(|s| s.as_str()).collect();
    let mut dcg = 0.0;

    for (i, doc) in retrieved.iter().take(k).enumerate() {
        let rel = if let Some(&grade) = relevance_grades.get(doc) {
            grade as f64
        } else if relevant_set.contains(doc.as_str()) {
            1.0
        } else {
            0.0
        };
        // DCG formula: rel_i / log2(i + 2) (1-indexed position)
        dcg += rel / (i as f64 + 2.0).log2();
    }
    dcg
}

/// Compute ideal DCG@k (best possible DCG given the relevant set).
fn ideal_dcg_at_k(relevant: &[String], relevance_grades: &HashMap<String, f32>, k: usize) -> f64 {
    let mut grades: Vec<f64> = relevant
        .iter()
        .map(|id| {
            relevance_grades
                .get(id)
                .copied()
                .map(|g| g as f64)
                .unwrap_or(1.0)
        })
        .collect();
    // Sort descending for ideal ordering
    grades.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));

    let mut idcg = 0.0;
    for (i, &rel) in grades.iter().take(k).enumerate() {
        idcg += rel / (i as f64 + 2.0).log2();
    }
    idcg
}

/// Compute precision: relevant_retrieved / total_retrieved.
pub fn precision(retrieved: &[String], relevant: &[String]) -> f64 {
    if retrieved.is_empty() {
        return 0.0;
    }
    let relevant_set: std::collections::HashSet<&str> =
        relevant.iter().map(|s| s.as_str()).collect();
    let relevant_found = retrieved
        .iter()
        .filter(|r| relevant_set.contains(r.as_str()))
        .count();
    relevant_found as f64 / retrieved.len() as f64
}

/// Compute F1 score: harmonic mean of precision and recall.
pub fn f1_score(prec: f64, rec: f64) -> f64 {
    if prec + rec == 0.0 {
        return 0.0;
    }
    2.0 * prec * rec / (prec + rec)
}

/// Compute a latency percentile from a sorted slice of microsecond values.
pub fn percentile(sorted_latencies: &[u64], p: f64) -> u64 {
    if sorted_latencies.is_empty() {
        return 0;
    }
    let idx = ((p / 100.0) * (sorted_latencies.len() as f64 - 1.0)).round() as usize;
    let idx = idx.min(sorted_latencies.len() - 1);
    sorted_latencies[idx]
}

// ---------------------------------------------------------------------------
// Aggregation
// ---------------------------------------------------------------------------

/// Aggregate per-query metrics into summary statistics.
pub fn aggregate_metrics(query_metrics: &[QueryMetrics], k_values: &[usize]) -> AggregateMetrics {
    let n = query_metrics.len();
    if n == 0 {
        return AggregateMetrics {
            mean_recall_at_k: k_values.iter().map(|&k| (k, 0.0)).collect(),
            mrr: 0.0,
            mean_ndcg_at_k: k_values.iter().map(|&k| (k, 0.0)).collect(),
            mean_precision: 0.0,
            mean_recall: 0.0,
            mean_f1: 0.0,
            latency_p50_us: 0,
            latency_p95_us: 0,
            latency_p99_us: 0,
            latency_mean_us: 0,
            num_queries: 0,
        };
    }

    // Mean Recall@k
    let mean_recall_at_k: Vec<(usize, f64)> = k_values
        .iter()
        .map(|&k| {
            let sum: f64 = query_metrics
                .iter()
                .map(|qm| {
                    qm.recall_at_k
                        .iter()
                        .find(|(kv, _)| *kv == k)
                        .map(|(_, v)| *v)
                        .unwrap_or(0.0)
                })
                .sum();
            (k, sum / n as f64)
        })
        .collect();

    // MRR
    let mrr: f64 = query_metrics
        .iter()
        .map(|qm| qm.reciprocal_rank)
        .sum::<f64>()
        / n as f64;

    // Mean NDCG@k
    let mean_ndcg_at_k: Vec<(usize, f64)> = k_values
        .iter()
        .map(|&k| {
            let sum: f64 = query_metrics
                .iter()
                .map(|qm| {
                    qm.ndcg_at_k
                        .iter()
                        .find(|(kv, _)| *kv == k)
                        .map(|(_, v)| *v)
                        .unwrap_or(0.0)
                })
                .sum();
            (k, sum / n as f64)
        })
        .collect();

    // Mean precision, recall, F1
    let mean_precision = query_metrics.iter().map(|qm| qm.precision).sum::<f64>() / n as f64;
    let mean_recall = query_metrics.iter().map(|qm| qm.recall).sum::<f64>() / n as f64;
    let mean_f1 = query_metrics.iter().map(|qm| qm.f1).sum::<f64>() / n as f64;

    // Latency percentiles
    let mut latencies: Vec<u64> = query_metrics.iter().map(|qm| qm.latency_us).collect();
    latencies.sort_unstable();
    let latency_p50_us = percentile(&latencies, 50.0);
    let latency_p95_us = percentile(&latencies, 95.0);
    let latency_p99_us = percentile(&latencies, 99.0);
    let latency_mean_us = latencies.iter().sum::<u64>() / n as u64;

    AggregateMetrics {
        mean_recall_at_k,
        mrr,
        mean_ndcg_at_k,
        mean_precision,
        mean_recall,
        mean_f1,
        latency_p50_us,
        latency_p95_us,
        latency_p99_us,
        latency_mean_us,
        num_queries: n,
    }
}

// ---------------------------------------------------------------------------
// Evaluator
// ---------------------------------------------------------------------------

/// Evaluation harness that runs queries against a retrieval function
/// and computes standard IR metrics against ground truth.
pub struct Evaluator {
    config: EvalConfig,
}

impl Evaluator {
    /// Create a new evaluator with the given configuration.
    pub fn new(config: EvalConfig) -> Self {
        Self { config }
    }

    /// Evaluate a dataset by running each query through the provided function.
    ///
    /// The `query_fn` takes a query string and returns a ranked list of
    /// retrieved document IDs (most relevant first).
    pub fn evaluate<F>(&self, dataset: &EvalDataset, query_fn: F) -> Result<EvalReport>
    where
        F: Fn(&str) -> Vec<String>,
    {
        let queries = if let Some(max) = self.config.max_queries {
            &dataset.queries[..max.min(dataset.queries.len())]
        } else {
            &dataset.queries
        };

        let mut query_metrics = Vec::with_capacity(queries.len());

        for q in queries {
            let start = Instant::now();
            let retrieved = query_fn(&q.query);
            let latency_us = start.elapsed().as_micros() as u64;

            let recall_values: Vec<(usize, f64)> = self
                .config
                .k_values
                .iter()
                .map(|&k| (k, recall_at_k(&retrieved, &q.relevant_ids, k)))
                .collect();

            let rr = reciprocal_rank(&retrieved, &q.relevant_ids);

            let ndcg_values: Vec<(usize, f64)> = self
                .config
                .k_values
                .iter()
                .map(|&k| {
                    (
                        k,
                        ndcg_at_k(&retrieved, &q.relevant_ids, &q.relevance_grades, k),
                    )
                })
                .collect();

            let prec = precision(&retrieved, &q.relevant_ids);
            let rec = if q.relevant_ids.is_empty() {
                0.0
            } else {
                recall_at_k(&retrieved, &q.relevant_ids, retrieved.len())
            };
            let f1 = f1_score(prec, rec);

            query_metrics.push(QueryMetrics {
                query_id: q.id.clone(),
                recall_at_k: recall_values,
                reciprocal_rank: rr,
                ndcg_at_k: ndcg_values,
                precision: prec,
                recall: rec,
                f1,
                latency_us,
                num_results: retrieved.len(),
                category: q.category.clone(),
            });
        }

        // Aggregate overall
        let aggregate = aggregate_metrics(&query_metrics, &self.config.k_values);

        // Per-category aggregation
        let mut per_category = HashMap::new();
        if self.config.per_category {
            let mut by_category: HashMap<String, Vec<&QueryMetrics>> = HashMap::new();
            for qm in &query_metrics {
                if let Some(cat) = &qm.category {
                    by_category.entry(cat.clone()).or_default().push(qm);
                }
            }
            for (cat, metrics_refs) in by_category {
                let owned: Vec<QueryMetrics> = metrics_refs.into_iter().cloned().collect();
                per_category.insert(cat, aggregate_metrics(&owned, &self.config.k_values));
            }
        }

        Ok(EvalReport {
            dataset_name: dataset.name.clone(),
            k_values: self.config.k_values.clone(),
            query_metrics,
            aggregate,
            per_category,
            seed: self.config.seed,
        })
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_retrieved() -> Vec<String> {
        vec![
            "d3".into(),
            "d1".into(),
            "d5".into(),
            "d2".into(),
            "d7".into(),
        ]
    }

    fn sample_relevant() -> Vec<String> {
        vec!["d1".into(), "d2".into(), "d4".into()]
    }

    // -- Recall@k tests --

    #[test]
    fn test_recall_at_1() {
        let retrieved = sample_retrieved();
        let relevant = sample_relevant();
        // Top-1 is "d3" which is NOT relevant → 0/3
        assert_eq!(recall_at_k(&retrieved, &relevant, 1), 0.0);
    }

    #[test]
    fn test_recall_at_2() {
        let retrieved = sample_retrieved();
        let relevant = sample_relevant();
        // Top-2: "d3", "d1" → d1 is relevant → 1/3
        let r = recall_at_k(&retrieved, &relevant, 2);
        assert!((r - 1.0 / 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_recall_at_5() {
        let retrieved = sample_retrieved();
        let relevant = sample_relevant();
        // Top-5: d3, d1, d5, d2, d7 → d1, d2 relevant → 2/3
        let r = recall_at_k(&retrieved, &relevant, 5);
        assert!((r - 2.0 / 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_recall_empty_relevant() {
        let retrieved = sample_retrieved();
        assert_eq!(recall_at_k(&retrieved, &[], 5), 0.0);
    }

    #[test]
    fn test_recall_empty_retrieved() {
        let relevant = sample_relevant();
        assert_eq!(recall_at_k(&[], &relevant, 5), 0.0);
    }

    // -- Reciprocal Rank tests --

    #[test]
    fn test_reciprocal_rank_first() {
        let retrieved = vec!["d1".into(), "d2".into()];
        let relevant = vec!["d1".into()];
        assert_eq!(reciprocal_rank(&retrieved, &relevant), 1.0);
    }

    #[test]
    fn test_reciprocal_rank_second() {
        let retrieved = sample_retrieved();
        let relevant = sample_relevant();
        // First relevant is "d1" at position 1 (0-indexed) → RR = 1/2
        assert_eq!(reciprocal_rank(&retrieved, &relevant), 0.5);
    }

    #[test]
    fn test_reciprocal_rank_none() {
        let retrieved = vec!["d10".into(), "d11".into()];
        let relevant = vec!["d1".into()];
        assert_eq!(reciprocal_rank(&retrieved, &relevant), 0.0);
    }

    // -- NDCG tests --

    #[test]
    fn test_ndcg_perfect_ranking() {
        let retrieved = vec!["d1".into(), "d2".into(), "d3".into()];
        let relevant = vec!["d1".into(), "d2".into(), "d3".into()];
        let grades = HashMap::new();
        // Perfect ranking → NDCG = 1.0
        let n = ndcg_at_k(&retrieved, &relevant, &grades, 3);
        assert!((n - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_ndcg_no_relevant() {
        let retrieved = vec!["d10".into(), "d11".into()];
        let relevant = vec!["d1".into()];
        let grades = HashMap::new();
        assert_eq!(ndcg_at_k(&retrieved, &relevant, &grades, 2), 0.0);
    }

    #[test]
    fn test_ndcg_empty_relevant() {
        let retrieved = vec!["d1".into()];
        let relevant = vec![];
        let grades = HashMap::new();
        assert_eq!(ndcg_at_k(&retrieved, &relevant, &grades, 1), 0.0);
    }

    #[test]
    fn test_ndcg_graded_relevance() {
        // d1 has grade 3.0, d2 has grade 1.0
        let retrieved = vec!["d2".into(), "d1".into()]; // Suboptimal order
        let relevant = vec!["d1".into(), "d2".into()];
        let mut grades = HashMap::new();
        grades.insert("d1".to_string(), 3.0);
        grades.insert("d2".to_string(), 1.0);

        let n = ndcg_at_k(&retrieved, &relevant, &grades, 2);
        // Should be < 1.0 because d2 (grade 1) is before d1 (grade 3)
        assert!(n < 1.0);
        assert!(n > 0.0);
    }

    // -- Precision tests --

    #[test]
    fn test_precision_basic() {
        let retrieved = sample_retrieved();
        let relevant = sample_relevant();
        // 2 relevant out of 5 retrieved → precision = 2/5
        let p = precision(&retrieved, &relevant);
        assert!((p - 0.4).abs() < 1e-10);
    }

    #[test]
    fn test_precision_empty_retrieved() {
        assert_eq!(precision(&[], &sample_relevant()), 0.0);
    }

    // -- F1 tests --

    #[test]
    fn test_f1_score() {
        let f = f1_score(0.5, 0.5);
        assert!((f - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_f1_perfect() {
        let f = f1_score(1.0, 1.0);
        assert!((f - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_f1_zero() {
        assert_eq!(f1_score(0.0, 0.0), 0.0);
    }

    // -- Percentile tests --

    #[test]
    fn test_percentile_basic() {
        let latencies = vec![10, 20, 30, 40, 50, 60, 70, 80, 90, 100];
        // P50 of 10 elements: index = round(0.5 * 9) = 5 → latencies[5] = 60
        assert_eq!(percentile(&latencies, 50.0), 60);
        assert_eq!(percentile(&latencies, 0.0), 10);
        assert_eq!(percentile(&latencies, 100.0), 100);
        // P95: index = round(0.95 * 9) = round(8.55) = 9 → latencies[9] = 100
        assert_eq!(percentile(&latencies, 95.0), 100);
    }

    #[test]
    fn test_percentile_empty() {
        assert_eq!(percentile(&[], 50.0), 0);
    }

    #[test]
    fn test_percentile_single() {
        assert_eq!(percentile(&[42], 95.0), 42);
    }

    // -- Aggregation tests --

    #[test]
    fn test_aggregate_empty() {
        let agg = aggregate_metrics(&[], &[5, 10]);
        assert_eq!(agg.num_queries, 0);
        assert_eq!(agg.mrr, 0.0);
    }

    #[test]
    fn test_aggregate_basic() {
        let metrics = vec![
            QueryMetrics {
                query_id: "q1".into(),
                recall_at_k: vec![(5, 0.8), (10, 1.0)],
                reciprocal_rank: 1.0,
                ndcg_at_k: vec![(5, 0.9), (10, 0.95)],
                precision: 0.8,
                recall: 1.0,
                f1: 0.888,
                latency_us: 100,
                num_results: 5,
                category: Some("single-hop".into()),
            },
            QueryMetrics {
                query_id: "q2".into(),
                recall_at_k: vec![(5, 0.6), (10, 0.8)],
                reciprocal_rank: 0.5,
                ndcg_at_k: vec![(5, 0.7), (10, 0.85)],
                precision: 0.6,
                recall: 0.8,
                f1: 0.685,
                latency_us: 200,
                num_results: 5,
                category: Some("multi-hop".into()),
            },
        ];
        let agg = aggregate_metrics(&metrics, &[5, 10]);
        assert_eq!(agg.num_queries, 2);
        assert!((agg.mrr - 0.75).abs() < 1e-10);
        assert!((agg.mean_recall_at_k[0].1 - 0.7).abs() < 1e-10); // (5, 0.7)
        assert!((agg.mean_recall_at_k[1].1 - 0.9).abs() < 1e-10); // (10, 0.9)
    }

    // -- Evaluator tests --

    #[test]
    fn test_evaluator_basic() {
        let dataset = EvalDataset::new(
            "test",
            vec![
                EvalDocument {
                    id: "d1".into(),
                    content: "Hello world".into(),
                    metadata: HashMap::new(),
                },
                EvalDocument {
                    id: "d2".into(),
                    content: "Goodbye world".into(),
                    metadata: HashMap::new(),
                },
            ],
            vec![EvalQuery {
                id: "q1".into(),
                query: "hello".into(),
                relevant_ids: vec!["d1".into()],
                relevance_grades: HashMap::new(),
                category: None,
                metadata: HashMap::new(),
            }],
        );

        let config = EvalConfig {
            k_values: vec![1, 5],
            ..Default::default()
        };

        let report = Evaluator::new(config)
            .evaluate(&dataset, |_query| vec!["d1".into(), "d2".into()])
            .unwrap();

        assert_eq!(report.aggregate.num_queries, 1);
        assert!((report.aggregate.mrr - 1.0).abs() < 1e-10);
        // Recall@1 = 1.0 (d1 found at position 0)
        assert!((report.aggregate.mean_recall_at_k[0].1 - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_evaluator_per_category() {
        let dataset = EvalDataset::new(
            "categorized",
            vec![],
            vec![
                EvalQuery {
                    id: "q1".into(),
                    query: "query1".into(),
                    relevant_ids: vec!["d1".into()],
                    relevance_grades: HashMap::new(),
                    category: Some("easy".into()),
                    metadata: HashMap::new(),
                },
                EvalQuery {
                    id: "q2".into(),
                    query: "query2".into(),
                    relevant_ids: vec!["d5".into()],
                    relevance_grades: HashMap::new(),
                    category: Some("hard".into()),
                    metadata: HashMap::new(),
                },
            ],
        );

        let config = EvalConfig {
            k_values: vec![5],
            per_category: true,
            ..Default::default()
        };

        let report = Evaluator::new(config)
            .evaluate(&dataset, |query| {
                if query == "query1" {
                    vec!["d1".into()]
                } else {
                    vec!["d99".into()]
                }
            })
            .unwrap();

        assert_eq!(report.per_category.len(), 2);
        assert!(report.per_category.contains_key("easy"));
        assert!(report.per_category.contains_key("hard"));

        let easy = &report.per_category["easy"];
        assert!((easy.mrr - 1.0).abs() < 1e-10); // Perfect retrieval
        let hard = &report.per_category["hard"];
        assert!((hard.mrr - 0.0).abs() < 1e-10); // No match
    }

    #[test]
    fn test_evaluator_max_queries() {
        let dataset = EvalDataset::new(
            "large",
            vec![],
            vec![
                EvalQuery {
                    id: "q1".into(),
                    query: "a".into(),
                    relevant_ids: vec!["d1".into()],
                    relevance_grades: HashMap::new(),
                    category: None,
                    metadata: HashMap::new(),
                },
                EvalQuery {
                    id: "q2".into(),
                    query: "b".into(),
                    relevant_ids: vec!["d2".into()],
                    relevance_grades: HashMap::new(),
                    category: None,
                    metadata: HashMap::new(),
                },
                EvalQuery {
                    id: "q3".into(),
                    query: "c".into(),
                    relevant_ids: vec!["d3".into()],
                    relevance_grades: HashMap::new(),
                    category: None,
                    metadata: HashMap::new(),
                },
            ],
        );

        let config = EvalConfig {
            k_values: vec![5],
            max_queries: Some(2),
            ..Default::default()
        };

        let report = Evaluator::new(config)
            .evaluate(&dataset, |_| vec![])
            .unwrap();

        assert_eq!(report.aggregate.num_queries, 2);
    }

    // -- Markdown output tests --

    #[test]
    fn test_report_to_markdown() {
        let report = EvalReport {
            dataset_name: "test".into(),
            k_values: vec![5, 10],
            query_metrics: vec![],
            aggregate: AggregateMetrics {
                mean_recall_at_k: vec![(5, 0.8), (10, 0.9)],
                mrr: 0.75,
                mean_ndcg_at_k: vec![(5, 0.85), (10, 0.92)],
                mean_precision: 0.7,
                mean_recall: 0.8,
                mean_f1: 0.74,
                latency_p50_us: 5000,
                latency_p95_us: 15000,
                latency_p99_us: 25000,
                latency_mean_us: 8000,
                num_queries: 100,
            },
            per_category: HashMap::new(),
            seed: 42,
        };

        let md = report.to_markdown();
        assert!(md.contains("# Evaluation Report: test"));
        assert!(md.contains("MRR"));
        assert!(md.contains("0.7500"));
        assert!(md.contains("Recall@k"));
        assert!(md.contains("NDCG@k"));
    }

    // -- JSON output test --

    #[test]
    fn test_report_to_json() {
        let report = EvalReport {
            dataset_name: "test".into(),
            k_values: vec![5],
            query_metrics: vec![],
            aggregate: AggregateMetrics {
                mean_recall_at_k: vec![(5, 0.8)],
                mrr: 0.75,
                mean_ndcg_at_k: vec![(5, 0.85)],
                mean_precision: 0.7,
                mean_recall: 0.8,
                mean_f1: 0.74,
                latency_p50_us: 5000,
                latency_p95_us: 15000,
                latency_p99_us: 25000,
                latency_mean_us: 8000,
                num_queries: 100,
            },
            per_category: HashMap::new(),
            seed: 42,
        };

        let json = report.to_json().unwrap();
        assert!(json.contains("\"dataset_name\": \"test\""));
        assert!(json.contains("\"mrr\": 0.75"));
    }

    // -- Dataset loading tests --

    #[test]
    fn test_dataset_new() {
        let ds = EvalDataset::new(
            "my_dataset",
            vec![EvalDocument {
                id: "d1".into(),
                content: "test".into(),
                metadata: HashMap::new(),
            }],
            vec![EvalQuery {
                id: "q1".into(),
                query: "test query".into(),
                relevant_ids: vec!["d1".into()],
                relevance_grades: HashMap::new(),
                category: None,
                metadata: HashMap::new(),
            }],
        );
        assert_eq!(ds.name, "my_dataset");
        assert_eq!(ds.documents.len(), 1);
        assert_eq!(ds.queries.len(), 1);
    }

    #[test]
    fn test_dataset_from_json() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("dataset.json");
        let data = serde_json::json!({
            "name": "test_dataset",
            "documents": [
                {"id": "d1", "content": "Hello world"},
                {"id": "d2", "content": "Goodbye world"}
            ],
            "queries": [
                {"id": "q1", "query": "hello", "relevant_ids": ["d1"]}
            ]
        });
        std::fs::write(&path, serde_json::to_string(&data).unwrap()).unwrap();

        let ds = EvalDataset::from_json(&path).unwrap();
        assert_eq!(ds.name, "test_dataset");
        assert_eq!(ds.documents.len(), 2);
        assert_eq!(ds.queries.len(), 1);
        assert_eq!(ds.queries[0].relevant_ids, vec!["d1"]);
    }

    #[test]
    fn test_dataset_from_jsonl() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("dataset.jsonl");
        let lines = [
            r#"{"name": "jsonl_test", "description": "A test dataset"}"#,
            r#"{"id": "d1", "content": "Hello world"}"#,
            r#"{"id": "d2", "content": "Goodbye"}"#,
            r#"{"id": "q1", "query": "hello", "relevant_ids": ["d1"]}"#,
            r#"{"id": "q2", "query": "bye", "relevant_ids": ["d2"], "category": "greetings"}"#,
        ];
        std::fs::write(&path, lines.join("\n")).unwrap();

        let ds = EvalDataset::from_jsonl(&path).unwrap();
        assert_eq!(ds.name, "jsonl_test");
        assert_eq!(ds.description.as_deref(), Some("A test dataset"));
        assert_eq!(ds.documents.len(), 2);
        assert_eq!(ds.queries.len(), 2);
        assert_eq!(ds.queries[1].category.as_deref(), Some("greetings"));
    }

    #[test]
    fn test_dataset_from_json_missing_file() {
        let result = EvalDataset::from_json("/nonexistent/path.json");
        assert!(result.is_err());
    }

    #[test]
    fn test_ndcg_reversed_order() {
        // Perfect: d1 (grade 3), d2 (grade 2), d3 (grade 1)
        // Reversed: d3 (grade 1), d2 (grade 2), d1 (grade 3)
        let retrieved = vec!["d3".into(), "d2".into(), "d1".into()];
        let relevant = vec!["d1".into(), "d2".into(), "d3".into()];
        let mut grades = HashMap::new();
        grades.insert("d1".to_string(), 3.0);
        grades.insert("d2".to_string(), 2.0);
        grades.insert("d3".to_string(), 1.0);

        let n_reversed = ndcg_at_k(&retrieved, &relevant, &grades, 3);
        // Now perfect order
        let retrieved_perfect = vec!["d1".into(), "d2".into(), "d3".into()];
        let n_perfect = ndcg_at_k(&retrieved_perfect, &relevant, &grades, 3);

        assert!((n_perfect - 1.0).abs() < 1e-10);
        assert!(n_reversed < n_perfect);
    }
}
