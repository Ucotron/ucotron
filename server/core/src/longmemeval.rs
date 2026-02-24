//! # LongMemEval Benchmark (ICLR 2025)
//!
//! Adapter and runner for the LongMemEval long-term conversational memory benchmark.
//! Evaluates 5 core memory abilities across 7 question subtypes:
//!
//! - **Information Extraction (IE)**: Recall facts from single sessions
//! - **Multi-Session Reasoning (MR)**: Synthesize across sessions
//! - **Knowledge Updates (KU)**: Recognize changed information
//! - **Temporal Reasoning (TR)**: Answer time-dependent queries
//! - **Abstention (ABS)**: Refuse unanswerable questions
//!
//! # Architecture
//!
//! 1. Parse LongMemEval JSON into [`LongMemEvalDataset`]
//! 2. Convert to [`EvalDataset`](crate::bench_eval::EvalDataset) (session-level granularity)
//! 3. Evaluate retrieval via [`Evaluator`](crate::bench_eval::Evaluator)
//! 4. Compare against published baselines
//!
//! # Usage
//!
//! ```rust,no_run
//! use ucotron_core::longmemeval::*;
//! use ucotron_core::bench_eval::*;
//!
//! // Load dataset
//! let dataset = LongMemEvalDataset::from_file("longmemeval_oracle.json").unwrap();
//!
//! // Convert to eval format (session-level)
//! let eval_ds = dataset.to_eval_dataset();
//!
//! // Run evaluation
//! let config = EvalConfig { k_values: vec![1, 5, 10], ..Default::default() };
//! let report = Evaluator::new(config)
//!     .evaluate(&eval_ds, |query| {
//!         // Your retrieval function here
//!         vec![]
//!     })
//!     .unwrap();
//!
//! // Extended report with baselines
//! let extended = LongMemEvalReport::new(report);
//! println!("{}", extended.to_markdown());
//! ```

use std::collections::HashMap;
use std::fs;
use std::path::Path;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

use crate::bench_eval::{EvalConfig, EvalDataset, EvalDocument, EvalQuery, EvalReport, Evaluator};

// ---------------------------------------------------------------------------
// LongMemEval raw data types (matching the HuggingFace JSON schema)
// ---------------------------------------------------------------------------

/// A single turn in a conversation session.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionTurn {
    /// Speaker role: "user" or "assistant".
    pub role: String,
    /// The turn content text.
    pub content: String,
    /// Whether this turn contains evidence for answering a question.
    #[serde(default)]
    pub has_answer: bool,
}

/// A single evaluation instance from the LongMemEval dataset.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LongMemEvalQuestion {
    /// Unique question identifier.
    pub question_id: String,
    /// Question type/subtype (e.g., "single-session-user", "multi-session", etc.).
    pub question_type: String,
    /// The question text.
    pub question: String,
    /// Ground-truth short answer.
    pub answer: String,
    /// Date the question is asked (YYYY-MM-DD).
    #[serde(default)]
    pub question_date: Option<String>,
    /// Session IDs in the haystack (all sessions the model sees).
    #[serde(default)]
    pub haystack_session_ids: Vec<String>,
    /// Dates corresponding to each haystack session.
    #[serde(default)]
    pub haystack_dates: Vec<String>,
    /// The actual conversation sessions.
    #[serde(default)]
    pub haystack_sessions: Vec<Vec<SessionTurn>>,
    /// Ground-truth session IDs that contain the answer evidence.
    #[serde(default)]
    pub answer_session_ids: Vec<String>,
}

/// The complete LongMemEval dataset parsed from JSON.
#[derive(Debug, Clone)]
pub struct LongMemEvalDataset {
    /// All evaluation instances.
    pub questions: Vec<LongMemEvalQuestion>,
    /// Dataset variant name (oracle, small, medium).
    pub variant: String,
}

// ---------------------------------------------------------------------------
// Question type classification
// ---------------------------------------------------------------------------

/// The 5 core memory ability categories in LongMemEval.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MemoryAbility {
    /// Information Extraction from single sessions.
    InformationExtraction,
    /// Multi-Session Reasoning across sessions.
    MultiSessionReasoning,
    /// Knowledge Update detection.
    KnowledgeUpdate,
    /// Temporal Reasoning over time-dependent data.
    TemporalReasoning,
    /// Abstention on unanswerable questions.
    Abstention,
}

impl MemoryAbility {
    /// Classify a question_type string into the core memory ability.
    pub fn from_question_type(qt: &str) -> Self {
        let qt_lower = qt.to_lowercase();
        if qt_lower.contains("_abs") || qt_lower == "abstention" {
            MemoryAbility::Abstention
        } else if qt_lower.contains("multi-session") || qt_lower.contains("multi_session") {
            MemoryAbility::MultiSessionReasoning
        } else if qt_lower.contains("knowledge-update") || qt_lower.contains("knowledge_update") {
            MemoryAbility::KnowledgeUpdate
        } else if qt_lower.contains("temporal") {
            MemoryAbility::TemporalReasoning
        } else {
            // single-session-user, single-session-assistant, single-session-preference
            MemoryAbility::InformationExtraction
        }
    }

    /// Human-readable label for reports.
    pub fn label(&self) -> &'static str {
        match self {
            MemoryAbility::InformationExtraction => "Information Extraction",
            MemoryAbility::MultiSessionReasoning => "Multi-Session Reasoning",
            MemoryAbility::KnowledgeUpdate => "Knowledge Update",
            MemoryAbility::TemporalReasoning => "Temporal Reasoning",
            MemoryAbility::Abstention => "Abstention",
        }
    }
}

// ---------------------------------------------------------------------------
// Document granularity for conversion
// ---------------------------------------------------------------------------

/// Controls how conversation sessions are converted to documents.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum DocumentGranularity {
    /// One document per session (concatenated turns). Best overall performance
    /// per the LongMemEval paper.
    #[default]
    Session,
    /// One document per turn (user or assistant message). Finer-grained but
    /// noisier retrieval.
    Turn,
}

// ---------------------------------------------------------------------------
// Dataset loading and conversion
// ---------------------------------------------------------------------------

impl LongMemEvalDataset {
    /// Load a LongMemEval dataset from a JSON file.
    ///
    /// Supports both array format (just a list of questions) and object format
    /// (with metadata wrapper).
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let content = fs::read_to_string(path.as_ref()).with_context(|| {
            format!(
                "Failed to read LongMemEval file: {}",
                path.as_ref().display()
            )
        })?;

        let variant = path
            .as_ref()
            .file_stem()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_else(|| "unknown".to_string());

        Self::from_json_str(&content, &variant)
    }

    /// Parse LongMemEval data from a JSON string.
    pub fn from_json_str(json: &str, variant: &str) -> Result<Self> {
        // Try parsing as an array of questions first
        let questions: Vec<LongMemEvalQuestion> = match serde_json::from_str(json) {
            Ok(qs) => qs,
            Err(_) => {
                // Try as an object with a "data" or "questions" field
                let value: serde_json::Value = serde_json::from_str(json)
                    .with_context(|| "Failed to parse LongMemEval JSON")?;
                if let Some(arr) = value.get("data").or_else(|| value.get("questions")) {
                    serde_json::from_value(arr.clone())
                        .with_context(|| "Failed to parse questions array")?
                } else {
                    anyhow::bail!(
                        "LongMemEval JSON must be an array or object with 'data'/'questions' key"
                    );
                }
            }
        };

        Ok(LongMemEvalDataset {
            questions,
            variant: variant.to_string(),
        })
    }

    /// Get summary statistics of the dataset.
    pub fn stats(&self) -> DatasetStats {
        let mut by_type: HashMap<String, usize> = HashMap::new();
        let mut by_ability: HashMap<MemoryAbility, usize> = HashMap::new();
        let mut total_sessions = 0usize;
        let mut total_turns = 0usize;

        for q in &self.questions {
            *by_type.entry(q.question_type.clone()).or_default() += 1;
            *by_ability
                .entry(MemoryAbility::from_question_type(&q.question_type))
                .or_default() += 1;
            total_sessions += q.haystack_sessions.len();
            total_turns += q.haystack_sessions.iter().map(|s| s.len()).sum::<usize>();
        }

        DatasetStats {
            num_questions: self.questions.len(),
            by_question_type: by_type,
            by_ability,
            total_sessions,
            total_turns,
            variant: self.variant.clone(),
        }
    }

    /// Convert the LongMemEval dataset to an [`EvalDataset`] for use with the
    /// evaluation harness.
    ///
    /// Uses session-level granularity by default (one document per session).
    /// The ground truth `relevant_ids` maps to `answer_session_ids`.
    pub fn to_eval_dataset(&self) -> EvalDataset {
        self.to_eval_dataset_with_granularity(DocumentGranularity::Session)
    }

    /// Convert with explicit granularity control.
    pub fn to_eval_dataset_with_granularity(
        &self,
        granularity: DocumentGranularity,
    ) -> EvalDataset {
        let mut documents = Vec::new();
        let mut queries = Vec::new();
        let mut seen_docs: HashMap<String, bool> = HashMap::new();

        for q in &self.questions {
            // Create documents from haystack sessions
            for (session_idx, session) in q.haystack_sessions.iter().enumerate() {
                let session_id = q
                    .haystack_session_ids
                    .get(session_idx)
                    .cloned()
                    .unwrap_or_else(|| format!("{}:session_{}", q.question_id, session_idx));

                let session_date = q
                    .haystack_dates
                    .get(session_idx)
                    .cloned()
                    .unwrap_or_default();

                match granularity {
                    DocumentGranularity::Session => {
                        // Deduplicate: same session_id seen across questions
                        if seen_docs.contains_key(&session_id) {
                            continue;
                        }
                        seen_docs.insert(session_id.clone(), true);

                        let content = session_to_text(session);
                        let mut metadata = HashMap::new();
                        if !session_date.is_empty() {
                            metadata.insert(
                                "date".to_string(),
                                serde_json::Value::String(session_date),
                            );
                        }
                        metadata.insert(
                            "source".to_string(),
                            serde_json::Value::String("longmemeval".to_string()),
                        );

                        documents.push(EvalDocument {
                            id: session_id,
                            content,
                            metadata,
                        });
                    }
                    DocumentGranularity::Turn => {
                        for (turn_idx, turn) in session.iter().enumerate() {
                            let turn_id = format!("{}:turn_{}", session_id, turn_idx);
                            if seen_docs.contains_key(&turn_id) {
                                continue;
                            }
                            seen_docs.insert(turn_id.clone(), true);

                            let content = format!("[{}] {}", turn.role, turn.content);
                            let mut metadata = HashMap::new();
                            metadata.insert(
                                "role".to_string(),
                                serde_json::Value::String(turn.role.clone()),
                            );
                            metadata.insert(
                                "session_id".to_string(),
                                serde_json::Value::String(session_id.clone()),
                            );
                            if !session_date.is_empty() {
                                metadata.insert(
                                    "date".to_string(),
                                    serde_json::Value::String(session_date.clone()),
                                );
                            }

                            documents.push(EvalDocument {
                                id: turn_id,
                                content,
                                metadata,
                            });
                        }
                    }
                }
            }

            // Create the query
            let ability = MemoryAbility::from_question_type(&q.question_type);
            let category = format!("{}:{}", ability.label(), q.question_type);

            let relevant_ids = match granularity {
                DocumentGranularity::Session => q.answer_session_ids.clone(),
                DocumentGranularity::Turn => {
                    // For turn-level, expand answer_session_ids to all turns with has_answer=true
                    let mut turn_ids = Vec::new();
                    for (session_idx, session) in q.haystack_sessions.iter().enumerate() {
                        let session_id = q
                            .haystack_session_ids
                            .get(session_idx)
                            .cloned()
                            .unwrap_or_else(|| {
                                format!("{}:session_{}", q.question_id, session_idx)
                            });

                        if q.answer_session_ids.contains(&session_id) {
                            for (turn_idx, turn) in session.iter().enumerate() {
                                if turn.has_answer {
                                    turn_ids.push(format!("{}:turn_{}", session_id, turn_idx));
                                }
                            }
                            // If no turns marked, include all turns in the answer session
                            if turn_ids.is_empty() {
                                for turn_idx in 0..session.len() {
                                    turn_ids.push(format!("{}:turn_{}", session_id, turn_idx));
                                }
                            }
                        }
                    }
                    turn_ids
                }
            };

            let mut query_metadata = HashMap::new();
            query_metadata.insert(
                "answer".to_string(),
                serde_json::Value::String(q.answer.clone()),
            );
            if let Some(ref date) = q.question_date {
                query_metadata.insert(
                    "question_date".to_string(),
                    serde_json::Value::String(date.clone()),
                );
            }
            query_metadata.insert(
                "question_type".to_string(),
                serde_json::Value::String(q.question_type.clone()),
            );

            queries.push(EvalQuery {
                id: q.question_id.clone(),
                query: q.question.clone(),
                relevant_ids,
                relevance_grades: HashMap::new(),
                category: Some(category),
                metadata: query_metadata,
            });
        }

        EvalDataset {
            name: format!("LongMemEval_{}", self.variant),
            description: Some(format!(
                "LongMemEval (ICLR 2025) - {} variant, {} questions, {} documents",
                self.variant,
                queries.len(),
                documents.len()
            )),
            documents,
            queries,
        }
    }

    /// Get questions for a specific memory ability.
    pub fn questions_by_ability(&self, ability: MemoryAbility) -> Vec<&LongMemEvalQuestion> {
        self.questions
            .iter()
            .filter(|q| MemoryAbility::from_question_type(&q.question_type) == ability)
            .collect()
    }
}

/// Summary statistics for a LongMemEval dataset.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetStats {
    pub num_questions: usize,
    pub by_question_type: HashMap<String, usize>,
    pub by_ability: HashMap<MemoryAbility, usize>,
    pub total_sessions: usize,
    pub total_turns: usize,
    pub variant: String,
}

// ---------------------------------------------------------------------------
// Published baselines for comparison
// ---------------------------------------------------------------------------

/// Published baseline results from the LongMemEval paper and related work.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BaselineResult {
    pub system: String,
    pub variant: String,
    pub recall_at_5: Option<f64>,
    pub ndcg_at_5: Option<f64>,
    pub qa_accuracy: Option<f64>,
    pub notes: String,
}

/// Get published baseline results for comparison.
pub fn published_baselines() -> Vec<BaselineResult> {
    vec![
        BaselineResult {
            system: "GPT-4o (oracle)".into(),
            variant: "oracle".into(),
            recall_at_5: None,
            ndcg_at_5: None,
            qa_accuracy: Some(87.0),
            notes: "Oracle retrieval (only evidence sessions)".into(),
        },
        BaselineResult {
            system: "GPT-4o (full history)".into(),
            variant: "S".into(),
            recall_at_5: None,
            ndcg_at_5: None,
            qa_accuracy: Some(60.6),
            notes: "Full history in context window".into(),
        },
        BaselineResult {
            system: "RAG K=V (session)".into(),
            variant: "M".into(),
            recall_at_5: Some(0.706),
            ndcg_at_5: Some(0.617),
            qa_accuracy: Some(78.3),
            notes: "Stella V5 1.5B embeddings, session granularity".into(),
        },
        BaselineResult {
            system: "RAG K=V+fact (session)".into(),
            variant: "M".into(),
            recall_at_5: Some(0.732),
            ndcg_at_5: Some(0.620),
            qa_accuracy: Some(86.2),
            notes: "Stella V5 1.5B + fact augmented keys, session granularity".into(),
        },
        BaselineResult {
            system: "RAG K=V (round)".into(),
            variant: "M".into(),
            recall_at_5: Some(0.582),
            ndcg_at_5: Some(0.481),
            qa_accuracy: Some(69.2),
            notes: "Stella V5 1.5B embeddings, round granularity".into(),
        },
        BaselineResult {
            system: "Zep".into(),
            variant: "S".into(),
            recall_at_5: None,
            ndcg_at_5: None,
            qa_accuracy: Some(71.2),
            notes: "Commercial system (reported 2025)".into(),
        },
        BaselineResult {
            system: "Mem0 (GPT-4o-mini)".into(),
            variant: "S".into(),
            recall_at_5: None,
            ndcg_at_5: None,
            qa_accuracy: Some(73.8),
            notes: "Via LiCoMemory comparison, GPT-4o-mini backbone".into(),
        },
        BaselineResult {
            system: "Naive RAG".into(),
            variant: "S".into(),
            recall_at_5: None,
            ndcg_at_5: None,
            qa_accuracy: Some(52.0),
            notes: "Simple retrieval baseline".into(),
        },
        BaselineResult {
            system: "EmergenceMem".into(),
            variant: "S".into(),
            recall_at_5: None,
            ndcg_at_5: None,
            qa_accuracy: Some(86.0),
            notes: "Emergence AI (2025 SOTA)".into(),
        },
    ]
}

// ---------------------------------------------------------------------------
// Extended report with baseline comparison
// ---------------------------------------------------------------------------

/// Extended evaluation report with LongMemEval-specific baseline comparisons.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LongMemEvalReport {
    /// The base evaluation report from the harness.
    pub eval_report: EvalReport,
    /// Published baselines for comparison.
    pub baselines: Vec<BaselineResult>,
    /// Per-ability breakdown (aggregated from category metrics).
    pub per_ability: HashMap<String, AbilityMetrics>,
    /// Number of tasks where Ucotron results are comparable or superior to baselines.
    pub tasks_passing: usize,
    /// Total tasks evaluated.
    pub total_tasks: usize,
}

/// Metrics aggregated by memory ability.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AbilityMetrics {
    pub ability: String,
    pub num_queries: usize,
    pub mrr: f64,
    pub mean_recall_at_5: f64,
    pub mean_ndcg_at_5: f64,
    pub mean_f1: f64,
    pub latency_p95_ms: f64,
}

impl LongMemEvalReport {
    /// Create an extended report from a base evaluation report.
    pub fn new(eval_report: EvalReport) -> Self {
        let baselines = published_baselines();
        let per_ability = Self::compute_ability_metrics(&eval_report);

        // Count passing tasks: we consider a task "comparable" if Recall@5 >= 0.50
        // (matching roughly the naive RAG baseline level)
        let tasks_passing = per_ability
            .values()
            .filter(|m| m.mean_recall_at_5 >= 0.50)
            .count();
        let total_tasks = per_ability.len();

        LongMemEvalReport {
            eval_report,
            baselines,
            per_ability,
            tasks_passing,
            total_tasks,
        }
    }

    /// Aggregate per-category metrics into per-ability metrics.
    fn compute_ability_metrics(report: &EvalReport) -> HashMap<String, AbilityMetrics> {
        let mut ability_map: HashMap<String, Vec<&str>> = HashMap::new();

        // Group categories by ability (category format: "Ability Label:question_type")
        for cat_name in report.per_category.keys() {
            let ability_label = cat_name.split(':').next().unwrap_or(cat_name).to_string();
            ability_map.entry(ability_label).or_default().push(cat_name);
        }

        let mut result = HashMap::new();

        for (ability_label, categories) in &ability_map {
            let mut total_queries = 0;
            let mut sum_mrr = 0.0;
            let mut sum_recall5 = 0.0;
            let mut sum_ndcg5 = 0.0;
            let mut sum_f1 = 0.0;
            let mut max_p95 = 0u64;

            for cat_name in categories {
                if let Some(metrics) = report.per_category.get(*cat_name) {
                    total_queries += metrics.num_queries;
                    sum_mrr += metrics.mrr * metrics.num_queries as f64;
                    sum_f1 += metrics.mean_f1 * metrics.num_queries as f64;
                    if metrics.latency_p95_us > max_p95 {
                        max_p95 = metrics.latency_p95_us;
                    }

                    // Extract Recall@5 and NDCG@5
                    let r5 = metrics
                        .mean_recall_at_k
                        .iter()
                        .find(|(k, _)| *k == 5)
                        .map(|(_, v)| *v)
                        .unwrap_or(0.0);
                    let n5 = metrics
                        .mean_ndcg_at_k
                        .iter()
                        .find(|(k, _)| *k == 5)
                        .map(|(_, v)| *v)
                        .unwrap_or(0.0);
                    sum_recall5 += r5 * metrics.num_queries as f64;
                    sum_ndcg5 += n5 * metrics.num_queries as f64;
                }
            }

            if total_queries > 0 {
                result.insert(
                    ability_label.clone(),
                    AbilityMetrics {
                        ability: ability_label.clone(),
                        num_queries: total_queries,
                        mrr: sum_mrr / total_queries as f64,
                        mean_recall_at_5: sum_recall5 / total_queries as f64,
                        mean_ndcg_at_5: sum_ndcg5 / total_queries as f64,
                        mean_f1: sum_f1 / total_queries as f64,
                        latency_p95_ms: max_p95 as f64 / 1000.0,
                    },
                );
            }
        }

        result
    }

    /// Render the full LongMemEval report as Markdown with baseline comparisons.
    pub fn to_markdown(&self) -> String {
        let mut md = String::new();

        md.push_str("# LongMemEval Benchmark Results\n\n");
        md.push_str(&format!(
            "**Dataset**: {}\n\n",
            self.eval_report.dataset_name
        ));
        md.push_str(&format!(
            "**Queries evaluated**: {}\n\n",
            self.eval_report.aggregate.num_queries
        ));
        md.push_str(&format!(
            "**Tasks passing (Recall@5 >= 0.50)**: {}/{}\n\n",
            self.tasks_passing, self.total_tasks
        ));

        // Overall retrieval metrics
        md.push_str("## Overall Retrieval Metrics\n\n");
        md.push_str("| Metric | Value |\n");
        md.push_str("|--------|-------|\n");
        md.push_str(&format!(
            "| MRR | {:.4} |\n",
            self.eval_report.aggregate.mrr
        ));
        md.push_str(&format!(
            "| Mean Precision | {:.4} |\n",
            self.eval_report.aggregate.mean_precision
        ));
        md.push_str(&format!(
            "| Mean Recall | {:.4} |\n",
            self.eval_report.aggregate.mean_recall
        ));
        md.push_str(&format!(
            "| Mean F1 | {:.4} |\n",
            self.eval_report.aggregate.mean_f1
        ));
        md.push_str(&format!(
            "| Latency P50 | {:.2}ms |\n",
            self.eval_report.aggregate.latency_p50_us as f64 / 1000.0
        ));
        md.push_str(&format!(
            "| Latency P95 | {:.2}ms |\n",
            self.eval_report.aggregate.latency_p95_us as f64 / 1000.0
        ));
        md.push('\n');

        // Recall@k and NDCG@k
        md.push_str("## Recall@k\n\n");
        md.push_str("| k | Recall |\n");
        md.push_str("|---|--------|\n");
        for (k, r) in &self.eval_report.aggregate.mean_recall_at_k {
            md.push_str(&format!("| {} | {:.4} |\n", k, r));
        }
        md.push('\n');

        md.push_str("## NDCG@k\n\n");
        md.push_str("| k | NDCG |\n");
        md.push_str("|---|------|\n");
        for (k, n) in &self.eval_report.aggregate.mean_ndcg_at_k {
            md.push_str(&format!("| {} | {:.4} |\n", k, n));
        }
        md.push('\n');

        // Per-ability breakdown
        if !self.per_ability.is_empty() {
            md.push_str("## Per-Ability Results\n\n");
            md.push_str("| Memory Ability | Queries | MRR | Recall@5 | NDCG@5 | F1 | P95 (ms) |\n");
            md.push_str("|----------------|---------|-----|----------|--------|----|---------|\n");

            let mut abilities: Vec<_> = self.per_ability.iter().collect();
            abilities.sort_by(|(a, _), (b, _)| a.cmp(b));

            for (_, metrics) in &abilities {
                md.push_str(&format!(
                    "| {} | {} | {:.4} | {:.4} | {:.4} | {:.4} | {:.2} |\n",
                    metrics.ability,
                    metrics.num_queries,
                    metrics.mrr,
                    metrics.mean_recall_at_5,
                    metrics.mean_ndcg_at_5,
                    metrics.mean_f1,
                    metrics.latency_p95_ms,
                ));
            }
            md.push('\n');
        }

        // Baseline comparison
        md.push_str("## Comparison with Published Baselines\n\n");
        md.push_str("| System | Variant | Recall@5 | NDCG@5 | QA Accuracy | Notes |\n");
        md.push_str("|--------|---------|----------|--------|-------------|-------|\n");

        // Add Ucotron row first
        let ucotron_r5 = self
            .eval_report
            .aggregate
            .mean_recall_at_k
            .iter()
            .find(|(k, _)| *k == 5)
            .map(|(_, v)| *v)
            .unwrap_or(0.0);
        let ucotron_n5 = self
            .eval_report
            .aggregate
            .mean_ndcg_at_k
            .iter()
            .find(|(k, _)| *k == 5)
            .map(|(_, v)| *v)
            .unwrap_or(0.0);
        md.push_str(&format!(
            "| **Ucotron** | **{}** | **{:.3}** | **{:.3}** | N/A | Local HNSW + graph expansion |\n",
            self.eval_report.dataset_name, ucotron_r5, ucotron_n5
        ));

        for baseline in &self.baselines {
            let r5 = baseline
                .recall_at_5
                .map(|v| format!("{:.3}", v))
                .unwrap_or_else(|| "N/A".to_string());
            let n5 = baseline
                .ndcg_at_5
                .map(|v| format!("{:.3}", v))
                .unwrap_or_else(|| "N/A".to_string());
            let qa = baseline
                .qa_accuracy
                .map(|v| format!("{:.1}%", v))
                .unwrap_or_else(|| "N/A".to_string());
            md.push_str(&format!(
                "| {} | {} | {} | {} | {} | {} |\n",
                baseline.system, baseline.variant, r5, n5, qa, baseline.notes,
            ));
        }
        md.push('\n');

        md.push_str("*Note: QA Accuracy requires LLM-as-judge (GPT-4o). ");
        md.push_str("Ucotron reports retrieval metrics only (Recall@k, NDCG@k, MRR).*\n");

        md
    }

    /// Serialize the full report to JSON.
    pub fn to_json(&self) -> Result<String> {
        serde_json::to_string_pretty(self)
            .map_err(|e| anyhow::anyhow!("JSON serialization error: {}", e))
    }
}

// ---------------------------------------------------------------------------
// LongMemEval benchmark runner
// ---------------------------------------------------------------------------

/// Configuration for running the LongMemEval benchmark.
#[derive(Debug, Clone)]
pub struct LongMemEvalConfig {
    /// Path to the LongMemEval JSON dataset file.
    pub dataset_path: String,
    /// Document granularity for conversion.
    pub granularity: DocumentGranularity,
    /// Evaluation config (k values, max queries, etc.).
    pub eval_config: EvalConfig,
}

impl Default for LongMemEvalConfig {
    fn default() -> Self {
        Self {
            dataset_path: "data/longmemeval_oracle.json".to_string(),
            granularity: DocumentGranularity::Session,
            eval_config: EvalConfig {
                k_values: vec![1, 5, 10, 20],
                per_category: true,
                ..Default::default()
            },
        }
    }
}

/// Run the LongMemEval benchmark with a custom query function.
///
/// This is the main entry point for running the benchmark.
/// The `query_fn` receives a query string and should return a ranked list
/// of document/session IDs (most relevant first).
///
/// # Arguments
///
/// * `config` - Benchmark configuration
/// * `query_fn` - Retrieval function: query text → ranked document IDs
///
/// # Returns
///
/// Extended report with per-ability breakdown and baseline comparisons.
pub fn run_benchmark<F>(config: &LongMemEvalConfig, query_fn: F) -> Result<LongMemEvalReport>
where
    F: Fn(&str) -> Vec<String>,
{
    // Load and parse dataset
    let dataset = LongMemEvalDataset::from_file(&config.dataset_path)?;

    // Convert to eval format
    let eval_dataset = dataset.to_eval_dataset_with_granularity(config.granularity);

    // Run evaluation
    let evaluator = Evaluator::new(config.eval_config.clone());
    let eval_report = evaluator.evaluate(&eval_dataset, query_fn)?;

    // Build extended report
    Ok(LongMemEvalReport::new(eval_report))
}

/// Run the benchmark from an already-loaded dataset (useful for testing).
pub fn run_benchmark_from_dataset<F>(
    dataset: &LongMemEvalDataset,
    granularity: DocumentGranularity,
    eval_config: EvalConfig,
    query_fn: F,
) -> Result<LongMemEvalReport>
where
    F: Fn(&str) -> Vec<String>,
{
    let eval_dataset = dataset.to_eval_dataset_with_granularity(granularity);
    let evaluator = Evaluator::new(eval_config);
    let eval_report = evaluator.evaluate(&eval_dataset, query_fn)?;
    Ok(LongMemEvalReport::new(eval_report))
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Convert a session (list of turns) to a single text document.
fn session_to_text(session: &[SessionTurn]) -> String {
    session
        .iter()
        .map(|turn| format!("[{}]: {}", turn.role, turn.content))
        .collect::<Vec<_>>()
        .join("\n")
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- Helper: create a minimal LongMemEval question --

    fn make_question(
        id: &str,
        question_type: &str,
        question: &str,
        answer: &str,
        session_ids: Vec<&str>,
        answer_ids: Vec<&str>,
        sessions: Vec<Vec<SessionTurn>>,
    ) -> LongMemEvalQuestion {
        LongMemEvalQuestion {
            question_id: id.to_string(),
            question_type: question_type.to_string(),
            question: question.to_string(),
            answer: answer.to_string(),
            question_date: Some("2024-03-15".to_string()),
            haystack_session_ids: session_ids.into_iter().map(|s| s.to_string()).collect(),
            haystack_dates: vec!["2023-06-10".to_string(), "2023-07-22".to_string()],
            haystack_sessions: sessions,
            answer_session_ids: answer_ids.into_iter().map(|s| s.to_string()).collect(),
        }
    }

    fn make_turn(role: &str, content: &str, has_answer: bool) -> SessionTurn {
        SessionTurn {
            role: role.to_string(),
            content: content.to_string(),
            has_answer,
        }
    }

    fn sample_dataset() -> LongMemEvalDataset {
        let sessions1 = vec![
            vec![
                make_turn("user", "What's the weather like?", false),
                make_turn("assistant", "It's sunny today.", false),
            ],
            vec![
                make_turn("user", "My name is Alice.", true),
                make_turn("assistant", "Nice to meet you, Alice!", true),
            ],
        ];

        let sessions2 = vec![
            vec![
                make_turn("user", "I moved to Paris last year.", true),
                make_turn("assistant", "Paris is beautiful!", true),
            ],
            vec![
                make_turn("user", "Tell me about cats.", false),
                make_turn("assistant", "Cats are domestic animals.", false),
            ],
        ];

        LongMemEvalDataset {
            questions: vec![
                make_question(
                    "q1",
                    "single-session-user",
                    "What is the user's name?",
                    "Alice",
                    vec!["s1", "s2"],
                    vec!["s2"],
                    sessions1,
                ),
                make_question(
                    "q2",
                    "knowledge-update",
                    "Where does the user live?",
                    "Paris",
                    vec!["s3", "s4"],
                    vec!["s3"],
                    sessions2,
                ),
            ],
            variant: "test".to_string(),
        }
    }

    // -- MemoryAbility classification tests --

    #[test]
    fn test_ability_classification_single_session() {
        assert_eq!(
            MemoryAbility::from_question_type("single-session-user"),
            MemoryAbility::InformationExtraction
        );
        assert_eq!(
            MemoryAbility::from_question_type("single-session-assistant"),
            MemoryAbility::InformationExtraction
        );
        assert_eq!(
            MemoryAbility::from_question_type("single-session-preference"),
            MemoryAbility::InformationExtraction
        );
    }

    #[test]
    fn test_ability_classification_multi_session() {
        assert_eq!(
            MemoryAbility::from_question_type("multi-session"),
            MemoryAbility::MultiSessionReasoning
        );
    }

    #[test]
    fn test_ability_classification_knowledge_update() {
        assert_eq!(
            MemoryAbility::from_question_type("knowledge-update"),
            MemoryAbility::KnowledgeUpdate
        );
    }

    #[test]
    fn test_ability_classification_temporal() {
        assert_eq!(
            MemoryAbility::from_question_type("temporal-reasoning"),
            MemoryAbility::TemporalReasoning
        );
    }

    #[test]
    fn test_ability_classification_abstention() {
        assert_eq!(
            MemoryAbility::from_question_type("single-session-user_abs"),
            MemoryAbility::Abstention
        );
        assert_eq!(
            MemoryAbility::from_question_type("multi-session_abs"),
            MemoryAbility::Abstention
        );
    }

    // -- Dataset stats tests --

    #[test]
    fn test_dataset_stats() {
        let ds = sample_dataset();
        let stats = ds.stats();

        assert_eq!(stats.num_questions, 2);
        assert_eq!(stats.total_sessions, 4);
        assert_eq!(stats.total_turns, 8);
        assert_eq!(stats.by_ability[&MemoryAbility::InformationExtraction], 1);
        assert_eq!(stats.by_ability[&MemoryAbility::KnowledgeUpdate], 1);
    }

    // -- Session to text conversion --

    #[test]
    fn test_session_to_text() {
        let session = vec![
            make_turn("user", "Hello!", false),
            make_turn("assistant", "Hi there!", false),
        ];
        let text = session_to_text(&session);
        assert_eq!(text, "[user]: Hello!\n[assistant]: Hi there!");
    }

    // -- EvalDataset conversion (session granularity) --

    #[test]
    fn test_to_eval_dataset_session_granularity() {
        let ds = sample_dataset();
        let eval = ds.to_eval_dataset();

        // 4 unique sessions
        assert_eq!(eval.documents.len(), 4);
        // 2 queries
        assert_eq!(eval.queries.len(), 2);

        // Check first query
        let q1 = &eval.queries[0];
        assert_eq!(q1.id, "q1");
        assert_eq!(q1.query, "What is the user's name?");
        assert_eq!(q1.relevant_ids, vec!["s2"]);
        assert!(q1
            .category
            .as_ref()
            .unwrap()
            .contains("Information Extraction"));

        // Check second query
        let q2 = &eval.queries[1];
        assert_eq!(q2.id, "q2");
        assert_eq!(q2.relevant_ids, vec!["s3"]);
        assert!(q2.category.as_ref().unwrap().contains("Knowledge Update"));
    }

    // -- EvalDataset conversion (turn granularity) --

    #[test]
    fn test_to_eval_dataset_turn_granularity() {
        let ds = sample_dataset();
        let eval = ds.to_eval_dataset_with_granularity(DocumentGranularity::Turn);

        // 4 sessions × 2 turns = 8 turn documents
        assert_eq!(eval.documents.len(), 8);
        assert_eq!(eval.queries.len(), 2);

        // Check first query: answer_session_ids = ["s2"], both turns have has_answer=true
        let q1 = &eval.queries[0];
        assert_eq!(q1.relevant_ids.len(), 2);
        assert!(q1.relevant_ids.contains(&"s2:turn_0".to_string()));
        assert!(q1.relevant_ids.contains(&"s2:turn_1".to_string()));
    }

    // -- Document content tests --

    #[test]
    fn test_document_content_format() {
        let ds = sample_dataset();
        let eval = ds.to_eval_dataset();

        let doc_s1 = eval.documents.iter().find(|d| d.id == "s1").unwrap();
        assert!(doc_s1.content.contains("[user]: What's the weather like?"));
        assert!(doc_s1.content.contains("[assistant]: It's sunny today."));
    }

    // -- Document metadata tests --

    #[test]
    fn test_document_metadata() {
        let ds = sample_dataset();
        let eval = ds.to_eval_dataset();

        let doc = &eval.documents[0];
        assert_eq!(
            doc.metadata.get("source").and_then(|v| v.as_str()),
            Some("longmemeval")
        );
    }

    // -- Query metadata tests --

    #[test]
    fn test_query_metadata() {
        let ds = sample_dataset();
        let eval = ds.to_eval_dataset();

        let q = &eval.queries[0];
        assert_eq!(
            q.metadata.get("answer").and_then(|v| v.as_str()),
            Some("Alice")
        );
        assert_eq!(
            q.metadata.get("question_date").and_then(|v| v.as_str()),
            Some("2024-03-15")
        );
    }

    // -- JSON parsing tests --

    #[test]
    fn test_parse_array_format() {
        let json = serde_json::to_string(&vec![LongMemEvalQuestion {
            question_id: "q1".into(),
            question_type: "single-session-user".into(),
            question: "Who am I?".into(),
            answer: "You are Alice".into(),
            question_date: None,
            haystack_session_ids: vec!["s1".into()],
            haystack_dates: vec![],
            haystack_sessions: vec![vec![SessionTurn {
                role: "user".into(),
                content: "My name is Alice".into(),
                has_answer: true,
            }]],
            answer_session_ids: vec!["s1".into()],
        }])
        .unwrap();

        let ds = LongMemEvalDataset::from_json_str(&json, "test").unwrap();
        assert_eq!(ds.questions.len(), 1);
        assert_eq!(ds.questions[0].question_id, "q1");
    }

    #[test]
    fn test_parse_object_format() {
        let json = r#"{"data": [{"question_id": "q1", "question_type": "multi-session", "question": "test", "answer": "ans", "haystack_sessions": [], "answer_session_ids": ["s1"]}]}"#;
        let ds = LongMemEvalDataset::from_json_str(json, "test").unwrap();
        assert_eq!(ds.questions.len(), 1);
    }

    #[test]
    fn test_parse_invalid_json() {
        let result = LongMemEvalDataset::from_json_str("not json", "test");
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_missing_questions_key() {
        let result = LongMemEvalDataset::from_json_str(r#"{"foo": "bar"}"#, "test");
        assert!(result.is_err());
    }

    // -- Baseline tests --

    #[test]
    fn test_published_baselines() {
        let baselines = published_baselines();
        assert!(!baselines.is_empty());
        // Should include at least GPT-4o and Mem0
        assert!(baselines.iter().any(|b| b.system.contains("GPT-4o")));
        assert!(baselines.iter().any(|b| b.system.contains("Mem0")));
    }

    // -- Run benchmark with mock query function --

    #[test]
    fn test_run_benchmark_from_dataset() {
        let ds = sample_dataset();
        let config = EvalConfig {
            k_values: vec![1, 5],
            per_category: true,
            ..Default::default()
        };

        // Perfect retrieval: always return the correct session
        let report =
            run_benchmark_from_dataset(&ds, DocumentGranularity::Session, config, |query| {
                if query.contains("name") {
                    vec!["s2".into(), "s1".into()]
                } else if query.contains("live") {
                    vec!["s3".into(), "s4".into()]
                } else {
                    vec![]
                }
            })
            .unwrap();

        // Both queries should have perfect Recall@1
        assert_eq!(report.eval_report.aggregate.num_queries, 2);
        assert!((report.eval_report.aggregate.mrr - 1.0).abs() < 1e-10);
        // Tasks passing should reflect good retrieval
        assert!(report.tasks_passing > 0);
    }

    #[test]
    fn test_run_benchmark_poor_retrieval() {
        let ds = sample_dataset();
        let config = EvalConfig {
            k_values: vec![1, 5],
            per_category: true,
            ..Default::default()
        };

        // Bad retrieval: return wrong sessions
        let report = run_benchmark_from_dataset(&ds, DocumentGranularity::Session, config, |_| {
            vec!["nonexistent_session".into()]
        })
        .unwrap();

        assert_eq!(report.eval_report.aggregate.num_queries, 2);
        assert!((report.eval_report.aggregate.mrr - 0.0).abs() < 1e-10);
        assert_eq!(report.tasks_passing, 0);
    }

    // -- Report generation tests --

    #[test]
    fn test_report_to_markdown() {
        let ds = sample_dataset();
        let config = EvalConfig {
            k_values: vec![1, 5, 10],
            per_category: true,
            ..Default::default()
        };

        let report =
            run_benchmark_from_dataset(&ds, DocumentGranularity::Session, config, |query| {
                if query.contains("name") {
                    vec!["s2".into()]
                } else {
                    vec!["s3".into()]
                }
            })
            .unwrap();

        let md = report.to_markdown();
        assert!(md.contains("# LongMemEval Benchmark Results"));
        assert!(md.contains("Ucotron"));
        assert!(md.contains("GPT-4o"));
        assert!(md.contains("Mem0"));
        assert!(md.contains("Per-Ability Results"));
        assert!(md.contains("Comparison with Published Baselines"));
    }

    #[test]
    fn test_report_to_json() {
        let ds = sample_dataset();
        let config = EvalConfig {
            k_values: vec![5],
            per_category: true,
            ..Default::default()
        };

        let report = run_benchmark_from_dataset(&ds, DocumentGranularity::Session, config, |_| {
            vec!["s2".into()]
        })
        .unwrap();

        let json = report.to_json().unwrap();
        assert!(json.contains("LongMemEval_test"));
        assert!(json.contains("baselines"));
        assert!(json.contains("per_ability"));
    }

    // -- Session deduplication test --

    #[test]
    fn test_session_dedup_across_questions() {
        // Two questions sharing the same sessions
        let shared_sessions = vec![
            vec![make_turn("user", "Hello", false)],
            vec![make_turn("user", "I'm Alice", true)],
        ];

        let ds = LongMemEvalDataset {
            questions: vec![
                make_question(
                    "q1",
                    "single-session-user",
                    "Name?",
                    "Alice",
                    vec!["shared_s1", "shared_s2"],
                    vec!["shared_s2"],
                    shared_sessions.clone(),
                ),
                make_question(
                    "q2",
                    "single-session-user",
                    "Name again?",
                    "Alice",
                    vec!["shared_s1", "shared_s2"],
                    vec!["shared_s2"],
                    shared_sessions,
                ),
            ],
            variant: "test".to_string(),
        };

        let eval = ds.to_eval_dataset();
        // Sessions should be deduplicated
        assert_eq!(eval.documents.len(), 2); // not 4
        assert_eq!(eval.queries.len(), 2);
    }

    // -- Questions by ability filter --

    #[test]
    fn test_questions_by_ability() {
        let ds = sample_dataset();
        let ie = ds.questions_by_ability(MemoryAbility::InformationExtraction);
        assert_eq!(ie.len(), 1);
        assert_eq!(ie[0].question_id, "q1");

        let ku = ds.questions_by_ability(MemoryAbility::KnowledgeUpdate);
        assert_eq!(ku.len(), 1);
        assert_eq!(ku[0].question_id, "q2");

        let abs = ds.questions_by_ability(MemoryAbility::Abstention);
        assert!(abs.is_empty());
    }

    // -- File loading test --

    #[test]
    fn test_from_file_not_found() {
        let result = LongMemEvalDataset::from_file("/nonexistent/longmemeval.json");
        assert!(result.is_err());
    }

    #[test]
    fn test_from_file_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test_dataset.json");

        let questions = vec![LongMemEvalQuestion {
            question_id: "q1".into(),
            question_type: "temporal-reasoning".into(),
            question: "When did I visit Paris?".into(),
            answer: "Last summer".into(),
            question_date: Some("2024-01-01".into()),
            haystack_session_ids: vec!["s1".into()],
            haystack_dates: vec!["2023-07-01".into()],
            haystack_sessions: vec![vec![SessionTurn {
                role: "user".into(),
                content: "I visited Paris last summer".into(),
                has_answer: true,
            }]],
            answer_session_ids: vec!["s1".into()],
        }];

        std::fs::write(&path, serde_json::to_string(&questions).unwrap()).unwrap();

        let ds = LongMemEvalDataset::from_file(&path).unwrap();
        assert_eq!(ds.questions.len(), 1);
        assert_eq!(ds.variant, "test_dataset");
        assert_eq!(
            MemoryAbility::from_question_type(&ds.questions[0].question_type),
            MemoryAbility::TemporalReasoning
        );
    }

    // -- Ability label test --

    #[test]
    fn test_ability_labels() {
        assert_eq!(
            MemoryAbility::InformationExtraction.label(),
            "Information Extraction"
        );
        assert_eq!(
            MemoryAbility::MultiSessionReasoning.label(),
            "Multi-Session Reasoning"
        );
        assert_eq!(MemoryAbility::KnowledgeUpdate.label(), "Knowledge Update");
        assert_eq!(
            MemoryAbility::TemporalReasoning.label(),
            "Temporal Reasoning"
        );
        assert_eq!(MemoryAbility::Abstention.label(), "Abstention");
    }

    // -- Extended report per-ability aggregation --

    #[test]
    fn test_per_ability_aggregation() {
        let ds = sample_dataset();
        let config = EvalConfig {
            k_values: vec![5],
            per_category: true,
            ..Default::default()
        };

        let report =
            run_benchmark_from_dataset(&ds, DocumentGranularity::Session, config, |query| {
                if query.contains("name") {
                    vec!["s2".into()]
                } else {
                    vec!["s3".into()]
                }
            })
            .unwrap();

        // Should have 2 ability categories
        assert_eq!(report.per_ability.len(), 2);
        assert!(report.per_ability.contains_key("Information Extraction"));
        assert!(report.per_ability.contains_key("Knowledge Update"));

        // Each ability should have 1 query
        let ie = &report.per_ability["Information Extraction"];
        assert_eq!(ie.num_queries, 1);
        assert!((ie.mrr - 1.0).abs() < 1e-10); // Perfect retrieval for this query
    }
}
