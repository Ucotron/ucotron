//! # LoCoMo Benchmark (ACL 2024)
//!
//! Adapter and runner for the LoCoMo long-term conversational memory benchmark.
//! Evaluates retrieval across 5 QA categories over multi-session conversations:
//!
//! - **Single-Hop (1)**: Factual recall from individual dialogue turns
//! - **Temporal (2)**: Time-dependent reasoning over conversation events
//! - **Open-Domain (3)**: Integration of conversation context with world knowledge
//! - **Multi-Hop (4)**: Synthesizing information across multiple sessions/turns
//! - **Adversarial (5)**: Misleading questions that should be rejected
//!
//! # Architecture
//!
//! 1. Parse LoCoMo JSON into [`LoCoMoDataset`]
//! 2. Convert to [`EvalDataset`](crate::bench_eval::EvalDataset) via session or turn granularity
//! 3. Evaluate retrieval via [`Evaluator`](crate::bench_eval::Evaluator)
//! 4. Compare against published baselines (RAG, MemoryBank, ReadAgent, Mem0, etc.)
//!
//! # Usage
//!
//! ```rust,no_run
//! use ucotron_core::locomo::*;
//! use ucotron_core::bench_eval::*;
//!
//! // Load dataset
//! let dataset = LoCoMoDataset::from_file("data/benchmarks/locomo/locomo10.json").unwrap();
//!
//! // Convert to eval format (turn-level, using evidence dia_ids)
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
//! let extended = LoCoMoReport::new(report);
//! println!("{}", extended.to_markdown());
//! ```

use std::collections::HashMap;
use std::fs;
use std::path::Path;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

use crate::bench_eval::{EvalConfig, EvalDataset, EvalDocument, EvalQuery, EvalReport, Evaluator};

// ---------------------------------------------------------------------------
// LoCoMo raw data types (matching the snap-research/locomo JSON schema)
// ---------------------------------------------------------------------------

/// A single dialogue turn within a conversation session.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DialogTurn {
    /// Speaker name (matches speaker_a or speaker_b).
    pub speaker: String,
    /// Dialogue identifier (e.g., "D1:3" = session 1, turn 3).
    pub dia_id: String,
    /// The dialogue text content.
    pub text: String,
    /// Optional image URL (multimodal conversations).
    #[serde(default)]
    pub img_url: Option<serde_json::Value>,
    /// Optional BLIP-generated image caption.
    #[serde(default)]
    pub blip_caption: Option<String>,
    /// Optional search query used to find the image.
    #[serde(default)]
    pub query: Option<String>,
}

/// A single QA annotation from the LoCoMo dataset.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoCoMoQA {
    /// The question text.
    pub question: String,
    /// Ground-truth answer.
    pub answer: String,
    /// List of dialogue IDs containing the answer evidence (e.g., ["D1:3", "D4:13"]).
    #[serde(default)]
    pub evidence: Vec<String>,
    /// QA category (1=single-hop, 2=temporal, 3=open-domain, 4=multi-hop, 5=adversarial).
    pub category: u8,
    /// Optional adversarial answer (for category 5 questions).
    #[serde(default)]
    pub adversarial_answer: Option<String>,
}

/// A single conversation sample from LoCoMo (one of ~10 conversations).
///
/// The conversation object uses dynamic keys like `session_1`, `session_2`, etc.
/// We parse it into a structured representation with ordered sessions.
#[derive(Debug, Clone)]
pub struct LoCoMoSample {
    /// Sample identifier.
    pub sample_id: String,
    /// Name of speaker A.
    pub speaker_a: String,
    /// Name of speaker B.
    pub speaker_b: String,
    /// Ordered list of sessions, each with (session_key, date_time, turns).
    pub sessions: Vec<ConversationSession>,
    /// QA annotations for this conversation.
    pub qa: Vec<LoCoMoQA>,
    /// Per-session event summaries (optional).
    pub event_summaries: HashMap<String, serde_json::Value>,
}

/// A single conversation session with its metadata.
#[derive(Debug, Clone)]
pub struct ConversationSession {
    /// Session key (e.g., "session_1").
    pub key: String,
    /// Session number (1-based).
    pub number: u32,
    /// Date/time string (e.g., "1:56 pm on 8 May, 2023").
    pub date_time: String,
    /// Dialogue turns in this session.
    pub turns: Vec<DialogTurn>,
}

/// The complete LoCoMo dataset parsed from JSON.
#[derive(Debug, Clone)]
pub struct LoCoMoDataset {
    /// All conversation samples.
    pub samples: Vec<LoCoMoSample>,
    /// Dataset variant name (e.g., "locomo10").
    pub variant: String,
}

// ---------------------------------------------------------------------------
// QA category classification
// ---------------------------------------------------------------------------

/// The 5 QA reasoning categories in LoCoMo.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum QACategory {
    /// Category 1: Factual recall from single dialogue turns.
    SingleHop,
    /// Category 2: Time-dependent reasoning.
    Temporal,
    /// Category 3: Integration with external/world knowledge.
    OpenDomain,
    /// Category 4: Synthesizing across multiple sessions.
    MultiHop,
    /// Category 5: Adversarial/misleading questions.
    Adversarial,
}

impl QACategory {
    /// Classify a numeric category into the QA category enum.
    pub fn from_category_id(id: u8) -> Self {
        match id {
            1 => QACategory::SingleHop,
            2 => QACategory::Temporal,
            3 => QACategory::OpenDomain,
            4 => QACategory::MultiHop,
            5 => QACategory::Adversarial,
            _ => QACategory::SingleHop, // default fallback
        }
    }

    /// Human-readable label for reports.
    pub fn label(&self) -> &'static str {
        match self {
            QACategory::SingleHop => "Single-Hop",
            QACategory::Temporal => "Temporal",
            QACategory::OpenDomain => "Open-Domain",
            QACategory::MultiHop => "Multi-Hop",
            QACategory::Adversarial => "Adversarial",
        }
    }

    /// Numeric category ID (matching the LoCoMo JSON).
    pub fn id(&self) -> u8 {
        match self {
            QACategory::SingleHop => 1,
            QACategory::Temporal => 2,
            QACategory::OpenDomain => 3,
            QACategory::MultiHop => 4,
            QACategory::Adversarial => 5,
        }
    }
}

// ---------------------------------------------------------------------------
// Document granularity for conversion
// ---------------------------------------------------------------------------

/// Controls how conversation sessions are converted to documents.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum DocumentGranularity {
    /// One document per session (concatenated turns).
    Session,
    /// One document per dialogue turn. Each turn becomes its own retrievable
    /// document, keyed by `dia_id`. This is the natural granularity for LoCoMo
    /// since evidence is specified at the turn level.
    #[default]
    Turn,
}

// ---------------------------------------------------------------------------
// Dataset loading and conversion
// ---------------------------------------------------------------------------

impl LoCoMoDataset {
    /// Load a LoCoMo dataset from a JSON file.
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let content = fs::read_to_string(path.as_ref())
            .with_context(|| format!("Failed to read LoCoMo file: {}", path.as_ref().display()))?;

        let variant = path
            .as_ref()
            .file_stem()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_else(|| "unknown".to_string());

        Self::from_json_str(&content, &variant)
    }

    /// Parse LoCoMo data from a JSON string.
    ///
    /// Supports both array format (list of samples) and single-object format.
    pub fn from_json_str(json: &str, variant: &str) -> Result<Self> {
        let value: serde_json::Value =
            serde_json::from_str(json).with_context(|| "Failed to parse LoCoMo JSON")?;

        let samples_values = match &value {
            serde_json::Value::Array(arr) => arr.clone(),
            serde_json::Value::Object(_) => {
                // Single sample or object with "data"/"samples" key
                if let Some(arr) = value.get("data").or_else(|| value.get("samples")) {
                    arr.as_array()
                        .ok_or_else(|| anyhow::anyhow!("'data'/'samples' field must be an array"))?
                        .clone()
                } else {
                    // Single sample
                    vec![value.clone()]
                }
            }
            _ => anyhow::bail!("LoCoMo JSON must be an array or object"),
        };

        let mut samples = Vec::with_capacity(samples_values.len());
        for (idx, sv) in samples_values.iter().enumerate() {
            samples.push(parse_sample(sv, idx)?);
        }

        Ok(LoCoMoDataset {
            samples,
            variant: variant.to_string(),
        })
    }

    /// Get summary statistics of the dataset.
    pub fn stats(&self) -> DatasetStats {
        let mut by_category: HashMap<QACategory, usize> = HashMap::new();
        let mut total_sessions = 0usize;
        let mut total_turns = 0usize;
        let mut total_qa = 0usize;

        for sample in &self.samples {
            total_sessions += sample.sessions.len();
            for session in &sample.sessions {
                total_turns += session.turns.len();
            }
            for qa in &sample.qa {
                let cat = QACategory::from_category_id(qa.category);
                *by_category.entry(cat).or_default() += 1;
                total_qa += 1;
            }
        }

        DatasetStats {
            num_samples: self.samples.len(),
            num_questions: total_qa,
            by_category,
            total_sessions,
            total_turns,
            variant: self.variant.clone(),
        }
    }

    /// Convert the LoCoMo dataset to an [`EvalDataset`] for the evaluation harness.
    ///
    /// Uses turn-level granularity by default (one document per dialogue turn),
    /// since LoCoMo evidence references specific `dia_id` values.
    pub fn to_eval_dataset(&self) -> EvalDataset {
        self.to_eval_dataset_with_granularity(DocumentGranularity::Turn)
    }

    /// Convert with explicit granularity control.
    pub fn to_eval_dataset_with_granularity(
        &self,
        granularity: DocumentGranularity,
    ) -> EvalDataset {
        let mut documents = Vec::new();
        let mut queries = Vec::new();
        let mut seen_docs: HashMap<String, bool> = HashMap::new();

        for sample in &self.samples {
            // Build a mapping from dia_id → session_key for session-level evidence mapping
            let mut dia_to_session: HashMap<String, String> = HashMap::new();

            // Create documents from conversation sessions
            for session in &sample.sessions {
                match granularity {
                    DocumentGranularity::Turn => {
                        for turn in &session.turns {
                            let doc_id = format!("{}:{}", sample.sample_id, turn.dia_id);
                            dia_to_session.insert(turn.dia_id.clone(), session.key.clone());

                            if seen_docs.contains_key(&doc_id) {
                                continue;
                            }
                            seen_docs.insert(doc_id.clone(), true);

                            let content = format!("[{}]: {}", turn.speaker, turn.text);
                            let mut metadata = HashMap::new();
                            metadata.insert(
                                "speaker".to_string(),
                                serde_json::Value::String(turn.speaker.clone()),
                            );
                            metadata.insert(
                                "dia_id".to_string(),
                                serde_json::Value::String(turn.dia_id.clone()),
                            );
                            metadata.insert(
                                "session".to_string(),
                                serde_json::Value::String(session.key.clone()),
                            );
                            metadata.insert(
                                "date".to_string(),
                                serde_json::Value::String(session.date_time.clone()),
                            );
                            metadata.insert(
                                "source".to_string(),
                                serde_json::Value::String("locomo".to_string()),
                            );
                            metadata.insert(
                                "sample_id".to_string(),
                                serde_json::Value::String(sample.sample_id.clone()),
                            );

                            documents.push(EvalDocument {
                                id: doc_id,
                                content,
                                metadata,
                            });
                        }
                    }
                    DocumentGranularity::Session => {
                        let doc_id = format!("{}:{}", sample.sample_id, session.key);
                        // Still build dia_to_session for evidence mapping
                        for turn in &session.turns {
                            dia_to_session.insert(turn.dia_id.clone(), session.key.clone());
                        }

                        if seen_docs.contains_key(&doc_id) {
                            continue;
                        }
                        seen_docs.insert(doc_id.clone(), true);

                        let content = session_to_text(&session.turns);
                        let mut metadata = HashMap::new();
                        metadata.insert(
                            "date".to_string(),
                            serde_json::Value::String(session.date_time.clone()),
                        );
                        metadata.insert(
                            "source".to_string(),
                            serde_json::Value::String("locomo".to_string()),
                        );
                        metadata.insert(
                            "sample_id".to_string(),
                            serde_json::Value::String(sample.sample_id.clone()),
                        );
                        metadata.insert(
                            "num_turns".to_string(),
                            serde_json::Value::Number(session.turns.len().into()),
                        );

                        documents.push(EvalDocument {
                            id: doc_id,
                            content,
                            metadata,
                        });
                    }
                }
            }

            // Create queries from QA annotations
            for (qa_idx, qa) in sample.qa.iter().enumerate() {
                let cat = QACategory::from_category_id(qa.category);
                let category = cat.label().to_string();
                let query_id = format!("{}:qa_{}", sample.sample_id, qa_idx);

                let relevant_ids = match granularity {
                    DocumentGranularity::Turn => {
                        // Evidence dia_ids map directly to turn-level documents
                        qa.evidence
                            .iter()
                            .map(|dia_id| format!("{}:{}", sample.sample_id, dia_id))
                            .collect()
                    }
                    DocumentGranularity::Session => {
                        // Map evidence dia_ids to session-level documents
                        let mut session_ids: Vec<String> = qa
                            .evidence
                            .iter()
                            .filter_map(|dia_id| {
                                dia_to_session
                                    .get(dia_id)
                                    .map(|s| format!("{}:{}", sample.sample_id, s))
                            })
                            .collect();
                        session_ids.sort();
                        session_ids.dedup();
                        session_ids
                    }
                };

                let mut query_metadata = HashMap::new();
                query_metadata.insert(
                    "answer".to_string(),
                    serde_json::Value::String(qa.answer.clone()),
                );
                query_metadata.insert(
                    "category_id".to_string(),
                    serde_json::Value::Number(qa.category.into()),
                );
                query_metadata.insert(
                    "sample_id".to_string(),
                    serde_json::Value::String(sample.sample_id.clone()),
                );
                if !qa.evidence.is_empty() {
                    query_metadata.insert(
                        "evidence".to_string(),
                        serde_json::Value::Array(
                            qa.evidence
                                .iter()
                                .map(|e| serde_json::Value::String(e.clone()))
                                .collect(),
                        ),
                    );
                }
                if let Some(ref adv) = qa.adversarial_answer {
                    query_metadata.insert(
                        "adversarial_answer".to_string(),
                        serde_json::Value::String(adv.clone()),
                    );
                }

                queries.push(EvalQuery {
                    id: query_id,
                    query: qa.question.clone(),
                    relevant_ids,
                    relevance_grades: HashMap::new(),
                    category: Some(category),
                    metadata: query_metadata,
                });
            }
        }

        EvalDataset {
            name: format!("LoCoMo_{}", self.variant),
            description: Some(format!(
                "LoCoMo (ACL 2024) - {} variant, {} samples, {} questions, {} documents",
                self.variant,
                self.samples.len(),
                queries.len(),
                documents.len()
            )),
            documents,
            queries,
        }
    }

    /// Get QA entries for a specific category across all samples.
    pub fn qa_by_category(&self, category: QACategory) -> Vec<(&LoCoMoSample, &LoCoMoQA)> {
        self.samples
            .iter()
            .flat_map(|s| s.qa.iter().map(move |qa| (s, qa)))
            .filter(|(_, qa)| QACategory::from_category_id(qa.category) == category)
            .collect()
    }
}

/// Summary statistics for a LoCoMo dataset.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetStats {
    pub num_samples: usize,
    pub num_questions: usize,
    pub by_category: HashMap<QACategory, usize>,
    pub total_sessions: usize,
    pub total_turns: usize,
    pub variant: String,
}

// ---------------------------------------------------------------------------
// Published baselines for comparison
// ---------------------------------------------------------------------------

/// Published baseline results from the LoCoMo paper and related work.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BaselineResult {
    pub system: String,
    pub overall_f1: Option<f64>,
    pub single_hop_f1: Option<f64>,
    pub multi_hop_f1: Option<f64>,
    pub temporal_f1: Option<f64>,
    pub open_domain_f1: Option<f64>,
    pub adversarial_f1: Option<f64>,
    /// LLM Judge score (0-1 scale) if available.
    pub judge_score: Option<f64>,
    pub notes: String,
}

/// Get published baseline results for comparison.
///
/// Includes results from the original LoCoMo paper (ACL 2024) and subsequent
/// evaluations by Mem0, Zep, and other memory systems.
pub fn published_baselines() -> Vec<BaselineResult> {
    vec![
        BaselineResult {
            system: "Human".into(),
            overall_f1: Some(88.0),
            single_hop_f1: None,
            multi_hop_f1: None,
            temporal_f1: None,
            open_domain_f1: None,
            adversarial_f1: None,
            judge_score: None,
            notes: "Human annotator upper bound".into(),
        },
        BaselineResult {
            system: "GPT-4-turbo (4K)".into(),
            overall_f1: Some(32.0),
            single_hop_f1: None,
            multi_hop_f1: None,
            temporal_f1: None,
            open_domain_f1: None,
            adversarial_f1: None,
            judge_score: None,
            notes: "Base LLM, 4K context window".into(),
        },
        BaselineResult {
            system: "GPT-3.5-turbo (16K)".into(),
            overall_f1: Some(37.8),
            single_hop_f1: None,
            multi_hop_f1: None,
            temporal_f1: None,
            open_domain_f1: None,
            adversarial_f1: None,
            judge_score: None,
            notes: "Long-context LLM baseline".into(),
        },
        BaselineResult {
            system: "MemoryBank".into(),
            overall_f1: None,
            single_hop_f1: None,
            multi_hop_f1: None,
            temporal_f1: None,
            open_domain_f1: None,
            adversarial_f1: None,
            judge_score: None,
            notes: "Three-part memory pipeline with decay (ACL 2024)".into(),
        },
        BaselineResult {
            system: "ReadAgent".into(),
            overall_f1: None,
            single_hop_f1: None,
            multi_hop_f1: None,
            temporal_f1: None,
            open_domain_f1: None,
            adversarial_f1: None,
            judge_score: None,
            notes: "Human-inspired text processing (ACL 2024)".into(),
        },
        BaselineResult {
            system: "Mem0".into(),
            overall_f1: None,
            single_hop_f1: Some(38.72),
            multi_hop_f1: Some(51.15),
            temporal_f1: None,
            open_domain_f1: None,
            adversarial_f1: None,
            judge_score: Some(0.6713),
            notes: "Mem0 with GPT-4o-mini backbone (2025)".into(),
        },
        BaselineResult {
            system: "Zep".into(),
            overall_f1: None,
            single_hop_f1: None,
            multi_hop_f1: None,
            temporal_f1: None,
            open_domain_f1: None,
            adversarial_f1: None,
            judge_score: Some(0.7514),
            notes: "Temporal knowledge graph platform (2025)".into(),
        },
        BaselineResult {
            system: "MemMachine".into(),
            overall_f1: None,
            single_hop_f1: None,
            multi_hop_f1: None,
            temporal_f1: None,
            open_domain_f1: None,
            adversarial_f1: None,
            judge_score: Some(0.8487),
            notes: "MemMachine SOTA on LoCoMo (2025)".into(),
        },
        BaselineResult {
            system: "LangMem".into(),
            overall_f1: None,
            single_hop_f1: None,
            multi_hop_f1: None,
            temporal_f1: None,
            open_domain_f1: None,
            adversarial_f1: None,
            judge_score: Some(0.5810),
            notes: "Open-source memory architecture".into(),
        },
    ]
}

// ---------------------------------------------------------------------------
// Extended report with baseline comparison
// ---------------------------------------------------------------------------

/// Extended evaluation report with LoCoMo-specific baseline comparisons.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoCoMoReport {
    /// The base evaluation report from the harness.
    pub eval_report: EvalReport,
    /// Published baselines for comparison.
    pub baselines: Vec<BaselineResult>,
    /// Per-category breakdown (aggregated from category metrics).
    pub per_category_results: HashMap<String, CategoryMetrics>,
    /// Number of categories where Ucotron results are comparable to baselines.
    pub categories_passing: usize,
    /// Total categories evaluated.
    pub total_categories: usize,
}

/// Metrics aggregated by QA category.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CategoryMetrics {
    pub category: String,
    pub num_queries: usize,
    pub mrr: f64,
    pub mean_recall_at_5: f64,
    pub mean_ndcg_at_5: f64,
    pub mean_f1: f64,
    pub latency_p95_ms: f64,
}

impl LoCoMoReport {
    /// Create an extended report from a base evaluation report.
    pub fn new(eval_report: EvalReport) -> Self {
        let baselines = published_baselines();
        let per_category_results = Self::compute_category_metrics(&eval_report);

        // Count passing categories: Recall@5 >= 0.40 (comparable to simple baselines)
        let categories_passing = per_category_results
            .values()
            .filter(|m| m.mean_recall_at_5 >= 0.40)
            .count();
        let total_categories = per_category_results.len();

        LoCoMoReport {
            eval_report,
            baselines,
            per_category_results,
            categories_passing,
            total_categories,
        }
    }

    /// Aggregate per-category metrics from the evaluation report.
    fn compute_category_metrics(report: &EvalReport) -> HashMap<String, CategoryMetrics> {
        let mut result = HashMap::new();

        for (cat_name, metrics) in &report.per_category {
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

            result.insert(
                cat_name.clone(),
                CategoryMetrics {
                    category: cat_name.clone(),
                    num_queries: metrics.num_queries,
                    mrr: metrics.mrr,
                    mean_recall_at_5: r5,
                    mean_ndcg_at_5: n5,
                    mean_f1: metrics.mean_f1,
                    latency_p95_ms: metrics.latency_p95_us as f64 / 1000.0,
                },
            );
        }

        result
    }

    /// Render the full LoCoMo report as Markdown with baseline comparisons.
    pub fn to_markdown(&self) -> String {
        let mut md = String::new();

        md.push_str("# LoCoMo Benchmark Results\n\n");
        md.push_str(&format!(
            "**Dataset**: {}\n\n",
            self.eval_report.dataset_name
        ));
        md.push_str(&format!(
            "**Queries evaluated**: {}\n\n",
            self.eval_report.aggregate.num_queries
        ));
        md.push_str(&format!(
            "**Categories passing (Recall@5 >= 0.40)**: {}/{}\n\n",
            self.categories_passing, self.total_categories
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

        // Per-category breakdown
        if !self.per_category_results.is_empty() {
            md.push_str("## Per-Category Results\n\n");
            md.push_str("| QA Category | Queries | MRR | Recall@5 | NDCG@5 | F1 | P95 (ms) |\n");
            md.push_str("|-------------|---------|-----|----------|--------|----|---------|\n");

            // Sort by category name for consistent output
            let mut cats: Vec<_> = self.per_category_results.iter().collect();
            cats.sort_by(|(a, _), (b, _)| a.cmp(b));

            for (_, metrics) in &cats {
                md.push_str(&format!(
                    "| {} | {} | {:.4} | {:.4} | {:.4} | {:.4} | {:.2} |\n",
                    metrics.category,
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
        md.push_str("| System | Overall F1 | Judge Score | Notes |\n");
        md.push_str("|--------|-----------|-------------|-------|\n");

        // Add Ucotron row first
        md.push_str(
            "| **Ucotron** | N/A | N/A | Local HNSW + graph expansion (retrieval-only) |\n",
        );

        for baseline in &self.baselines {
            let f1 = baseline
                .overall_f1
                .map(|v| format!("{:.1}", v))
                .unwrap_or_else(|| "N/A".to_string());
            let judge = baseline
                .judge_score
                .map(|v| format!("{:.4}", v))
                .unwrap_or_else(|| "N/A".to_string());
            md.push_str(&format!(
                "| {} | {} | {} | {} |\n",
                baseline.system, f1, judge, baseline.notes,
            ));
        }
        md.push('\n');

        md.push_str("*Note: F1 and Judge scores require LLM-as-judge evaluation. ");
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
// LoCoMo benchmark runner
// ---------------------------------------------------------------------------

/// Configuration for running the LoCoMo benchmark.
#[derive(Debug, Clone)]
pub struct LoCoMoConfig {
    /// Path to the LoCoMo JSON dataset file.
    pub dataset_path: String,
    /// Document granularity for conversion.
    pub granularity: DocumentGranularity,
    /// Evaluation config (k values, max queries, etc.).
    pub eval_config: EvalConfig,
}

impl Default for LoCoMoConfig {
    fn default() -> Self {
        Self {
            dataset_path: "data/benchmarks/locomo/locomo10.json".to_string(),
            granularity: DocumentGranularity::Turn,
            eval_config: EvalConfig {
                k_values: vec![1, 5, 10, 20],
                per_category: true,
                ..Default::default()
            },
        }
    }
}

/// Run the LoCoMo benchmark with a custom query function.
///
/// The `query_fn` receives a query string and should return a ranked list
/// of document IDs (most relevant first).
pub fn run_benchmark<F>(config: &LoCoMoConfig, query_fn: F) -> Result<LoCoMoReport>
where
    F: Fn(&str) -> Vec<String>,
{
    let dataset = LoCoMoDataset::from_file(&config.dataset_path)?;
    let eval_dataset = dataset.to_eval_dataset_with_granularity(config.granularity);
    let evaluator = Evaluator::new(config.eval_config.clone());
    let eval_report = evaluator.evaluate(&eval_dataset, query_fn)?;
    Ok(LoCoMoReport::new(eval_report))
}

/// Run the benchmark from an already-loaded dataset (useful for testing).
pub fn run_benchmark_from_dataset<F>(
    dataset: &LoCoMoDataset,
    granularity: DocumentGranularity,
    eval_config: EvalConfig,
    query_fn: F,
) -> Result<LoCoMoReport>
where
    F: Fn(&str) -> Vec<String>,
{
    let eval_dataset = dataset.to_eval_dataset_with_granularity(granularity);
    let evaluator = Evaluator::new(eval_config);
    let eval_report = evaluator.evaluate(&eval_dataset, query_fn)?;
    Ok(LoCoMoReport::new(eval_report))
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Convert a session (list of turns) to a single text document.
fn session_to_text(turns: &[DialogTurn]) -> String {
    turns
        .iter()
        .map(|turn| format!("[{}]: {}", turn.speaker, turn.text))
        .collect::<Vec<_>>()
        .join("\n")
}

/// Parse a single LoCoMo sample from a JSON value.
///
/// The LoCoMo JSON uses dynamic keys like `session_1`, `session_2`, etc.
/// for conversations, so we parse the object manually.
fn parse_sample(value: &serde_json::Value, fallback_idx: usize) -> Result<LoCoMoSample> {
    let obj = value
        .as_object()
        .ok_or_else(|| anyhow::anyhow!("LoCoMo sample must be an object"))?;

    let sample_id = obj
        .get("sample_id")
        .and_then(|v| v.as_str())
        .or_else(|| obj.get("sample_id").and_then(|v| v.as_u64()).map(|_| ""))
        .map(|s| s.to_string())
        .unwrap_or_else(|| format!("sample_{}", fallback_idx));

    // Handle numeric sample_id
    let sample_id = if sample_id.is_empty() {
        obj.get("sample_id")
            .map(|v| v.to_string().trim_matches('"').to_string())
            .unwrap_or_else(|| format!("sample_{}", fallback_idx))
    } else {
        sample_id
    };

    // Parse conversation
    let conversation = obj
        .get("conversation")
        .and_then(|v| v.as_object())
        .ok_or_else(|| anyhow::anyhow!("Missing 'conversation' field in sample {}", sample_id))?;

    let speaker_a = conversation
        .get("speaker_a")
        .and_then(|v| v.as_str())
        .unwrap_or("Speaker A")
        .to_string();

    let speaker_b = conversation
        .get("speaker_b")
        .and_then(|v| v.as_str())
        .unwrap_or("Speaker B")
        .to_string();

    // Extract sessions (dynamic keys: session_1, session_2, ...)
    let mut sessions = Vec::new();
    let mut session_num = 1u32;
    loop {
        let session_key = format!("session_{}", session_num);
        let date_key = format!("session_{}_date_time", session_num);

        let turns_value = match conversation.get(&session_key) {
            Some(v) => v,
            None => break,
        };

        let date_time = conversation
            .get(&date_key)
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();

        let turns: Vec<DialogTurn> = serde_json::from_value(turns_value.clone())
            .with_context(|| format!("Failed to parse turns in {}", session_key))?;

        sessions.push(ConversationSession {
            key: session_key,
            number: session_num,
            date_time,
            turns,
        });

        session_num += 1;
    }

    // Parse QA annotations
    let qa: Vec<LoCoMoQA> = obj
        .get("qa")
        .map(|v| serde_json::from_value(v.clone()))
        .transpose()
        .with_context(|| format!("Failed to parse QA for sample {}", sample_id))?
        .unwrap_or_default();

    // Parse event summaries (if present)
    let event_summaries = obj
        .get("event_summary")
        .and_then(|v| v.as_object())
        .map(|o| {
            o.iter()
                .map(|(k, v)| (k.clone(), v.clone()))
                .collect::<HashMap<_, _>>()
        })
        .unwrap_or_default();

    Ok(LoCoMoSample {
        sample_id,
        speaker_a,
        speaker_b,
        sessions,
        qa,
        event_summaries,
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bench_eval::EvalConfig;

    // -- Fixture helpers --

    fn make_turn(speaker: &str, dia_id: &str, text: &str) -> DialogTurn {
        DialogTurn {
            speaker: speaker.to_string(),
            dia_id: dia_id.to_string(),
            text: text.to_string(),
            img_url: None,
            blip_caption: None,
            query: None,
        }
    }

    fn make_qa(question: &str, answer: &str, evidence: Vec<&str>, category: u8) -> LoCoMoQA {
        LoCoMoQA {
            question: question.to_string(),
            answer: answer.to_string(),
            evidence: evidence.into_iter().map(String::from).collect(),
            category,
            adversarial_answer: None,
        }
    }

    fn sample_dataset() -> LoCoMoDataset {
        let sessions = vec![
            ConversationSession {
                key: "session_1".to_string(),
                number: 1,
                date_time: "1:56 pm on 8 May, 2023".to_string(),
                turns: vec![
                    make_turn("Alice", "D1:1", "Hey! I just started learning piano."),
                    make_turn(
                        "Bob",
                        "D1:2",
                        "That's great! How long have you been playing?",
                    ),
                    make_turn(
                        "Alice",
                        "D1:3",
                        "About two weeks. I'm taking lessons on Tuesdays.",
                    ),
                ],
            },
            ConversationSession {
                key: "session_2".to_string(),
                number: 2,
                date_time: "3:14 pm on 25 May, 2023".to_string(),
                turns: vec![
                    make_turn("Alice", "D2:1", "I learned my first song!"),
                    make_turn("Bob", "D2:2", "Which song?"),
                    make_turn("Alice", "D2:3", "Für Elise by Beethoven."),
                ],
            },
            ConversationSession {
                key: "session_3".to_string(),
                number: 3,
                date_time: "10:00 am on 15 June, 2023".to_string(),
                turns: vec![
                    make_turn("Bob", "D3:1", "How's the piano going?"),
                    make_turn(
                        "Alice",
                        "D3:2",
                        "I switched to guitar instead. Piano was too hard.",
                    ),
                ],
            },
        ];

        let qa = vec![
            make_qa(
                "When did Alice start learning piano?",
                "About two weeks before 8 May 2023",
                vec!["D1:3"],
                1, // single-hop
            ),
            make_qa(
                "What was the first song Alice learned?",
                "Für Elise by Beethoven",
                vec!["D2:3"],
                1, // single-hop
            ),
            make_qa(
                "When did Alice mention learning her first song?",
                "25 May, 2023",
                vec!["D2:1"],
                2, // temporal
            ),
            make_qa(
                "Did Alice continue with piano or switch instruments?",
                "She switched to guitar",
                vec!["D1:3", "D3:2"],
                4, // multi-hop
            ),
            make_qa(
                "What instrument does Bob play?",
                "Not mentioned in conversations",
                vec![],
                5, // adversarial
            ),
        ];

        LoCoMoDataset {
            samples: vec![LoCoMoSample {
                sample_id: "test_001".to_string(),
                speaker_a: "Alice".to_string(),
                speaker_b: "Bob".to_string(),
                sessions,
                qa,
                event_summaries: HashMap::new(),
            }],
            variant: "test".to_string(),
        }
    }

    // -- QA category tests --

    #[test]
    fn test_category_from_id() {
        assert_eq!(QACategory::from_category_id(1), QACategory::SingleHop);
        assert_eq!(QACategory::from_category_id(2), QACategory::Temporal);
        assert_eq!(QACategory::from_category_id(3), QACategory::OpenDomain);
        assert_eq!(QACategory::from_category_id(4), QACategory::MultiHop);
        assert_eq!(QACategory::from_category_id(5), QACategory::Adversarial);
        // Unknown defaults to SingleHop
        assert_eq!(QACategory::from_category_id(99), QACategory::SingleHop);
    }

    #[test]
    fn test_category_labels() {
        assert_eq!(QACategory::SingleHop.label(), "Single-Hop");
        assert_eq!(QACategory::Temporal.label(), "Temporal");
        assert_eq!(QACategory::OpenDomain.label(), "Open-Domain");
        assert_eq!(QACategory::MultiHop.label(), "Multi-Hop");
        assert_eq!(QACategory::Adversarial.label(), "Adversarial");
    }

    #[test]
    fn test_category_ids_roundtrip() {
        for id in 1..=5 {
            let cat = QACategory::from_category_id(id);
            assert_eq!(cat.id(), id);
        }
    }

    // -- Dataset stats tests --

    #[test]
    fn test_dataset_stats() {
        let ds = sample_dataset();
        let stats = ds.stats();

        assert_eq!(stats.num_samples, 1);
        assert_eq!(stats.num_questions, 5);
        assert_eq!(stats.total_sessions, 3);
        assert_eq!(stats.total_turns, 8);
        assert_eq!(stats.by_category.get(&QACategory::SingleHop), Some(&2));
        assert_eq!(stats.by_category.get(&QACategory::Temporal), Some(&1));
        assert_eq!(stats.by_category.get(&QACategory::MultiHop), Some(&1));
        assert_eq!(stats.by_category.get(&QACategory::Adversarial), Some(&1));
    }

    // -- JSON parsing tests --

    #[test]
    fn test_parse_array_format() {
        let json = r#"[
            {
                "sample_id": "s1",
                "conversation": {
                    "speaker_a": "Alice",
                    "speaker_b": "Bob",
                    "session_1_date_time": "1:00 pm on 1 Jan, 2024",
                    "session_1": [
                        {"speaker": "Alice", "dia_id": "D1:1", "text": "Hello"}
                    ]
                },
                "qa": [
                    {"question": "Who said hello?", "answer": "Alice", "evidence": ["D1:1"], "category": 1}
                ]
            }
        ]"#;

        let ds = LoCoMoDataset::from_json_str(json, "test").unwrap();
        assert_eq!(ds.samples.len(), 1);
        assert_eq!(ds.samples[0].sample_id, "s1");
        assert_eq!(ds.samples[0].speaker_a, "Alice");
        assert_eq!(ds.samples[0].sessions.len(), 1);
        assert_eq!(ds.samples[0].sessions[0].turns.len(), 1);
        assert_eq!(ds.samples[0].qa.len(), 1);
        assert_eq!(ds.samples[0].qa[0].evidence, vec!["D1:1"]);
    }

    #[test]
    fn test_parse_single_object_format() {
        let json = r#"{
            "sample_id": "s1",
            "conversation": {
                "speaker_a": "X",
                "speaker_b": "Y",
                "session_1_date_time": "noon",
                "session_1": [
                    {"speaker": "X", "dia_id": "D1:1", "text": "Hi"}
                ]
            },
            "qa": []
        }"#;

        let ds = LoCoMoDataset::from_json_str(json, "single").unwrap();
        assert_eq!(ds.samples.len(), 1);
        assert_eq!(ds.samples[0].speaker_a, "X");
    }

    #[test]
    fn test_parse_multiple_sessions() {
        let json = r#"[{
            "sample_id": "multi",
            "conversation": {
                "speaker_a": "A",
                "speaker_b": "B",
                "session_1_date_time": "day 1",
                "session_1": [{"speaker": "A", "dia_id": "D1:1", "text": "one"}],
                "session_2_date_time": "day 2",
                "session_2": [{"speaker": "B", "dia_id": "D2:1", "text": "two"}],
                "session_3_date_time": "day 3",
                "session_3": [{"speaker": "A", "dia_id": "D3:1", "text": "three"}]
            },
            "qa": []
        }]"#;

        let ds = LoCoMoDataset::from_json_str(json, "test").unwrap();
        assert_eq!(ds.samples[0].sessions.len(), 3);
        assert_eq!(ds.samples[0].sessions[0].number, 1);
        assert_eq!(ds.samples[0].sessions[1].number, 2);
        assert_eq!(ds.samples[0].sessions[2].number, 3);
        assert_eq!(ds.samples[0].sessions[0].date_time, "day 1");
    }

    #[test]
    fn test_parse_invalid_json() {
        let result = LoCoMoDataset::from_json_str("not valid json", "bad");
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_adversarial_qa() {
        let json = r#"[{
            "sample_id": "adv",
            "conversation": {
                "speaker_a": "A",
                "speaker_b": "B",
                "session_1_date_time": "now",
                "session_1": [{"speaker": "A", "dia_id": "D1:1", "text": "hi"}]
            },
            "qa": [
                {
                    "question": "trick question?",
                    "answer": "no answer",
                    "evidence": [],
                    "category": 5,
                    "adversarial_answer": "wrong answer"
                }
            ]
        }]"#;

        let ds = LoCoMoDataset::from_json_str(json, "test").unwrap();
        assert_eq!(ds.samples[0].qa[0].category, 5);
        assert_eq!(
            ds.samples[0].qa[0].adversarial_answer.as_deref(),
            Some("wrong answer")
        );
    }

    #[test]
    fn test_parse_from_file_roundtrip() {
        let ds = sample_dataset();
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test_locomo.json");

        // Serialize to the LoCoMo JSON format
        let json = serialize_dataset_to_json(&ds);
        std::fs::write(&path, &json).unwrap();

        // Reload
        let loaded = LoCoMoDataset::from_file(&path).unwrap();
        assert_eq!(loaded.samples.len(), ds.samples.len());
        assert_eq!(loaded.samples[0].qa.len(), ds.samples[0].qa.len());
        assert_eq!(
            loaded.samples[0].sessions.len(),
            ds.samples[0].sessions.len()
        );
    }

    // -- Conversion tests --

    #[test]
    fn test_to_eval_dataset_turn_granularity() {
        let ds = sample_dataset();
        let eval = ds.to_eval_dataset_with_granularity(DocumentGranularity::Turn);

        assert_eq!(eval.name, "LoCoMo_test");
        // 8 turns across 3 sessions
        assert_eq!(eval.documents.len(), 8);
        // 5 QA questions
        assert_eq!(eval.queries.len(), 5);

        // Check document IDs are prefixed with sample_id
        assert!(eval.documents[0].id.starts_with("test_001:"));
        // Check content format
        assert!(eval.documents[0].content.starts_with("[Alice]:"));
    }

    #[test]
    fn test_to_eval_dataset_session_granularity() {
        let ds = sample_dataset();
        let eval = ds.to_eval_dataset_with_granularity(DocumentGranularity::Session);

        // 3 sessions
        assert_eq!(eval.documents.len(), 3);
        // 5 QA questions
        assert_eq!(eval.queries.len(), 5);

        // Check document IDs
        assert_eq!(eval.documents[0].id, "test_001:session_1");
        assert_eq!(eval.documents[1].id, "test_001:session_2");
        assert_eq!(eval.documents[2].id, "test_001:session_3");

        // Session content should contain all turns
        assert!(eval.documents[0].content.contains("[Alice]:"));
        assert!(eval.documents[0].content.contains("[Bob]:"));
    }

    #[test]
    fn test_turn_level_evidence_mapping() {
        let ds = sample_dataset();
        let eval = ds.to_eval_dataset_with_granularity(DocumentGranularity::Turn);

        // First QA: evidence=["D1:3"] → relevant_ids=["test_001:D1:3"]
        assert_eq!(eval.queries[0].relevant_ids, vec!["test_001:D1:3"]);

        // Multi-hop QA: evidence=["D1:3", "D3:2"]
        assert_eq!(
            eval.queries[3].relevant_ids,
            vec!["test_001:D1:3", "test_001:D3:2"]
        );

        // Adversarial QA: no evidence
        assert!(eval.queries[4].relevant_ids.is_empty());
    }

    #[test]
    fn test_session_level_evidence_mapping() {
        let ds = sample_dataset();
        let eval = ds.to_eval_dataset_with_granularity(DocumentGranularity::Session);

        // First QA: evidence=["D1:3"] → session_1
        assert_eq!(eval.queries[0].relevant_ids, vec!["test_001:session_1"]);

        // Multi-hop QA: evidence=["D1:3", "D3:2"] → session_1, session_3
        let mut expected = vec!["test_001:session_1", "test_001:session_3"];
        expected.sort();
        let mut actual = eval.queries[3].relevant_ids.clone();
        actual.sort();
        assert_eq!(actual, expected);
    }

    #[test]
    fn test_document_metadata_turn() {
        let ds = sample_dataset();
        let eval = ds.to_eval_dataset_with_granularity(DocumentGranularity::Turn);

        let doc = &eval.documents[0];
        assert_eq!(
            doc.metadata.get("speaker").and_then(|v| v.as_str()),
            Some("Alice")
        );
        assert_eq!(
            doc.metadata.get("dia_id").and_then(|v| v.as_str()),
            Some("D1:1")
        );
        assert_eq!(
            doc.metadata.get("session").and_then(|v| v.as_str()),
            Some("session_1")
        );
        assert_eq!(
            doc.metadata.get("source").and_then(|v| v.as_str()),
            Some("locomo")
        );
        assert_eq!(
            doc.metadata.get("sample_id").and_then(|v| v.as_str()),
            Some("test_001")
        );
    }

    #[test]
    fn test_document_metadata_session() {
        let ds = sample_dataset();
        let eval = ds.to_eval_dataset_with_granularity(DocumentGranularity::Session);

        let doc = &eval.documents[0];
        assert_eq!(
            doc.metadata.get("source").and_then(|v| v.as_str()),
            Some("locomo")
        );
        assert_eq!(
            doc.metadata.get("num_turns").and_then(|v| v.as_u64()),
            Some(3)
        );
        assert_eq!(
            doc.metadata.get("date").and_then(|v| v.as_str()),
            Some("1:56 pm on 8 May, 2023")
        );
    }

    #[test]
    fn test_query_metadata() {
        let ds = sample_dataset();
        let eval = ds.to_eval_dataset_with_granularity(DocumentGranularity::Turn);

        let q = &eval.queries[0];
        assert_eq!(
            q.metadata.get("answer").and_then(|v| v.as_str()),
            Some("About two weeks before 8 May 2023")
        );
        assert_eq!(
            q.metadata.get("category_id").and_then(|v| v.as_u64()),
            Some(1)
        );
        assert!(q.metadata.contains_key("evidence"));

        // Category label
        assert_eq!(q.category.as_deref(), Some("Single-Hop"));
    }

    #[test]
    fn test_query_categories_set_correctly() {
        let ds = sample_dataset();
        let eval = ds.to_eval_dataset();

        let categories: Vec<_> = eval.queries.iter().map(|q| q.category.as_deref()).collect();
        assert_eq!(
            categories,
            vec![
                Some("Single-Hop"),
                Some("Single-Hop"),
                Some("Temporal"),
                Some("Multi-Hop"),
                Some("Adversarial"),
            ]
        );
    }

    #[test]
    fn test_dedup_across_samples() {
        // Create dataset with 2 samples sharing no documents
        let ds = LoCoMoDataset {
            samples: vec![
                LoCoMoSample {
                    sample_id: "s1".to_string(),
                    speaker_a: "A".to_string(),
                    speaker_b: "B".to_string(),
                    sessions: vec![ConversationSession {
                        key: "session_1".to_string(),
                        number: 1,
                        date_time: "day 1".to_string(),
                        turns: vec![make_turn("A", "D1:1", "Hello from s1")],
                    }],
                    qa: vec![make_qa("Q1?", "A1", vec!["D1:1"], 1)],
                    event_summaries: HashMap::new(),
                },
                LoCoMoSample {
                    sample_id: "s2".to_string(),
                    speaker_a: "X".to_string(),
                    speaker_b: "Y".to_string(),
                    sessions: vec![ConversationSession {
                        key: "session_1".to_string(),
                        number: 1,
                        date_time: "day 2".to_string(),
                        turns: vec![make_turn("X", "D1:1", "Hello from s2")],
                    }],
                    qa: vec![make_qa("Q2?", "A2", vec!["D1:1"], 2)],
                    event_summaries: HashMap::new(),
                },
            ],
            variant: "dedup_test".to_string(),
        };

        let eval = ds.to_eval_dataset_with_granularity(DocumentGranularity::Turn);
        // 2 documents (different sample_ids make them unique: s1:D1:1, s2:D1:1)
        assert_eq!(eval.documents.len(), 2);
        assert_eq!(eval.queries.len(), 2);
    }

    // -- QA filtering test --

    #[test]
    fn test_qa_by_category() {
        let ds = sample_dataset();
        let single_hop = ds.qa_by_category(QACategory::SingleHop);
        assert_eq!(single_hop.len(), 2);

        let temporal = ds.qa_by_category(QACategory::Temporal);
        assert_eq!(temporal.len(), 1);

        let adversarial = ds.qa_by_category(QACategory::Adversarial);
        assert_eq!(adversarial.len(), 1);

        let open_domain = ds.qa_by_category(QACategory::OpenDomain);
        assert_eq!(open_domain.len(), 0);
    }

    // -- Benchmark execution tests --

    #[test]
    fn test_run_benchmark_from_dataset_perfect() {
        let ds = sample_dataset();
        let config = EvalConfig {
            k_values: vec![1, 5],
            per_category: true,
            ..Default::default()
        };

        // Perfect retrieval: return exact evidence docs
        let eval_ds = ds.to_eval_dataset_with_granularity(DocumentGranularity::Turn);
        let doc_ids: HashMap<String, Vec<String>> = eval_ds
            .queries
            .iter()
            .map(|q| (q.query.clone(), q.relevant_ids.clone()))
            .collect();

        let report = run_benchmark_from_dataset(&ds, DocumentGranularity::Turn, config, |query| {
            doc_ids.get(query).cloned().unwrap_or_default()
        })
        .unwrap();

        // With perfect retrieval, non-adversarial queries should have high recall
        // Adversarial has empty evidence, so it's a special case
        assert!(report.eval_report.aggregate.mrr > 0.5);
    }

    #[test]
    fn test_run_benchmark_from_dataset_poor() {
        let ds = sample_dataset();
        let config = EvalConfig {
            k_values: vec![1, 5],
            per_category: true,
            ..Default::default()
        };

        // Return irrelevant docs
        let report = run_benchmark_from_dataset(&ds, DocumentGranularity::Turn, config, |_| {
            vec!["nonexistent_doc".to_string()]
        })
        .unwrap();

        // Poor retrieval should yield low metrics
        assert!(report.eval_report.aggregate.mean_f1 < 0.1);
    }

    #[test]
    fn test_per_category_aggregation() {
        let ds = sample_dataset();
        let config = EvalConfig {
            k_values: vec![5],
            per_category: true,
            ..Default::default()
        };

        let eval_ds = ds.to_eval_dataset_with_granularity(DocumentGranularity::Turn);
        let doc_ids: HashMap<String, Vec<String>> = eval_ds
            .queries
            .iter()
            .map(|q| (q.query.clone(), q.relevant_ids.clone()))
            .collect();

        let report = run_benchmark_from_dataset(&ds, DocumentGranularity::Turn, config, |query| {
            doc_ids.get(query).cloned().unwrap_or_default()
        })
        .unwrap();

        // Should have per-category results
        assert!(!report.per_category_results.is_empty());
        // Categories present: Single-Hop, Temporal, Multi-Hop, Adversarial
        assert!(report.per_category_results.contains_key("Single-Hop"));
        assert!(report.per_category_results.contains_key("Temporal"));
        assert!(report.per_category_results.contains_key("Multi-Hop"));
        assert!(report.per_category_results.contains_key("Adversarial"));
    }

    // -- Report generation tests --

    #[test]
    fn test_report_to_markdown() {
        let ds = sample_dataset();
        let config = EvalConfig {
            k_values: vec![1, 5],
            per_category: true,
            ..Default::default()
        };

        let report =
            run_benchmark_from_dataset(&ds, DocumentGranularity::Turn, config, |_| vec![]).unwrap();

        let md = report.to_markdown();
        assert!(md.contains("# LoCoMo Benchmark Results"));
        assert!(md.contains("## Overall Retrieval Metrics"));
        assert!(md.contains("## Per-Category Results"));
        assert!(md.contains("## Comparison with Published Baselines"));
        assert!(md.contains("**Ucotron**"));
        assert!(md.contains("Human"));
        assert!(md.contains("Mem0"));
    }

    #[test]
    fn test_report_to_json() {
        let ds = sample_dataset();
        let config = EvalConfig {
            k_values: vec![5],
            per_category: true,
            ..Default::default()
        };

        let report =
            run_benchmark_from_dataset(&ds, DocumentGranularity::Turn, config, |_| vec![]).unwrap();

        let json = report.to_json().unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert!(parsed.get("eval_report").is_some());
        assert!(parsed.get("baselines").is_some());
        assert!(parsed.get("per_category_results").is_some());
    }

    // -- Baseline tests --

    #[test]
    fn test_published_baselines() {
        let baselines = published_baselines();
        assert!(baselines.len() >= 5);

        let system_names: Vec<_> = baselines.iter().map(|b| b.system.as_str()).collect();
        assert!(system_names.contains(&"Human"));
        assert!(system_names.contains(&"Mem0"));
        assert!(system_names.contains(&"Zep"));
        assert!(system_names.contains(&"MemMachine"));
    }

    // -- Helper for file roundtrip tests --

    /// Serialize a LoCoMoDataset back to JSON (test helper).
    fn serialize_dataset_to_json(ds: &LoCoMoDataset) -> String {
        let mut samples = Vec::new();

        for sample in &ds.samples {
            let mut conv = serde_json::Map::new();
            conv.insert(
                "speaker_a".into(),
                serde_json::Value::String(sample.speaker_a.clone()),
            );
            conv.insert(
                "speaker_b".into(),
                serde_json::Value::String(sample.speaker_b.clone()),
            );

            for session in &sample.sessions {
                conv.insert(
                    format!("{}_date_time", session.key),
                    serde_json::Value::String(session.date_time.clone()),
                );
                conv.insert(
                    session.key.clone(),
                    serde_json::to_value(&session.turns).unwrap(),
                );
            }

            let mut obj = serde_json::Map::new();
            obj.insert(
                "sample_id".into(),
                serde_json::Value::String(sample.sample_id.clone()),
            );
            obj.insert("conversation".into(), serde_json::Value::Object(conv));
            obj.insert("qa".into(), serde_json::to_value(&sample.qa).unwrap());

            samples.push(serde_json::Value::Object(obj));
        }

        serde_json::to_string_pretty(&samples).unwrap()
    }
}
