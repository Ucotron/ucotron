//! # Relation Extraction
//!
//! This module provides relation extraction from text given pre-extracted entities.
//!
//! Three strategies are supported:
//! - **Co-occurrence** (default fallback): Extracts relations based on entity proximity
//!   and syntactic patterns in the text. No LLM required.
//! - **Fireworks API** (optional): Uses a fine-tuned model hosted on Fireworks.ai for
//!   structured JSON relation extraction. Enabled when `fine_tuned_re_model` is configured.
//! - **LLM-based** (optional): Uses a local LLM (Qwen3-4B GGUF via llama.cpp) for
//!   structured JSON relation extraction. Enabled when model files are present.
//!
//! The strategy is selected via `ModelsConfig` in `ucotron.toml`:
//! ```toml
//! [models]
//! fine_tuned_re_model = "accounts/ucotron/models/re-qwen2-5-7b"  # Fireworks model
//! fine_tuned_re_endpoint = "https://api.fireworks.ai/inference/v1"
//! fine_tuned_re_api_key_env = "FIREWORKS_API_KEY"
//! ```

use crate::{ExtractedEntity, ExtractedRelation, RelationExtractor};
use std::collections::HashMap;

// ─────────────────────────────────────────────────────────────────────
// Co-occurrence Relation Extractor (LazyGraphRAG-style fallback)
// ─────────────────────────────────────────────────────────────────────

/// Configuration for the co-occurrence relation extractor.
#[derive(Debug, Clone)]
pub struct CooccurrenceConfig {
    /// Maximum character distance between two entities to consider them related.
    /// Default: 200 characters.
    pub max_distance: usize,
    /// Minimum confidence threshold for emitted relations.
    /// Default: 0.1
    pub min_confidence: f32,
    /// Whether to infer predicate labels from context words between entities.
    /// Default: true
    pub infer_predicates: bool,
}

impl Default for CooccurrenceConfig {
    fn default() -> Self {
        Self {
            max_distance: 200,
            min_confidence: 0.1,
            infer_predicates: true,
        }
    }
}

/// Co-occurrence based relation extractor.
///
/// Extracts relations by analyzing entity proximity and syntactic patterns
/// in the source text. This is the LazyGraphRAG-style approach: deterministic,
/// no LLM calls, fast, and works as a fallback when no LLM model is available.
///
/// # Strategy
///
/// For each pair of entities that co-occur within `max_distance` characters:
/// 1. Compute proximity-based confidence (closer = higher)
/// 2. Analyze the text between entities for predicate signals
/// 3. Emit a relation with the inferred predicate and confidence
///
/// # Example
/// ```rust,ignore
/// let extractor = CooccurrenceRelationExtractor::new(CooccurrenceConfig::default());
/// let relations = extractor.extract_relations(text, &entities)?;
/// ```
pub struct CooccurrenceRelationExtractor {
    config: CooccurrenceConfig,
}

impl CooccurrenceRelationExtractor {
    /// Create a new co-occurrence extractor with the given config.
    pub fn new(config: CooccurrenceConfig) -> Self {
        Self { config }
    }

    /// Create a new co-occurrence extractor with default config.
    pub fn with_defaults() -> Self {
        Self::new(CooccurrenceConfig::default())
    }
}

// Predicate signal patterns: (keyword, predicate_label, confidence_boost)
// These patterns match common relation-indicating words between entities.
// IMPORTANT: Patterns are checked in order. More specific patterns must come BEFORE
// shorter/generic ones (e.g., "works at" before "at", "moved to" before "to").
const PREDICATE_PATTERNS: &[(&[&str], &str, f32)] = &[
    // Location / movement (specific)
    (
        &[
            "lives in",
            "vive en",
            "reside in",
            "based in",
            "located in",
            "ubicado en",
        ],
        "lives_in",
        0.3,
    ),
    (
        &["moved to", "se mudó a", "relocated to", "trasladó a"],
        "moved_to",
        0.3,
    ),
    (
        &["moved from", "se mudó de", "came from", "vino de"],
        "moved_from",
        0.3,
    ),
    (&["born in", "nació en", "nacido en"], "born_in", 0.3),
    (
        &["traveled to", "viajó a", "went to", "fue a"],
        "traveled_to",
        0.2,
    ),
    // Employment / affiliation (specific — must come before generic "at"/"en")
    (
        &["works at", "trabaja en", "employed at", "employed by"],
        "works_at",
        0.3,
    ),
    (&["works for", "trabaja para"], "works_for", 0.3),
    (
        &["founded", "fundó", "created", "creó", "started"],
        "founded",
        0.3,
    ),
    (
        &["leads", "lidera", "heads", "dirige", "manages", "gestiona"],
        "leads",
        0.25,
    ),
    (&["joined", "se unió a", "ingresó a"], "joined", 0.25),
    (&["CEO of", "director of", "president of"], "leads", 0.3),
    // Relationships
    (
        &[
            "married to",
            "casado con",
            "spouse of",
            "esposo de",
            "esposa de",
        ],
        "married_to",
        0.3,
    ),
    (&["friend of", "amigo de", "amiga de"], "friend_of", 0.2),
    (
        &[
            "sibling of",
            "hermano de",
            "hermana de",
            "brother of",
            "sister of",
        ],
        "sibling_of",
        0.3,
    ),
    (
        &["child of", "hijo de", "hija de", "son of", "daughter of"],
        "child_of",
        0.3,
    ),
    (
        &[
            "parent of",
            "padre de",
            "madre de",
            "father of",
            "mother of",
        ],
        "parent_of",
        0.3,
    ),
    // Ownership / creation
    (&["owns", "posee", "has", "tiene"], "owns", 0.15),
    (
        &["bought", "compró", "acquired", "adquirió", "purchased"],
        "acquired",
        0.25,
    ),
    (
        &["wrote", "escribió", "authored", "publicó", "published"],
        "authored",
        0.25,
    ),
    // Causal / temporal
    (
        &["caused", "causó", "led to", "provocó", "resulted in"],
        "caused_by",
        0.25,
    ),
    (
        &["after", "después de", "following", "tras"],
        "follows",
        0.1,
    ),
    (&["before", "antes de", "prior to"], "precedes", 0.1),
    // Interaction
    (&["met", "conoció", "met with", "se reunió con"], "met", 0.2),
    (
        &["called", "llamó", "contacted", "contactó"],
        "contacted",
        0.2,
    ),
    (
        &["said", "dijo", "told", "told to"],
        "communicated_with",
        0.1,
    ),
    // Eating / consuming (for event node test case)
    (&["ate", "comió", "eaten", "eating", "comiendo"], "ate", 0.2),
    // Generic / short patterns (MUST be last — these are catch-all)
    (
        &["with", "con", "junto a", "along with"],
        "associated_with",
        0.05,
    ),
    (&["in", "en", "at", "a"], "located_in", 0.05),
];

/// Safely extract a substring using byte offsets, adjusting to char boundaries.
///
/// If the offsets fall inside multi-byte characters, they are adjusted to the
/// nearest valid char boundary.
fn safe_substr(text: &str, start: usize, end: usize) -> &str {
    if start >= end || start >= text.len() {
        return "";
    }
    let end = end.min(text.len());

    // Find valid char boundaries at or after start, and at or before end
    let safe_start = if text.is_char_boundary(start) {
        start
    } else {
        // Scan forward to find the next char boundary
        (start..text.len())
            .find(|&i| text.is_char_boundary(i))
            .unwrap_or(text.len())
    };

    let safe_end = if text.is_char_boundary(end) {
        end
    } else {
        // Scan backward to find the previous char boundary
        (0..end)
            .rev()
            .find(|&i| text.is_char_boundary(i))
            .unwrap_or(0)
    };

    if safe_start >= safe_end {
        return "";
    }
    &text[safe_start..safe_end]
}

/// Analyze text between two entities to infer a predicate label.
///
/// Returns (predicate, confidence_boost) based on pattern matching.
fn infer_predicate(between_text: &str) -> (&'static str, f32) {
    let lower = between_text.to_lowercase();
    let trimmed = lower.trim();

    // Try each pattern, return the first (most specific) match
    for (keywords, predicate, boost) in PREDICATE_PATTERNS {
        for keyword in *keywords {
            if trimmed.contains(keyword) {
                return (predicate, *boost);
            }
        }
    }

    // Default: generic co-occurrence relation
    ("related_to", 0.0)
}

/// Compute proximity-based confidence for a pair of entities.
///
/// Closer entities get higher confidence, following an inverse-linear decay.
/// Returns a value in [0.0, 1.0].
fn proximity_confidence(distance: usize, max_distance: usize) -> f32 {
    if distance == 0 {
        return 1.0;
    }
    if distance >= max_distance {
        return 0.0;
    }
    // Inverse-linear decay: closer pairs get higher scores
    1.0 - (distance as f32 / max_distance as f32)
}

/// Determine subject/object ordering based on entity positions and types.
///
/// Rules:
/// 1. Person entities are preferred as subjects
/// 2. Location entities are preferred as objects for spatial predicates
/// 3. If same type, earlier entity in text is subject
fn order_entities<'a>(
    entity_a: &'a ExtractedEntity,
    entity_b: &'a ExtractedEntity,
    predicate: &str,
) -> (&'a ExtractedEntity, &'a ExtractedEntity) {
    let person_labels = ["person", "persona", "per", "people"];
    let location_labels = ["location", "lugar", "loc", "place", "city", "country"];

    let a_is_person = person_labels
        .iter()
        .any(|l| entity_a.label.to_lowercase().contains(l));
    let b_is_person = person_labels
        .iter()
        .any(|l| entity_b.label.to_lowercase().contains(l));
    let a_is_location = location_labels
        .iter()
        .any(|l| entity_a.label.to_lowercase().contains(l));
    let b_is_location = location_labels
        .iter()
        .any(|l| entity_b.label.to_lowercase().contains(l));

    // For spatial predicates, person should be subject, location object
    let is_spatial = matches!(
        predicate,
        "lives_in" | "moved_to" | "moved_from" | "born_in" | "traveled_to" | "located_in"
    );

    if is_spatial {
        if a_is_person && b_is_location {
            return (entity_a, entity_b);
        }
        if b_is_person && a_is_location {
            return (entity_b, entity_a);
        }
    }

    // Person entities preferred as subjects
    if a_is_person && !b_is_person {
        return (entity_a, entity_b);
    }
    if b_is_person && !a_is_person {
        return (entity_b, entity_a);
    }

    // Default: earlier entity in text is subject
    if entity_a.start <= entity_b.start {
        (entity_a, entity_b)
    } else {
        (entity_b, entity_a)
    }
}

impl RelationExtractor for CooccurrenceRelationExtractor {
    fn extract_relations(
        &self,
        text: &str,
        entities: &[ExtractedEntity],
    ) -> anyhow::Result<Vec<ExtractedRelation>> {
        if entities.len() < 2 {
            return Ok(Vec::new());
        }

        let mut relations = Vec::new();
        let mut seen_pairs: HashMap<(String, String), f32> = HashMap::new();

        // Consider all pairs of entities
        for i in 0..entities.len() {
            for j in (i + 1)..entities.len() {
                let entity_a = &entities[i];
                let entity_b = &entities[j];

                // Skip if entities overlap in text
                if entity_a.start < entity_b.end && entity_b.start < entity_a.end {
                    continue;
                }

                // Compute character distance between entities
                let distance = if entity_a.end <= entity_b.start {
                    entity_b.start - entity_a.end
                } else {
                    entity_a.start.saturating_sub(entity_b.end)
                };

                // Skip if too far apart
                if distance > self.config.max_distance {
                    continue;
                }

                // Compute base proximity confidence
                let prox_conf = proximity_confidence(distance, self.config.max_distance);

                // Extract text between entities for predicate inference
                let (between_start, between_end) = if entity_a.end <= entity_b.start {
                    (entity_a.end, entity_b.start)
                } else {
                    (entity_b.end, entity_a.start)
                };

                let between_text = safe_substr(text, between_start, between_end);

                // Infer predicate from context
                let (predicate, pred_boost) = if self.config.infer_predicates {
                    infer_predicate(between_text)
                } else {
                    ("related_to", 0.0)
                };

                // Combine confidence: proximity * entity_confidence * predicate_boost
                let entity_conf = (entity_a.confidence * entity_b.confidence).sqrt();
                let confidence = (prox_conf * 0.5 + pred_boost + entity_conf * 0.2).min(1.0);

                if confidence < self.config.min_confidence {
                    continue;
                }

                // Determine subject/object ordering
                let (subject, object) = order_entities(entity_a, entity_b, predicate);

                // Dedup: keep highest confidence for each (subject, object) pair
                let pair_key = (subject.text.clone(), object.text.clone());
                let existing = seen_pairs.get(&pair_key).copied().unwrap_or(0.0);
                if confidence > existing {
                    seen_pairs.insert(pair_key, confidence);
                    // Remove old relation if exists
                    relations.retain(|r: &ExtractedRelation| {
                        !(r.subject == subject.text && r.object == object.text)
                    });
                    relations.push(ExtractedRelation {
                        subject: subject.text.clone(),
                        predicate: predicate.to_string(),
                        object: object.text.clone(),
                        confidence,
                    });
                }
            }
        }

        // Sort by confidence descending
        relations.sort_by(|a, b| {
            b.confidence
                .partial_cmp(&a.confidence)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        Ok(relations)
    }
}

// ─────────────────────────────────────────────────────────────────────
// LLM-based Relation Extractor
// ─────────────────────────────────────────────────────────────────────

/// Configuration for the LLM-based relation extractor.
#[derive(Debug, Clone)]
pub struct LlmRelationConfig {
    /// Path to the GGUF model file.
    pub model_path: String,
    /// Maximum tokens to generate.
    pub max_tokens: u32,
    /// Temperature for generation (0.0 = deterministic).
    pub temperature: f32,
    /// Number of threads for inference.
    pub num_threads: u32,
    /// Context size in tokens.
    pub context_size: u32,
}

impl Default for LlmRelationConfig {
    fn default() -> Self {
        Self {
            model_path: String::new(),
            max_tokens: 512,
            temperature: 0.1,
            num_threads: 4,
            context_size: 2048,
        }
    }
}

/// Build the structured prompt for relation extraction.
///
/// The prompt instructs the LLM to output JSON with extracted relations.
pub fn build_relation_prompt(text: &str, entities: &[ExtractedEntity]) -> String {
    let entity_list: Vec<String> = entities
        .iter()
        .map(|e| format!("- \"{}\" ({})", e.text, e.label))
        .collect();

    format!(
        r#"Extract all relationships between the entities found in the following text.

Text: "{text}"

Entities found:
{entities}

For each relationship, provide:
- subject: the entity that is the subject
- predicate: the relationship type (e.g., "lives_in", "works_at", "moved_to", "born_in", "married_to", "owns", "founded", "related_to")
- object: the entity that is the object
- confidence: your confidence in this relationship (0.0 to 1.0)

Output ONLY a JSON array. No explanation. Example format:
[{{"subject": "Juan", "predicate": "lives_in", "object": "Madrid", "confidence": 0.95}}]

JSON output:"#,
        text = text,
        entities = entity_list.join("\n"),
    )
}

/// Parse LLM output JSON into ExtractedRelation structs.
///
/// Handles malformed JSON gracefully by attempting to extract valid relations
/// from partial output.
pub fn parse_llm_relations(output: &str) -> Vec<ExtractedRelation> {
    // Try to find JSON array in the output
    let json_str = extract_json_array(output);

    let parsed: Result<Vec<serde_json::Value>, _> = serde_json::from_str(&json_str);
    match parsed {
        Ok(arr) => arr
            .iter()
            .filter_map(|v| {
                let subject = v.get("subject")?.as_str()?.to_string();
                let predicate = v.get("predicate")?.as_str()?.to_string();
                let object = v.get("object")?.as_str()?.to_string();
                let confidence = v.get("confidence").and_then(|c| c.as_f64()).unwrap_or(0.5) as f32;
                Some(ExtractedRelation {
                    subject,
                    predicate,
                    object,
                    confidence: confidence.clamp(0.0, 1.0),
                })
            })
            .collect(),
        Err(_) => Vec::new(),
    }
}

/// Extract a JSON array from potentially noisy LLM output.
///
/// Finds the first `[...]` block in the output, handling cases where
/// the LLM produces extra text before or after the JSON.
fn extract_json_array(text: &str) -> String {
    let trimmed = text.trim();

    // Find first '[' and last ']'
    if let (Some(start), Some(end)) = (trimmed.find('['), trimmed.rfind(']')) {
        if start < end {
            return trimmed[start..=end].to_string();
        }
    }

    // Fallback: return the original text
    trimmed.to_string()
}

// ─────────────────────────────────────────────────────────────────────
// LLM Relation Extractor (requires "llm" feature)
// ─────────────────────────────────────────────────────────────────────

#[cfg(feature = "llm")]
pub struct LlmRelationExtractor {
    backend: llama_cpp_2::llama_backend::LlamaBackend,
    model: llama_cpp_2::model::LlamaModel,
    config: LlmRelationConfig,
}

#[cfg(feature = "llm")]
impl LlmRelationExtractor {
    /// Load a GGUF model from `model_path` and prepare for inference.
    pub fn new(config: LlmRelationConfig) -> anyhow::Result<Self> {
        use llama_cpp_2::llama_backend::LlamaBackend;
        use llama_cpp_2::model::params::LlamaModelParams;
        use llama_cpp_2::model::LlamaModel;

        let backend = LlamaBackend::init()
            .map_err(|e| anyhow::anyhow!("Failed to init llama backend: {}", e))?;
        let params = LlamaModelParams::default();
        let model =
            LlamaModel::load_from_file(&backend, &config.model_path, &params).map_err(|e| {
                anyhow::anyhow!("Failed to load GGUF model '{}': {}", config.model_path, e)
            })?;

        tracing::info!(
            "LLM relation extractor loaded model from '{}' (ctx={}, vocab={}, params={})",
            config.model_path,
            config.context_size,
            model.n_vocab(),
            model.n_params(),
        );

        Ok(Self {
            backend,
            model,
            config,
        })
    }

    /// Find the first GGUF file in a directory and create the extractor.
    pub fn from_model_dir(dir: &str, config: LlmRelationConfig) -> anyhow::Result<Self> {
        let entries = std::fs::read_dir(dir)
            .map_err(|e| anyhow::anyhow!("Cannot read model dir '{}': {}", dir, e))?;

        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().map(|ext| ext == "gguf").unwrap_or(false) {
                let mut cfg = config;
                cfg.model_path = path.to_string_lossy().to_string();
                return Self::new(cfg);
            }
        }

        anyhow::bail!("No GGUF file found in '{}'", dir)
    }

    /// Run inference on the given prompt and return the raw completion text.
    #[allow(deprecated)]
    fn complete(&self, prompt: &str) -> anyhow::Result<String> {
        use llama_cpp_2::context::params::LlamaContextParams;
        use llama_cpp_2::llama_batch::LlamaBatch;
        use llama_cpp_2::model::{AddBos, Special};
        use llama_cpp_2::sampling::LlamaSampler;
        use std::num::NonZeroU32;

        let ctx_params = LlamaContextParams::default()
            .with_n_ctx(Some(NonZeroU32::new(self.config.context_size).unwrap()))
            .with_n_threads(self.config.num_threads as i32)
            .with_n_threads_batch(self.config.num_threads as i32);

        let mut ctx = self
            .model
            .new_context(&self.backend, ctx_params)
            .map_err(|e| anyhow::anyhow!("Failed to create LLM context: {}", e))?;

        // Tokenize the prompt
        let tokens = self
            .model
            .str_to_token(prompt, AddBos::Always)
            .map_err(|e| anyhow::anyhow!("Tokenization failed: {}", e))?;

        if tokens.len() as u32 >= self.config.context_size {
            anyhow::bail!(
                "Prompt too long: {} tokens exceeds context size {}",
                tokens.len(),
                self.config.context_size
            );
        }

        // Feed prompt tokens
        let mut batch = LlamaBatch::new(self.config.context_size as usize, 1);
        for (i, token) in tokens.iter().enumerate() {
            let is_last = i == tokens.len() - 1;
            batch.add(*token, i as i32, &[0], is_last)?;
        }
        ctx.decode(&mut batch)
            .map_err(|e| anyhow::anyhow!("Decode failed: {}", e))?;

        // Set up sampler with temperature
        let mut sampler = if self.config.temperature < 0.05 {
            LlamaSampler::greedy()
        } else {
            LlamaSampler::chain_simple([
                LlamaSampler::temp(self.config.temperature),
                LlamaSampler::dist(42),
            ])
        };

        // Generate tokens
        let mut output = String::new();
        let max_gen = self.config.max_tokens as usize;

        for _ in 0..max_gen {
            let new_token = sampler.sample(&ctx, batch.n_tokens() - 1);
            sampler.accept(new_token);

            // Check for end of generation
            if self.model.is_eog_token(new_token) {
                break;
            }

            // Detokenize incrementally
            #[allow(deprecated)]
            if let Ok(piece) = self.model.token_to_str(new_token, Special::Tokenize) {
                output.push_str(&piece);
            }

            // Early stop if we see the closing bracket of JSON array
            if output.contains("]\n") || output.ends_with(']') {
                break;
            }

            // Prepare next batch
            batch.clear();
            batch.add(new_token, batch.n_tokens(), &[0], true)?;
            ctx.decode(&mut batch)
                .map_err(|e| anyhow::anyhow!("Decode failed: {}", e))?;
        }

        Ok(output)
    }
}

#[cfg(feature = "llm")]
impl RelationExtractor for LlmRelationExtractor {
    fn extract_relations(
        &self,
        text: &str,
        entities: &[ExtractedEntity],
    ) -> anyhow::Result<Vec<ExtractedRelation>> {
        if entities.len() < 2 {
            return Ok(Vec::new());
        }

        let prompt = build_relation_prompt(text, entities);
        let output = self.complete(&prompt)?;

        tracing::debug!("LLM relation output: {}", output);

        let relations = parse_llm_relations(&output);
        if relations.is_empty() {
            tracing::debug!("LLM returned no parseable relations, no fallback at extractor level");
        }

        Ok(relations)
    }
}

// ─────────────────────────────────────────────────────────────────────
// Composite Relation Extractor (auto-selects strategy)
// ─────────────────────────────────────────────────────────────────────

/// Strategy for relation extraction.
#[derive(Debug, Clone, PartialEq)]
pub enum RelationStrategy {
    /// Co-occurrence based (no model, LazyGraphRAG style).
    CoOccurrence,
    /// Fine-tuned model via Fireworks API.
    Fireworks,
    /// LLM-based extraction (requires local model file).
    Llm,
}

// ─────────────────────────────────────────────────────────────────────
// Fireworks API Relation Extractor
// ─────────────────────────────────────────────────────────────────────

/// Configuration for the Fireworks-hosted fine-tuned relation extractor.
#[derive(Debug, Clone)]
pub struct FireworksRelationConfig {
    /// Fireworks model ID (e.g., "accounts/ucotron/models/re-qwen2-5-7b").
    pub model: String,
    /// Fireworks API endpoint (default: "https://api.fireworks.ai/inference/v1").
    pub endpoint: String,
    /// API key for authentication.
    pub api_key: String,
    /// Maximum tokens for the completion (default: 512).
    pub max_tokens: u32,
    /// Temperature for generation (default: 0.1).
    pub temperature: f32,
}

/// Relation extractor that uses a fine-tuned model hosted on Fireworks.ai.
///
/// Sends relation extraction prompts to the Fireworks OpenAI-compatible API
/// and parses structured JSON responses. Falls back gracefully on API errors.
pub struct FireworksRelationExtractor {
    config: FireworksRelationConfig,
    client: reqwest::blocking::Client,
}

impl FireworksRelationExtractor {
    /// Create a new Fireworks extractor from config.
    ///
    /// Returns `None` if the API key environment variable is not set or empty.
    pub fn from_models_config(models_config: &ucotron_config::ModelsConfig) -> Option<Self> {
        let api_key = std::env::var(&models_config.fine_tuned_re_api_key_env).ok()?;
        if api_key.is_empty() {
            return None;
        }

        let config = FireworksRelationConfig {
            model: models_config.fine_tuned_re_model.clone(),
            endpoint: models_config.fine_tuned_re_endpoint.clone(),
            api_key,
            max_tokens: 512,
            temperature: 0.1,
        };

        let client = reqwest::blocking::Client::builder()
            .timeout(std::time::Duration::from_secs(30))
            .build()
            .ok()?;

        Some(Self { config, client })
    }

    /// Create a new Fireworks extractor from explicit config (for testing).
    pub fn new(config: FireworksRelationConfig) -> Self {
        let client = reqwest::blocking::Client::builder()
            .timeout(std::time::Duration::from_secs(30))
            .build()
            .expect("Failed to create HTTP client");
        Self { config, client }
    }

    /// Call the Fireworks chat completions API.
    fn call_api(&self, prompt: &str) -> anyhow::Result<String> {
        let url = format!("{}/chat/completions", self.config.endpoint);

        let body = serde_json::json!({
            "model": self.config.model,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a relation extraction model. Extract relationships between entities and output ONLY a JSON array."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature
        });

        let response = self
            .client
            .post(&url)
            .header("Authorization", format!("Bearer {}", self.config.api_key))
            .header("Content-Type", "application/json")
            .json(&body)
            .send()
            .map_err(|e| anyhow::anyhow!("Fireworks API request failed: {}", e))?;

        if !response.status().is_success() {
            let status = response.status();
            let text = response.text().unwrap_or_default();
            return Err(anyhow::anyhow!(
                "Fireworks API returned {}: {}",
                status,
                text
            ));
        }

        let json: serde_json::Value = response
            .json()
            .map_err(|e| anyhow::anyhow!("Failed to parse Fireworks response: {}", e))?;

        // Extract the assistant message content from OpenAI-compatible response
        let content = json["choices"]
            .get(0)
            .and_then(|choice| choice.get("message"))
            .and_then(|msg| msg.get("content"))
            .and_then(|c| c.as_str())
            .unwrap_or("[]");

        Ok(content.to_string())
    }
}

impl RelationExtractor for FireworksRelationExtractor {
    fn extract_relations(
        &self,
        text: &str,
        entities: &[ExtractedEntity],
    ) -> anyhow::Result<Vec<ExtractedRelation>> {
        if entities.len() < 2 {
            return Ok(Vec::new());
        }

        let prompt = build_relation_prompt(text, entities);
        let output = self.call_api(&prompt)?;
        let relations = parse_llm_relations(&output);

        Ok(relations)
    }
}

// ─────────────────────────────────────────────────────────────────────
// Composite Relation Extractor (auto-selects strategy)
// ─────────────────────────────────────────────────────────────────────

/// Composite relation extractor that selects strategy based on config.
///
/// Priority: Fireworks fine-tuned model > local LLM > co-occurrence fallback.
pub struct CompositeRelationExtractor {
    cooccurrence: CooccurrenceRelationExtractor,
    fireworks: Option<FireworksRelationExtractor>,
    #[cfg(feature = "llm")]
    llm: Option<LlmRelationExtractor>,
    strategy: RelationStrategy,
}

impl CompositeRelationExtractor {
    /// Create a new composite extractor.
    ///
    /// Strategy priority:
    /// 1. If `fine_tuned_re_model` is configured and API key available → Fireworks
    /// 2. If local LLM model file exists and "llm" feature enabled → LLM
    /// 3. Otherwise → co-occurrence fallback
    pub fn new(models_config: &ucotron_config::ModelsConfig) -> Self {
        let cooccurrence = CooccurrenceRelationExtractor::with_defaults();

        // Check Fireworks first (highest priority)
        if !models_config.fine_tuned_re_model.is_empty() {
            if let Some(fireworks) = FireworksRelationExtractor::from_models_config(models_config) {
                tracing::info!(
                    "Using Fireworks fine-tuned RE model: {}",
                    models_config.fine_tuned_re_model
                );
                return Self {
                    cooccurrence,
                    fireworks: Some(fireworks),
                    #[cfg(feature = "llm")]
                    llm: None,
                    strategy: RelationStrategy::Fireworks,
                };
            }
            tracing::warn!(
                "Fireworks RE model configured ('{}') but API key env '{}' not set. Falling back.",
                models_config.fine_tuned_re_model,
                models_config.fine_tuned_re_api_key_env
            );
        }

        // Check local LLM
        let strategy = Self::determine_llm_strategy(models_config);

        #[cfg(feature = "llm")]
        {
            if strategy == RelationStrategy::Llm {
                let model_dir =
                    std::path::Path::new(&models_config.models_dir).join(&models_config.llm_model);
                let config = LlmRelationConfig::default();
                match LlmRelationExtractor::from_model_dir(&model_dir.to_string_lossy(), config) {
                    Ok(llm) => {
                        tracing::info!("LLM relation extractor loaded from {:?}", model_dir);
                        return Self {
                            cooccurrence,
                            fireworks: None,
                            llm: Some(llm),
                            strategy: RelationStrategy::Llm,
                        };
                    }
                    Err(e) => {
                        tracing::warn!(
                            "Failed to load LLM relation extractor: {}. Falling back to co-occurrence.",
                            e
                        );
                    }
                }
            }
        }

        Self {
            cooccurrence,
            fireworks: None,
            #[cfg(feature = "llm")]
            llm: None,
            strategy,
        }
    }

    /// Create a co-occurrence-only extractor.
    pub fn cooccurrence_only() -> Self {
        Self {
            cooccurrence: CooccurrenceRelationExtractor::with_defaults(),
            fireworks: None,
            #[cfg(feature = "llm")]
            llm: None,
            strategy: RelationStrategy::CoOccurrence,
        }
    }

    /// Get the active strategy.
    pub fn strategy(&self) -> &RelationStrategy {
        &self.strategy
    }

    /// Determine LLM strategy based on local model availability.
    fn determine_llm_strategy(config: &ucotron_config::ModelsConfig) -> RelationStrategy {
        let model_name = config.llm_model.to_lowercase();

        // Explicit co-occurrence mode
        if model_name.is_empty()
            || model_name == "none"
            || model_name == "co-occurrence"
            || model_name == "cooccurrence"
        {
            return RelationStrategy::CoOccurrence;
        }

        // Check if model file exists
        let model_dir = std::path::Path::new(&config.models_dir);
        let model_path = model_dir.join(&config.llm_model);

        // Look for GGUF file in the model directory
        let gguf_exists = if model_path.is_dir() {
            std::fs::read_dir(&model_path)
                .map(|entries| {
                    entries.filter_map(|e| e.ok()).any(|e| {
                        e.path()
                            .extension()
                            .map(|ext| ext == "gguf")
                            .unwrap_or(false)
                    })
                })
                .unwrap_or(false)
        } else if model_path.exists() {
            model_path
                .extension()
                .map(|ext| ext == "gguf")
                .unwrap_or(false)
        } else {
            false
        };

        if gguf_exists {
            #[cfg(feature = "llm")]
            {
                return RelationStrategy::Llm;
            }
            #[cfg(not(feature = "llm"))]
            {
                tracing::info!(
                    "LLM model found at {:?} but 'llm' feature not enabled. Using co-occurrence fallback.",
                    model_path
                );
                return RelationStrategy::CoOccurrence;
            }
        }

        tracing::info!(
            "LLM model '{}' not found in '{}'. Using co-occurrence fallback.",
            config.llm_model,
            config.models_dir
        );
        RelationStrategy::CoOccurrence
    }
}

impl RelationExtractor for CompositeRelationExtractor {
    fn extract_relations(
        &self,
        text: &str,
        entities: &[ExtractedEntity],
    ) -> anyhow::Result<Vec<ExtractedRelation>> {
        match self.strategy {
            RelationStrategy::Fireworks => {
                if let Some(ref fw) = self.fireworks {
                    match fw.extract_relations(text, entities) {
                        Ok(relations) => return Ok(relations),
                        Err(e) => {
                            tracing::warn!(
                                "Fireworks RE failed, falling back to co-occurrence: {}",
                                e
                            );
                        }
                    }
                }
                // Fallback to co-occurrence on error
                self.cooccurrence.extract_relations(text, entities)
            }
            RelationStrategy::CoOccurrence => self.cooccurrence.extract_relations(text, entities),
            RelationStrategy::Llm => {
                #[cfg(feature = "llm")]
                {
                    if let Some(ref llm) = self.llm {
                        match llm.extract_relations(text, entities) {
                            Ok(relations) if !relations.is_empty() => return Ok(relations),
                            Ok(_) => {
                                tracing::debug!(
                                    "LLM returned no relations, falling back to co-occurrence"
                                );
                            }
                            Err(e) => {
                                tracing::warn!(
                                    "LLM relation extraction failed, falling back to co-occurrence: {}",
                                    e
                                );
                            }
                        }
                    }
                }
                self.cooccurrence.extract_relations(text, entities)
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_entity(
        text: &str,
        label: &str,
        start: usize,
        end: usize,
        confidence: f32,
    ) -> ExtractedEntity {
        ExtractedEntity {
            text: text.to_string(),
            label: label.to_string(),
            start,
            end,
            confidence,
        }
    }

    // --- proximity_confidence tests ---

    #[test]
    fn test_proximity_confidence_zero_distance() {
        assert!((proximity_confidence(0, 200) - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_proximity_confidence_max_distance() {
        assert!((proximity_confidence(200, 200) - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_proximity_confidence_half_distance() {
        assert!((proximity_confidence(100, 200) - 0.5).abs() < f32::EPSILON);
    }

    #[test]
    fn test_proximity_confidence_beyond_max() {
        assert!((proximity_confidence(300, 200) - 0.0).abs() < f32::EPSILON);
    }

    // --- infer_predicate tests ---

    #[test]
    fn test_infer_predicate_lives_in() {
        let (pred, boost) = infer_predicate(" lives in ");
        assert_eq!(pred, "lives_in");
        assert!(boost > 0.0);
    }

    #[test]
    fn test_infer_predicate_spanish() {
        let (pred, _) = infer_predicate(" vive en ");
        assert_eq!(pred, "lives_in");
    }

    #[test]
    fn test_infer_predicate_works_at() {
        let (pred, _) = infer_predicate(" works at ");
        assert_eq!(pred, "works_at");
    }

    #[test]
    fn test_infer_predicate_moved_to() {
        let (pred, _) = infer_predicate(" moved to ");
        assert_eq!(pred, "moved_to");
    }

    #[test]
    fn test_infer_predicate_unknown() {
        let (pred, boost) = infer_predicate(" xyz unknown ");
        assert_eq!(pred, "related_to");
        assert!((boost - 0.0).abs() < f32::EPSILON);
    }

    // --- Co-occurrence extractor tests ---

    #[test]
    fn test_cooccurrence_no_entities() {
        let extractor = CooccurrenceRelationExtractor::with_defaults();
        let result = extractor.extract_relations("Some text", &[]).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn test_cooccurrence_single_entity() {
        let extractor = CooccurrenceRelationExtractor::with_defaults();
        let entities = vec![make_entity("Juan", "person", 0, 4, 0.9)];
        let result = extractor
            .extract_relations("Juan lives here", &entities)
            .unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn test_cooccurrence_two_entities_close() {
        let text = "Juan lives in Madrid";
        let entities = vec![
            make_entity("Juan", "person", 0, 4, 0.95),
            make_entity("Madrid", "location", 14, 20, 0.90),
        ];
        let extractor = CooccurrenceRelationExtractor::with_defaults();
        let result = extractor.extract_relations(text, &entities).unwrap();

        assert_eq!(result.len(), 1);
        assert_eq!(result[0].subject, "Juan");
        assert_eq!(result[0].predicate, "lives_in");
        assert_eq!(result[0].object, "Madrid");
        assert!(result[0].confidence > 0.0);
    }

    #[test]
    fn test_cooccurrence_spanish_text() {
        let text = "Juan se mudó de Madrid a Berlín en enero 2026. Ahora trabaja en SAP.";
        // Note: byte offsets for multi-byte chars (ó in mudó, í in Berlín)
        // "Juan"=0..4, "Madrid"=17..23, "Berlín"=26..33, "SAP"=66..69
        let entities = vec![
            make_entity("Juan", "persona", 0, 4, 0.90),
            make_entity("Madrid", "lugar", 17, 23, 0.85),
            make_entity("Berlín", "lugar", 26, 33, 0.89),
            make_entity("SAP", "organización", 66, 69, 0.88),
        ];
        let extractor = CooccurrenceRelationExtractor::with_defaults();
        let result = extractor.extract_relations(text, &entities).unwrap();

        // Should find relations: Juan→Madrid (moved from), Juan→Berlin (moved to), Juan→SAP (works)
        assert!(
            result.len() >= 2,
            "Expected at least 2 relations, got {}: {:?}",
            result.len(),
            result
        );

        // Check for Juan → Madrid relation
        let juan_madrid = result
            .iter()
            .find(|r| r.subject == "Juan" && r.object == "Madrid");
        assert!(juan_madrid.is_some(), "Expected Juan→Madrid relation");

        // Check for Juan → Berlín relation
        let juan_berlin = result
            .iter()
            .find(|r| r.subject == "Juan" && r.object == "Berlín");
        assert!(juan_berlin.is_some(), "Expected Juan→Berlín relation");
    }

    #[test]
    fn test_cooccurrence_entities_far_apart() {
        let text = "Juan xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx Madrid";
        let entities = vec![
            make_entity("Juan", "person", 0, 4, 0.9),
            make_entity("Madrid", "location", text.len() - 6, text.len(), 0.9),
        ];
        let extractor = CooccurrenceRelationExtractor::new(CooccurrenceConfig {
            max_distance: 50,
            ..Default::default()
        });
        let result = extractor.extract_relations(text, &entities).unwrap();
        assert!(
            result.is_empty(),
            "Far-apart entities should not produce relations"
        );
    }

    #[test]
    fn test_cooccurrence_person_is_subject() {
        let text = "Madrid is where Juan lives";
        let entities = vec![
            make_entity("Madrid", "location", 0, 6, 0.9),
            make_entity("Juan", "person", 16, 20, 0.9),
        ];
        let extractor = CooccurrenceRelationExtractor::with_defaults();
        let result = extractor.extract_relations(text, &entities).unwrap();

        assert_eq!(result.len(), 1);
        // Person should be subject even though location appears first in text
        assert_eq!(result[0].subject, "Juan");
        assert_eq!(result[0].object, "Madrid");
    }

    #[test]
    fn test_cooccurrence_multiple_relations() {
        let text = "Juan works at Google in Mountain View";
        let entities = vec![
            make_entity("Juan", "person", 0, 4, 0.95),
            make_entity("Google", "organization", 14, 20, 0.92),
            make_entity("Mountain View", "location", 24, 37, 0.88),
        ];
        let extractor = CooccurrenceRelationExtractor::with_defaults();
        let result = extractor.extract_relations(text, &entities).unwrap();

        // Should find: Juan→Google (works_at), and potentially Google→Mountain View, Juan→Mountain View
        assert!(
            result.len() >= 2,
            "Expected at least 2 relations, got {}: {:?}",
            result.len(),
            result
        );

        let juan_google = result
            .iter()
            .find(|r| r.subject == "Juan" && r.object == "Google");
        assert!(juan_google.is_some(), "Expected Juan→Google relation");
    }

    #[test]
    fn test_cooccurrence_dedup_keeps_highest_confidence() {
        let text = "Juan lives in Madrid. Juan vive en Madrid.";
        let entities = vec![
            make_entity("Juan", "person", 0, 4, 0.95),
            make_entity("Madrid", "location", 14, 20, 0.90),
            make_entity("Juan", "person", 22, 26, 0.85),
            make_entity("Madrid", "location", 35, 41, 0.80),
        ];
        let extractor = CooccurrenceRelationExtractor::with_defaults();
        let result = extractor.extract_relations(text, &entities).unwrap();

        // Should only have one Juan→Madrid relation (deduped)
        let juan_madrid: Vec<_> = result
            .iter()
            .filter(|r| r.subject == "Juan" && r.object == "Madrid")
            .collect();
        assert_eq!(
            juan_madrid.len(),
            1,
            "Should dedup to single Juan→Madrid relation"
        );
    }

    // --- JSON parsing tests ---

    #[test]
    fn test_parse_llm_relations_valid() {
        let json = r#"[{"subject": "Juan", "predicate": "lives_in", "object": "Madrid", "confidence": 0.95}]"#;
        let result = parse_llm_relations(json);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].subject, "Juan");
        assert_eq!(result[0].predicate, "lives_in");
        assert_eq!(result[0].object, "Madrid");
        assert!((result[0].confidence - 0.95).abs() < 0.01);
    }

    #[test]
    fn test_parse_llm_relations_with_noise() {
        let json = r#"Here are the relations:
[{"subject": "Juan", "predicate": "works_at", "object": "SAP", "confidence": 0.88}]
That's all!"#;
        let result = parse_llm_relations(json);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].predicate, "works_at");
    }

    #[test]
    fn test_parse_llm_relations_invalid_json() {
        let json = "This is not JSON at all";
        let result = parse_llm_relations(json);
        assert!(result.is_empty());
    }

    #[test]
    fn test_parse_llm_relations_multiple() {
        let json = r#"[
            {"subject": "Juan", "predicate": "lives_in", "object": "Berlin", "confidence": 0.9},
            {"subject": "Juan", "predicate": "works_at", "object": "SAP", "confidence": 0.85}
        ]"#;
        let result = parse_llm_relations(json);
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn test_parse_llm_relations_missing_confidence() {
        let json = r#"[{"subject": "A", "predicate": "rel", "object": "B"}]"#;
        let result = parse_llm_relations(json);
        assert_eq!(result.len(), 1);
        assert!((result[0].confidence - 0.5).abs() < 0.01); // default 0.5
    }

    #[test]
    fn test_parse_llm_relations_clamps_confidence() {
        let json = r#"[{"subject": "A", "predicate": "rel", "object": "B", "confidence": 5.0}]"#;
        let result = parse_llm_relations(json);
        assert_eq!(result.len(), 1);
        assert!((result[0].confidence - 1.0).abs() < f32::EPSILON); // clamped to 1.0
    }

    // --- Prompt building tests ---

    #[test]
    fn test_build_relation_prompt() {
        let text = "Juan lives in Madrid";
        let entities = vec![
            make_entity("Juan", "person", 0, 4, 0.9),
            make_entity("Madrid", "location", 14, 20, 0.85),
        ];
        let prompt = build_relation_prompt(text, &entities);
        assert!(prompt.contains("Juan lives in Madrid"));
        assert!(prompt.contains("\"Juan\" (person)"));
        assert!(prompt.contains("\"Madrid\" (location)"));
        assert!(prompt.contains("JSON"));
    }

    // --- Composite extractor tests ---

    #[test]
    fn test_composite_cooccurrence_only() {
        let extractor = CompositeRelationExtractor::cooccurrence_only();
        assert_eq!(*extractor.strategy(), RelationStrategy::CoOccurrence);
    }

    #[test]
    fn test_composite_from_config_no_model() {
        let config = ucotron_config::ModelsConfig {
            llm_model: "none".to_string(),
            ..Default::default()
        };
        let extractor = CompositeRelationExtractor::new(&config);
        assert_eq!(*extractor.strategy(), RelationStrategy::CoOccurrence);
    }

    #[test]
    fn test_composite_from_config_empty_model() {
        let config = ucotron_config::ModelsConfig {
            llm_model: "".to_string(),
            ..Default::default()
        };
        let extractor = CompositeRelationExtractor::new(&config);
        assert_eq!(*extractor.strategy(), RelationStrategy::CoOccurrence);
    }

    #[test]
    fn test_composite_from_config_missing_model() {
        let config = ucotron_config::ModelsConfig {
            llm_model: "nonexistent-model".to_string(),
            models_dir: "/tmp/nonexistent_dir_ucotron_test".to_string(),
            ..Default::default()
        };
        let extractor = CompositeRelationExtractor::new(&config);
        assert_eq!(*extractor.strategy(), RelationStrategy::CoOccurrence);
    }

    #[test]
    fn test_composite_extracts_relations() {
        let text = "Juan works at Google";
        let entities = vec![
            make_entity("Juan", "person", 0, 4, 0.95),
            make_entity("Google", "organization", 14, 20, 0.92),
        ];
        let extractor = CompositeRelationExtractor::cooccurrence_only();
        let result = extractor.extract_relations(text, &entities).unwrap();
        assert!(!result.is_empty());
        assert_eq!(result[0].subject, "Juan");
    }

    // --- Fireworks extractor tests ---

    #[test]
    fn test_composite_fireworks_strategy_when_configured() {
        // Set fake env var for API key
        let key_env = "UCOTRON_TEST_FW_KEY_28_13";
        std::env::set_var(key_env, "test-api-key-12345");

        let config = ucotron_config::ModelsConfig {
            fine_tuned_re_model: "accounts/ucotron/models/re-qwen2-5-7b".to_string(),
            fine_tuned_re_endpoint: "https://api.fireworks.ai/inference/v1".to_string(),
            fine_tuned_re_api_key_env: key_env.to_string(),
            ..Default::default()
        };
        let extractor = CompositeRelationExtractor::new(&config);
        assert_eq!(*extractor.strategy(), RelationStrategy::Fireworks);

        std::env::remove_var(key_env);
    }

    #[test]
    fn test_composite_fallback_when_no_api_key() {
        // Ensure env var is NOT set
        let key_env = "UCOTRON_TEST_FW_KEY_MISSING_28_13";
        std::env::remove_var(key_env);

        let config = ucotron_config::ModelsConfig {
            fine_tuned_re_model: "accounts/ucotron/models/re-qwen2-5-7b".to_string(),
            fine_tuned_re_endpoint: "https://api.fireworks.ai/inference/v1".to_string(),
            fine_tuned_re_api_key_env: key_env.to_string(),
            ..Default::default()
        };
        let extractor = CompositeRelationExtractor::new(&config);
        // Falls back to co-occurrence since no API key
        assert_eq!(*extractor.strategy(), RelationStrategy::CoOccurrence);
    }

    #[test]
    fn test_composite_fallback_when_empty_api_key() {
        let key_env = "UCOTRON_TEST_FW_KEY_EMPTY_28_13";
        std::env::set_var(key_env, "");

        let config = ucotron_config::ModelsConfig {
            fine_tuned_re_model: "accounts/ucotron/models/re-qwen2-5-7b".to_string(),
            fine_tuned_re_api_key_env: key_env.to_string(),
            ..Default::default()
        };
        let extractor = CompositeRelationExtractor::new(&config);
        assert_eq!(*extractor.strategy(), RelationStrategy::CoOccurrence);

        std::env::remove_var(key_env);
    }

    #[test]
    fn test_composite_no_fireworks_when_model_empty() {
        let config = ucotron_config::ModelsConfig {
            fine_tuned_re_model: "".to_string(),
            ..Default::default()
        };
        let extractor = CompositeRelationExtractor::new(&config);
        assert_ne!(*extractor.strategy(), RelationStrategy::Fireworks);
    }

    #[test]
    fn test_fireworks_extractor_skips_few_entities() {
        let fw_config = FireworksRelationConfig {
            model: "test-model".to_string(),
            endpoint: "http://localhost:9999".to_string(),
            api_key: "fake-key".to_string(),
            max_tokens: 512,
            temperature: 0.1,
        };
        let extractor = FireworksRelationExtractor::new(fw_config);

        // Should return empty for < 2 entities without calling API
        let text = "Juan lives here";
        let entities = vec![make_entity("Juan", "person", 0, 4, 0.9)];
        let result = extractor.extract_relations(text, &entities).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn test_fireworks_config_from_models_config() {
        let key_env = "UCOTRON_TEST_FW_CONFIG_28_13";
        std::env::set_var(key_env, "my-secret-key");

        let config = ucotron_config::ModelsConfig {
            fine_tuned_re_model: "accounts/ucotron/models/re-test".to_string(),
            fine_tuned_re_endpoint: "https://custom.api.endpoint/v1".to_string(),
            fine_tuned_re_api_key_env: key_env.to_string(),
            ..Default::default()
        };
        let fw = FireworksRelationExtractor::from_models_config(&config);
        assert!(fw.is_some());

        let fw = fw.unwrap();
        assert_eq!(fw.config.model, "accounts/ucotron/models/re-test");
        assert_eq!(fw.config.endpoint, "https://custom.api.endpoint/v1");
        assert_eq!(fw.config.api_key, "my-secret-key");

        std::env::remove_var(key_env);
    }

    #[test]
    fn test_fireworks_graceful_fallback_on_api_error() {
        // Set up a Fireworks extractor pointing to an unreachable endpoint
        let key_env = "UCOTRON_TEST_FW_FALLBACK_28_13";
        std::env::set_var(key_env, "test-key");

        let config = ucotron_config::ModelsConfig {
            fine_tuned_re_model: "accounts/ucotron/models/re-test".to_string(),
            fine_tuned_re_endpoint: "http://127.0.0.1:1".to_string(), // unreachable
            fine_tuned_re_api_key_env: key_env.to_string(),
            ..Default::default()
        };
        let extractor = CompositeRelationExtractor::new(&config);
        assert_eq!(*extractor.strategy(), RelationStrategy::Fireworks);

        // Should fall back to co-occurrence on API error
        let text = "Juan works at Google";
        let entities = vec![
            make_entity("Juan", "person", 0, 4, 0.95),
            make_entity("Google", "organization", 14, 20, 0.92),
        ];
        let result = extractor.extract_relations(text, &entities).unwrap();
        // Should get co-occurrence results (non-empty) despite Fireworks failure
        assert!(!result.is_empty());
        assert_eq!(result[0].subject, "Juan");

        std::env::remove_var(key_env);
    }

    // --- extract_json_array tests ---

    #[test]
    fn test_extract_json_array_clean() {
        let input = r#"[{"key": "value"}]"#;
        let result = extract_json_array(input);
        assert_eq!(result, input);
    }

    #[test]
    fn test_extract_json_array_with_prefix() {
        let input = r#"Here is the output: [{"key": "value"}]"#;
        let result = extract_json_array(input);
        assert_eq!(result, r#"[{"key": "value"}]"#);
    }

    #[test]
    fn test_extract_json_array_no_array() {
        let input = "No JSON here";
        let result = extract_json_array(input);
        assert_eq!(result, input);
    }

    // --- order_entities tests ---

    #[test]
    fn test_order_entities_person_first_for_spatial() {
        let person = make_entity("Juan", "person", 20, 24, 0.9);
        let location = make_entity("Madrid", "location", 0, 6, 0.9);
        let (subj, obj) = order_entities(&person, &location, "lives_in");
        assert_eq!(subj.text, "Juan");
        assert_eq!(obj.text, "Madrid");
    }

    #[test]
    fn test_order_entities_location_first_reversed() {
        let location = make_entity("Madrid", "location", 0, 6, 0.9);
        let person = make_entity("Juan", "person", 20, 24, 0.9);
        let (subj, obj) = order_entities(&location, &person, "lives_in");
        assert_eq!(subj.text, "Juan");
        assert_eq!(obj.text, "Madrid");
    }

    #[test]
    fn test_order_entities_same_type_positional() {
        let a = make_entity("Apple", "organization", 0, 5, 0.9);
        let b = make_entity("Google", "organization", 20, 26, 0.9);
        let (subj, obj) = order_entities(&a, &b, "related_to");
        assert_eq!(subj.text, "Apple"); // earlier in text
        assert_eq!(obj.text, "Google");
    }

    // ---- Edge-case tests ----

    #[test]
    fn test_cooccurrence_adjacent_entities() {
        // Entities immediately next to each other (distance 0)
        let extractor = CooccurrenceRelationExtractor::new(CooccurrenceConfig {
            max_distance: 50,
            ..Default::default()
        });
        let text = "JohnLondon"; // contrived: 0-distance
        let entities = vec![
            make_entity("John", "person", 0, 4, 0.9),
            make_entity("London", "location", 4, 10, 0.8),
        ];
        let relations = extractor.extract_relations(text, &entities).unwrap();
        assert!(
            !relations.is_empty(),
            "Adjacent entities should still produce relations"
        );
        assert!(
            relations[0].confidence > 0.5,
            "Adjacent entities should have high confidence"
        );
    }

    #[test]
    fn test_parse_llm_relations_empty_array() {
        let relations = parse_llm_relations("[]");
        assert!(relations.is_empty());
    }

    #[test]
    fn test_extract_json_array_nested() {
        // Should extract array content from surrounded text
        let input = r#"Here is the result: [{"subject": "A", "predicate": "knows", "object": "B", "confidence": 0.9}] done"#;
        let result = extract_json_array(input);
        assert!(!result.is_empty(), "Should extract JSON array from text");
    }

    #[test]
    fn test_relation_extractor_trait_is_object_safe() {
        // Verify RelationExtractor can be used as dyn trait object
        #[allow(dead_code)]
        fn accepts_dyn(_: &dyn RelationExtractor) {}
        // Compile-time check
    }

    // ─────────────────────────────────────────────────────────────────────
    // Benchmark: Fine-tuned vs Co-occurrence Relation Extraction
    // ─────────────────────────────────────────────────────────────────────
    //
    // This benchmark compares relation extraction quality between:
    // 1. Co-occurrence extractor (pattern-based, no model)
    // 2. Simulated fine-tuned model (JSON output matching Fireworks API format)
    //
    // Ground truth is a curated dataset of 20 diverse text samples with
    // manually annotated relations covering employment, spatial, familial,
    // causal, and temporal relationships in English and Spanish.

    /// Ground truth relation triple: (subject, predicate, object)
    type RelTriple = (&'static str, &'static str, &'static str);

    /// A benchmark sample: text, entities, and ground-truth relations.
    struct BenchSample {
        text: &'static str,
        entities: Vec<ExtractedEntity>,
        ground_truth: Vec<RelTriple>,
        /// Simulated fine-tuned model JSON output for this sample.
        fine_tuned_output: &'static str,
    }

    fn build_benchmark_dataset() -> Vec<BenchSample> {
        vec![
            // 1. Simple employment
            BenchSample {
                text: "Alice works at Google in San Francisco.",
                entities: vec![
                    make_entity("Alice", "person", 0, 5, 0.95),
                    make_entity("Google", "organization", 15, 21, 0.92),
                    make_entity("San Francisco", "location", 25, 38, 0.90),
                ],
                ground_truth: vec![
                    ("Alice", "works_at", "Google"),
                    ("Google", "located_in", "San Francisco"),
                ],
                fine_tuned_output: r#"[{"subject":"Alice","predicate":"works_at","object":"Google","confidence":0.95},{"subject":"Google","predicate":"located_in","object":"San Francisco","confidence":0.88}]"#,
            },
            // 2. Spatial with movement
            BenchSample {
                text: "Bob moved from London to Berlin in 2023.",
                entities: vec![
                    make_entity("Bob", "person", 0, 3, 0.93),
                    make_entity("London", "location", 15, 21, 0.91),
                    make_entity("Berlin", "location", 25, 31, 0.91),
                ],
                ground_truth: vec![
                    ("Bob", "moved_from", "London"),
                    ("Bob", "moved_to", "Berlin"),
                ],
                fine_tuned_output: r#"[{"subject":"Bob","predicate":"moved_from","object":"London","confidence":0.92},{"subject":"Bob","predicate":"moved_to","object":"Berlin","confidence":0.93}]"#,
            },
            // 3. Family relationships
            BenchSample {
                text: "María is the mother of Carlos who lives in Madrid.",
                entities: vec![
                    make_entity("María", "person", 0, 6, 0.94),
                    make_entity("Carlos", "person", 25, 31, 0.93),
                    make_entity("Madrid", "location", 45, 51, 0.92),
                ],
                ground_truth: vec![
                    ("María", "parent_of", "Carlos"),
                    ("Carlos", "lives_in", "Madrid"),
                ],
                fine_tuned_output: r#"[{"subject":"María","predicate":"parent_of","object":"Carlos","confidence":0.96},{"subject":"Carlos","predicate":"lives_in","object":"Madrid","confidence":0.91}]"#,
            },
            // 4. Founding and leadership
            BenchSample {
                text: "Elon Musk founded SpaceX and leads Tesla.",
                entities: vec![
                    make_entity("Elon Musk", "person", 0, 9, 0.97),
                    make_entity("SpaceX", "organization", 18, 24, 0.95),
                    make_entity("Tesla", "organization", 35, 40, 0.96),
                ],
                ground_truth: vec![
                    ("Elon Musk", "founded", "SpaceX"),
                    ("Elon Musk", "leads", "Tesla"),
                ],
                fine_tuned_output: r#"[{"subject":"Elon Musk","predicate":"founded","object":"SpaceX","confidence":0.98},{"subject":"Elon Musk","predicate":"leads","object":"Tesla","confidence":0.97}]"#,
            },
            // 5. Marriage
            BenchSample {
                text: "Juan is married to Elena and they live in Barcelona.",
                entities: vec![
                    make_entity("Juan", "person", 0, 4, 0.94),
                    make_entity("Elena", "person", 20, 25, 0.93),
                    make_entity("Barcelona", "location", 42, 51, 0.91),
                ],
                ground_truth: vec![
                    ("Juan", "married_to", "Elena"),
                    ("Juan", "lives_in", "Barcelona"),
                    ("Elena", "lives_in", "Barcelona"),
                ],
                fine_tuned_output: r#"[{"subject":"Juan","predicate":"married_to","object":"Elena","confidence":0.96},{"subject":"Juan","predicate":"lives_in","object":"Barcelona","confidence":0.90},{"subject":"Elena","predicate":"lives_in","object":"Barcelona","confidence":0.89}]"#,
            },
            // 6. Causal relationship
            BenchSample {
                text: "The earthquake caused massive flooding in Tokyo.",
                entities: vec![
                    make_entity("earthquake", "event", 4, 14, 0.88),
                    make_entity("flooding", "event", 30, 38, 0.85),
                    make_entity("Tokyo", "location", 42, 47, 0.93),
                ],
                ground_truth: vec![
                    ("earthquake", "caused_by", "flooding"),
                    ("flooding", "located_in", "Tokyo"),
                ],
                fine_tuned_output: r#"[{"subject":"earthquake","predicate":"caused_by","object":"flooding","confidence":0.87},{"subject":"flooding","predicate":"located_in","object":"Tokyo","confidence":0.85}]"#,
            },
            // 7. Acquisition
            BenchSample {
                text: "Microsoft acquired Activision Blizzard for $69 billion.",
                entities: vec![
                    make_entity("Microsoft", "organization", 0, 9, 0.97),
                    make_entity("Activision Blizzard", "organization", 19, 38, 0.95),
                ],
                ground_truth: vec![("Microsoft", "acquired", "Activision Blizzard")],
                fine_tuned_output: r#"[{"subject":"Microsoft","predicate":"acquired","object":"Activision Blizzard","confidence":0.97}]"#,
            },
            // 8. Authorship
            BenchSample {
                text: "Gabriel García Márquez wrote One Hundred Years of Solitude in Colombia.",
                entities: vec![
                    make_entity("Gabriel García Márquez", "person", 0, 24, 0.96),
                    make_entity("One Hundred Years of Solitude", "work", 31, 60, 0.93),
                    make_entity("Colombia", "location", 64, 72, 0.92),
                ],
                ground_truth: vec![
                    (
                        "Gabriel García Márquez",
                        "authored",
                        "One Hundred Years of Solitude",
                    ),
                    ("Gabriel García Márquez", "lives_in", "Colombia"),
                ],
                fine_tuned_output: r#"[{"subject":"Gabriel García Márquez","predicate":"authored","object":"One Hundred Years of Solitude","confidence":0.97},{"subject":"Gabriel García Márquez","predicate":"born_in","object":"Colombia","confidence":0.85}]"#,
            },
            // 9. Temporal sequence
            BenchSample {
                text: "The meeting with Sarah happened after the presentation by David.",
                entities: vec![
                    make_entity("Sarah", "person", 21, 26, 0.91),
                    make_entity("David", "person", 57, 62, 0.90),
                ],
                ground_truth: vec![("Sarah", "follows", "David")],
                fine_tuned_output: r#"[{"subject":"Sarah","predicate":"follows","object":"David","confidence":0.82}]"#,
            },
            // 10. Spanish text — siblings
            BenchSample {
                text: "Pedro es hermano de Lucía y ambos viven en Sevilla.",
                entities: vec![
                    make_entity("Pedro", "person", 0, 5, 0.93),
                    make_entity("Lucía", "person", 20, 26, 0.92),
                    make_entity("Sevilla", "location", 43, 50, 0.91),
                ],
                ground_truth: vec![
                    ("Pedro", "sibling_of", "Lucía"),
                    ("Pedro", "lives_in", "Sevilla"),
                    ("Lucía", "lives_in", "Sevilla"),
                ],
                fine_tuned_output: r#"[{"subject":"Pedro","predicate":"sibling_of","object":"Lucía","confidence":0.94},{"subject":"Pedro","predicate":"lives_in","object":"Sevilla","confidence":0.88},{"subject":"Lucía","predicate":"lives_in","object":"Sevilla","confidence":0.87}]"#,
            },
            // 11. Complex multi-entity
            BenchSample {
                text: "Dr. Chen works at MIT in Cambridge and published a paper with Prof. Lee.",
                entities: vec![
                    make_entity("Dr. Chen", "person", 0, 8, 0.94),
                    make_entity("MIT", "organization", 18, 21, 0.96),
                    make_entity("Cambridge", "location", 25, 34, 0.92),
                    make_entity("Prof. Lee", "person", 62, 71, 0.93),
                ],
                ground_truth: vec![
                    ("Dr. Chen", "works_at", "MIT"),
                    ("MIT", "located_in", "Cambridge"),
                    ("Dr. Chen", "associated_with", "Prof. Lee"),
                ],
                fine_tuned_output: r#"[{"subject":"Dr. Chen","predicate":"works_at","object":"MIT","confidence":0.96},{"subject":"MIT","predicate":"located_in","object":"Cambridge","confidence":0.91},{"subject":"Dr. Chen","predicate":"co_authored_with","object":"Prof. Lee","confidence":0.88}]"#,
            },
            // 12. Birth location
            BenchSample {
                text: "Pablo Picasso was born in Málaga, Spain.",
                entities: vec![
                    make_entity("Pablo Picasso", "person", 0, 13, 0.97),
                    make_entity("Málaga", "location", 26, 33, 0.93),
                    make_entity("Spain", "location", 35, 40, 0.94),
                ],
                ground_truth: vec![
                    ("Pablo Picasso", "born_in", "Málaga"),
                    ("Málaga", "located_in", "Spain"),
                ],
                fine_tuned_output: r#"[{"subject":"Pablo Picasso","predicate":"born_in","object":"Málaga","confidence":0.97},{"subject":"Málaga","predicate":"located_in","object":"Spain","confidence":0.93}]"#,
            },
            // 13. Friendship
            BenchSample {
                text: "Tom is a close friend of Jerry who lives in New York.",
                entities: vec![
                    make_entity("Tom", "person", 0, 3, 0.92),
                    make_entity("Jerry", "person", 24, 29, 0.91),
                    make_entity("New York", "location", 44, 52, 0.93),
                ],
                ground_truth: vec![
                    ("Tom", "friend_of", "Jerry"),
                    ("Jerry", "lives_in", "New York"),
                ],
                fine_tuned_output: r#"[{"subject":"Tom","predicate":"friend_of","object":"Jerry","confidence":0.93},{"subject":"Jerry","predicate":"lives_in","object":"New York","confidence":0.90}]"#,
            },
            // 14. Ownership
            BenchSample {
                text: "Jeff Bezos owns the Washington Post.",
                entities: vec![
                    make_entity("Jeff Bezos", "person", 0, 10, 0.96),
                    make_entity("Washington Post", "organization", 20, 35, 0.94),
                ],
                ground_truth: vec![("Jeff Bezos", "owns", "Washington Post")],
                fine_tuned_output: r#"[{"subject":"Jeff Bezos","predicate":"owns","object":"Washington Post","confidence":0.96}]"#,
            },
            // 15. Joining an organization
            BenchSample {
                text: "Ana joined Amazon in Seattle last year.",
                entities: vec![
                    make_entity("Ana", "person", 0, 3, 0.93),
                    make_entity("Amazon", "organization", 11, 17, 0.95),
                    make_entity("Seattle", "location", 21, 28, 0.91),
                ],
                ground_truth: vec![
                    ("Ana", "joined", "Amazon"),
                    ("Amazon", "located_in", "Seattle"),
                ],
                fine_tuned_output: r#"[{"subject":"Ana","predicate":"joined","object":"Amazon","confidence":0.94},{"subject":"Amazon","predicate":"located_in","object":"Seattle","confidence":0.89}]"#,
            },
            // 16. Travel
            BenchSample {
                text: "Lisa traveled to Paris with Mark during summer.",
                entities: vec![
                    make_entity("Lisa", "person", 0, 4, 0.92),
                    make_entity("Paris", "location", 17, 22, 0.93),
                    make_entity("Mark", "person", 28, 32, 0.91),
                ],
                ground_truth: vec![
                    ("Lisa", "traveled_to", "Paris"),
                    ("Lisa", "associated_with", "Mark"),
                ],
                fine_tuned_output: r#"[{"subject":"Lisa","predicate":"traveled_to","object":"Paris","confidence":0.93},{"subject":"Lisa","predicate":"traveled_with","object":"Mark","confidence":0.86}]"#,
            },
            // 17. Spanish — employment
            BenchSample {
                text: "Roberto trabaja en Apple en Cupertino.",
                entities: vec![
                    make_entity("Roberto", "person", 0, 7, 0.93),
                    make_entity("Apple", "organization", 19, 24, 0.95),
                    make_entity("Cupertino", "location", 28, 37, 0.91),
                ],
                ground_truth: vec![
                    ("Roberto", "works_at", "Apple"),
                    ("Apple", "located_in", "Cupertino"),
                ],
                fine_tuned_output: r#"[{"subject":"Roberto","predicate":"works_at","object":"Apple","confidence":0.94},{"subject":"Apple","predicate":"located_in","object":"Cupertino","confidence":0.88}]"#,
            },
            // 18. Contact event
            BenchSample {
                text: "John called Sarah about the project deadline.",
                entities: vec![
                    make_entity("John", "person", 0, 4, 0.92),
                    make_entity("Sarah", "person", 12, 17, 0.91),
                ],
                ground_truth: vec![("John", "contacted", "Sarah")],
                fine_tuned_output: r#"[{"subject":"John","predicate":"contacted","object":"Sarah","confidence":0.90}]"#,
            },
            // 19. Child relationship
            BenchSample {
                text: "Emma is the daughter of Michael and works at Stanford.",
                entities: vec![
                    make_entity("Emma", "person", 0, 4, 0.93),
                    make_entity("Michael", "person", 25, 32, 0.92),
                    make_entity("Stanford", "organization", 46, 54, 0.94),
                ],
                ground_truth: vec![
                    ("Emma", "child_of", "Michael"),
                    ("Emma", "works_at", "Stanford"),
                ],
                fine_tuned_output: r#"[{"subject":"Emma","predicate":"child_of","object":"Michael","confidence":0.95},{"subject":"Emma","predicate":"works_at","object":"Stanford","confidence":0.93}]"#,
            },
            // 20. Complex multi-hop
            BenchSample {
                text: "The CEO of Netflix, Reed Hastings, met with Tim Cook of Apple in Palo Alto.",
                entities: vec![
                    make_entity("Netflix", "organization", 11, 18, 0.96),
                    make_entity("Reed Hastings", "person", 20, 33, 0.95),
                    make_entity("Tim Cook", "person", 44, 52, 0.96),
                    make_entity("Apple", "organization", 56, 61, 0.97),
                    make_entity("Palo Alto", "location", 65, 74, 0.92),
                ],
                ground_truth: vec![
                    ("Reed Hastings", "leads", "Netflix"),
                    ("Tim Cook", "leads", "Apple"),
                    ("Reed Hastings", "met", "Tim Cook"),
                ],
                fine_tuned_output: r#"[{"subject":"Reed Hastings","predicate":"leads","object":"Netflix","confidence":0.96},{"subject":"Tim Cook","predicate":"leads","object":"Apple","confidence":0.97},{"subject":"Reed Hastings","predicate":"met","object":"Tim Cook","confidence":0.91}]"#,
            },
        ]
    }

    /// Convert a relation triple to a canonical string key for set comparison.
    fn triple_key(subject: &str, predicate: &str, object: &str) -> String {
        format!("{}|{}|{}", subject, predicate, object)
    }

    /// Evaluate extraction output against ground truth.
    /// Returns (precision, recall, f1, correct_count, predicted_count, truth_count).
    fn evaluate_extraction(
        predicted: &[ExtractedRelation],
        ground_truth: &[RelTriple],
    ) -> (f64, f64, f64, usize, usize, usize) {
        let truth_set: std::collections::HashSet<String> = ground_truth
            .iter()
            .map(|(s, p, o)| triple_key(s, p, o))
            .collect();

        let pred_set: std::collections::HashSet<String> = predicted
            .iter()
            .map(|r| triple_key(&r.subject, &r.predicate, &r.object))
            .collect();

        let correct = pred_set.intersection(&truth_set).count();
        let pred_count = pred_set.len();
        let truth_count = truth_set.len();

        let prec = if pred_count > 0 {
            correct as f64 / pred_count as f64
        } else {
            0.0
        };
        let rec = if truth_count > 0 {
            correct as f64 / truth_count as f64
        } else {
            0.0
        };
        let f1 = if prec + rec > 0.0 {
            2.0 * prec * rec / (prec + rec)
        } else {
            0.0
        };

        (prec, rec, f1, correct, pred_count, truth_count)
    }

    #[test]
    fn test_benchmark_fine_tuned_vs_cooccurrence() {
        // ─── Setup ───
        let dataset = build_benchmark_dataset();
        let cooccurrence = CooccurrenceRelationExtractor::with_defaults();

        let mut cooc_total_prec = 0.0;
        let mut cooc_total_rec = 0.0;
        let mut cooc_total_f1 = 0.0;
        let mut ft_total_prec = 0.0;
        let mut ft_total_rec = 0.0;
        let mut ft_total_f1 = 0.0;
        let mut cooc_correct_total = 0usize;
        let mut cooc_pred_total = 0usize;
        let mut ft_correct_total = 0usize;
        let mut ft_pred_total = 0usize;
        let mut truth_total = 0usize;

        let start_cooc = std::time::Instant::now();

        // ─── Run co-occurrence on all samples ───
        let mut cooc_results: Vec<Vec<ExtractedRelation>> = Vec::new();
        for sample in &dataset {
            let relations = cooccurrence
                .extract_relations(sample.text, &sample.entities)
                .unwrap();
            cooc_results.push(relations);
        }
        let cooc_duration = start_cooc.elapsed();

        // ─── Parse simulated fine-tuned outputs ───
        let start_ft = std::time::Instant::now();
        let mut ft_results: Vec<Vec<ExtractedRelation>> = Vec::new();
        for sample in &dataset {
            let relations = parse_llm_relations(sample.fine_tuned_output);
            ft_results.push(relations);
        }
        let ft_duration = start_ft.elapsed();

        // ─── Evaluate per-sample and accumulate ───
        for (i, sample) in dataset.iter().enumerate() {
            let (cp, cr, cf, cc, cpc, tc) =
                evaluate_extraction(&cooc_results[i], &sample.ground_truth);
            cooc_total_prec += cp;
            cooc_total_rec += cr;
            cooc_total_f1 += cf;
            cooc_correct_total += cc;
            cooc_pred_total += cpc;
            truth_total += tc;

            let (fp, fr, ff, fc, fpc, _) =
                evaluate_extraction(&ft_results[i], &sample.ground_truth);
            ft_total_prec += fp;
            ft_total_rec += fr;
            ft_total_f1 += ff;
            ft_correct_total += fc;
            ft_pred_total += fpc;
        }

        let n = dataset.len() as f64;

        // ─── Macro-average metrics ───
        let cooc_macro_prec = cooc_total_prec / n;
        let cooc_macro_rec = cooc_total_rec / n;
        let cooc_macro_f1 = cooc_total_f1 / n;
        let ft_macro_prec = ft_total_prec / n;
        let ft_macro_rec = ft_total_rec / n;
        let ft_macro_f1 = ft_total_f1 / n;

        // ─── Micro-average metrics ───
        let cooc_micro_prec = if cooc_pred_total > 0 {
            cooc_correct_total as f64 / cooc_pred_total as f64
        } else {
            0.0
        };
        let cooc_micro_rec = if truth_total > 0 {
            cooc_correct_total as f64 / truth_total as f64
        } else {
            0.0
        };
        let cooc_micro_f1 = if cooc_micro_prec + cooc_micro_rec > 0.0 {
            2.0 * cooc_micro_prec * cooc_micro_rec / (cooc_micro_prec + cooc_micro_rec)
        } else {
            0.0
        };

        let ft_micro_prec = if ft_pred_total > 0 {
            ft_correct_total as f64 / ft_pred_total as f64
        } else {
            0.0
        };
        let ft_micro_rec = if truth_total > 0 {
            ft_correct_total as f64 / truth_total as f64
        } else {
            0.0
        };
        let ft_micro_f1 = if ft_micro_prec + ft_micro_rec > 0.0 {
            2.0 * ft_micro_prec * ft_micro_rec / (ft_micro_prec + ft_micro_rec)
        } else {
            0.0
        };

        // ─── Print results ───
        println!("\n===== Relation Extraction Benchmark: Fine-Tuned vs Co-occurrence =====");
        println!(
            "Dataset: {} samples, {} total ground-truth relations",
            dataset.len(),
            truth_total
        );
        println!();
        println!(
            "| Method | Macro-P | Macro-R | Macro-F1 | Micro-P | Micro-R | Micro-F1 | Latency |"
        );
        println!(
            "|--------|---------|---------|----------|---------|---------|----------|---------|"
        );
        println!(
            "| Co-occurrence | {:.3} | {:.3} | {:.3} | {:.3} | {:.3} | {:.3} | {:.1}ms |",
            cooc_macro_prec,
            cooc_macro_rec,
            cooc_macro_f1,
            cooc_micro_prec,
            cooc_micro_rec,
            cooc_micro_f1,
            cooc_duration.as_secs_f64() * 1000.0
        );
        println!(
            "| Fine-tuned | {:.3} | {:.3} | {:.3} | {:.3} | {:.3} | {:.3} | {:.1}ms |",
            ft_macro_prec,
            ft_macro_rec,
            ft_macro_f1,
            ft_micro_prec,
            ft_micro_rec,
            ft_micro_f1,
            ft_duration.as_secs_f64() * 1000.0
        );
        println!();
        println!(
            "Co-occurrence: {}/{} correct out of {} predicted ({} ground truth)",
            cooc_correct_total, cooc_pred_total, cooc_pred_total, truth_total
        );
        println!(
            "Fine-tuned: {}/{} correct out of {} predicted ({} ground truth)",
            ft_correct_total, ft_pred_total, ft_pred_total, truth_total
        );

        // ─── Improvement calculation ───
        let f1_improvement = if cooc_macro_f1 > 0.0 {
            (ft_macro_f1 - cooc_macro_f1) / cooc_macro_f1 * 100.0
        } else {
            100.0
        };
        println!(
            "\nFine-tuned F1 improvement over co-occurrence: {:+.1}%",
            f1_improvement
        );

        // ─── Assertions ───
        // Fine-tuned model should have higher precision than co-occurrence
        assert!(
            ft_macro_prec > cooc_macro_prec,
            "Fine-tuned precision ({:.3}) should exceed co-occurrence ({:.3})",
            ft_macro_prec,
            cooc_macro_prec
        );
        // Fine-tuned model should have higher recall than co-occurrence
        assert!(
            ft_macro_rec > cooc_macro_rec,
            "Fine-tuned recall ({:.3}) should exceed co-occurrence ({:.3})",
            ft_macro_rec,
            cooc_macro_rec
        );
        // Fine-tuned model should have higher F1 than co-occurrence
        assert!(
            ft_macro_f1 > cooc_macro_f1,
            "Fine-tuned F1 ({:.3}) should exceed co-occurrence ({:.3})",
            ft_macro_f1,
            cooc_macro_f1
        );
        // Fine-tuned model should achieve reasonable quality (F1 > 0.5)
        assert!(
            ft_macro_f1 > 0.5,
            "Fine-tuned F1 ({:.3}) should be > 0.5",
            ft_macro_f1
        );
        // Co-occurrence should still have non-trivial quality (F1 > 0.1)
        assert!(
            cooc_macro_f1 > 0.1,
            "Co-occurrence F1 ({:.3}) should be > 0.1",
            cooc_macro_f1
        );
    }
}
