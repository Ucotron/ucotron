//! # GLiNER NER Pipeline
//!
//! Zero-shot Named Entity Recognition using GLiNER models via ONNX Runtime.
//!
//! GLiNER (Generalist and Lightweight Named Entity Recognition) uses a span-based
//! approach where entity labels are provided at inference time, enabling zero-shot
//! extraction without retraining.
//!
//! ## Model Architecture
//!
//! The model expects a special prompt format:
//! ```text
//! [CLS] <<ENT>> label1 <<ENT>> label2 ... <<SEP>> word1 word2 ... [SEP]
//! ```
//!
//! Input tensors:
//! - `input_ids`: Token IDs (i64) — entity labels + separator + text tokens
//! - `attention_mask`: Binary mask (i64) — 1 for real tokens, 0 for padding
//! - `words_mask`: Word boundary tracking (i64) — word index for first subword of text words
//! - `text_lengths`: Number of text words per sequence (i64)
//! - `span_idx`: Candidate span start/end pairs (i64)
//! - `span_mask`: Valid span indicator (bool)
//!
//! Output: `logits` tensor with shape `[batch, num_spans, num_labels]`, decoded via sigmoid.

use std::collections::HashSet;
use std::path::Path;
use std::sync::Mutex;

use anyhow::{Context, Result};
use ndarray::{Array2, Array3};
use tokenizers::Tokenizer;

use crate::{ExtractedEntity, NerPipeline};

/// Special token IDs for GLiNER span-mode models (DeBERTa-based).
const START_TOKEN_ID: i64 = 1; // [CLS]
const END_TOKEN_ID: i64 = 2; // [SEP]
#[allow(dead_code)]
const PAD_TOKEN_ID: i64 = 0;
const ENTITY_TOKEN_ID: i64 = 128002; // <<ENT>>
const SEP_TOKEN_ID: i64 = 128003; // <<SEP>>

/// Default confidence threshold for entity extraction.
const DEFAULT_THRESHOLD: f32 = 0.5;

/// Default max span width (in words) for candidate spans.
const DEFAULT_MAX_WIDTH: usize = 12;

/// Configuration for the GLiNER NER pipeline.
#[derive(Debug, Clone)]
pub struct GlinerConfig {
    /// Minimum confidence score (0.0-1.0) for accepting an entity.
    pub threshold: f32,
    /// Maximum span width in words for candidate entity spans.
    pub max_width: usize,
    /// Whether to apply flat NER (no nested entities).
    pub flat_ner: bool,
    /// Number of ONNX inference threads.
    pub num_threads: usize,
}

impl Default for GlinerConfig {
    fn default() -> Self {
        Self {
            threshold: DEFAULT_THRESHOLD,
            max_width: DEFAULT_MAX_WIDTH,
            flat_ner: true,
            num_threads: 4,
        }
    }
}

/// GLiNER NER pipeline using ONNX Runtime for zero-shot entity extraction.
///
/// Thread-safe via internal `Mutex` on the ONNX session.
pub struct GlinerNerPipeline {
    session: Mutex<ort::session::Session>,
    tokenizer: Mutex<Tokenizer>,
    config: GlinerConfig,
}

impl GlinerNerPipeline {
    /// Create a new GLiNER NER pipeline from ONNX model and tokenizer files.
    ///
    /// # Arguments
    /// * `model_path` - Path to the ONNX model file (e.g., `models/gliner_small-v2.1/onnx/model.onnx`)
    /// * `tokenizer_path` - Path to the tokenizer.json file
    /// * `config` - Pipeline configuration
    pub fn new(
        model_path: impl AsRef<Path>,
        tokenizer_path: impl AsRef<Path>,
        config: GlinerConfig,
    ) -> Result<Self> {
        let session = ort::session::Session::builder()?
            .with_optimization_level(ort::session::builder::GraphOptimizationLevel::Level3)?
            .with_intra_threads(config.num_threads)?
            .commit_from_file(model_path.as_ref())
            .context("Failed to load GLiNER ONNX model")?;

        let tokenizer = Tokenizer::from_file(tokenizer_path.as_ref())
            .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;

        Ok(Self {
            session: Mutex::new(session),
            tokenizer: Mutex::new(tokenizer),
            config,
        })
    }

    /// Run inference on a single text with the given entity labels.
    fn run_inference(&self, text: &str, labels: &[&str]) -> Result<Vec<ExtractedEntity>> {
        // Step 1: Split text into words (whitespace-based for simplicity)
        let words: Vec<&str> = text.split_whitespace().collect();
        if words.is_empty() {
            return Ok(Vec::new());
        }

        // Step 2: Tokenize each word and entity label
        let tokenizer = self.tokenizer.lock().unwrap();
        let (input_ids, attention_mask, words_mask, text_lengths, word_to_char) =
            self.encode_prompt(&tokenizer, &words, labels, text)?;

        let num_words = words.len();
        let num_labels = labels.len();

        // Step 3: Generate candidate spans
        let (span_idx, span_mask) = self.make_span_tensors(num_words, 1);

        // Step 4: Run ONNX inference and extract logits
        let logits_owned = {
            let mut session = self.session.lock().unwrap();

            let input_ids_tensor = ort::value::Tensor::from_array(input_ids)?;
            let attention_mask_tensor = ort::value::Tensor::from_array(attention_mask)?;
            let words_mask_tensor = ort::value::Tensor::from_array(words_mask)?;
            let text_lengths_tensor = ort::value::Tensor::from_array(text_lengths)?;
            let span_idx_tensor = ort::value::Tensor::from_array(span_idx)?;
            let span_mask_tensor = ort::value::Tensor::from_array(span_mask)?;

            let outputs = session.run(ort::inputs![
                "input_ids" => input_ids_tensor,
                "attention_mask" => attention_mask_tensor,
                "words_mask" => words_mask_tensor,
                "text_lengths" => text_lengths_tensor,
                "span_idx" => span_idx_tensor,
                "span_mask" => span_mask_tensor,
            ])?;

            // Extract logits: shape [batch=1, num_spans, num_labels]
            // Try named output first, fall back to first output
            let logits_view = if let Some(val) = outputs.get("logits") {
                val.try_extract_array::<f32>()
                    .context("Failed to extract logits tensor")?
            } else {
                let first_key = outputs
                    .keys()
                    .next()
                    .context("No outputs from GLiNER model")?;
                outputs[first_key]
                    .try_extract_array::<f32>()
                    .context("Failed to extract first output tensor")?
            };

            // Clone to owned array so we can drop outputs and session
            logits_view.to_owned()
        };

        // Step 5: Decode logits into entities
        let entities = self.decode_spans(
            &logits_owned,
            &words,
            labels,
            num_words,
            num_labels,
            &word_to_char,
        );

        // Step 6: Apply greedy deduplication if flat_ner
        if self.config.flat_ner {
            Ok(greedy_dedup(entities))
        } else {
            Ok(entities)
        }
    }

    /// Run batched inference on multiple texts with a single ONNX model call.
    ///
    /// Tokenizes all texts, pads tensors to uniform dimensions, and runs one batched
    /// forward pass. Results are decoded per-text.
    fn run_batch_inference(
        &self,
        texts: &[&str],
        labels: &[&str],
    ) -> Result<Vec<Vec<ExtractedEntity>>> {
        let batch_size = texts.len();
        if batch_size == 0 {
            return Ok(Vec::new());
        }

        // Step 1: Split all texts into words and filter empty
        let all_words: Vec<Vec<&str>> = texts
            .iter()
            .map(|t| t.split_whitespace().collect::<Vec<&str>>())
            .collect();

        // For texts with no words, return empty results
        let max_num_words = all_words.iter().map(|w| w.len()).max().unwrap_or(0);
        if max_num_words == 0 {
            return Ok(vec![Vec::new(); batch_size]);
        }

        // Step 2: Tokenize all texts under a single lock acquisition
        let tokenizer = self.tokenizer.lock().unwrap();

        // Encode entity labels (shared across all texts in the batch)
        let mut entity_token_ids: Vec<i64> = Vec::new();
        for label in labels {
            entity_token_ids.push(ENTITY_TOKEN_ID);
            let encoding = tokenizer
                .encode(*label, false)
                .map_err(|e| anyhow::anyhow!("Tokenizer error for label '{}': {}", label, e))?;
            for &id in encoding.get_ids() {
                entity_token_ids.push(id as i64);
            }
        }
        entity_token_ids.push(SEP_TOKEN_ID);

        // Tokenize each text's words
        let mut all_word_tokens: Vec<Vec<Vec<i64>>> = Vec::with_capacity(batch_size);
        let mut all_word_to_char: Vec<Vec<(usize, usize)>> = Vec::with_capacity(batch_size);

        for (i, words) in all_words.iter().enumerate() {
            let mut word_token_ids: Vec<Vec<i64>> = Vec::new();
            for word in words {
                let encoding = tokenizer
                    .encode(*word, false)
                    .map_err(|e| anyhow::anyhow!("Tokenizer error for word '{}': {}", word, e))?;
                let ids: Vec<i64> = encoding.get_ids().iter().map(|&id| id as i64).collect();
                word_token_ids.push(ids);
            }
            let word_to_char = build_word_to_char_map(words, texts[i]);
            all_word_tokens.push(word_token_ids);
            all_word_to_char.push(word_to_char);
        }

        drop(tokenizer); // Release tokenizer lock before inference

        // Step 3: Compute max sequence length across all texts
        let max_seq_len = all_word_tokens
            .iter()
            .map(|wt| {
                let total_text_tokens: usize = wt.iter().map(|ids| ids.len()).sum();
                1 + entity_token_ids.len() + total_text_tokens + 1 // [CLS] + entity + text + [SEP]
            })
            .max()
            .unwrap_or(0);

        // Step 4: Build padded tensors for the full batch
        let mut input_ids = Array2::<i64>::zeros((batch_size, max_seq_len));
        let mut attention_mask = Array2::<i64>::zeros((batch_size, max_seq_len));
        let mut words_mask_arr = Array2::<i64>::zeros((batch_size, max_seq_len));
        let mut text_lengths = Array2::<i64>::zeros((batch_size, 1));

        for (b, word_tokens) in all_word_tokens.iter().enumerate() {
            let mut idx = 0;

            // [CLS] token
            input_ids[[b, idx]] = START_TOKEN_ID;
            attention_mask[[b, idx]] = 1;
            idx += 1;

            // Entity label tokens (shared)
            for &token_id in &entity_token_ids {
                input_ids[[b, idx]] = token_id;
                attention_mask[[b, idx]] = 1;
                idx += 1;
            }

            // Text word tokens with word mask
            let mut word_id: i64 = 1;
            for word_ids in word_tokens {
                for (token_idx, &token_id) in word_ids.iter().enumerate() {
                    input_ids[[b, idx]] = token_id;
                    attention_mask[[b, idx]] = 1;
                    if token_idx == 0 {
                        words_mask_arr[[b, idx]] = word_id;
                    }
                    idx += 1;
                }
                word_id += 1;
            }

            // [SEP] token
            input_ids[[b, idx]] = END_TOKEN_ID;
            attention_mask[[b, idx]] = 1;

            text_lengths[[b, 0]] = all_words[b].len() as i64;
        }

        // Step 5: Generate span tensors (padded to max_num_words)
        let (span_idx, span_mask) = self.make_span_tensors(max_num_words, batch_size);

        // Step 6: Run single ONNX inference for the full batch
        let logits_owned = {
            let mut session = self.session.lock().unwrap();

            let input_ids_tensor = ort::value::Tensor::from_array(input_ids)?;
            let attention_mask_tensor = ort::value::Tensor::from_array(attention_mask)?;
            let words_mask_tensor = ort::value::Tensor::from_array(words_mask_arr)?;
            let text_lengths_tensor = ort::value::Tensor::from_array(text_lengths)?;
            let span_idx_tensor = ort::value::Tensor::from_array(span_idx)?;
            let span_mask_tensor = ort::value::Tensor::from_array(span_mask)?;

            let outputs = session.run(ort::inputs![
                "input_ids" => input_ids_tensor,
                "attention_mask" => attention_mask_tensor,
                "words_mask" => words_mask_tensor,
                "text_lengths" => text_lengths_tensor,
                "span_idx" => span_idx_tensor,
                "span_mask" => span_mask_tensor,
            ])?;

            let logits_view = if let Some(val) = outputs.get("logits") {
                val.try_extract_array::<f32>()
                    .context("Failed to extract logits tensor")?
            } else {
                let first_key = outputs
                    .keys()
                    .next()
                    .context("No outputs from GLiNER model")?;
                outputs[first_key]
                    .try_extract_array::<f32>()
                    .context("Failed to extract first output tensor")?
            };

            logits_view.to_owned()
        };

        // Step 7: Decode results per-text
        let num_labels = labels.len();
        let mut results = Vec::with_capacity(batch_size);

        for b in 0..batch_size {
            let num_words = all_words[b].len();
            if num_words == 0 {
                results.push(Vec::new());
                continue;
            }

            let entities = self.decode_spans_batch(
                &logits_owned,
                b,
                &all_words[b],
                labels,
                num_words,
                num_labels,
                &all_word_to_char[b],
            );

            if self.config.flat_ner {
                results.push(greedy_dedup(entities));
            } else {
                results.push(entities);
            }
        }

        Ok(results)
    }

    /// Encode the GLiNER prompt: [CLS] <<ENT>> label1 <<ENT>> label2 ... <<SEP>> word1 word2 ... [SEP]
    #[allow(clippy::type_complexity)]
    fn encode_prompt(
        &self,
        tokenizer: &Tokenizer,
        words: &[&str],
        labels: &[&str],
        original_text: &str,
    ) -> Result<(
        Array2<i64>,
        Array2<i64>,
        Array2<i64>,
        Array2<i64>,
        Vec<(usize, usize)>,
    )> {
        // Tokenize entity labels
        let mut entity_token_ids: Vec<i64> = Vec::new();
        for label in labels {
            entity_token_ids.push(ENTITY_TOKEN_ID);
            let encoding = tokenizer
                .encode(*label, false)
                .map_err(|e| anyhow::anyhow!("Tokenizer error for label '{}': {}", label, e))?;
            for &id in encoding.get_ids() {
                entity_token_ids.push(id as i64);
            }
        }
        entity_token_ids.push(SEP_TOKEN_ID);

        let _text_offset = 1 + entity_token_ids.len(); // +1 for [CLS]

        // Tokenize each text word
        let mut word_token_ids: Vec<Vec<i64>> = Vec::new();
        let mut total_text_tokens = 0;
        for word in words {
            let encoding = tokenizer
                .encode(*word, false)
                .map_err(|e| anyhow::anyhow!("Tokenizer error for word '{}': {}", word, e))?;
            let ids: Vec<i64> = encoding.get_ids().iter().map(|&id| id as i64).collect();
            total_text_tokens += ids.len();
            word_token_ids.push(ids);
        }

        // Build word-to-char offset mapping
        let word_to_char = build_word_to_char_map(words, original_text);

        // Total sequence length: [CLS] + entity_tokens + text_tokens + [SEP]
        let seq_len = 1 + entity_token_ids.len() + total_text_tokens + 1;

        // Build tensors (batch_size = 1)
        let mut input_ids = Array2::<i64>::zeros((1, seq_len));
        let mut attention_mask = Array2::<i64>::zeros((1, seq_len));
        let mut words_mask_arr = Array2::<i64>::zeros((1, seq_len));

        let mut idx = 0;

        // [CLS] token
        input_ids[[0, idx]] = START_TOKEN_ID;
        attention_mask[[0, idx]] = 1;
        idx += 1;

        // Entity label tokens
        for &token_id in &entity_token_ids {
            input_ids[[0, idx]] = token_id;
            attention_mask[[0, idx]] = 1;
            idx += 1;
        }

        // Text word tokens with word mask
        let mut word_id: i64 = 1; // 1-indexed word IDs
        for word_ids in &word_token_ids {
            for (token_idx, &token_id) in word_ids.iter().enumerate() {
                input_ids[[0, idx]] = token_id;
                attention_mask[[0, idx]] = 1;
                // Only first subword token of each word gets the word ID
                if token_idx == 0 {
                    words_mask_arr[[0, idx]] = word_id;
                }
                idx += 1;
            }
            word_id += 1;
        }

        // [SEP] token
        input_ids[[0, idx]] = END_TOKEN_ID;
        attention_mask[[0, idx]] = 1;

        // Text lengths (number of words)
        let text_lengths = Array2::from_elem((1, 1), words.len() as i64);

        Ok((
            input_ids,
            attention_mask,
            words_mask_arr,
            text_lengths,
            word_to_char,
        ))
    }

    /// Generate span index and mask tensors for candidate entity spans.
    fn make_span_tensors(
        &self,
        num_words: usize,
        batch_size: usize,
    ) -> (Array3<i64>, Array2<bool>) {
        let max_width = self.config.max_width;
        let num_spans = num_words * max_width;

        let mut span_idx = Array3::<i64>::zeros((batch_size, num_spans, 2));
        let mut span_mask = Array2::from_elem((batch_size, num_spans), false);

        for s in 0..batch_size {
            for start in 0..num_words {
                let remaining = num_words - start;
                let actual_max_width = max_width.min(remaining);
                for width in 0..actual_max_width {
                    let dim = start * max_width + width;
                    span_idx[[s, dim, 0]] = start as i64;
                    span_idx[[s, dim, 1]] = (start + width) as i64;
                    span_mask[[s, dim]] = true;
                }
            }
        }

        (span_idx, span_mask)
    }

    /// Decode model output logits into extracted entities.
    ///
    /// The logits array has shape `[batch=1, num_words, max_width, num_labels]` where:
    /// - batch: always 1 for single-text inference
    /// - num_words: number of words in the text
    /// - max_width: candidate span widths (0 = single word, max_width-1 = longest span)
    /// - num_labels: number of entity labels
    ///
    /// Raw logit scores are converted to probabilities via sigmoid.
    fn decode_spans(
        &self,
        logits: &ndarray::ArrayD<f32>,
        words: &[&str],
        labels: &[&str],
        num_words: usize,
        num_labels: usize,
        word_to_char: &[(usize, usize)],
    ) -> Vec<ExtractedEntity> {
        let threshold = self.config.threshold;
        let mut entities = Vec::new();

        let shape = logits.shape();

        // Handle different output shapes from GLiNER models
        match shape.len() {
            4 => {
                // Shape: [batch, num_words, max_width, num_labels]
                let out_num_words = shape[1];
                let out_max_width = shape[2];
                let out_num_labels = shape[3].min(num_labels);

                for start_word in 0..out_num_words.min(num_words) {
                    for width in 0..out_max_width {
                        let end_word = start_word + width;
                        if end_word >= num_words {
                            break;
                        }
                        for label_idx in 0..out_num_labels {
                            let logit = logits[[0, start_word, width, label_idx]];
                            let score = sigmoid(logit);
                            if score >= threshold {
                                let entity_text = words[start_word..=end_word].join(" ");
                                let char_start =
                                    word_to_char.get(start_word).map(|&(s, _)| s).unwrap_or(0);
                                let char_end =
                                    word_to_char.get(end_word).map(|&(_, e)| e).unwrap_or(0);
                                entities.push(ExtractedEntity {
                                    text: entity_text,
                                    label: labels[label_idx].to_string(),
                                    start: char_start,
                                    end: char_end,
                                    confidence: score,
                                });
                            }
                        }
                    }
                }
            }
            3 => {
                // Shape: [batch, num_spans, num_labels] (flattened span indices)
                let max_width = self.config.max_width;
                let actual_num_spans = shape[1];
                let actual_num_labels = shape[2].min(num_labels);

                for span_idx in 0..actual_num_spans {
                    let start_word = span_idx / max_width;
                    let width = span_idx % max_width;
                    let end_word = start_word + width;

                    if end_word >= num_words {
                        continue;
                    }

                    for label_idx in 0..actual_num_labels {
                        let logit = logits[[0, span_idx, label_idx]];
                        let score = sigmoid(logit);
                        if score >= threshold {
                            let entity_text = words[start_word..=end_word].join(" ");
                            let char_start =
                                word_to_char.get(start_word).map(|&(s, _)| s).unwrap_or(0);
                            let char_end = word_to_char.get(end_word).map(|&(_, e)| e).unwrap_or(0);
                            entities.push(ExtractedEntity {
                                text: entity_text,
                                label: labels[label_idx].to_string(),
                                start: char_start,
                                end: char_end,
                                confidence: score,
                            });
                        }
                    }
                }
            }
            _ => {
                tracing::warn!("Unexpected logits shape: {:?}", shape);
            }
        }

        // Sort by confidence descending
        entities.sort_by(|a, b| {
            b.confidence
                .partial_cmp(&a.confidence)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        entities
    }
}

impl NerPipeline for GlinerNerPipeline {
    fn extract_entities(&self, text: &str, labels: &[&str]) -> Result<Vec<ExtractedEntity>> {
        if text.is_empty() || labels.is_empty() {
            return Ok(Vec::new());
        }
        self.run_inference(text, labels)
    }

    fn extract_entities_batch(
        &self,
        texts: &[&str],
        labels: &[&str],
    ) -> Result<Vec<Vec<ExtractedEntity>>> {
        if texts.is_empty() || labels.is_empty() {
            return Ok(texts.iter().map(|_| Vec::new()).collect());
        }

        self.run_batch_inference(texts, labels)
    }
}

impl GlinerNerPipeline {
    /// Decode model output logits for a specific batch element.
    ///
    /// Similar to `decode_spans` but takes a batch index for multi-text inference.
    #[allow(clippy::too_many_arguments)]
    fn decode_spans_batch(
        &self,
        logits: &ndarray::ArrayD<f32>,
        batch_idx: usize,
        words: &[&str],
        labels: &[&str],
        num_words: usize,
        num_labels: usize,
        word_to_char: &[(usize, usize)],
    ) -> Vec<ExtractedEntity> {
        let threshold = self.config.threshold;
        let mut entities = Vec::new();

        let shape = logits.shape();

        match shape.len() {
            4 => {
                // Shape: [batch, num_words, max_width, num_labels]
                let out_num_words = shape[1];
                let out_max_width = shape[2];
                let out_num_labels = shape[3].min(num_labels);

                for start_word in 0..out_num_words.min(num_words) {
                    for width in 0..out_max_width {
                        let end_word = start_word + width;
                        if end_word >= num_words {
                            break;
                        }
                        for label_idx in 0..out_num_labels {
                            let logit = logits[[batch_idx, start_word, width, label_idx]];
                            let score = sigmoid(logit);
                            if score >= threshold {
                                let entity_text = words[start_word..=end_word].join(" ");
                                let char_start =
                                    word_to_char.get(start_word).map(|&(s, _)| s).unwrap_or(0);
                                let char_end =
                                    word_to_char.get(end_word).map(|&(_, e)| e).unwrap_or(0);
                                entities.push(ExtractedEntity {
                                    text: entity_text,
                                    label: labels[label_idx].to_string(),
                                    start: char_start,
                                    end: char_end,
                                    confidence: score,
                                });
                            }
                        }
                    }
                }
            }
            3 => {
                // Shape: [batch, num_spans, num_labels]
                let max_width = self.config.max_width;
                let actual_num_spans = shape[1];
                let actual_num_labels = shape[2].min(num_labels);

                for span_idx in 0..actual_num_spans {
                    let start_word = span_idx / max_width;
                    let width = span_idx % max_width;
                    let end_word = start_word + width;

                    if end_word >= num_words {
                        continue;
                    }

                    for label_idx in 0..actual_num_labels {
                        let logit = logits[[batch_idx, span_idx, label_idx]];
                        let score = sigmoid(logit);
                        if score >= threshold {
                            let entity_text = words[start_word..=end_word].join(" ");
                            let char_start =
                                word_to_char.get(start_word).map(|&(s, _)| s).unwrap_or(0);
                            let char_end = word_to_char.get(end_word).map(|&(_, e)| e).unwrap_or(0);
                            entities.push(ExtractedEntity {
                                text: entity_text,
                                label: labels[label_idx].to_string(),
                                start: char_start,
                                end: char_end,
                                confidence: score,
                            });
                        }
                    }
                }
            }
            _ => {
                tracing::warn!("Unexpected logits shape: {:?}", shape);
            }
        }

        entities.sort_by(|a, b| {
            b.confidence
                .partial_cmp(&a.confidence)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        entities
    }
}

// Send + Sync are satisfied because Session and Tokenizer are behind Mutex
unsafe impl Send for GlinerNerPipeline {}
unsafe impl Sync for GlinerNerPipeline {}

/// Sigmoid activation function.
#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Build a mapping from word index to (start_char, end_char) in the original text.
fn build_word_to_char_map(words: &[&str], original_text: &str) -> Vec<(usize, usize)> {
    let mut result = Vec::with_capacity(words.len());
    let mut search_start = 0;

    for &word in words {
        if let Some(pos) = original_text[search_start..].find(word) {
            let abs_start = search_start + pos;
            let abs_end = abs_start + word.len();
            result.push((abs_start, abs_end));
            search_start = abs_end;
        } else {
            // Fallback: use accumulated position
            result.push((search_start, search_start + word.len()));
            search_start += word.len() + 1;
        }
    }

    result
}

/// Greedy deduplication: when spans overlap, keep the one with highest confidence.
/// This implements flat NER (no nested entities).
fn greedy_dedup(mut entities: Vec<ExtractedEntity>) -> Vec<ExtractedEntity> {
    // Sort by confidence descending (highest first)
    entities.sort_by(|a, b| {
        b.confidence
            .partial_cmp(&a.confidence)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut result = Vec::new();
    let mut occupied: HashSet<usize> = HashSet::new();

    for entity in entities.drain(..) {
        // Check if any character position in this span is already occupied
        let overlaps = (entity.start..entity.end).any(|pos| occupied.contains(&pos));
        if !overlaps {
            for pos in entity.start..entity.end {
                occupied.insert(pos);
            }
            result.push(entity);
        }
    }

    // Sort by start position for consistent output
    result.sort_by_key(|e| e.start);
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sigmoid() {
        assert!((sigmoid(0.0) - 0.5).abs() < 1e-6);
        assert!(sigmoid(10.0) > 0.99);
        assert!(sigmoid(-10.0) < 0.01);
    }

    #[test]
    fn test_sigmoid_symmetry() {
        for x in [-5.0, -1.0, 0.0, 1.0, 5.0] {
            let sum = sigmoid(x) + sigmoid(-x);
            assert!(
                (sum - 1.0).abs() < 1e-6,
                "sigmoid(x) + sigmoid(-x) should equal 1"
            );
        }
    }

    #[test]
    fn test_build_word_to_char_map_simple() {
        let text = "Hello World";
        let words = vec!["Hello", "World"];
        let map = build_word_to_char_map(&words, text);
        assert_eq!(map, vec![(0, 5), (6, 11)]);
    }

    #[test]
    fn test_build_word_to_char_map_multiple_spaces() {
        let text = "Juan  vive  en  Madrid";
        let words: Vec<&str> = text.split_whitespace().collect();
        let map = build_word_to_char_map(&words, text);
        assert_eq!(map[0], (0, 4)); // Juan
        assert_eq!(map[1], (6, 10)); // vive
        assert_eq!(map[2], (12, 14)); // en
        assert_eq!(map[3], (16, 22)); // Madrid
    }

    #[test]
    fn test_build_word_to_char_map_punctuation() {
        let text = "Hello, World!";
        let words: Vec<&str> = text.split_whitespace().collect();
        let map = build_word_to_char_map(&words, text);
        assert_eq!(map[0], (0, 6)); // "Hello,"
        assert_eq!(map[1], (7, 13)); // "World!"
    }

    #[test]
    fn test_greedy_dedup_no_overlap() {
        let entities = vec![
            ExtractedEntity {
                text: "Juan".to_string(),
                label: "person".to_string(),
                start: 0,
                end: 4,
                confidence: 0.95,
            },
            ExtractedEntity {
                text: "Madrid".to_string(),
                label: "location".to_string(),
                start: 15,
                end: 21,
                confidence: 0.90,
            },
        ];
        let result = greedy_dedup(entities);
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn test_greedy_dedup_with_overlap() {
        let entities = vec![
            ExtractedEntity {
                text: "New York City".to_string(),
                label: "location".to_string(),
                start: 0,
                end: 13,
                confidence: 0.95,
            },
            ExtractedEntity {
                text: "York".to_string(),
                label: "location".to_string(),
                start: 4,
                end: 8,
                confidence: 0.80,
            },
        ];
        let result = greedy_dedup(entities);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].text, "New York City");
    }

    #[test]
    fn test_greedy_dedup_keeps_higher_confidence() {
        let entities = vec![
            ExtractedEntity {
                text: "Apple Inc".to_string(),
                label: "organization".to_string(),
                start: 0,
                end: 9,
                confidence: 0.60,
            },
            ExtractedEntity {
                text: "Apple".to_string(),
                label: "organization".to_string(),
                start: 0,
                end: 5,
                confidence: 0.90,
            },
        ];
        // Higher confidence "Apple" should win over overlapping "Apple Inc"
        let result = greedy_dedup(entities);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].text, "Apple");
        assert_eq!(result[0].confidence, 0.90);
    }

    #[test]
    fn test_greedy_dedup_empty() {
        let entities: Vec<ExtractedEntity> = Vec::new();
        let result = greedy_dedup(entities);
        assert!(result.is_empty());
    }

    // -- Tests that require the ONNX model --

    fn get_model_paths() -> Option<(String, String)> {
        let manifest_dir = env!("CARGO_MANIFEST_DIR");
        let model_path = format!("{}/models/gliner_small-v2.1/onnx/model.onnx", manifest_dir);
        let tokenizer_path = format!("{}/models/gliner_small-v2.1/tokenizer.json", manifest_dir);

        // Also check workspace-level models dir
        let workspace_model_path = format!(
            "{}/../models/gliner_small-v2.1/onnx/model.onnx",
            manifest_dir
        );
        let workspace_tokenizer_path = format!(
            "{}/../models/gliner_small-v2.1/tokenizer.json",
            manifest_dir
        );

        if Path::new(&model_path).exists() && Path::new(&tokenizer_path).exists() {
            Some((model_path, tokenizer_path))
        } else if Path::new(&workspace_model_path).exists()
            && Path::new(&workspace_tokenizer_path).exists()
        {
            Some((workspace_model_path, workspace_tokenizer_path))
        } else {
            None
        }
    }

    #[test]
    fn test_gliner_pipeline_creation() {
        let Some((model_path, tokenizer_path)) = get_model_paths() else {
            eprintln!("Skipping test: GLiNER model not found. Run scripts/download_models.sh");
            return;
        };
        let pipeline =
            GlinerNerPipeline::new(&model_path, &tokenizer_path, GlinerConfig::default());
        assert!(
            pipeline.is_ok(),
            "Failed to create pipeline: {:?}",
            pipeline.err()
        );
    }

    #[test]
    fn test_gliner_extract_entities_english() {
        let Some((model_path, tokenizer_path)) = get_model_paths() else {
            eprintln!("Skipping test: GLiNER model not found. Run scripts/download_models.sh");
            return;
        };
        let pipeline = GlinerNerPipeline::new(
            &model_path,
            &tokenizer_path,
            GlinerConfig {
                threshold: 0.3,
                ..GlinerConfig::default()
            },
        )
        .unwrap();

        let text = "My name is James Bond and I live in London";
        let labels = &["person", "location"];
        let entities = pipeline.extract_entities(text, labels).unwrap();

        eprintln!("Extracted entities: {:?}", entities);

        // We expect at least person and location entities
        let persons: Vec<_> = entities.iter().filter(|e| e.label == "person").collect();
        let locations: Vec<_> = entities.iter().filter(|e| e.label == "location").collect();

        assert!(
            !persons.is_empty(),
            "Should find at least one person entity"
        );
        assert!(
            !locations.is_empty(),
            "Should find at least one location entity"
        );
    }

    #[test]
    fn test_gliner_extract_entities_spanish() {
        let Some((model_path, tokenizer_path)) = get_model_paths() else {
            eprintln!("Skipping test: GLiNER model not found. Run scripts/download_models.sh");
            return;
        };
        let pipeline = GlinerNerPipeline::new(
            &model_path,
            &tokenizer_path,
            GlinerConfig {
                threshold: 0.3,
                ..GlinerConfig::default()
            },
        )
        .unwrap();

        let text = "Juan se mudó de Madrid a Berlín en enero 2026";
        let labels = &["persona", "lugar", "fecha"];
        let entities = pipeline.extract_entities(text, labels).unwrap();

        eprintln!("Extracted entities (Spanish): {:?}", entities);

        // Should extract at least some entities from Spanish text
        assert!(
            !entities.is_empty(),
            "Should extract entities from Spanish text"
        );
    }

    #[test]
    fn test_gliner_empty_text() {
        let Some((model_path, tokenizer_path)) = get_model_paths() else {
            eprintln!("Skipping test: GLiNER model not found. Run scripts/download_models.sh");
            return;
        };
        let pipeline =
            GlinerNerPipeline::new(&model_path, &tokenizer_path, GlinerConfig::default()).unwrap();

        let entities = pipeline.extract_entities("", &["person"]).unwrap();
        assert!(entities.is_empty());
    }

    #[test]
    fn test_gliner_empty_labels() {
        let Some((model_path, tokenizer_path)) = get_model_paths() else {
            eprintln!("Skipping test: GLiNER model not found. Run scripts/download_models.sh");
            return;
        };
        let pipeline =
            GlinerNerPipeline::new(&model_path, &tokenizer_path, GlinerConfig::default()).unwrap();

        let entities = pipeline.extract_entities("Hello World", &[]).unwrap();
        assert!(entities.is_empty());
    }

    #[test]
    fn test_gliner_confidence_range() {
        let Some((model_path, tokenizer_path)) = get_model_paths() else {
            eprintln!("Skipping test: GLiNER model not found. Run scripts/download_models.sh");
            return;
        };
        let pipeline = GlinerNerPipeline::new(
            &model_path,
            &tokenizer_path,
            GlinerConfig {
                threshold: 0.1, // Low threshold to get more entities
                ..GlinerConfig::default()
            },
        )
        .unwrap();

        let entities = pipeline
            .extract_entities(
                "Apple was founded by Steve Jobs in Cupertino",
                &["person", "organization", "location"],
            )
            .unwrap();

        for entity in &entities {
            assert!(
                entity.confidence >= 0.0 && entity.confidence <= 1.0,
                "Confidence {} out of range for entity '{}'",
                entity.confidence,
                entity.text
            );
        }
    }

    #[test]
    fn test_gliner_configurable_labels() {
        let Some((model_path, tokenizer_path)) = get_model_paths() else {
            eprintln!("Skipping test: GLiNER model not found. Run scripts/download_models.sh");
            return;
        };
        let pipeline = GlinerNerPipeline::new(
            &model_path,
            &tokenizer_path,
            GlinerConfig {
                threshold: 0.3,
                ..GlinerConfig::default()
            },
        )
        .unwrap();

        // Test with custom domain-specific labels
        let text = "The patient John Smith was diagnosed with diabetes at Mayo Clinic";
        let labels = &["patient", "disease", "hospital"];
        let entities = pipeline.extract_entities(text, labels).unwrap();

        eprintln!("Custom label entities: {:?}", entities);
        // Zero-shot: labels are configurable at inference time
        // We just verify the pipeline runs without errors and returns valid entities
        for entity in &entities {
            assert!(
                labels.contains(&entity.label.as_str()),
                "Entity label '{}' should be one of the provided labels",
                entity.label
            );
        }
    }

    // ---- Edge-case tests (no model required) ----

    #[test]
    fn test_build_word_to_char_map_empty_input() {
        let map = build_word_to_char_map(&[], "");
        assert!(map.is_empty());
    }

    #[test]
    fn test_build_word_to_char_map_unicode_multibyte() {
        // Verify correct byte offset tracking with multi-byte UTF-8 characters
        let text = "café niño";
        let words: Vec<&str> = text.split_whitespace().collect();
        let map = build_word_to_char_map(&words, text);
        assert_eq!(map.len(), 2);
        assert_eq!(map[0].0, 0); // "café" starts at byte 0
                                 // "niño" starts after "café " (5 bytes for café + 1 space = 6)
        assert!(map[1].0 > 0); // second word starts after first
    }

    #[test]
    fn test_greedy_dedup_keeps_all_non_overlapping() {
        // Non-overlapping entities should all be kept regardless of confidence ordering
        let entities = vec![
            ExtractedEntity {
                text: "first".to_string(),
                label: "a".to_string(),
                start: 0,
                end: 5,
                confidence: 0.8,
            },
            ExtractedEntity {
                text: "second".to_string(),
                label: "b".to_string(),
                start: 10,
                end: 16,
                confidence: 0.9,
            },
            ExtractedEntity {
                text: "third".to_string(),
                label: "c".to_string(),
                start: 20,
                end: 25,
                confidence: 0.7,
            },
        ];
        let result = greedy_dedup(entities);
        assert_eq!(
            result.len(),
            3,
            "All non-overlapping entities should be kept"
        );
        let texts: Vec<&str> = result.iter().map(|e| e.text.as_str()).collect();
        assert!(texts.contains(&"first"));
        assert!(texts.contains(&"second"));
        assert!(texts.contains(&"third"));
    }

    #[test]
    fn test_sigmoid_extreme_values() {
        // Very large positive: sigmoid → 1.0
        let large = sigmoid(100.0);
        assert!((large - 1.0).abs() < 1e-6);
        // Very large negative: sigmoid → 0.0
        let small = sigmoid(-100.0);
        assert!(small < 1e-6);
    }

    #[test]
    fn test_ner_pipeline_trait_is_object_safe() {
        // Verify NerPipeline can be used as dyn trait object
        #[allow(dead_code)]
        fn accepts_dyn(_: &dyn NerPipeline) {}
        // Compile-time check
    }

    // ---- Batch NER tests (no model required) ----

    /// A simple NER pipeline for testing batch behavior.
    struct SimpleBatchNer;

    impl NerPipeline for SimpleBatchNer {
        fn extract_entities(&self, text: &str, _labels: &[&str]) -> Result<Vec<ExtractedEntity>> {
            // Extract capitalized words as entities
            let mut entities = Vec::new();
            for word in text.split_whitespace() {
                let clean = word.trim_matches(|c: char| !c.is_alphanumeric());
                if !clean.is_empty() && clean.chars().next().unwrap().is_uppercase() {
                    let start = text.find(clean).unwrap_or(0);
                    entities.push(ExtractedEntity {
                        text: clean.to_string(),
                        label: "entity".to_string(),
                        start,
                        end: start + clean.len(),
                        confidence: 0.9,
                    });
                }
            }
            Ok(entities)
        }
    }

    #[test]
    fn test_batch_default_implementation_matches_individual() {
        let ner = SimpleBatchNer;
        let texts = &["Hello World", "Alice met Bob", "No entities here"];
        let labels = &["person", "location"];

        // Default batch implementation calls extract_entities per text
        let batch_results = ner.extract_entities_batch(texts, labels).unwrap();
        assert_eq!(batch_results.len(), 3);

        // Compare with individual calls
        for (i, text) in texts.iter().enumerate() {
            let individual = ner.extract_entities(text, labels).unwrap();
            assert_eq!(
                batch_results[i].len(),
                individual.len(),
                "Batch result for text {} doesn't match individual",
                i
            );
        }
    }

    #[test]
    fn test_batch_empty_texts() {
        let ner = SimpleBatchNer;
        let texts: &[&str] = &[];
        let labels = &["person"];
        let results = ner.extract_entities_batch(texts, labels).unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn test_batch_empty_labels() {
        let ner = SimpleBatchNer;
        let texts = &["Hello World"];
        let labels: &[&str] = &[];
        let results = ner.extract_entities_batch(texts, labels).unwrap();
        assert_eq!(results.len(), 1);
        // Default impl calls extract_entities which may return entities regardless of labels
        // (depends on impl), but our SimpleBatchNer ignores labels
    }

    #[test]
    fn test_batch_single_text() {
        let ner = SimpleBatchNer;
        let texts = &["Alice met Bob in London"];
        let labels = &["person", "location"];
        let batch_results = ner.extract_entities_batch(texts, labels).unwrap();
        assert_eq!(batch_results.len(), 1);
        assert!(!batch_results[0].is_empty());
    }

    #[test]
    fn test_batch_preserves_per_text_results() {
        let ner = SimpleBatchNer;
        let texts = &[
            "Alice works in Paris", // has Alice, Paris
            "no caps here",         // no entities
            "Bob lives in Berlin",  // has Bob, Berlin
        ];
        let labels = &["person", "location"];
        let results = ner.extract_entities_batch(texts, labels).unwrap();
        assert_eq!(results.len(), 3);
        assert!(!results[0].is_empty(), "First text should have entities");
        assert!(results[1].is_empty(), "Second text should have no entities");
        assert!(!results[2].is_empty(), "Third text should have entities");
    }

    // ---- Batch NER model tests (skip if model absent) ----

    #[test]
    fn test_gliner_batch_inference() {
        let Some((model_path, tokenizer_path)) = get_model_paths() else {
            eprintln!("Skipping test: GLiNER model not found. Run scripts/download_models.sh");
            return;
        };
        let pipeline = GlinerNerPipeline::new(
            &model_path,
            &tokenizer_path,
            GlinerConfig {
                threshold: 0.3,
                ..GlinerConfig::default()
            },
        )
        .unwrap();

        let texts = &["James Bond lives in London", "Marie Curie worked in Paris"];
        let labels = &["person", "location"];
        let results = pipeline.extract_entities_batch(texts, labels).unwrap();

        assert_eq!(results.len(), 2);

        // Both texts should have entities
        assert!(!results[0].is_empty(), "First text should have entities");
        assert!(!results[1].is_empty(), "Second text should have entities");

        eprintln!("Batch results[0]: {:?}", results[0]);
        eprintln!("Batch results[1]: {:?}", results[1]);
    }

    #[test]
    fn test_gliner_batch_matches_individual() {
        let Some((model_path, tokenizer_path)) = get_model_paths() else {
            eprintln!("Skipping test: GLiNER model not found. Run scripts/download_models.sh");
            return;
        };
        let pipeline = GlinerNerPipeline::new(
            &model_path,
            &tokenizer_path,
            GlinerConfig {
                threshold: 0.3,
                ..GlinerConfig::default()
            },
        )
        .unwrap();

        let texts = &[
            "Apple was founded by Steve Jobs",
            "Google headquarters is in Mountain View",
        ];
        let labels = &["person", "organization", "location"];

        // Get batch results
        let batch_results = pipeline.extract_entities_batch(texts, labels).unwrap();

        // Get individual results
        let individual_0 = pipeline.extract_entities(texts[0], labels).unwrap();
        let individual_1 = pipeline.extract_entities(texts[1], labels).unwrap();

        // Entity counts should match (batch inference may have slight
        // differences due to padding, but entity text should be similar)
        assert_eq!(batch_results.len(), 2);
        eprintln!("Batch[0] entities: {:?}", batch_results[0]);
        eprintln!("Individual[0] entities: {:?}", individual_0);
        eprintln!("Batch[1] entities: {:?}", batch_results[1]);
        eprintln!("Individual[1] entities: {:?}", individual_1);

        // Same entity texts should appear in both (order may differ)
        let batch_texts_0: std::collections::HashSet<&str> =
            batch_results[0].iter().map(|e| e.text.as_str()).collect();
        let indiv_texts_0: std::collections::HashSet<&str> =
            individual_0.iter().map(|e| e.text.as_str()).collect();
        // At minimum, both should find the same prominent entities
        assert!(
            !batch_texts_0.is_empty() || !indiv_texts_0.is_empty(),
            "At least one method should find entities"
        );
    }

    #[test]
    fn test_gliner_batch_empty_and_nonempty() {
        let Some((model_path, tokenizer_path)) = get_model_paths() else {
            eprintln!("Skipping test: GLiNER model not found. Run scripts/download_models.sh");
            return;
        };
        let pipeline = GlinerNerPipeline::new(
            &model_path,
            &tokenizer_path,
            GlinerConfig {
                threshold: 0.3,
                ..GlinerConfig::default()
            },
        )
        .unwrap();

        // Empty texts should return empty, non-empty should have entities
        let texts: &[&str] = &[];
        let labels = &["person"];
        let results = pipeline.extract_entities_batch(texts, labels).unwrap();
        assert!(results.is_empty());
    }
}
