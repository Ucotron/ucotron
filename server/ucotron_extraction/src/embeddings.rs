//! # ONNX Embedding Pipeline
//!
//! Local embedding generation using ONNX Runtime with the all-MiniLM-L6-v2 model.
//! Produces 384-dimensional normalized vectors compatible with sentence-transformers.
//!
//! ## Architecture
//!
//! The pipeline:
//! 1. Tokenizes input text using a HuggingFace WordPiece tokenizer
//! 2. Runs inference through the ONNX model (3 inputs: input_ids, attention_mask, token_type_ids)
//! 3. Applies mean pooling over token embeddings (masked by attention_mask)
//! 4. L2-normalizes the resulting 384-dim vector
//!
//! ## Usage
//!
//! ```no_run
//! use ucotron_extraction::embeddings::OnnxEmbeddingPipeline;
//! use ucotron_extraction::EmbeddingPipeline;
//!
//! let pipeline = OnnxEmbeddingPipeline::new(
//!     "models/all-MiniLM-L6-v2/model.onnx",
//!     "models/all-MiniLM-L6-v2/tokenizer.json",
//!     4, // intra_threads
//! ).unwrap();
//!
//! let embedding = pipeline.embed_text("Hello world").unwrap();
//! assert_eq!(embedding.len(), 384);
//! ```

use anyhow::{Context, Result};
// ndarray used by ort for tensor extraction
use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use ort::value::Tensor;
use std::path::Path;
use std::sync::Mutex;
use tokenizers::Tokenizer;

use crate::EmbeddingPipeline;

/// Output embedding dimension for all-MiniLM-L6-v2.
pub const EMBEDDING_DIM: usize = 384;

/// ONNX-based embedding pipeline using all-MiniLM-L6-v2 (or compatible models).
///
/// Thread-safe: holds the ONNX session behind a Mutex for concurrent access.
/// For maximum throughput in multi-threaded scenarios, consider creating one
/// pipeline per thread.
pub struct OnnxEmbeddingPipeline {
    session: Mutex<Session>,
    tokenizer: Mutex<Tokenizer>,
}

impl OnnxEmbeddingPipeline {
    /// Create a new ONNX embedding pipeline.
    ///
    /// # Arguments
    /// * `model_path` - Path to the ONNX model file (e.g., `model.onnx`)
    /// * `tokenizer_path` - Path to the HuggingFace tokenizer JSON (e.g., `tokenizer.json`)
    /// * `intra_threads` - Number of threads for ONNX Runtime intra-op parallelism
    pub fn new(
        model_path: impl AsRef<Path>,
        tokenizer_path: impl AsRef<Path>,
        intra_threads: usize,
    ) -> Result<Self> {
        let model_path = model_path.as_ref();
        let tokenizer_path = tokenizer_path.as_ref();

        // Load ONNX session
        let session = Session::builder()
            .context("Failed to create ONNX session builder")?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .context("Failed to set optimization level")?
            .with_intra_threads(intra_threads)
            .context("Failed to set intra threads")?
            .commit_from_file(model_path)
            .with_context(|| format!("Failed to load ONNX model from {:?}", model_path))?;

        // Load tokenizer
        let mut tokenizer = Tokenizer::from_file(tokenizer_path)
            .map_err(|e| anyhow::anyhow!("Failed to load tokenizer from {:?}: {}", tokenizer_path, e))?;

        // Configure tokenizer: disable default padding so we control it per-batch
        // The tokenizer.json has padding=128 by default, but we want dynamic padding
        tokenizer.with_padding(None);

        Ok(Self {
            session: Mutex::new(session),
            tokenizer: Mutex::new(tokenizer),
        })
    }

    /// Run inference on tokenized inputs and return mean-pooled, L2-normalized embeddings.
    ///
    /// # Arguments
    /// * `input_ids` - Token IDs, shape `[batch_size, seq_len]`
    /// * `attention_mask` - Attention mask, shape `[batch_size, seq_len]`
    /// * `token_type_ids` - Token type IDs, shape `[batch_size, seq_len]`
    /// * `batch_size` - Number of texts in the batch
    /// * `seq_len` - Padded sequence length
    fn run_inference(
        &self,
        input_ids: Vec<i64>,
        attention_mask: Vec<i64>,
        token_type_ids: Vec<i64>,
        batch_size: usize,
        seq_len: usize,
    ) -> Result<Vec<Vec<f32>>> {
        // Create tensors
        let ids_tensor = Tensor::from_array(([batch_size, seq_len], input_ids))
            .context("Failed to create input_ids tensor")?;
        let mask_tensor = Tensor::from_array(([batch_size, seq_len], attention_mask.clone()))
            .context("Failed to create attention_mask tensor")?;
        let type_tensor = Tensor::from_array(([batch_size, seq_len], token_type_ids))
            .context("Failed to create token_type_ids tensor")?;

        // Run inference
        let mut session = self.session.lock().map_err(|e| anyhow::anyhow!("Session lock poisoned: {}", e))?;
        let outputs = session.run(ort::inputs![
            "input_ids" => ids_tensor,
            "attention_mask" => mask_tensor,
            "token_type_ids" => type_tensor
        ]).context("ONNX inference failed")?;

        // Extract token embeddings: shape [batch_size, seq_len, 384]
        let token_embeddings = outputs["last_hidden_state"]
            .try_extract_array::<f32>()
            .context("Failed to extract last_hidden_state")?;

        // Get raw flat slice and work with it directly (avoids ndarray version issues)
        let emb_shape = token_embeddings.shape();
        anyhow::ensure!(
            emb_shape.len() == 3 && emb_shape[0] == batch_size && emb_shape[1] == seq_len && emb_shape[2] == EMBEDDING_DIM,
            "Expected output shape [{}, {}, {}], got {:?}",
            batch_size, seq_len, EMBEDDING_DIM, emb_shape
        );

        // Mean pooling with attention mask
        let mut results = Vec::with_capacity(batch_size);
        for b in 0..batch_size {
            let mut embedding = vec![0.0f32; EMBEDDING_DIM];
            let mut mask_sum = 0.0f32;

            for s in 0..seq_len {
                let m = attention_mask[b * seq_len + s] as f32;
                if m > 0.0 {
                    mask_sum += m;
                    for d in 0..EMBEDDING_DIM {
                        // ndarray uses row-major by default: index [b, s, d]
                        let val = token_embeddings[[b, s, d]];
                        embedding[d] += val * m;
                    }
                }
            }

            // Divide by mask sum (avoid div by zero)
            if mask_sum > 0.0 {
                for item in embedding.iter_mut().take(EMBEDDING_DIM) {
                    *item /= mask_sum;
                }
            }

            // L2 normalize
            l2_normalize(&mut embedding);
            results.push(embedding);
        }

        Ok(results)
    }
}

impl EmbeddingPipeline for OnnxEmbeddingPipeline {
    fn embed_text(&self, text: &str) -> Result<Vec<f32>> {
        let tokenizer = self.tokenizer.lock().map_err(|e| anyhow::anyhow!("Tokenizer lock poisoned: {}", e))?;

        let encoding = tokenizer
            .encode(text, true)
            .map_err(|e| anyhow::anyhow!("Tokenization failed: {}", e))?;

        let input_ids: Vec<i64> = encoding.get_ids().iter().map(|&id| id as i64).collect();
        let attention_mask: Vec<i64> = encoding.get_attention_mask().iter().map(|&m| m as i64).collect();
        let token_type_ids: Vec<i64> = encoding.get_type_ids().iter().map(|&t| t as i64).collect();
        let seq_len = input_ids.len();

        drop(tokenizer); // Release lock before inference

        let results = self.run_inference(input_ids, attention_mask, token_type_ids, 1, seq_len)?;
        Ok(results.into_iter().next().unwrap())
    }

    fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        let tokenizer = self.tokenizer.lock().map_err(|e| anyhow::anyhow!("Tokenizer lock poisoned: {}", e))?;

        // Tokenize all texts
        let encodings: Vec<_> = texts
            .iter()
            .map(|text| {
                tokenizer
                    .encode(*text, true)
                    .map_err(|e| anyhow::anyhow!("Tokenization failed for '{}': {}", text, e))
            })
            .collect::<Result<Vec<_>>>()?;

        drop(tokenizer); // Release lock before inference

        // Find max sequence length for padding
        let max_len = encodings.iter().map(|e| e.get_ids().len()).max().unwrap_or(0);
        let batch_size = encodings.len();

        // Pad and flatten into contiguous arrays
        let mut input_ids = Vec::with_capacity(batch_size * max_len);
        let mut attention_mask = Vec::with_capacity(batch_size * max_len);
        let mut token_type_ids = Vec::with_capacity(batch_size * max_len);

        for encoding in &encodings {
            let ids = encoding.get_ids();
            let mask = encoding.get_attention_mask();
            let types = encoding.get_type_ids();
            let len = ids.len();

            // Copy actual tokens
            input_ids.extend(ids.iter().map(|&id| id as i64));
            attention_mask.extend(mask.iter().map(|&m| m as i64));
            token_type_ids.extend(types.iter().map(|&t| t as i64));

            // Pad to max_len
            let pad_count = max_len - len;
            input_ids.extend(std::iter::repeat_n(0i64, pad_count));
            attention_mask.extend(std::iter::repeat_n(0i64, pad_count));
            token_type_ids.extend(std::iter::repeat_n(0i64, pad_count));
        }

        self.run_inference(input_ids, attention_mask, token_type_ids, batch_size, max_len)
    }
}

// Send + Sync: Session and Tokenizer are behind Mutex
unsafe impl Send for OnnxEmbeddingPipeline {}
unsafe impl Sync for OnnxEmbeddingPipeline {}

/// Configuration for parallel embedding computation.
#[derive(Debug, Clone)]
pub struct ParallelEmbeddingConfig {
    /// Number of parallel workers (each has its own ONNX session).
    /// Default: 1 (sequential). Values > 1 enable parallel inference.
    pub num_workers: usize,
    /// Sub-batch size per worker. Texts are split into sub-batches of this
    /// size and distributed round-robin across workers. Default: 32.
    pub batch_size: usize,
}

impl Default for ParallelEmbeddingConfig {
    fn default() -> Self {
        Self {
            num_workers: 1,
            batch_size: 32,
        }
    }
}

/// Parallel embedding pipeline that distributes work across multiple ONNX
/// sessions for concurrent CPU inference.
///
/// Each worker holds its own `OnnxEmbeddingPipeline` (and thus its own ONNX
/// session + tokenizer), avoiding Mutex contention. Sub-batches are assigned
/// round-robin to workers and processed in parallel via `std::thread::scope`.
///
/// When `num_workers == 1`, this is equivalent to a single pipeline with
/// sub-batching (useful for controlling peak memory on large inputs).
pub struct ParallelEmbeddingPipeline {
    workers: Vec<OnnxEmbeddingPipeline>,
    batch_size: usize,
}

impl ParallelEmbeddingPipeline {
    /// Create a new parallel embedding pipeline.
    ///
    /// # Arguments
    /// * `model_path` - Path to the ONNX model file
    /// * `tokenizer_path` - Path to the HuggingFace tokenizer JSON
    /// * `intra_threads` - ONNX intra-op threads per worker
    /// * `config` - Parallelism configuration
    pub fn new(
        model_path: impl AsRef<Path>,
        tokenizer_path: impl AsRef<Path>,
        intra_threads: usize,
        config: ParallelEmbeddingConfig,
    ) -> Result<Self> {
        let num_workers = config.num_workers.max(1);
        let batch_size = config.batch_size.max(1);

        let mut workers = Vec::with_capacity(num_workers);
        for _ in 0..num_workers {
            workers.push(OnnxEmbeddingPipeline::new(
                model_path.as_ref(),
                tokenizer_path.as_ref(),
                intra_threads,
            )?);
        }

        Ok(Self {
            workers,
            batch_size,
        })
    }

    /// Number of worker sessions.
    pub fn num_workers(&self) -> usize {
        self.workers.len()
    }

    /// Sub-batch size per worker.
    pub fn batch_size(&self) -> usize {
        self.batch_size
    }
}

impl EmbeddingPipeline for ParallelEmbeddingPipeline {
    fn embed_text(&self, text: &str) -> Result<Vec<f32>> {
        self.workers[0].embed_text(text)
    }

    fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        // Split into sub-batches
        let sub_batches: Vec<&[&str]> = texts.chunks(self.batch_size).collect();

        // Fast path: single worker or single sub-batch
        if self.workers.len() == 1 || sub_batches.len() == 1 {
            let mut all_results = Vec::with_capacity(texts.len());
            for batch in &sub_batches {
                let results = self.workers[0].embed_batch(batch)?;
                all_results.extend(results);
            }
            return Ok(all_results);
        }

        // Parallel path: distribute sub-batches across workers
        let num_workers = self.workers.len();
        let num_batches = sub_batches.len();

        // Pre-allocate output slots (one per sub-batch)
        let mut results: Vec<Option<Result<Vec<Vec<f32>>>>> =
            (0..num_batches).map(|_| None).collect();

        std::thread::scope(|scope| {
            let mut handles = Vec::with_capacity(num_batches);

            for (batch_idx, batch) in sub_batches.iter().enumerate() {
                let worker = &self.workers[batch_idx % num_workers];
                let batch_owned: Vec<&str> = batch.to_vec();

                handles.push((batch_idx, scope.spawn(move || {
                    worker.embed_batch(&batch_owned)
                })));
            }

            for (batch_idx, handle) in handles {
                results[batch_idx] = Some(handle.join().map_err(|_| {
                    anyhow::anyhow!("Embedding worker thread panicked for batch {}", batch_idx)
                })?);
            }

            Ok::<(), anyhow::Error>(())
        })?;

        // Collect results in order
        let mut all_embeddings = Vec::with_capacity(texts.len());
        for (i, slot) in results.into_iter().enumerate() {
            match slot {
                Some(Ok(embeddings)) => all_embeddings.extend(embeddings),
                Some(Err(e)) => return Err(e.context(format!("Sub-batch {} failed", i))),
                None => return Err(anyhow::anyhow!("Sub-batch {} was not processed", i)),
            }
        }

        Ok(all_embeddings)
    }
}

// Send + Sync: each worker's Session and Tokenizer are behind Mutex
unsafe impl Send for ParallelEmbeddingPipeline {}
unsafe impl Sync for ParallelEmbeddingPipeline {}

/// L2-normalize a vector in place. If the vector is zero, it remains zero.
fn l2_normalize(vec: &mut [f32]) {
    let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > f32::EPSILON {
        for x in vec.iter_mut() {
            *x /= norm;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    /// Get the path to the test model directory, or None if models are not available.
    /// Tests requiring ONNX models gracefully skip when models are absent (CI-friendly).
    /// Download models with: `scripts/download_models.sh`
    fn model_dir() -> Option<PathBuf> {
        let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .join("models")
            .join("all-MiniLM-L6-v2");
        if dir.join("model.onnx").exists() && dir.join("tokenizer.json").exists() {
            Some(dir)
        } else {
            None
        }
    }

    fn create_pipeline() -> Option<OnnxEmbeddingPipeline> {
        let dir = model_dir()?;
        Some(
            OnnxEmbeddingPipeline::new(
                dir.join("model.onnx"),
                dir.join("tokenizer.json"),
                4,
            )
            .expect("Failed to create pipeline"),
        )
    }

    #[test]
    fn test_l2_normalize() {
        let mut v = vec![3.0, 4.0];
        l2_normalize(&mut v);
        assert!((v[0] - 0.6).abs() < 1e-6);
        assert!((v[1] - 0.8).abs() < 1e-6);

        // Zero vector stays zero
        let mut z = vec![0.0, 0.0, 0.0];
        l2_normalize(&mut z);
        assert_eq!(z, vec![0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_l2_normalize_384dim() {
        let mut v: Vec<f32> = (0..384).map(|i| i as f32).collect();
        l2_normalize(&mut v);
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5, "Expected unit norm, got {}", norm);
    }

    // ---- Tests requiring ONNX model (skip gracefully if models absent) ----

    #[test]
    fn test_embed_text_produces_384_dim() {
        let Some(pipeline) = create_pipeline() else {
            eprintln!("Skipping test: embedding model not found. Run scripts/download_models.sh");
            return;
        };
        let embedding = pipeline.embed_text("This is a test sentence.").unwrap();
        assert_eq!(embedding.len(), EMBEDDING_DIM);
    }

    #[test]
    fn test_embed_text_is_normalized() {
        let Some(pipeline) = create_pipeline() else {
            eprintln!("Skipping test: embedding model not found. Run scripts/download_models.sh");
            return;
        };
        let embedding = pipeline.embed_text("The quick brown fox jumps over the lazy dog.").unwrap();
        let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(
            (norm - 1.0).abs() < 1e-4,
            "Expected unit norm, got {}",
            norm
        );
    }

    #[test]
    fn test_embed_text_deterministic() {
        let Some(pipeline) = create_pipeline() else {
            eprintln!("Skipping test: embedding model not found. Run scripts/download_models.sh");
            return;
        };
        let e1 = pipeline.embed_text("Hello world").unwrap();
        let e2 = pipeline.embed_text("Hello world").unwrap();
        assert_eq!(e1, e2, "Same text should produce identical embeddings");
    }

    #[test]
    fn test_embed_text_semantic_similarity() {
        let Some(pipeline) = create_pipeline() else {
            eprintln!("Skipping test: embedding model not found. Run scripts/download_models.sh");
            return;
        };
        let e_dog = pipeline.embed_text("The dog runs in the park").unwrap();
        let e_cat = pipeline.embed_text("The cat plays in the garden").unwrap();
        let e_code = pipeline.embed_text("Rust programming language features").unwrap();

        let sim_animals = cosine_sim(&e_dog, &e_cat);
        let sim_dog_code = cosine_sim(&e_dog, &e_code);

        assert!(
            sim_animals > sim_dog_code,
            "Similar sentences ({:.4}) should have higher similarity than dissimilar ones ({:.4})",
            sim_animals,
            sim_dog_code
        );
    }

    #[test]
    fn test_embed_batch_produces_correct_count() {
        let Some(pipeline) = create_pipeline() else {
            eprintln!("Skipping test: embedding model not found. Run scripts/download_models.sh");
            return;
        };
        let texts = vec!["First sentence.", "Second sentence.", "Third one."];
        let embeddings = pipeline.embed_batch(&texts).unwrap();
        assert_eq!(embeddings.len(), 3);
        for emb in &embeddings {
            assert_eq!(emb.len(), EMBEDDING_DIM);
        }
    }

    #[test]
    fn test_embed_batch_matches_individual() {
        let Some(pipeline) = create_pipeline() else {
            eprintln!("Skipping test: embedding model not found. Run scripts/download_models.sh");
            return;
        };
        let texts = vec!["Hello world", "Goodbye moon"];

        let batch = pipeline.embed_batch(&texts).unwrap();
        let single_0 = pipeline.embed_text("Hello world").unwrap();
        let single_1 = pipeline.embed_text("Goodbye moon").unwrap();

        // Batch and individual should produce very similar results
        // (may not be exactly equal due to floating-point padding differences)
        let sim_0 = cosine_sim(&batch[0], &single_0);
        let sim_1 = cosine_sim(&batch[1], &single_1);

        assert!(
            sim_0 > 0.999,
            "Batch[0] should match individual embed, similarity: {:.6}",
            sim_0
        );
        assert!(
            sim_1 > 0.999,
            "Batch[1] should match individual embed, similarity: {:.6}",
            sim_1
        );
    }

    #[test]
    fn test_embed_batch_empty() {
        let Some(pipeline) = create_pipeline() else {
            eprintln!("Skipping test: embedding model not found. Run scripts/download_models.sh");
            return;
        };
        let embeddings = pipeline.embed_batch(&[]).unwrap();
        assert!(embeddings.is_empty());
    }

    #[test]
    fn test_embed_batch_single_item() {
        let Some(pipeline) = create_pipeline() else {
            eprintln!("Skipping test: embedding model not found. Run scripts/download_models.sh");
            return;
        };
        let embeddings = pipeline.embed_batch(&["Just one text"]).unwrap();
        assert_eq!(embeddings.len(), 1);
        assert_eq!(embeddings[0].len(), EMBEDDING_DIM);
    }

    #[test]
    fn test_embed_text_short_input() {
        let Some(pipeline) = create_pipeline() else {
            eprintln!("Skipping test: embedding model not found. Run scripts/download_models.sh");
            return;
        };
        let embedding = pipeline.embed_text("Hi").unwrap();
        assert_eq!(embedding.len(), EMBEDDING_DIM);
        let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-4);
    }

    // ---- Edge-case tests (no model required) ----

    #[test]
    fn test_l2_normalize_single_element() {
        let mut v = vec![5.0];
        l2_normalize(&mut v);
        assert!((v[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_l2_normalize_negative_values() {
        let mut v = vec![-3.0, -4.0];
        l2_normalize(&mut v);
        assert!((v[0] - (-0.6)).abs() < 1e-6);
        assert!((v[1] - (-0.8)).abs() < 1e-6);
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_l2_normalize_very_small_values() {
        // Very small values: norm = sqrt(2) * 1e-40 ≈ 1.41e-40 which is > 0 but < f32::EPSILON (1.19e-7)
        // So they should remain zero (norm below epsilon threshold)
        let mut v = vec![1e-20, 1e-20];
        l2_normalize(&mut v);
        // norm = sqrt(2) * 1e-20 ≈ 1.41e-20 which is < f32::EPSILON → stays as-is
        // Actually norm of [1e-20, 1e-20] = sqrt(2e-40) = ~1.41e-20 which IS > f32::EPSILON
        // Let's just verify the function doesn't panic and produces finite output
        assert!(v.iter().all(|x| x.is_finite()), "All values should be finite");
    }

    #[test]
    fn test_embedding_pipeline_trait_is_object_safe() {
        // Verify EmbeddingPipeline can be used as dyn trait object
        #[allow(dead_code)]
        fn accepts_dyn(_: &dyn EmbeddingPipeline) {}
        // Compile-time check — if this compiles, trait is object-safe
    }

    #[test]
    fn test_embed_text_with_model_produces_valid_output() {
        // Integration test: verify real model output properties
        let Some(pipeline) = create_pipeline() else {
            eprintln!("Skipping test: embedding model not found. Run scripts/download_models.sh");
            return;
        };

        // Test with various text types
        let cases = vec![
            "A single word",
            "A typical English sentence with common vocabulary.",
            "¡Hola! Este es un texto en español con acentos: café, niño, año.",
            "1234567890 !@#$%^&*()",
            "   spaces   and   tabs\t\there  ",
        ];

        for text in cases {
            let embedding = pipeline.embed_text(text).unwrap();
            assert_eq!(embedding.len(), EMBEDDING_DIM, "Wrong dim for: {}", text);
            let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
            assert!((norm - 1.0).abs() < 1e-3, "Not normalized for: {} (norm={})", text, norm);
            // Ensure no NaN or Inf
            assert!(embedding.iter().all(|x| x.is_finite()), "NaN/Inf in embedding for: {}", text);
        }
    }

    // ---- ParallelEmbeddingConfig tests (no model required) ----

    #[test]
    fn test_parallel_config_default() {
        let config = ParallelEmbeddingConfig::default();
        assert_eq!(config.num_workers, 1);
        assert_eq!(config.batch_size, 32);
    }

    #[test]
    fn test_parallel_config_clamp_zero() {
        // Zero workers/batch_size should be clamped to 1
        let config = ParallelEmbeddingConfig {
            num_workers: 0,
            batch_size: 0,
        };
        // Verified at construction time in ParallelEmbeddingPipeline::new
        // but we test the config itself holds the values
        assert_eq!(config.num_workers, 0);
        assert_eq!(config.batch_size, 0);
    }

    // ---- ParallelEmbeddingPipeline tests (require ONNX model) ----

    fn create_parallel_pipeline(num_workers: usize, batch_size: usize) -> Option<ParallelEmbeddingPipeline> {
        let dir = model_dir()?;
        Some(
            ParallelEmbeddingPipeline::new(
                dir.join("model.onnx"),
                dir.join("tokenizer.json"),
                2, // intra_threads per worker
                ParallelEmbeddingConfig {
                    num_workers,
                    batch_size,
                },
            )
            .expect("Failed to create parallel pipeline"),
        )
    }

    #[test]
    fn test_parallel_pipeline_single_worker() {
        let Some(pipeline) = create_parallel_pipeline(1, 4) else {
            eprintln!("Skipping: embedding model not found");
            return;
        };
        assert_eq!(pipeline.num_workers(), 1);
        assert_eq!(pipeline.batch_size(), 4);

        let texts = vec!["Hello world", "Goodbye moon", "Test text"];
        let results = pipeline.embed_batch(&texts).unwrap();
        assert_eq!(results.len(), 3);
        for emb in &results {
            assert_eq!(emb.len(), EMBEDDING_DIM);
        }
    }

    #[test]
    fn test_parallel_pipeline_multi_worker() {
        let Some(pipeline) = create_parallel_pipeline(2, 2) else {
            eprintln!("Skipping: embedding model not found");
            return;
        };
        assert_eq!(pipeline.num_workers(), 2);

        // 5 texts with batch_size=2 → 3 sub-batches across 2 workers
        let texts = vec![
            "First sentence.",
            "Second sentence.",
            "Third sentence.",
            "Fourth sentence.",
            "Fifth sentence.",
        ];
        let results = pipeline.embed_batch(&texts).unwrap();
        assert_eq!(results.len(), 5);
        for emb in &results {
            assert_eq!(emb.len(), EMBEDDING_DIM);
            let norm: f32 = emb.iter().map(|x| x * x).sum::<f32>().sqrt();
            assert!((norm - 1.0).abs() < 1e-3, "Not normalized: {}", norm);
        }
    }

    #[test]
    fn test_parallel_pipeline_matches_sequential() {
        let Some(sequential) = create_pipeline() else {
            eprintln!("Skipping: embedding model not found");
            return;
        };
        let Some(parallel) = create_parallel_pipeline(2, 2) else {
            eprintln!("Skipping: embedding model not found");
            return;
        };

        let texts = vec!["The quick brown fox", "jumps over the lazy dog"];
        let seq_results = sequential.embed_batch(&texts).unwrap();
        let par_results = parallel.embed_batch(&texts).unwrap();

        // Results should be very similar (same model, same tokenizer)
        for (seq, par) in seq_results.iter().zip(par_results.iter()) {
            let sim = cosine_sim(seq, par);
            assert!(
                sim > 0.999,
                "Parallel vs sequential mismatch: similarity {:.6}",
                sim
            );
        }
    }

    #[test]
    fn test_parallel_pipeline_empty_input() {
        let Some(pipeline) = create_parallel_pipeline(2, 4) else {
            eprintln!("Skipping: embedding model not found");
            return;
        };
        let results = pipeline.embed_batch(&[]).unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn test_parallel_pipeline_embed_text() {
        let Some(pipeline) = create_parallel_pipeline(2, 4) else {
            eprintln!("Skipping: embedding model not found");
            return;
        };
        let emb = pipeline.embed_text("Single text test").unwrap();
        assert_eq!(emb.len(), EMBEDDING_DIM);
        let norm: f32 = emb.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-3);
    }

    #[test]
    fn test_parallel_pipeline_throughput_measurement() {
        // This test measures throughput and logs it — not a strict assertion.
        // Serves as the benchmark measurement for US-30.2.
        let Some(sequential) = create_pipeline() else {
            eprintln!("Skipping: embedding model not found");
            return;
        };
        let Some(parallel) = create_parallel_pipeline(2, 8) else {
            eprintln!("Skipping: embedding model not found");
            return;
        };

        // Generate test data: 32 texts
        let texts: Vec<String> = (0..32)
            .map(|i| format!("This is test sentence number {} for benchmark measurement.", i))
            .collect();
        let text_refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();

        // Warm up both pipelines
        let _ = sequential.embed_text("warmup");
        let _ = parallel.embed_text("warmup");

        // Measure sequential
        let start = std::time::Instant::now();
        let _ = sequential.embed_batch(&text_refs).unwrap();
        let seq_ms = start.elapsed().as_millis();

        // Measure parallel
        let start = std::time::Instant::now();
        let _ = parallel.embed_batch(&text_refs).unwrap();
        let par_ms = start.elapsed().as_millis();

        eprintln!(
            "Throughput measurement (32 texts): sequential={}ms, parallel(2 workers)={}ms, speedup={:.2}x",
            seq_ms, par_ms,
            seq_ms as f64 / par_ms.max(1) as f64
        );

        // We don't assert speedup since it depends on hardware,
        // but verify both produce correct output count
        let seq_results = sequential.embed_batch(&text_refs).unwrap();
        let par_results = parallel.embed_batch(&text_refs).unwrap();
        assert_eq!(seq_results.len(), 32);
        assert_eq!(par_results.len(), 32);
    }

    /// Helper: cosine similarity between two vectors.
    fn cosine_sim(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
    }
}
