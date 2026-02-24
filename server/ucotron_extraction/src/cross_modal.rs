//! Cross-modal projection layer for bridging CLIP and MiniLM embedding spaces.
//!
//! Provides [`ProjectionLayerPipeline`] that projects 512-dim CLIP embeddings into
//! 384-dim MiniLM space, enabling image-to-text semantic search by comparing projected
//! image embeddings against text embeddings in the same vector space.
//!
//! ## Architecture
//!
//! The projection is an MLP with architecture: `512 → 1024 → 512 → 384`:
//! - Layer 1: Linear(512, 1024) + ReLU
//! - Layer 2: Linear(1024, 512) + ReLU
//! - Layer 3: Linear(512, 384) (no activation)
//! - Output: L2-normalized 384-dim vector
//!
//! ## Model Format
//!
//! Expects an ONNX model (`projection_layer.onnx`) with:
//! - Input: `input` tensor of shape `[batch, 512]` (f32)
//! - Output: tensor of shape `[batch, 384]` (f32)
//!
//! ## Training
//!
//! The projection layer is trained on paired (CLIP, MiniLM) embeddings
//! using cosine similarity loss. See `scripts/train_projection_layer.py`.

use std::path::Path;
use std::sync::Mutex;

use anyhow::{Context, Result};
use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use ort::value::Tensor;

use crate::image::{l2_normalize, CLIP_EMBED_DIM};

/// MiniLM embedding dimensionality (all-MiniLM-L6-v2).
pub const MINILM_EMBED_DIM: usize = 384;

/// Configuration for the projection layer pipeline.
#[derive(Debug, Clone)]
pub struct ProjectionConfig {
    /// Number of ONNX intra-op threads (default: 4).
    pub num_threads: usize,
    /// Input dimensionality (CLIP space, default: 512).
    pub input_dim: usize,
    /// Output dimensionality (MiniLM space, default: 384).
    pub output_dim: usize,
}

impl Default for ProjectionConfig {
    fn default() -> Self {
        Self {
            num_threads: 4,
            input_dim: CLIP_EMBED_DIM,
            output_dim: MINILM_EMBED_DIM,
        }
    }
}

/// ONNX-backed projection layer that bridges CLIP (512-dim) to MiniLM (384-dim) space.
///
/// This enables image-to-text search: project a CLIP image embedding into MiniLM space,
/// then compare it against text embeddings from the text vector index.
pub struct ProjectionLayerPipeline {
    session: Mutex<Session>,
    config: ProjectionConfig,
}

// SAFETY: Session is behind a Mutex, ensuring exclusive access.
unsafe impl Send for ProjectionLayerPipeline {}
unsafe impl Sync for ProjectionLayerPipeline {}

impl ProjectionLayerPipeline {
    /// Create a new projection layer pipeline from an ONNX model.
    ///
    /// # Arguments
    /// - `model_path` — Path to `projection_layer.onnx`
    /// - `config` — Pipeline configuration
    pub fn new(model_path: impl AsRef<Path>, config: ProjectionConfig) -> Result<Self> {
        let session = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(config.num_threads)?
            .commit_from_file(model_path.as_ref())
            .context("Failed to load projection layer ONNX model")?;

        Ok(Self {
            session: Mutex::new(session),
            config,
        })
    }

    /// Project a single CLIP embedding (512-dim) into MiniLM space (384-dim).
    ///
    /// Returns an L2-normalized 384-dim vector.
    pub fn project(&self, clip_embedding: &[f32]) -> Result<Vec<f32>> {
        anyhow::ensure!(
            clip_embedding.len() == self.config.input_dim,
            "Expected {}-dim input, got {}",
            self.config.input_dim,
            clip_embedding.len()
        );

        let results = self.run_projection(&[clip_embedding.to_vec()])?;
        Ok(results.into_iter().next().unwrap())
    }

    /// Project a batch of CLIP embeddings into MiniLM space.
    ///
    /// Returns L2-normalized 384-dim vectors.
    pub fn project_batch(&self, clip_embeddings: &[Vec<f32>]) -> Result<Vec<Vec<f32>>> {
        if clip_embeddings.is_empty() {
            return Ok(Vec::new());
        }

        for (i, emb) in clip_embeddings.iter().enumerate() {
            anyhow::ensure!(
                emb.len() == self.config.input_dim,
                "Embedding {} has dim {}, expected {}",
                i,
                emb.len(),
                self.config.input_dim
            );
        }

        self.run_projection(clip_embeddings)
    }

    /// Run the ONNX projection model on a batch of embeddings.
    fn run_projection(&self, embeddings: &[Vec<f32>]) -> Result<Vec<Vec<f32>>> {
        let batch_size = embeddings.len();
        let input_dim = self.config.input_dim;
        let output_dim = self.config.output_dim;

        // Flatten embeddings into contiguous array
        let mut flat_input = Vec::with_capacity(batch_size * input_dim);
        for emb in embeddings {
            flat_input.extend_from_slice(emb);
        }

        // Create input tensor [batch_size, input_dim]
        let input_tensor = Tensor::from_array(([batch_size, input_dim], flat_input))
            .context("Failed to create projection input tensor")?;

        // Run inference
        let mut session = self
            .session
            .lock()
            .map_err(|e| anyhow::anyhow!("Session lock poisoned: {}", e))?;

        let outputs = session
            .run(ort::inputs!["input" => input_tensor])
            .context("Projection layer inference failed")?;

        // Extract output tensor — shape [batch_size, output_dim]
        let projected = outputs[0]
            .try_extract_array::<f32>()
            .context("Failed to extract projected embeddings")?;

        let projected_owned = projected.to_owned();
        let shape = projected_owned.shape();

        anyhow::ensure!(
            shape.len() == 2 && shape[0] == batch_size && shape[1] == output_dim,
            "Unexpected output shape {:?}, expected [{}, {}]",
            shape,
            batch_size,
            output_dim
        );

        // Convert to Vec<Vec<f32>> and L2 normalize each vector
        let mut results = Vec::with_capacity(batch_size);
        for b in 0..batch_size {
            let mut vec = Vec::with_capacity(output_dim);
            for d in 0..output_dim {
                vec.push(projected_owned[[b, d]]);
            }
            results.push(l2_normalize(&vec));
        }

        Ok(results)
    }

    /// Get the pipeline configuration.
    pub fn config(&self) -> &ProjectionConfig {
        &self.config
    }
}

/// Cross-modal projection trait for projecting embeddings across spaces.
///
/// This trait abstracts the projection operation, allowing both ONNX-backed
/// and mock implementations for testing.
pub trait CrossModalProjection: Send + Sync {
    /// Project a single embedding from source space to target space.
    fn project(&self, embedding: &[f32]) -> Result<Vec<f32>>;

    /// Project a batch of embeddings.
    fn project_batch(&self, embeddings: &[Vec<f32>]) -> Result<Vec<Vec<f32>>>;

    /// Source embedding dimensionality.
    fn input_dim(&self) -> usize;

    /// Target embedding dimensionality.
    fn output_dim(&self) -> usize;
}

impl CrossModalProjection for ProjectionLayerPipeline {
    fn project(&self, embedding: &[f32]) -> Result<Vec<f32>> {
        ProjectionLayerPipeline::project(self, embedding)
    }

    fn project_batch(&self, embeddings: &[Vec<f32>]) -> Result<Vec<Vec<f32>>> {
        ProjectionLayerPipeline::project_batch(self, embeddings)
    }

    fn input_dim(&self) -> usize {
        self.config.input_dim
    }

    fn output_dim(&self) -> usize {
        self.config.output_dim
    }
}

/// Mock projection layer for testing without ONNX models.
///
/// Performs a simple linear projection by truncating/padding and normalizing,
/// which preserves relative similarity ordering for testing purposes.
#[cfg(test)]
pub struct MockProjectionLayer {
    pub input_dim: usize,
    pub output_dim: usize,
}

#[cfg(test)]
impl MockProjectionLayer {
    pub fn new() -> Self {
        Self {
            input_dim: CLIP_EMBED_DIM,
            output_dim: MINILM_EMBED_DIM,
        }
    }
}

#[cfg(test)]
impl CrossModalProjection for MockProjectionLayer {
    fn project(&self, embedding: &[f32]) -> Result<Vec<f32>> {
        anyhow::ensure!(
            embedding.len() == self.input_dim,
            "Expected {}-dim input, got {}",
            self.input_dim,
            embedding.len()
        );
        // Simple projection: take first output_dim elements (truncation)
        let mut projected: Vec<f32> = embedding.iter().take(self.output_dim).copied().collect();
        projected.resize(self.output_dim, 0.0);
        Ok(l2_normalize(&projected))
    }

    fn project_batch(&self, embeddings: &[Vec<f32>]) -> Result<Vec<Vec<f32>>> {
        embeddings.iter().map(|e| self.project(e)).collect()
    }

    fn input_dim(&self) -> usize {
        self.input_dim
    }

    fn output_dim(&self) -> usize {
        self.output_dim
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::image::cosine_similarity;

    // --- Unit tests (no ONNX model required) ---

    #[test]
    fn test_projection_config_default() {
        let cfg = ProjectionConfig::default();
        assert_eq!(cfg.num_threads, 4);
        assert_eq!(cfg.input_dim, 512);
        assert_eq!(cfg.output_dim, 384);
    }

    #[test]
    fn test_projection_config_custom() {
        let cfg = ProjectionConfig {
            num_threads: 2,
            input_dim: 256,
            output_dim: 128,
        };
        assert_eq!(cfg.num_threads, 2);
        assert_eq!(cfg.input_dim, 256);
        assert_eq!(cfg.output_dim, 128);
    }

    #[test]
    fn test_minilm_embed_dim() {
        assert_eq!(MINILM_EMBED_DIM, 384);
    }

    #[test]
    fn test_mock_projection_dimensions() {
        let mock = MockProjectionLayer::new();
        assert_eq!(mock.input_dim(), CLIP_EMBED_DIM);
        assert_eq!(mock.output_dim(), MINILM_EMBED_DIM);
    }

    #[test]
    fn test_mock_projection_single() {
        let mock = MockProjectionLayer::new();
        let input = vec![0.1; CLIP_EMBED_DIM];
        let output = mock.project(&input).unwrap();

        assert_eq!(output.len(), MINILM_EMBED_DIM);
        // Should be L2 normalized
        let norm: f32 = output.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5, "Output not normalized: {}", norm);
    }

    #[test]
    fn test_mock_projection_batch() {
        let mock = MockProjectionLayer::new();
        let inputs = vec![vec![0.1; CLIP_EMBED_DIM], vec![0.2; CLIP_EMBED_DIM]];
        let outputs = mock.project_batch(&inputs).unwrap();

        assert_eq!(outputs.len(), 2);
        for out in &outputs {
            assert_eq!(out.len(), MINILM_EMBED_DIM);
            let norm: f32 = out.iter().map(|x| x * x).sum::<f32>().sqrt();
            assert!((norm - 1.0).abs() < 1e-5);
        }
    }

    #[test]
    fn test_mock_projection_empty_batch() {
        let mock = MockProjectionLayer::new();
        let outputs = mock.project_batch(&[]).unwrap();
        assert!(outputs.is_empty());
    }

    #[test]
    fn test_mock_projection_wrong_dim() {
        let mock = MockProjectionLayer::new();
        let input = vec![0.1; 256]; // Wrong dimension
        let result = mock.project(&input);
        assert!(result.is_err());
    }

    #[test]
    fn test_mock_projection_preserves_relative_similarity() {
        let mock = MockProjectionLayer::new();

        // Two similar CLIP embeddings
        let a = vec![0.1; CLIP_EMBED_DIM];
        let mut b = vec![0.1; CLIP_EMBED_DIM];
        b[0] = 0.11; // Slightly different

        // One dissimilar embedding
        let mut c = vec![-0.1; CLIP_EMBED_DIM];
        c[0] = 0.5;

        let proj_a = mock.project(&l2_normalize(&a)).unwrap();
        let proj_b = mock.project(&l2_normalize(&b)).unwrap();
        let proj_c = mock.project(&l2_normalize(&c)).unwrap();

        let sim_ab = cosine_similarity(&proj_a, &proj_b);
        let sim_ac = cosine_similarity(&proj_a, &proj_c);

        // a and b should be more similar than a and c
        assert!(
            sim_ab > sim_ac,
            "Expected sim(a,b)={} > sim(a,c)={}",
            sim_ab,
            sim_ac
        );
    }

    #[test]
    fn test_trait_object_safety() {
        let mock = MockProjectionLayer::new();
        let _boxed: Box<dyn CrossModalProjection> = Box::new(mock);
    }

    // --- Model-dependent tests (skip gracefully if model absent) ---

    fn projection_model_path() -> String {
        let base = std::env::var("UCOTRON_MODELS_DIR").unwrap_or_else(|_| "models".to_string());
        format!("{}/projection_layer.onnx", base)
    }

    #[test]
    fn test_projection_pipeline_loads() {
        let path = projection_model_path();
        if !Path::new(&path).exists() {
            eprintln!("SKIP: Projection model not found at {}", path);
            return;
        }
        let pipeline = ProjectionLayerPipeline::new(&path, ProjectionConfig::default());
        assert!(
            pipeline.is_ok(),
            "Failed to load projection model: {:?}",
            pipeline.err()
        );
    }

    #[test]
    fn test_projection_pipeline_project() {
        let path = projection_model_path();
        if !Path::new(&path).exists() {
            eprintln!("SKIP: Projection model not found at {}", path);
            return;
        }
        let pipeline = ProjectionLayerPipeline::new(&path, ProjectionConfig::default()).unwrap();

        let input = l2_normalize(&vec![0.1; CLIP_EMBED_DIM]);
        let output = pipeline.project(&input).unwrap();

        assert_eq!(output.len(), MINILM_EMBED_DIM);
        let norm: f32 = output.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(
            (norm - 1.0).abs() < 1e-4,
            "Output not normalized: {}",
            norm
        );
    }

    #[test]
    fn test_projection_pipeline_batch() {
        let path = projection_model_path();
        if !Path::new(&path).exists() {
            eprintln!("SKIP: Projection model not found at {}", path);
            return;
        }
        let pipeline = ProjectionLayerPipeline::new(&path, ProjectionConfig::default()).unwrap();

        let inputs = vec![
            l2_normalize(&vec![0.1; CLIP_EMBED_DIM]),
            l2_normalize(&vec![0.2; CLIP_EMBED_DIM]),
        ];
        let outputs = pipeline.project_batch(&inputs).unwrap();

        assert_eq!(outputs.len(), 2);
        for out in &outputs {
            assert_eq!(out.len(), MINILM_EMBED_DIM);
        }
    }

    #[test]
    fn test_projection_pipeline_wrong_dim() {
        let path = projection_model_path();
        if !Path::new(&path).exists() {
            eprintln!("SKIP: Projection model not found at {}", path);
            return;
        }
        let pipeline = ProjectionLayerPipeline::new(&path, ProjectionConfig::default()).unwrap();

        let input = vec![0.1; 256]; // Wrong dimension
        let result = pipeline.project(&input);
        assert!(result.is_err());
    }
}
