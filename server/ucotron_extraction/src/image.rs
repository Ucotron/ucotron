//! CLIP ONNX pipeline for image embedding and cross-modal search.
//!
//! Provides [`ClipImagePipeline`] for encoding images into 512-dimensional vectors
//! and [`ClipTextPipeline`] for encoding text queries into the same space,
//! enabling cross-modal (text-to-image and image-to-image) semantic search.
//!
//! ## Model Format
//!
//! Expects two ONNX models exported from CLIP ViT-B/32:
//! - **Visual encoder** (`visual_model.onnx`): `pixel_values [1,3,224,224]` → `image_embeds [1,512]`
//! - **Text encoder** (`text_model.onnx`): `input_ids [1,N]` + `attention_mask [1,N]` → `text_embeds [1,512]`
//!
//! ## Image Preprocessing
//!
//! Images are preprocessed to match CLIP's expected input:
//! 1. Resize to 224×224 (bicubic interpolation)
//! 2. Center crop if aspect ratio differs
//! 3. Convert to RGB float32 in [0,1]
//! 4. Normalize with CLIP means `[0.48145466, 0.4578275, 0.40821073]`
//!    and stds `[0.26862954, 0.26130258, 0.27577711]`
//! 5. Layout: CHW (channels-first)

use std::path::Path;
use std::sync::Mutex;

use anyhow::{Context, Result};
// ndarray used indirectly via try_extract_array
use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use ort::value::Tensor;

/// CLIP image input size (ViT-B/32).
pub const CLIP_IMAGE_SIZE: usize = 224;

/// CLIP embedding dimensionality (ViT-B/32).
pub const CLIP_EMBED_DIM: usize = 512;

/// CLIP normalization means (ImageNet-derived, used by OpenAI CLIP).
const CLIP_MEAN: [f32; 3] = [0.48145466, 0.4578275, 0.40821073];

/// CLIP normalization standard deviations.
#[allow(clippy::excessive_precision)]
const CLIP_STD: [f32; 3] = [0.26862954, 0.26130258, 0.27577711];

/// Maximum CLIP text token sequence length.
const CLIP_MAX_TEXT_LEN: usize = 77;

/// CLIP special token IDs (from the CLIP tokenizer).
const CLIP_SOT_TOKEN: i64 = 49406; // <|startoftext|>
const CLIP_EOT_TOKEN: i64 = 49407; // <|endoftext|>

/// Configuration for the CLIP ONNX pipeline.
#[derive(Debug, Clone)]
pub struct ClipConfig {
    /// Number of ONNX intra-op threads (default: 4).
    pub num_threads: usize,
    /// Image input size (default: 224).
    pub image_size: usize,
}

impl Default for ClipConfig {
    fn default() -> Self {
        Self {
            num_threads: 4,
            image_size: CLIP_IMAGE_SIZE,
        }
    }
}

/// CLIP visual encoder — converts images to 512-dim embedding vectors.
pub struct ClipImagePipeline {
    session: Mutex<Session>,
    config: ClipConfig,
}

impl ClipImagePipeline {
    /// Create a new CLIP image pipeline from an ONNX visual model.
    ///
    /// # Arguments
    /// - `model_path` — Path to `visual_model.onnx`
    /// - `config` — Pipeline configuration
    pub fn new(model_path: impl AsRef<Path>, config: ClipConfig) -> Result<Self> {
        let session = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(config.num_threads)?
            .commit_from_file(model_path.as_ref())
            .context("Failed to load CLIP visual ONNX model")?;

        Ok(Self {
            session: Mutex::new(session),
            config,
        })
    }

    /// Embed a single image from raw bytes (JPEG, PNG, etc.).
    ///
    /// Returns an L2-normalized 512-dim vector.
    pub fn embed_image_bytes(&self, bytes: &[u8]) -> Result<Vec<f32>> {
        let img = image::load_from_memory(bytes).context("Failed to decode image")?;
        self.embed_image(&img)
    }

    /// Embed a single image from a file path.
    pub fn embed_image_file(&self, path: impl AsRef<Path>) -> Result<Vec<f32>> {
        let img = image::open(path.as_ref())
            .with_context(|| format!("Failed to open image: {:?}", path.as_ref()))?;
        self.embed_image(&img)
    }

    /// Embed a DynamicImage.
    pub fn embed_image(&self, img: &image::DynamicImage) -> Result<Vec<f32>> {
        let pixel_values = preprocess_image(img, self.config.image_size);
        self.run_visual_encoder(&pixel_values)
    }

    /// Embed a batch of images from raw bytes.
    pub fn embed_batch_bytes(&self, images: &[&[u8]]) -> Result<Vec<Vec<f32>>> {
        let mut results = Vec::with_capacity(images.len());
        for bytes in images {
            results.push(self.embed_image_bytes(bytes)?);
        }
        Ok(results)
    }

    /// Run the visual encoder on preprocessed pixel values.
    fn run_visual_encoder(&self, pixel_values: &[f32]) -> Result<Vec<f32>> {
        let size = self.config.image_size;
        let tensor = Tensor::from_array(([1usize, 3, size, size], pixel_values.to_vec()))
            .context("Failed to create pixel_values tensor")?;

        let mut session = self.session.lock().unwrap();
        let inputs = ort::inputs![tensor];
        let outputs = session
            .run(inputs)
            .context("CLIP visual encoder inference failed")?;

        // Extract image embeddings — shape [1, 512]
        let arr = outputs[0]
            .try_extract_array::<f32>()
            .context("Failed to extract image embeddings")?;

        let embedding: Vec<f32> = arr
            .as_slice()
            .context("Embedding array not contiguous")?
            .to_vec();

        // L2 normalize
        Ok(l2_normalize(&embedding))
    }
}

/// CLIP text encoder — converts text queries to 512-dim embedding vectors
/// in the same space as images, enabling cross-modal search.
pub struct ClipTextPipeline {
    session: Mutex<Session>,
    tokenizer: Mutex<tokenizers::Tokenizer>,
}

impl ClipTextPipeline {
    /// Create a new CLIP text pipeline.
    ///
    /// # Arguments
    /// - `model_path` — Path to `text_model.onnx`
    /// - `tokenizer_path` — Path to CLIP `tokenizer.json`
    /// - `num_threads` — ONNX intra-op threads
    pub fn new(
        model_path: impl AsRef<Path>,
        tokenizer_path: impl AsRef<Path>,
        num_threads: usize,
    ) -> Result<Self> {
        let session = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(num_threads)?
            .commit_from_file(model_path.as_ref())
            .context("Failed to load CLIP text ONNX model")?;

        let tokenizer = tokenizers::Tokenizer::from_file(tokenizer_path.as_ref())
            .map_err(|e| anyhow::anyhow!("Failed to load CLIP tokenizer: {}", e))?;

        Ok(Self {
            session: Mutex::new(session),
            tokenizer: Mutex::new(tokenizer),
        })
    }

    /// Encode a text query into CLIP's embedding space.
    ///
    /// Returns an L2-normalized 512-dim vector.
    pub fn embed_text(&self, text: &str) -> Result<Vec<f32>> {
        let (input_ids, attention_mask) = self.tokenize(text)?;
        self.run_text_encoder(&input_ids, &attention_mask)
    }

    /// Encode a batch of text queries.
    pub fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        let mut results = Vec::with_capacity(texts.len());
        for text in texts {
            results.push(self.embed_text(text)?);
        }
        Ok(results)
    }

    /// Tokenize text for CLIP's text encoder.
    ///
    /// CLIP uses a simple BPE tokenizer with SOT/EOT tokens,
    /// padded/truncated to 77 tokens.
    fn tokenize(&self, text: &str) -> Result<(Vec<i64>, Vec<i64>)> {
        let tokenizer = self.tokenizer.lock().unwrap();
        let encoding = tokenizer
            .encode(text, false)
            .map_err(|e| anyhow::anyhow!("CLIP tokenization failed: {}", e))?;
        drop(tokenizer);

        let token_ids: Vec<i64> = encoding.get_ids().iter().map(|&id| id as i64).collect();

        // Build CLIP-format sequence: [SOT] tokens... [EOT] [PAD...]
        let max_content_len = CLIP_MAX_TEXT_LEN - 2; // Reserve SOT + EOT
        let truncated = if token_ids.len() > max_content_len {
            &token_ids[..max_content_len]
        } else {
            &token_ids
        };

        let mut input_ids = Vec::with_capacity(CLIP_MAX_TEXT_LEN);
        input_ids.push(CLIP_SOT_TOKEN);
        input_ids.extend_from_slice(truncated);
        input_ids.push(CLIP_EOT_TOKEN);

        let real_len = input_ids.len();
        // Pad to CLIP_MAX_TEXT_LEN with 0
        input_ids.resize(CLIP_MAX_TEXT_LEN, 0);

        let mut attention_mask = vec![0i64; CLIP_MAX_TEXT_LEN];
        #[allow(clippy::needless_range_loop)]
        for i in 0..real_len {
            attention_mask[i] = 1;
        }

        Ok((input_ids, attention_mask))
    }

    /// Run the text encoder.
    fn run_text_encoder(&self, input_ids: &[i64], attention_mask: &[i64]) -> Result<Vec<f32>> {
        let ids_tensor = Tensor::from_array(([1usize, CLIP_MAX_TEXT_LEN], input_ids.to_vec()))
            .context("Failed to create input_ids tensor")?;
        let mask_tensor =
            Tensor::from_array(([1usize, CLIP_MAX_TEXT_LEN], attention_mask.to_vec()))
                .context("Failed to create attention_mask tensor")?;

        let mut session = self.session.lock().unwrap();
        let inputs = ort::inputs![ids_tensor, mask_tensor];
        let outputs = session
            .run(inputs)
            .context("CLIP text encoder inference failed")?;

        // Extract text embeddings — shape [1, 512]
        let arr = outputs[0]
            .try_extract_array::<f32>()
            .context("Failed to extract text embeddings")?;

        let embedding: Vec<f32> = arr
            .as_slice()
            .context("Text embedding array not contiguous")?
            .to_vec();

        Ok(l2_normalize(&embedding))
    }
}

// ---------------------------------------------------------------------------
// Image preprocessing
// ---------------------------------------------------------------------------

/// Preprocess an image for CLIP's visual encoder.
///
/// 1. Center-crop to square
/// 2. Resize to `size × size` (bilinear)
/// 3. Convert to f32 [0, 1]
/// 4. Normalize with CLIP means/stds
/// 5. Return as flat CHW array
pub fn preprocess_image(img: &image::DynamicImage, size: usize) -> Vec<f32> {
    use image::imageops::FilterType;

    // Center crop to square
    let (w, h) = (img.width(), img.height());
    let crop_size = w.min(h);
    let x_offset = (w - crop_size) / 2;
    let y_offset = (h - crop_size) / 2;
    let cropped = img.crop_imm(x_offset, y_offset, crop_size, crop_size);

    // Resize to target size
    let resized = cropped.resize_exact(size as u32, size as u32, FilterType::Triangle);

    // Convert to RGB
    let rgb = resized.to_rgb8();

    // Convert to float CHW with CLIP normalization
    let mut pixel_values = vec![0.0f32; 3 * size * size];

    for y in 0..size {
        for x in 0..size {
            let pixel = rgb.get_pixel(x as u32, y as u32);
            for c in 0..3 {
                let val = pixel[c] as f32 / 255.0;
                let normalized = (val - CLIP_MEAN[c]) / CLIP_STD[c];
                pixel_values[c * size * size + y * size + x] = normalized;
            }
        }
    }

    pixel_values
}

/// L2-normalize a vector to unit length.
pub fn l2_normalize(v: &[f32]) -> Vec<f32> {
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm < f32::EPSILON {
        return v.to_vec();
    }
    v.iter().map(|x| x / norm).collect()
}

/// Compute cosine similarity between two L2-normalized vectors.
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

// Implement the extraction traits for CLIP pipelines.

impl crate::ImageEmbeddingPipeline for ClipImagePipeline {
    fn embed_image_bytes(&self, bytes: &[u8]) -> Result<Vec<f32>> {
        ClipImagePipeline::embed_image_bytes(self, bytes)
    }

    fn embed_image_file(&self, path: &Path) -> Result<Vec<f32>> {
        ClipImagePipeline::embed_image_file(self, path)
    }
}

impl crate::CrossModalTextEncoder for ClipTextPipeline {
    fn embed_text(&self, text: &str) -> Result<Vec<f32>> {
        ClipTextPipeline::embed_text(self, text)
    }
}

/// Result of indexing an image.
#[derive(Debug, Clone)]
pub struct ImageIndexResult {
    /// Node ID assigned to this image in the graph.
    pub node_id: u64,
    /// The CLIP embedding (512-dim).
    pub embedding: Vec<f32>,
    /// Image dimensions (width, height).
    pub dimensions: (u32, u32),
    /// Image format detected.
    pub format: String,
}

/// Result of a cross-modal search (text query → image results).
#[derive(Debug, Clone)]
pub struct CrossModalSearchResult {
    /// Node ID of the matched image.
    pub node_id: u64,
    /// Cosine similarity score.
    pub score: f32,
    /// Content/description stored for this image node.
    pub content: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_l2_normalize() {
        let v = vec![3.0, 4.0];
        let n = l2_normalize(&v);
        let norm: f32 = n.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-6);
        assert!((n[0] - 0.6).abs() < 1e-6);
        assert!((n[1] - 0.8).abs() < 1e-6);
    }

    #[test]
    fn test_l2_normalize_zero_vector() {
        let v = vec![0.0, 0.0, 0.0];
        let n = l2_normalize(&v);
        assert_eq!(n, vec![0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_cosine_similarity_identical() {
        let a = l2_normalize(&[1.0, 2.0, 3.0]);
        let sim = cosine_similarity(&a, &a);
        assert!((sim - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity_orthogonal() {
        let a = l2_normalize(&[1.0, 0.0]);
        let b = l2_normalize(&[0.0, 1.0]);
        let sim = cosine_similarity(&a, &b);
        assert!(sim.abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity_opposite() {
        let a = l2_normalize(&[1.0, 0.0]);
        let b = l2_normalize(&[-1.0, 0.0]);
        let sim = cosine_similarity(&a, &b);
        assert!((sim + 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_preprocess_image_output_shape() {
        // Create a 100x150 RGB image
        let img = image::DynamicImage::new_rgb8(100, 150);
        let pixels = preprocess_image(&img, CLIP_IMAGE_SIZE);
        assert_eq!(pixels.len(), 3 * CLIP_IMAGE_SIZE * CLIP_IMAGE_SIZE);
    }

    #[test]
    fn test_preprocess_image_square() {
        let img = image::DynamicImage::new_rgb8(224, 224);
        let pixels = preprocess_image(&img, CLIP_IMAGE_SIZE);
        assert_eq!(pixels.len(), 3 * 224 * 224);
    }

    #[test]
    fn test_preprocess_image_landscape() {
        // Wide image — should be center-cropped then resized
        let img = image::DynamicImage::new_rgb8(640, 480);
        let pixels = preprocess_image(&img, CLIP_IMAGE_SIZE);
        assert_eq!(pixels.len(), 3 * 224 * 224);
    }

    #[test]
    fn test_preprocess_image_portrait() {
        // Tall image
        let img = image::DynamicImage::new_rgb8(480, 640);
        let pixels = preprocess_image(&img, CLIP_IMAGE_SIZE);
        assert_eq!(pixels.len(), 3 * 224 * 224);
    }

    #[test]
    fn test_preprocess_image_normalization_range() {
        // Create a white image (all 255)
        let mut imgbuf = image::RgbImage::new(224, 224);
        for pixel in imgbuf.pixels_mut() {
            *pixel = image::Rgb([255, 255, 255]);
        }
        let img = image::DynamicImage::ImageRgb8(imgbuf);
        let pixels = preprocess_image(&img, CLIP_IMAGE_SIZE);

        // After normalization, white pixels (1.0) should be (1.0 - mean) / std
        let expected_r = (1.0 - CLIP_MEAN[0]) / CLIP_STD[0];
        let expected_g = (1.0 - CLIP_MEAN[1]) / CLIP_STD[1];
        let expected_b = (1.0 - CLIP_MEAN[2]) / CLIP_STD[2];

        // Check first pixel of each channel
        let s = CLIP_IMAGE_SIZE * CLIP_IMAGE_SIZE;
        assert!((pixels[0] - expected_r).abs() < 1e-4, "R channel mismatch");
        assert!((pixels[s] - expected_g).abs() < 1e-4, "G channel mismatch");
        assert!(
            (pixels[2 * s] - expected_b).abs() < 1e-4,
            "B channel mismatch"
        );
    }

    #[test]
    fn test_preprocess_image_black_normalization() {
        // Black image (all 0)
        let img = image::DynamicImage::new_rgb8(224, 224);
        let pixels = preprocess_image(&img, CLIP_IMAGE_SIZE);

        let expected_r = (0.0 - CLIP_MEAN[0]) / CLIP_STD[0];
        assert!((pixels[0] - expected_r).abs() < 1e-4);
    }

    #[test]
    fn test_clip_config_default() {
        let cfg = ClipConfig::default();
        assert_eq!(cfg.num_threads, 4);
        assert_eq!(cfg.image_size, 224);
    }

    #[test]
    fn test_clip_embed_dim() {
        assert_eq!(CLIP_EMBED_DIM, 512);
    }

    #[test]
    fn test_clip_image_size() {
        assert_eq!(CLIP_IMAGE_SIZE, 224);
    }

    #[test]
    fn test_image_index_result() {
        let result = ImageIndexResult {
            node_id: 42,
            embedding: vec![0.1; CLIP_EMBED_DIM],
            dimensions: (640, 480),
            format: "jpeg".to_string(),
        };
        assert_eq!(result.node_id, 42);
        assert_eq!(result.embedding.len(), CLIP_EMBED_DIM);
        assert_eq!(result.dimensions, (640, 480));
    }

    #[test]
    fn test_cross_modal_search_result() {
        let result = CrossModalSearchResult {
            node_id: 10,
            score: 0.85,
            content: "A photo of a cat".to_string(),
        };
        assert_eq!(result.node_id, 10);
        assert!((result.score - 0.85).abs() < 1e-6);
    }

    // --- Model-dependent tests (skip gracefully if models absent) ---

    fn clip_model_dir() -> String {
        let base = std::env::var("UCOTRON_MODELS_DIR").unwrap_or_else(|_| "models".to_string());
        format!("{}/clip-vit-base-patch32", base)
    }

    fn clip_visual_model_path() -> String {
        format!("{}/visual_model.onnx", clip_model_dir())
    }

    fn clip_text_model_path() -> String {
        format!("{}/text_model.onnx", clip_model_dir())
    }

    fn clip_tokenizer_path() -> String {
        format!("{}/tokenizer.json", clip_model_dir())
    }

    #[test]
    fn test_clip_image_pipeline_loads() {
        let path = clip_visual_model_path();
        if !std::path::Path::new(&path).exists() {
            eprintln!("SKIP: CLIP visual model not found at {}", path);
            return;
        }
        let pipeline = ClipImagePipeline::new(&path, ClipConfig::default());
        assert!(
            pipeline.is_ok(),
            "Failed to load CLIP visual model: {:?}",
            pipeline.err()
        );
    }

    #[test]
    fn test_clip_image_embed_synthetic() {
        let path = clip_visual_model_path();
        if !std::path::Path::new(&path).exists() {
            eprintln!("SKIP: CLIP visual model not found at {}", path);
            return;
        }
        let pipeline = ClipImagePipeline::new(&path, ClipConfig::default()).unwrap();

        // Create a synthetic red image
        let mut imgbuf = image::RgbImage::new(100, 100);
        for pixel in imgbuf.pixels_mut() {
            *pixel = image::Rgb([255, 0, 0]);
        }
        let img = image::DynamicImage::ImageRgb8(imgbuf);
        let embedding = pipeline.embed_image(&img).unwrap();

        assert_eq!(embedding.len(), CLIP_EMBED_DIM);
        // Should be L2-normalized
        let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(
            (norm - 1.0).abs() < 1e-4,
            "Embedding not normalized: {}",
            norm
        );
    }

    #[test]
    fn test_clip_text_pipeline_loads() {
        let model_path = clip_text_model_path();
        let tok_path = clip_tokenizer_path();
        if !std::path::Path::new(&model_path).exists() || !std::path::Path::new(&tok_path).exists()
        {
            eprintln!("SKIP: CLIP text model not found");
            return;
        }
        let pipeline = ClipTextPipeline::new(&model_path, &tok_path, 4);
        assert!(
            pipeline.is_ok(),
            "Failed to load CLIP text model: {:?}",
            pipeline.err()
        );
    }

    #[test]
    fn test_clip_cross_modal_similarity() {
        let vis_path = clip_visual_model_path();
        let txt_path = clip_text_model_path();
        let tok_path = clip_tokenizer_path();
        if !std::path::Path::new(&vis_path).exists()
            || !std::path::Path::new(&txt_path).exists()
            || !std::path::Path::new(&tok_path).exists()
        {
            eprintln!("SKIP: CLIP models not found");
            return;
        }

        let image_pipeline = ClipImagePipeline::new(&vis_path, ClipConfig::default()).unwrap();
        let text_pipeline = ClipTextPipeline::new(&txt_path, &tok_path, 4).unwrap();

        // Create a red image
        let mut imgbuf = image::RgbImage::new(100, 100);
        for pixel in imgbuf.pixels_mut() {
            *pixel = image::Rgb([255, 0, 0]);
        }
        let img = image::DynamicImage::ImageRgb8(imgbuf);
        let img_emb = image_pipeline.embed_image(&img).unwrap();

        // Text query
        let text_emb = text_pipeline.embed_text("a red image").unwrap();

        assert_eq!(img_emb.len(), CLIP_EMBED_DIM);
        assert_eq!(text_emb.len(), CLIP_EMBED_DIM);

        let sim = cosine_similarity(&img_emb, &text_emb);
        // Cross-modal similarity should be non-zero
        assert!(sim.abs() > 0.0, "Cross-modal similarity should be non-zero");
    }
}
