//! # Ucotron Extraction
//!
//! Extraction pipeline for the Ucotron cognitive trust framework.
//!
//! This crate provides:
//! - **Embedding generation** via ONNX Runtime (all-MiniLM-L6-v2) — [`embeddings::OnnxEmbeddingPipeline`]
//! - **Image embedding** via CLIP ONNX (ViT-B/32) — [`image::ClipImagePipeline`] + [`image::ClipTextPipeline`]
//! - **Cross-modal projection** via ONNX MLP (CLIP→MiniLM) — [`cross_modal::ProjectionLayerPipeline`]
//! - **Audio transcription** via Whisper ONNX (voice-to-text) — [`audio::WhisperOnnxPipeline`]
//! - **Video frame extraction** via FFmpeg — [`video::FfmpegVideoPipeline`]
//! - **Document OCR** via pdf\_extract + Tesseract CLI — [`ocr`] module
//! - **Cross-modal search** orchestrator for unified multi-modal queries — [`cross_modal_search::CrossModalSearch`]
//! - **Fine-tuning dataset generation** from knowledge graph — [`fine_tuning`] module
//! - Named Entity Recognition via GLiNER (ONNX) — *US-7.2*
//! - Relation Extraction via Qwen3 (candle or llama.cpp) — *US-7.3*
//! - Ingestion orchestration (text → graph) — *US-7.4*
//! - Retrieval orchestration (query → context) — *US-7.5*
//! - Consolidation worker (background "dreaming") — *US-7.6*
//!
//! # Test Infrastructure
//!
//! Tests are split into two categories:
//!
//! **Mock-based tests (no models required, CI-safe):**
//! - `relations::tests` (32 tests) — co-occurrence, JSON parsing, prompt building
//! - `ingestion::tests` (19 tests) — chunking, pipeline with mock backends
//! - `retrieval::tests` (20 tests) — scoring, filtering, context assembly with mocks
//! - `consolidation::tests` (23 tests) — community detection, entity merge, decay with mocks
//! - `embeddings::tests` (6 tests) — L2 normalization, trait safety
//! - `ner::tests` (11 tests) — sigmoid, dedup, word mapping helpers
//! - `audio::tests` (20 tests) — mel filterbank, WAV loading, resampling, token decoding
//! - `image::tests` (15 tests) — preprocessing, normalization, L2 normalize, cosine similarity
//! - `cross_modal::tests` (11 tests) — projection config, mock projection, dimension validation, trait safety
//! - `cross_modal_search::tests` (41 tests) — all 6 query types, fusion, error handling, filtering, metrics, text-to-image convenience, image-to-text convenience, standalone functions
//! - `fine_tuning::tests` (24 tests) — dataset generation, SFT formatting, JSONL I/O, helpers
//! - `ocr::tests` (20 tests) — text cleaning, page splitting, chunking, config, format handling
//!
//! **ONNX model-dependent tests (skip gracefully if models absent):**
//! - `embeddings::tests` (10 tests) — real OnnxEmbeddingPipeline tests
//!   - Requires: `models/all-MiniLM-L6-v2/model.onnx` + `tokenizer.json`
//! - `ner::tests` (7 tests) — real GlinerNerPipeline tests
//!   - Requires: `models/gliner_small-v2.1/onnx/model.onnx` + `tokenizer.json`
//! - `audio::tests` (3 tests) — real WhisperOnnxPipeline tests
//!   - Requires: `models/whisper-tiny/encoder.onnx` + `decoder.onnx` + `tokens.txt`
//! - `image::tests` (4 tests) — real ClipImagePipeline + ClipTextPipeline tests
//!   - Requires: `models/clip-vit-base-patch32/visual_model.onnx` + `text_model.onnx` + `tokenizer.json`
//!
//! **System-dependent tests (skip if Tesseract not installed):**
//! - `ocr::tests` — Tesseract OCR image processing tests
//!   - Requires: `tesseract` binary on PATH
//!
//! Download models: `scripts/download_models.sh`

pub mod audio;
pub mod consolidation;
pub mod cross_modal;
pub mod cross_modal_search;
pub mod embeddings;
pub mod fine_tuning;
pub mod image;
pub mod ingestion;
pub mod ner;
pub mod ocr;
pub mod relations;
pub mod reranker;
pub mod retrieval;
pub mod video;

/// Embedding pipeline trait for generating vector representations of text.
///
/// Implementors produce 384-dimensional normalized vectors compatible with
/// the sentence-transformers ecosystem.
pub trait EmbeddingPipeline: Send + Sync {
    /// Generate an embedding for a single text.
    fn embed_text(&self, text: &str) -> anyhow::Result<Vec<f32>>;

    /// Generate embeddings for a batch of texts.
    fn embed_batch(&self, texts: &[&str]) -> anyhow::Result<Vec<Vec<f32>>>;
}

/// Named entity extracted from text.
#[derive(Debug, Clone)]
pub struct ExtractedEntity {
    /// The entity text as it appears in the source.
    pub text: String,
    /// Entity label (e.g., "persona", "lugar", "organización").
    pub label: String,
    /// Start character offset in the source text.
    pub start: usize,
    /// End character offset in the source text.
    pub end: usize,
    /// Extraction confidence score (0.0-1.0).
    pub confidence: f32,
}

/// NER pipeline trait for extracting named entities from text.
pub trait NerPipeline: Send + Sync {
    /// Extract entities from text with the given label set.
    fn extract_entities(&self, text: &str, labels: &[&str])
        -> anyhow::Result<Vec<ExtractedEntity>>;

    /// Extract entities from a batch of texts with the given label set.
    ///
    /// Default implementation calls `extract_entities` per text. Implementations
    /// may override this for true batched inference (single model call).
    fn extract_entities_batch(
        &self,
        texts: &[&str],
        labels: &[&str],
    ) -> anyhow::Result<Vec<Vec<ExtractedEntity>>> {
        texts
            .iter()
            .map(|text| self.extract_entities(text, labels))
            .collect()
    }
}

/// A relation extracted between two entities.
#[derive(Debug, Clone)]
pub struct ExtractedRelation {
    /// Subject entity text.
    pub subject: String,
    /// Predicate (relationship type).
    pub predicate: String,
    /// Object entity text.
    pub object: String,
    /// Extraction confidence score (0.0-1.0).
    pub confidence: f32,
}

/// Relation extraction pipeline trait.
pub trait RelationExtractor: Send + Sync {
    /// Extract relations from text given pre-extracted entities.
    fn extract_relations(
        &self,
        text: &str,
        entities: &[ExtractedEntity],
    ) -> anyhow::Result<Vec<ExtractedRelation>>;
}

/// Image embedding pipeline trait for generating vector representations of images.
///
/// Implementors produce 512-dimensional normalized vectors compatible with
/// CLIP's joint vision-language embedding space, enabling cross-modal search.
pub trait ImageEmbeddingPipeline: Send + Sync {
    /// Generate an embedding for a single image from raw bytes (JPEG, PNG, etc.).
    fn embed_image_bytes(&self, bytes: &[u8]) -> anyhow::Result<Vec<f32>>;

    /// Generate an embedding for a single image from a file path.
    fn embed_image_file(&self, path: &std::path::Path) -> anyhow::Result<Vec<f32>>;
}

/// Cross-modal text encoder trait for encoding text queries into the image embedding space.
///
/// Used in combination with [`ImageEmbeddingPipeline`] to enable text-to-image search.
pub trait CrossModalTextEncoder: Send + Sync {
    /// Encode a text query into the same embedding space as images.
    fn embed_text(&self, text: &str) -> anyhow::Result<Vec<f32>>;
}

/// Audio transcription pipeline trait for converting speech to text.
///
/// Implementors take audio input (file or raw samples) and produce
/// transcribed text that can be fed into the ingestion pipeline.
pub trait TranscriptionPipeline: Send + Sync {
    /// Transcribe audio from a WAV file.
    fn transcribe_file(&self, path: &std::path::Path)
        -> anyhow::Result<audio::TranscriptionResult>;

    /// Transcribe audio from raw f32 samples at a given sample rate.
    fn transcribe_samples(
        &self,
        samples: &[f32],
        sample_rate: u32,
    ) -> anyhow::Result<audio::TranscriptionResult>;
}

/// Document OCR pipeline trait for extracting text from PDFs and scanned documents.
///
/// Implementors handle both structured PDFs (text extraction) and scanned
/// documents/images (OCR via Tesseract or ONNX models).
pub trait DocumentOcrPipeline: Send + Sync {
    /// Process a document from raw bytes with a filename hint.
    ///
    /// The filename is used to detect the document format (pdf, jpg, png, etc.).
    fn process_document(
        &self,
        data: &[u8],
        filename: &str,
    ) -> anyhow::Result<ocr::DocumentExtractionResult>;

    /// Process a document from a file path.
    fn process_file(&self, path: &std::path::Path)
        -> anyhow::Result<ocr::DocumentExtractionResult>;
}

/// Default implementation of [`DocumentOcrPipeline`] using pdf_extract + Tesseract CLI.
#[derive(Default)]
pub struct DefaultDocumentOcrPipeline {
    config: ocr::OcrConfig,
}

impl DefaultDocumentOcrPipeline {
    /// Create a new pipeline with default configuration.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a new pipeline with custom configuration.
    pub fn with_config(config: ocr::OcrConfig) -> Self {
        Self { config }
    }
}

impl DocumentOcrPipeline for DefaultDocumentOcrPipeline {
    fn process_document(
        &self,
        data: &[u8],
        filename: &str,
    ) -> anyhow::Result<ocr::DocumentExtractionResult> {
        ocr::process_document(data, filename, &self.config)
    }

    fn process_file(
        &self,
        path: &std::path::Path,
    ) -> anyhow::Result<ocr::DocumentExtractionResult> {
        let filename = path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("document.pdf");
        let data = std::fs::read(path)?;
        ocr::process_document(&data, filename, &self.config)
    }
}

/// Video frame extraction pipeline trait for extracting keyframes from video files.
///
/// Implementors decode video streams and extract frames at configurable rates,
/// with scene change detection for identifying keyframes.
pub trait VideoPipeline: Send + Sync {
    /// Extract frames from a video file at the configured FPS rate.
    ///
    /// Returns extracted frames with timestamps and optional scene change scores.
    fn extract_frames(
        &self,
        path: &std::path::Path,
    ) -> anyhow::Result<video::FrameExtractionResult>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extracted_entity_creation() {
        let entity = ExtractedEntity {
            text: "Madrid".to_string(),
            label: "lugar".to_string(),
            start: 10,
            end: 16,
            confidence: 0.95,
        };
        assert_eq!(entity.text, "Madrid");
        assert_eq!(entity.label, "lugar");
        assert_eq!(entity.start, 10);
        assert_eq!(entity.end, 16);
        assert!((entity.confidence - 0.95).abs() < f32::EPSILON);
    }

    #[test]
    fn test_extracted_relation_creation() {
        let relation = ExtractedRelation {
            subject: "Juan".to_string(),
            predicate: "lives_in".to_string(),
            object: "Berlin".to_string(),
            confidence: 0.88,
        };
        assert_eq!(relation.subject, "Juan");
        assert_eq!(relation.predicate, "lives_in");
        assert_eq!(relation.object, "Berlin");
    }
}
