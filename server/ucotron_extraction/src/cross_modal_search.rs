//! Cross-modal search orchestrator for unified multi-modal queries.
//!
//! Provides [`CrossModalSearch`] that dispatches queries to the correct
//! index (text 384-dim or visual 512-dim) based on the query modality,
//! using projection layers to bridge between embedding spaces.
//!
//! # Supported Query Types
//!
//! - **Text**: Standard text search in the 384-dim MiniLM index
//! - **TextToImage**: Text query → CLIP text encoder → search visual 512-dim index
//! - **Image**: Image bytes → CLIP image encoder → search visual 512-dim index
//! - **ImageToText**: Image bytes → CLIP image encoder → projection → search text 384-dim index
//! - **Audio**: Audio transcript text → search text 384-dim index
//! - **Video**: Dual search — visual frames in CLIP index + transcript in text index → fuse results

use std::collections::HashMap;
use std::time::Instant;

use anyhow::{Context, Result};
use ucotron_core::backends::BackendRegistry;
use ucotron_core::NodeId;

use crate::cross_modal::CrossModalProjection;
use crate::{CrossModalTextEncoder, EmbeddingPipeline, ImageEmbeddingPipeline};

/// A cross-modal query specifying the modality and search data.
#[derive(Debug)]
pub enum CrossModalQuery<'a> {
    /// Standard text search in the 384-dim MiniLM space.
    Text {
        /// The text query string.
        text: &'a str,
    },

    /// Text-to-image search: encode text with CLIP text encoder, search visual index.
    TextToImage {
        /// The text query describing the desired image.
        text: &'a str,
    },

    /// Image search: encode image with CLIP, search visual index for similar images.
    Image {
        /// Raw image bytes (JPEG, PNG, etc.).
        image_bytes: &'a [u8],
    },

    /// Image-to-text search: encode image with CLIP, project to MiniLM space, search text index.
    ImageToText {
        /// Raw image bytes (JPEG, PNG, etc.).
        image_bytes: &'a [u8],
    },

    /// Audio search: use pre-transcribed text to search the text index.
    Audio {
        /// Transcribed text from audio.
        transcript: &'a str,
    },

    /// Video search: dual search using visual frames + transcript, fuse results.
    Video {
        /// CLIP embeddings of extracted video frames (512-dim each).
        frame_embeddings: &'a [Vec<f32>],
        /// Transcribed text from the video's audio track.
        transcript: Option<&'a str>,
    },
}

/// Configuration for cross-modal search.
#[derive(Debug, Clone)]
pub struct CrossModalSearchConfig {
    /// Number of results to return from each sub-search.
    pub top_k: usize,
    /// Weight for text index results when fusing (0.0 - 1.0).
    pub text_weight: f32,
    /// Weight for visual index results when fusing (0.0 - 1.0).
    pub visual_weight: f32,
    /// Minimum similarity threshold for results.
    pub min_similarity: f32,
}

impl Default for CrossModalSearchConfig {
    fn default() -> Self {
        Self {
            top_k: 10,
            text_weight: 0.5,
            visual_weight: 0.5,
            min_similarity: 0.0,
        }
    }
}

/// A single search result with source information.
#[derive(Debug, Clone)]
pub struct CrossModalResult {
    /// The node ID of the result.
    pub node_id: NodeId,
    /// Overall similarity score (0.0 - 1.0).
    pub score: f32,
    /// Which index produced this result.
    pub source: ResultSource,
}

/// Indicates which index a result came from.
#[derive(Debug, Clone, PartialEq)]
pub enum ResultSource {
    /// Found in the 384-dim text (MiniLM) index.
    TextIndex,
    /// Found in the 512-dim visual (CLIP) index.
    VisualIndex,
    /// Found in both indices (fused score).
    Fused,
}

/// Timing metrics for a cross-modal search operation.
#[derive(Debug, Clone, Default)]
pub struct CrossModalSearchMetrics {
    /// Time spent encoding the query (embedding/projection), in microseconds.
    pub query_encoding_us: u64,
    /// Time spent searching the text index, in microseconds.
    pub text_search_us: u64,
    /// Time spent searching the visual index, in microseconds.
    pub visual_search_us: u64,
    /// Time spent fusing results, in microseconds.
    pub fusion_us: u64,
    /// Total search time, in microseconds.
    pub total_us: u64,
    /// Number of results from text index before fusion.
    pub text_result_count: usize,
    /// Number of results from visual index before fusion.
    pub visual_result_count: usize,
    /// Number of final results after fusion and filtering.
    pub final_result_count: usize,
}

/// Full search response including results and metrics.
#[derive(Debug)]
pub struct CrossModalSearchResponse {
    /// Ranked results sorted by descending score.
    pub results: Vec<CrossModalResult>,
    /// Timing and count metrics for the search.
    pub metrics: CrossModalSearchMetrics,
}

/// Cross-modal search orchestrator that dispatches queries to the correct index.
///
/// Holds references to the backend registry (text + visual indices), embedding
/// pipelines, and an optional projection layer for bridging CLIP↔MiniLM spaces.
pub struct CrossModalSearch<'a> {
    registry: &'a BackendRegistry,
    text_embedder: &'a dyn EmbeddingPipeline,
    image_embedder: Option<&'a dyn ImageEmbeddingPipeline>,
    clip_text_encoder: Option<&'a dyn CrossModalTextEncoder>,
    projection: Option<&'a dyn CrossModalProjection>,
    config: CrossModalSearchConfig,
}

impl<'a> CrossModalSearch<'a> {
    /// Create a new cross-modal search orchestrator.
    ///
    /// # Arguments
    /// - `registry` — Backend registry with text vector + optional visual vector backends
    /// - `text_embedder` — MiniLM 384-dim text embedding pipeline
    /// - `config` — Search configuration
    pub fn new(
        registry: &'a BackendRegistry,
        text_embedder: &'a dyn EmbeddingPipeline,
        config: CrossModalSearchConfig,
    ) -> Self {
        Self {
            registry,
            text_embedder,
            image_embedder: None,
            clip_text_encoder: None,
            projection: None,
            config,
        }
    }

    /// Set the CLIP image embedding pipeline (required for Image and ImageToText queries).
    pub fn with_image_embedder(mut self, embedder: &'a dyn ImageEmbeddingPipeline) -> Self {
        self.image_embedder = Some(embedder);
        self
    }

    /// Set the CLIP text encoder (required for TextToImage queries).
    pub fn with_clip_text_encoder(mut self, encoder: &'a dyn CrossModalTextEncoder) -> Self {
        self.clip_text_encoder = Some(encoder);
        self
    }

    /// Set the projection layer (required for ImageToText queries).
    pub fn with_projection(mut self, projection: &'a dyn CrossModalProjection) -> Self {
        self.projection = Some(projection);
        self
    }

    /// Execute a cross-modal search query.
    ///
    /// Dispatches to the correct index based on the query type:
    /// - `Text` / `Audio` → text index (384-dim MiniLM)
    /// - `TextToImage` → visual index (512-dim CLIP) via CLIP text encoder
    /// - `Image` → visual index (512-dim CLIP) via CLIP image encoder
    /// - `ImageToText` → text index (384-dim MiniLM) via CLIP image encoder + projection
    /// - `Video` → dual search (visual + text) with result fusion
    pub fn search(&self, query: &CrossModalQuery) -> Result<CrossModalSearchResponse> {
        let total_start = Instant::now();
        let mut metrics = CrossModalSearchMetrics::default();

        let results = match query {
            CrossModalQuery::Text { text } => self.search_text(text, &mut metrics)?,
            CrossModalQuery::TextToImage { text } => {
                self.search_text_to_image(text, &mut metrics)?
            }
            CrossModalQuery::Image { image_bytes } => {
                self.search_image(image_bytes, &mut metrics)?
            }
            CrossModalQuery::ImageToText { image_bytes } => {
                self.search_image_to_text(image_bytes, &mut metrics)?
            }
            CrossModalQuery::Audio { transcript } => self.search_audio(transcript, &mut metrics)?,
            CrossModalQuery::Video {
                frame_embeddings,
                transcript,
            } => self.search_video(frame_embeddings, *transcript, &mut metrics)?,
        };

        metrics.final_result_count = results.len();
        metrics.total_us = total_start.elapsed().as_micros() as u64;

        Ok(CrossModalSearchResponse { results, metrics })
    }

    /// Text query → embed with MiniLM → search text index.
    fn search_text(
        &self,
        text: &str,
        metrics: &mut CrossModalSearchMetrics,
    ) -> Result<Vec<CrossModalResult>> {
        let encode_start = Instant::now();
        let query_vec = self
            .text_embedder
            .embed_text(text)
            .context("Failed to embed text query")?;
        metrics.query_encoding_us = encode_start.elapsed().as_micros() as u64;

        let search_start = Instant::now();
        let results = self
            .registry
            .vector()
            .search(&query_vec, self.config.top_k)
            .context("Text index search failed")?;
        metrics.text_search_us = search_start.elapsed().as_micros() as u64;
        metrics.text_result_count = results.len();

        Ok(self.filter_results(
            results
                .into_iter()
                .map(|(id, score)| CrossModalResult {
                    node_id: id,
                    score,
                    source: ResultSource::TextIndex,
                })
                .collect(),
        ))
    }

    /// Text query → encode with CLIP text encoder → search visual index.
    fn search_text_to_image(
        &self,
        text: &str,
        metrics: &mut CrossModalSearchMetrics,
    ) -> Result<Vec<CrossModalResult>> {
        let encoder = self
            .clip_text_encoder
            .context("CLIP text encoder required for TextToImage queries")?;

        let visual_backend = self
            .registry
            .visual()
            .context("Visual vector backend required for TextToImage queries")?;

        let encode_start = Instant::now();
        let clip_vec = encoder
            .embed_text(text)
            .context("Failed to encode text with CLIP")?;
        metrics.query_encoding_us = encode_start.elapsed().as_micros() as u64;

        let search_start = Instant::now();
        let results = visual_backend
            .search_visual(&clip_vec, self.config.top_k)
            .context("Visual index search failed")?;
        metrics.visual_search_us = search_start.elapsed().as_micros() as u64;
        metrics.visual_result_count = results.len();

        Ok(self.filter_results(
            results
                .into_iter()
                .map(|(id, score)| CrossModalResult {
                    node_id: id,
                    score,
                    source: ResultSource::VisualIndex,
                })
                .collect(),
        ))
    }

    /// Image bytes → encode with CLIP image encoder → search visual index.
    fn search_image(
        &self,
        image_bytes: &[u8],
        metrics: &mut CrossModalSearchMetrics,
    ) -> Result<Vec<CrossModalResult>> {
        let embedder = self
            .image_embedder
            .context("CLIP image embedder required for Image queries")?;

        let visual_backend = self
            .registry
            .visual()
            .context("Visual vector backend required for Image queries")?;

        let encode_start = Instant::now();
        let clip_vec = embedder
            .embed_image_bytes(image_bytes)
            .context("Failed to embed image with CLIP")?;
        metrics.query_encoding_us = encode_start.elapsed().as_micros() as u64;

        let search_start = Instant::now();
        let results = visual_backend
            .search_visual(&clip_vec, self.config.top_k)
            .context("Visual index search failed")?;
        metrics.visual_search_us = search_start.elapsed().as_micros() as u64;
        metrics.visual_result_count = results.len();

        Ok(self.filter_results(
            results
                .into_iter()
                .map(|(id, score)| CrossModalResult {
                    node_id: id,
                    score,
                    source: ResultSource::VisualIndex,
                })
                .collect(),
        ))
    }

    /// Image bytes → CLIP encoder → projection (512→384) → search text index.
    fn search_image_to_text(
        &self,
        image_bytes: &[u8],
        metrics: &mut CrossModalSearchMetrics,
    ) -> Result<Vec<CrossModalResult>> {
        let embedder = self
            .image_embedder
            .context("CLIP image embedder required for ImageToText queries")?;

        let projection = self
            .projection
            .context("Projection layer required for ImageToText queries")?;

        let encode_start = Instant::now();
        let clip_vec = embedder
            .embed_image_bytes(image_bytes)
            .context("Failed to embed image with CLIP")?;
        let text_vec = projection
            .project(&clip_vec)
            .context("Failed to project CLIP embedding to text space")?;
        metrics.query_encoding_us = encode_start.elapsed().as_micros() as u64;

        let search_start = Instant::now();
        let results = self
            .registry
            .vector()
            .search(&text_vec, self.config.top_k)
            .context("Text index search failed")?;
        metrics.text_search_us = search_start.elapsed().as_micros() as u64;
        metrics.text_result_count = results.len();

        Ok(self.filter_results(
            results
                .into_iter()
                .map(|(id, score)| CrossModalResult {
                    node_id: id,
                    score,
                    source: ResultSource::TextIndex,
                })
                .collect(),
        ))
    }

    /// Audio transcript → embed with MiniLM → search text index (same as Text).
    fn search_audio(
        &self,
        transcript: &str,
        metrics: &mut CrossModalSearchMetrics,
    ) -> Result<Vec<CrossModalResult>> {
        // Audio search is equivalent to text search on the transcript
        self.search_text(transcript, metrics)
    }

    /// Video: dual search on visual frames + transcript text, fuse results.
    fn search_video(
        &self,
        frame_embeddings: &[Vec<f32>],
        transcript: Option<&str>,
        metrics: &mut CrossModalSearchMetrics,
    ) -> Result<Vec<CrossModalResult>> {
        let mut visual_results: Vec<(NodeId, f32)> = Vec::new();
        let mut text_results: Vec<(NodeId, f32)> = Vec::new();

        // Visual search: search each frame embedding in the visual index, aggregate
        if !frame_embeddings.is_empty() {
            if let Some(visual_backend) = self.registry.visual() {
                let search_start = Instant::now();

                let mut frame_scores: HashMap<NodeId, f32> = HashMap::new();
                for frame_emb in frame_embeddings {
                    let results = visual_backend
                        .search_visual(frame_emb, self.config.top_k)
                        .context("Visual index search failed for video frame")?;
                    for (id, score) in results {
                        let entry = frame_scores.entry(id).or_insert(0.0);
                        if score > *entry {
                            *entry = score;
                        }
                    }
                }

                visual_results = frame_scores.into_iter().collect();
                visual_results
                    .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                visual_results.truncate(self.config.top_k);

                metrics.visual_search_us = search_start.elapsed().as_micros() as u64;
                metrics.visual_result_count = visual_results.len();
            }
        }

        // Text search: embed transcript and search text index
        if let Some(transcript_text) = transcript {
            if !transcript_text.is_empty() {
                let encode_start = Instant::now();
                let query_vec = self
                    .text_embedder
                    .embed_text(transcript_text)
                    .context("Failed to embed video transcript")?;
                metrics.query_encoding_us = encode_start.elapsed().as_micros() as u64;

                let search_start = Instant::now();
                text_results = self
                    .registry
                    .vector()
                    .search(&query_vec, self.config.top_k)
                    .context("Text index search failed for video transcript")?;
                metrics.text_search_us = search_start.elapsed().as_micros() as u64;
                metrics.text_result_count = text_results.len();
            }
        }

        // Fuse results from both indices
        let fusion_start = Instant::now();
        let fused = self.fuse_results(&text_results, &visual_results);
        metrics.fusion_us = fusion_start.elapsed().as_micros() as u64;

        Ok(self.filter_results(fused))
    }

    /// Fuse results from text and visual indices using weighted score combination.
    fn fuse_results(
        &self,
        text_results: &[(NodeId, f32)],
        visual_results: &[(NodeId, f32)],
    ) -> Vec<CrossModalResult> {
        let mut scores: HashMap<NodeId, (f32, f32)> = HashMap::new();

        // Text results
        for &(id, score) in text_results {
            scores.entry(id).or_insert((0.0, 0.0)).0 = score;
        }

        // Visual results
        for &(id, score) in visual_results {
            scores.entry(id).or_insert((0.0, 0.0)).1 = score;
        }

        let mut results: Vec<CrossModalResult> = scores
            .into_iter()
            .map(|(id, (text_score, visual_score))| {
                let has_text = text_score > 0.0;
                let has_visual = visual_score > 0.0;

                let combined =
                    text_score * self.config.text_weight + visual_score * self.config.visual_weight;

                let source = if has_text && has_visual {
                    ResultSource::Fused
                } else if has_text {
                    ResultSource::TextIndex
                } else {
                    ResultSource::VisualIndex
                };

                CrossModalResult {
                    node_id: id,
                    score: combined,
                    source,
                }
            })
            .collect();

        results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        results.truncate(self.config.top_k);
        results
    }

    /// Filter results by minimum similarity threshold.
    fn filter_results(&self, results: Vec<CrossModalResult>) -> Vec<CrossModalResult> {
        if self.config.min_similarity <= 0.0 {
            return results;
        }
        results
            .into_iter()
            .filter(|r| r.score >= self.config.min_similarity)
            .collect()
    }

    /// Convenience method: perform a text-to-image search in one call.
    ///
    /// Encodes the text query using the CLIP text encoder and searches the
    /// 512-dim visual index for image nodes with the highest cosine similarity.
    ///
    /// This is equivalent to calling `search(&CrossModalQuery::TextToImage { text })`,
    /// but provides a simpler API for the most common cross-modal query pattern.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - No CLIP text encoder has been set (see [`with_clip_text_encoder`])
    /// - No visual vector backend is present in the registry
    /// - The encoding or search fails
    pub fn text_to_image(&self, query: &str) -> Result<CrossModalSearchResponse> {
        self.search(&CrossModalQuery::TextToImage { text: query })
    }

    /// Convenience method: perform a text-to-image search with custom top_k.
    ///
    /// Temporarily overrides the configured `top_k` for this single query,
    /// then restores it. Useful when the caller needs a different result count
    /// than the default.
    pub fn text_to_image_top_k(
        &self,
        query: &str,
        top_k: usize,
    ) -> Result<CrossModalSearchResponse> {
        let total_start = std::time::Instant::now();
        let mut metrics = CrossModalSearchMetrics::default();

        let encoder = self
            .clip_text_encoder
            .context("CLIP text encoder required for TextToImage queries")?;

        let visual_backend = self
            .registry
            .visual()
            .context("Visual vector backend required for TextToImage queries")?;

        let encode_start = std::time::Instant::now();
        let clip_vec = encoder
            .embed_text(query)
            .context("Failed to encode text with CLIP")?;
        metrics.query_encoding_us = encode_start.elapsed().as_micros() as u64;

        let search_start = std::time::Instant::now();
        let results = visual_backend
            .search_visual(&clip_vec, top_k)
            .context("Visual index search failed")?;
        metrics.visual_search_us = search_start.elapsed().as_micros() as u64;
        metrics.visual_result_count = results.len();

        let filtered = self.filter_results(
            results
                .into_iter()
                .map(|(id, score)| CrossModalResult {
                    node_id: id,
                    score,
                    source: ResultSource::VisualIndex,
                })
                .collect(),
        );

        metrics.final_result_count = filtered.len();
        metrics.total_us = total_start.elapsed().as_micros() as u64;

        Ok(CrossModalSearchResponse {
            results: filtered,
            metrics,
        })
    }

    /// Convenience method: perform an image-to-text search in one call.
    ///
    /// Encodes the image using the CLIP image encoder, projects the 512-dim CLIP
    /// embedding to the 384-dim MiniLM space via the projection layer, then searches
    /// the text index for the most semantically similar text memories.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - No CLIP image embedder has been set (see [`with_image_embedder`])
    /// - No projection layer has been set (see [`with_projection`])
    /// - The encoding, projection, or search fails
    pub fn image_to_text(&self, image_bytes: &[u8]) -> Result<CrossModalSearchResponse> {
        self.search(&CrossModalQuery::ImageToText { image_bytes })
    }

    /// Convenience method: perform an image-to-text search with custom top_k.
    ///
    /// Temporarily overrides the configured `top_k` for this single query.
    pub fn image_to_text_top_k(
        &self,
        image_bytes: &[u8],
        top_k: usize,
    ) -> Result<CrossModalSearchResponse> {
        let total_start = std::time::Instant::now();
        let mut metrics = CrossModalSearchMetrics::default();

        let embedder = self
            .image_embedder
            .context("CLIP image embedder required for ImageToText queries")?;

        let projection = self
            .projection
            .context("Projection layer required for ImageToText queries")?;

        let encode_start = std::time::Instant::now();
        let clip_vec = embedder
            .embed_image_bytes(image_bytes)
            .context("Failed to embed image with CLIP")?;
        let text_vec = projection
            .project(&clip_vec)
            .context("Failed to project CLIP embedding to text space")?;
        metrics.query_encoding_us = encode_start.elapsed().as_micros() as u64;

        let search_start = std::time::Instant::now();
        let results = self
            .registry
            .vector()
            .search(&text_vec, top_k)
            .context("Text index search failed")?;
        metrics.text_search_us = search_start.elapsed().as_micros() as u64;
        metrics.text_result_count = results.len();

        let filtered = self.filter_results(
            results
                .into_iter()
                .map(|(id, score)| CrossModalResult {
                    node_id: id,
                    score,
                    source: ResultSource::TextIndex,
                })
                .collect(),
        );

        metrics.final_result_count = filtered.len();
        metrics.total_us = total_start.elapsed().as_micros() as u64;

        Ok(CrossModalSearchResponse {
            results: filtered,
            metrics,
        })
    }
}

/// Perform a standalone text-to-image search without constructing a full orchestrator.
///
/// This is a convenience function for the common case where you only need
/// text-to-image search and have the required components available.
///
/// # Arguments
/// - `registry` — Backend registry with a visual vector backend
/// - `clip_text_encoder` — CLIP text encoder for embedding text queries
/// - `text_embedder` — MiniLM text embedder (required by `CrossModalSearch`, but not
///   used for text-to-image queries)
/// - `query` — The text query describing the desired image
/// - `top_k` — Maximum number of results
/// - `min_similarity` — Minimum similarity threshold (0.0 to disable)
///
/// # Returns
/// A list of `(node_id, similarity_score)` pairs sorted by descending similarity.
pub fn text_to_image_search(
    registry: &BackendRegistry,
    clip_text_encoder: &dyn CrossModalTextEncoder,
    text_embedder: &dyn EmbeddingPipeline,
    query: &str,
    top_k: usize,
    min_similarity: f32,
) -> Result<Vec<(NodeId, f32)>> {
    let searcher = CrossModalSearch::new(
        registry,
        text_embedder,
        CrossModalSearchConfig {
            top_k,
            min_similarity,
            ..Default::default()
        },
    )
    .with_clip_text_encoder(clip_text_encoder);

    let response = searcher.text_to_image(query)?;

    Ok(response
        .results
        .into_iter()
        .map(|r| (r.node_id, r.score))
        .collect())
}

/// Perform a standalone image-to-text search without constructing a full orchestrator.
///
/// Encodes the image with the CLIP image embedder, projects the resulting 512-dim
/// CLIP embedding into the 384-dim MiniLM text space via the projection layer,
/// then searches the text index for the most semantically similar text memories.
///
/// # Arguments
/// - `registry` — Backend registry with a text vector backend
/// - `image_embedder` — CLIP image encoder for embedding image bytes
/// - `projection` — Projection layer for bridging CLIP→MiniLM spaces
/// - `text_embedder` — MiniLM text embedder (required by `CrossModalSearch` constructor)
/// - `image_bytes` — Raw image bytes (JPEG, PNG, etc.)
/// - `top_k` — Maximum number of results
/// - `min_similarity` — Minimum similarity threshold (0.0 to disable)
///
/// # Returns
/// A list of `(node_id, similarity_score)` pairs sorted by descending similarity.
pub fn image_to_text_search(
    registry: &BackendRegistry,
    image_embedder: &dyn ImageEmbeddingPipeline,
    projection: &dyn CrossModalProjection,
    text_embedder: &dyn EmbeddingPipeline,
    image_bytes: &[u8],
    top_k: usize,
    min_similarity: f32,
) -> Result<Vec<(NodeId, f32)>> {
    let searcher = CrossModalSearch::new(
        registry,
        text_embedder,
        CrossModalSearchConfig {
            top_k,
            min_similarity,
            ..Default::default()
        },
    )
    .with_image_embedder(image_embedder)
    .with_projection(projection);

    let response = searcher.image_to_text(image_bytes)?;

    Ok(response
        .results
        .into_iter()
        .map(|r| (r.node_id, r.score))
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use anyhow::Result;
    use std::collections::HashMap;
    use std::sync::Mutex;
    use ucotron_core::backends::VisualVectorBackend;

    // --- Mock backends ---

    struct MockTextEmbedder {
        embeddings: HashMap<String, Vec<f32>>,
        default_dim: usize,
    }

    impl MockTextEmbedder {
        fn new() -> Self {
            Self {
                embeddings: HashMap::new(),
                default_dim: 384,
            }
        }

        fn with_embedding(mut self, text: &str, embedding: Vec<f32>) -> Self {
            self.embeddings.insert(text.to_string(), embedding);
            self
        }
    }

    impl EmbeddingPipeline for MockTextEmbedder {
        fn embed_text(&self, text: &str) -> Result<Vec<f32>> {
            Ok(self
                .embeddings
                .get(text)
                .cloned()
                .unwrap_or_else(|| vec![0.1; self.default_dim]))
        }

        fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
            texts.iter().map(|t| self.embed_text(t)).collect()
        }
    }

    struct MockImageEmbedder {
        embedding: Vec<f32>,
    }

    impl MockImageEmbedder {
        fn new(embedding: Vec<f32>) -> Self {
            Self { embedding }
        }
    }

    impl ImageEmbeddingPipeline for MockImageEmbedder {
        fn embed_image_bytes(&self, _bytes: &[u8]) -> Result<Vec<f32>> {
            Ok(self.embedding.clone())
        }

        fn embed_image_file(&self, _path: &std::path::Path) -> Result<Vec<f32>> {
            Ok(self.embedding.clone())
        }
    }

    struct MockClipTextEncoder {
        embedding: Vec<f32>,
    }

    impl MockClipTextEncoder {
        fn new(embedding: Vec<f32>) -> Self {
            Self { embedding }
        }
    }

    impl CrossModalTextEncoder for MockClipTextEncoder {
        fn embed_text(&self, _text: &str) -> Result<Vec<f32>> {
            Ok(self.embedding.clone())
        }
    }

    struct MockProjection {
        output: Vec<f32>,
    }

    impl MockProjection {
        fn new(output: Vec<f32>) -> Self {
            Self { output }
        }
    }

    impl CrossModalProjection for MockProjection {
        fn project(&self, _embedding: &[f32]) -> Result<Vec<f32>> {
            Ok(self.output.clone())
        }

        fn project_batch(&self, embeddings: &[Vec<f32>]) -> Result<Vec<Vec<f32>>> {
            Ok(vec![self.output.clone(); embeddings.len()])
        }

        fn input_dim(&self) -> usize {
            512
        }

        fn output_dim(&self) -> usize {
            384
        }
    }

    struct MockVectorBackend {
        data: Mutex<HashMap<NodeId, Vec<f32>>>,
    }

    impl MockVectorBackend {
        fn with_data(data: Vec<(NodeId, Vec<f32>)>) -> Self {
            let map: HashMap<_, _> = data.into_iter().collect();
            Self {
                data: Mutex::new(map),
            }
        }
    }

    impl ucotron_core::backends::VectorBackend for MockVectorBackend {
        fn upsert_embeddings(&self, items: &[(NodeId, Vec<f32>)]) -> Result<()> {
            let mut map = self.data.lock().unwrap();
            for (id, vec) in items {
                map.insert(*id, vec.clone());
            }
            Ok(())
        }

        fn search(&self, query: &[f32], top_k: usize) -> Result<Vec<(NodeId, f32)>> {
            let map = self.data.lock().unwrap();
            let mut results: Vec<(NodeId, f32)> = map
                .iter()
                .map(|(&id, vec)| {
                    let sim: f32 = query.iter().zip(vec.iter()).map(|(a, b)| a * b).sum();
                    (id, sim)
                })
                .collect();
            results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            results.truncate(top_k);
            Ok(results)
        }

        fn delete(&self, ids: &[NodeId]) -> Result<()> {
            let mut map = self.data.lock().unwrap();
            for id in ids {
                map.remove(id);
            }
            Ok(())
        }
    }

    struct MockVisualBackend {
        data: Mutex<HashMap<NodeId, Vec<f32>>>,
    }

    impl MockVisualBackend {
        fn with_data(data: Vec<(NodeId, Vec<f32>)>) -> Self {
            let map: HashMap<_, _> = data.into_iter().collect();
            Self {
                data: Mutex::new(map),
            }
        }
    }

    impl VisualVectorBackend for MockVisualBackend {
        fn upsert_visual_embeddings(&self, items: &[(NodeId, Vec<f32>)]) -> Result<()> {
            let mut map = self.data.lock().unwrap();
            for (id, vec) in items {
                map.insert(*id, vec.clone());
            }
            Ok(())
        }

        fn search_visual(&self, query: &[f32], top_k: usize) -> Result<Vec<(NodeId, f32)>> {
            let map = self.data.lock().unwrap();
            let mut results: Vec<(NodeId, f32)> = map
                .iter()
                .map(|(&id, vec)| {
                    let sim: f32 = query.iter().zip(vec.iter()).map(|(a, b)| a * b).sum();
                    (id, sim)
                })
                .collect();
            results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            results.truncate(top_k);
            Ok(results)
        }

        fn delete_visual(&self, ids: &[NodeId]) -> Result<()> {
            let mut map = self.data.lock().unwrap();
            for id in ids {
                map.remove(id);
            }
            Ok(())
        }
    }

    // Minimal GraphBackend mock
    struct MockGraph;

    impl ucotron_core::backends::GraphBackend for MockGraph {
        fn upsert_nodes(&self, _nodes: &[ucotron_core::Node]) -> Result<()> {
            Ok(())
        }
        fn upsert_edges(&self, _edges: &[ucotron_core::Edge]) -> Result<()> {
            Ok(())
        }
        fn get_node(&self, _id: NodeId) -> Result<Option<ucotron_core::Node>> {
            Ok(None)
        }
        fn get_neighbors(&self, _id: NodeId, _hops: u8) -> Result<Vec<ucotron_core::Node>> {
            Ok(vec![])
        }
        fn find_path(&self, _src: NodeId, _tgt: NodeId, _max: u32) -> Result<Option<Vec<NodeId>>> {
            Ok(None)
        }
        fn get_community(&self, _id: NodeId) -> Result<Vec<NodeId>> {
            Ok(vec![])
        }
        fn delete_nodes(&self, _ids: &[NodeId]) -> Result<()> {
            Ok(())
        }
        fn get_all_nodes(&self) -> Result<Vec<ucotron_core::Node>> {
            Ok(vec![])
        }
        fn get_all_edges(&self) -> Result<Vec<(NodeId, NodeId, f32)>> {
            Ok(vec![])
        }
        fn get_all_edges_full(&self) -> Result<Vec<ucotron_core::Edge>> {
            Ok(vec![])
        }
        fn store_community_assignments(
            &self,
            _assignments: &std::collections::HashMap<
                ucotron_core::NodeId,
                ucotron_core::community::CommunityId,
            >,
        ) -> Result<()> {
            Ok(())
        }
    }

    // --- Helper functions ---

    fn normalized_vec(dim: usize, dominant_idx: usize) -> Vec<f32> {
        let mut v = vec![0.0f32; dim];
        v[dominant_idx] = 1.0;
        v
    }

    fn build_registry_text_only(text_data: Vec<(NodeId, Vec<f32>)>) -> BackendRegistry {
        BackendRegistry::new(
            Box::new(MockVectorBackend::with_data(text_data)),
            Box::new(MockGraph),
        )
    }

    fn build_registry_dual(
        text_data: Vec<(NodeId, Vec<f32>)>,
        visual_data: Vec<(NodeId, Vec<f32>)>,
    ) -> BackendRegistry {
        BackendRegistry::with_visual(
            Box::new(MockVectorBackend::with_data(text_data)),
            Box::new(MockGraph),
            Box::new(MockVisualBackend::with_data(visual_data)),
        )
    }

    // --- Tests ---

    #[test]
    fn test_text_search_returns_results() {
        let registry = build_registry_text_only(vec![
            (1, normalized_vec(384, 0)),
            (2, normalized_vec(384, 1)),
            (3, normalized_vec(384, 2)),
        ]);

        let embedder =
            MockTextEmbedder::new().with_embedding("hello world", normalized_vec(384, 0));

        let searcher = CrossModalSearch::new(
            &registry,
            &embedder,
            CrossModalSearchConfig {
                top_k: 2,
                ..Default::default()
            },
        );

        let response = searcher
            .search(&CrossModalQuery::Text {
                text: "hello world",
            })
            .unwrap();

        assert_eq!(response.results.len(), 2);
        assert_eq!(response.results[0].node_id, 1); // Best match
        assert_eq!(response.results[0].source, ResultSource::TextIndex);
        assert!(response.metrics.query_encoding_us > 0 || response.metrics.text_search_us > 0);
        assert_eq!(response.metrics.text_result_count, 2);
    }

    #[test]
    fn test_text_to_image_search() {
        let registry = build_registry_dual(
            vec![(1, normalized_vec(384, 0))],
            vec![(10, normalized_vec(512, 0)), (11, normalized_vec(512, 1))],
        );

        let embedder = MockTextEmbedder::new();
        let clip_encoder = MockClipTextEncoder::new(normalized_vec(512, 0));

        let searcher = CrossModalSearch::new(
            &registry,
            &embedder,
            CrossModalSearchConfig {
                top_k: 5,
                ..Default::default()
            },
        )
        .with_clip_text_encoder(&clip_encoder);

        let response = searcher
            .search(&CrossModalQuery::TextToImage { text: "a cute cat" })
            .unwrap();

        assert!(!response.results.is_empty());
        assert_eq!(response.results[0].node_id, 10); // Best visual match
        assert_eq!(response.results[0].source, ResultSource::VisualIndex);
        assert_eq!(response.metrics.visual_result_count, 2);
    }

    #[test]
    fn test_image_search_visual_index() {
        let registry = build_registry_dual(
            vec![],
            vec![(20, normalized_vec(512, 0)), (21, normalized_vec(512, 3))],
        );

        let embedder = MockTextEmbedder::new();
        let image_embedder = MockImageEmbedder::new(normalized_vec(512, 0));

        let searcher =
            CrossModalSearch::new(&registry, &embedder, CrossModalSearchConfig::default())
                .with_image_embedder(&image_embedder);

        let response = searcher
            .search(&CrossModalQuery::Image {
                image_bytes: b"fake_png",
            })
            .unwrap();

        assert!(!response.results.is_empty());
        assert_eq!(response.results[0].node_id, 20); // Best visual match
        assert_eq!(response.results[0].source, ResultSource::VisualIndex);
    }

    #[test]
    fn test_image_to_text_search() {
        let registry = build_registry_text_only(vec![
            (1, normalized_vec(384, 0)),
            (2, normalized_vec(384, 5)),
        ]);

        let embedder = MockTextEmbedder::new();
        let image_embedder = MockImageEmbedder::new(vec![0.1; 512]);
        let projection = MockProjection::new(normalized_vec(384, 0));

        let searcher =
            CrossModalSearch::new(&registry, &embedder, CrossModalSearchConfig::default())
                .with_image_embedder(&image_embedder)
                .with_projection(&projection);

        let response = searcher
            .search(&CrossModalQuery::ImageToText {
                image_bytes: b"fake_png",
            })
            .unwrap();

        assert!(!response.results.is_empty());
        assert_eq!(response.results[0].node_id, 1); // Best text match after projection
        assert_eq!(response.results[0].source, ResultSource::TextIndex);
    }

    #[test]
    fn test_audio_search_uses_text_index() {
        let registry = build_registry_text_only(vec![
            (1, normalized_vec(384, 0)),
            (2, normalized_vec(384, 1)),
        ]);

        let embedder = MockTextEmbedder::new()
            .with_embedding("the cat sat on the mat", normalized_vec(384, 0));

        let searcher = CrossModalSearch::new(
            &registry,
            &embedder,
            CrossModalSearchConfig {
                top_k: 1,
                ..Default::default()
            },
        );

        let response = searcher
            .search(&CrossModalQuery::Audio {
                transcript: "the cat sat on the mat",
            })
            .unwrap();

        assert_eq!(response.results.len(), 1);
        assert_eq!(response.results[0].node_id, 1);
        assert_eq!(response.results[0].source, ResultSource::TextIndex);
    }

    #[test]
    fn test_video_search_dual_fusion() {
        let registry = build_registry_dual(
            vec![(1, normalized_vec(384, 0)), (2, normalized_vec(384, 1))],
            vec![
                (1, normalized_vec(512, 0)), // Same node in both indices
                (3, normalized_vec(512, 2)), // Only in visual index
            ],
        );

        let embedder =
            MockTextEmbedder::new().with_embedding("video transcript", normalized_vec(384, 0));

        let searcher = CrossModalSearch::new(
            &registry,
            &embedder,
            CrossModalSearchConfig {
                top_k: 10,
                text_weight: 0.5,
                visual_weight: 0.5,
                ..Default::default()
            },
        );

        let frame_embs = vec![normalized_vec(512, 0)];

        let response = searcher
            .search(&CrossModalQuery::Video {
                frame_embeddings: &frame_embs,
                transcript: Some("video transcript"),
            })
            .unwrap();

        assert!(!response.results.is_empty());
        // Node 1 is in both indices, so it should have a fused score
        let node1_result = response.results.iter().find(|r| r.node_id == 1);
        assert!(node1_result.is_some());
        assert_eq!(node1_result.unwrap().source, ResultSource::Fused);
        assert!(response.metrics.text_result_count > 0);
        assert!(response.metrics.visual_result_count > 0);
        assert!(response.metrics.fusion_us < 1_000_000); // sanity check
    }

    #[test]
    fn test_video_search_visual_only() {
        let registry = build_registry_dual(
            vec![(1, normalized_vec(384, 0))],
            vec![(10, normalized_vec(512, 0))],
        );

        let embedder = MockTextEmbedder::new();

        let searcher =
            CrossModalSearch::new(&registry, &embedder, CrossModalSearchConfig::default());

        let frame_embs = vec![normalized_vec(512, 0)];

        let response = searcher
            .search(&CrossModalQuery::Video {
                frame_embeddings: &frame_embs,
                transcript: None,
            })
            .unwrap();

        assert!(!response.results.is_empty());
        assert_eq!(response.results[0].node_id, 10);
        assert_eq!(response.results[0].source, ResultSource::VisualIndex);
    }

    #[test]
    fn test_text_to_image_without_encoder_fails() {
        let registry = build_registry_dual(vec![], vec![(1, vec![1.0; 512])]);
        let embedder = MockTextEmbedder::new();

        let searcher =
            CrossModalSearch::new(&registry, &embedder, CrossModalSearchConfig::default());
        // No clip text encoder set

        let result = searcher.search(&CrossModalQuery::TextToImage { text: "test" });
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("CLIP text encoder required"));
    }

    #[test]
    fn test_image_search_without_embedder_fails() {
        let registry = build_registry_dual(vec![], vec![(1, vec![1.0; 512])]);
        let embedder = MockTextEmbedder::new();

        let searcher =
            CrossModalSearch::new(&registry, &embedder, CrossModalSearchConfig::default());
        // No image embedder set

        let result = searcher.search(&CrossModalQuery::Image {
            image_bytes: b"fake",
        });
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("CLIP image embedder required"));
    }

    #[test]
    fn test_image_to_text_without_projection_fails() {
        let registry = build_registry_text_only(vec![(1, vec![0.1; 384])]);
        let embedder = MockTextEmbedder::new();
        let image_embedder = MockImageEmbedder::new(vec![0.1; 512]);

        let searcher =
            CrossModalSearch::new(&registry, &embedder, CrossModalSearchConfig::default())
                .with_image_embedder(&image_embedder);
        // No projection layer set

        let result = searcher.search(&CrossModalQuery::ImageToText {
            image_bytes: b"fake",
        });
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("Projection layer required"));
    }

    #[test]
    fn test_min_similarity_filter() {
        let registry = build_registry_text_only(vec![
            (1, normalized_vec(384, 0)),
            (2, vec![0.01; 384]), // Low similarity
        ]);

        let embedder = MockTextEmbedder::new().with_embedding("test", normalized_vec(384, 0));

        let searcher = CrossModalSearch::new(
            &registry,
            &embedder,
            CrossModalSearchConfig {
                top_k: 10,
                min_similarity: 0.5,
                ..Default::default()
            },
        );

        let response = searcher
            .search(&CrossModalQuery::Text { text: "test" })
            .unwrap();

        // Only node 1 should pass the 0.5 threshold
        assert_eq!(response.results.len(), 1);
        assert_eq!(response.results[0].node_id, 1);
    }

    #[test]
    fn test_metrics_populated() {
        let registry = build_registry_text_only(vec![(1, normalized_vec(384, 0))]);
        let embedder = MockTextEmbedder::new();

        let searcher =
            CrossModalSearch::new(&registry, &embedder, CrossModalSearchConfig::default());

        let response = searcher
            .search(&CrossModalQuery::Text { text: "hello" })
            .unwrap();

        assert!(response.metrics.total_us > 0);
        assert_eq!(response.metrics.final_result_count, response.results.len());
    }

    #[test]
    fn test_empty_video_frames_and_transcript() {
        let registry = build_registry_text_only(vec![(1, normalized_vec(384, 0))]);
        let embedder = MockTextEmbedder::new();

        let searcher =
            CrossModalSearch::new(&registry, &embedder, CrossModalSearchConfig::default());

        let response = searcher
            .search(&CrossModalQuery::Video {
                frame_embeddings: &[],
                transcript: None,
            })
            .unwrap();

        // No frames and no transcript → empty results
        assert!(response.results.is_empty());
    }

    #[test]
    fn test_config_defaults() {
        let config = CrossModalSearchConfig::default();
        assert_eq!(config.top_k, 10);
        assert!((config.text_weight - 0.5).abs() < f32::EPSILON);
        assert!((config.visual_weight - 0.5).abs() < f32::EPSILON);
        assert!((config.min_similarity - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_result_source_variants() {
        assert_ne!(ResultSource::TextIndex, ResultSource::VisualIndex);
        assert_ne!(ResultSource::TextIndex, ResultSource::Fused);
        assert_ne!(ResultSource::VisualIndex, ResultSource::Fused);
    }

    // --- US-33.15: Text-to-image search tests ---

    #[test]
    fn test_text_to_image_convenience_method() {
        let registry = build_registry_dual(
            vec![(1, normalized_vec(384, 0))],
            vec![
                (10, normalized_vec(512, 0)),
                (11, normalized_vec(512, 1)),
                (12, normalized_vec(512, 2)),
            ],
        );

        let embedder = MockTextEmbedder::new();
        let clip_encoder = MockClipTextEncoder::new(normalized_vec(512, 0));

        let searcher = CrossModalSearch::new(
            &registry,
            &embedder,
            CrossModalSearchConfig {
                top_k: 5,
                ..Default::default()
            },
        )
        .with_clip_text_encoder(&clip_encoder);

        // Use the convenience method instead of constructing CrossModalQuery manually
        let response = searcher.text_to_image("a sunset over mountains").unwrap();

        assert!(!response.results.is_empty());
        assert_eq!(response.results[0].node_id, 10); // Best visual match
        assert_eq!(response.results[0].source, ResultSource::VisualIndex);
        assert!(response.results[0].score > 0.0);
        assert!(response.metrics.visual_search_us > 0 || response.metrics.query_encoding_us > 0);
    }

    #[test]
    fn test_text_to_image_top_k_override() {
        let registry = build_registry_dual(
            vec![],
            vec![
                (10, normalized_vec(512, 0)),
                (11, normalized_vec(512, 1)),
                (12, normalized_vec(512, 2)),
                (13, normalized_vec(512, 3)),
                (14, normalized_vec(512, 4)),
            ],
        );

        let embedder = MockTextEmbedder::new();
        let clip_encoder = MockClipTextEncoder::new(normalized_vec(512, 0));

        // Config says top_k=2, but we override to 3
        let searcher = CrossModalSearch::new(
            &registry,
            &embedder,
            CrossModalSearchConfig {
                top_k: 2,
                ..Default::default()
            },
        )
        .with_clip_text_encoder(&clip_encoder);

        let response = searcher.text_to_image_top_k("test query", 3).unwrap();
        assert_eq!(response.results.len(), 3);
        assert_eq!(response.metrics.final_result_count, 3);
    }

    #[test]
    fn test_text_to_image_with_min_similarity_filter() {
        let registry = build_registry_dual(
            vec![],
            vec![
                (10, normalized_vec(512, 0)), // Will have score 1.0
                (11, vec![0.001; 512]),       // Will have very low similarity
            ],
        );

        let embedder = MockTextEmbedder::new();
        let clip_encoder = MockClipTextEncoder::new(normalized_vec(512, 0));

        let searcher = CrossModalSearch::new(
            &registry,
            &embedder,
            CrossModalSearchConfig {
                top_k: 10,
                min_similarity: 0.5,
                ..Default::default()
            },
        )
        .with_clip_text_encoder(&clip_encoder);

        let response = searcher.text_to_image("find similar images").unwrap();

        // Only node 10 should pass the 0.5 threshold (score ≈ 1.0)
        assert_eq!(response.results.len(), 1);
        assert_eq!(response.results[0].node_id, 10);
        assert!(response.results[0].score >= 0.5);
    }

    #[test]
    fn test_text_to_image_empty_visual_index() {
        let registry = build_registry_dual(
            vec![(1, normalized_vec(384, 0))],
            vec![], // Empty visual index
        );

        let embedder = MockTextEmbedder::new();
        let clip_encoder = MockClipTextEncoder::new(normalized_vec(512, 0));

        let searcher =
            CrossModalSearch::new(&registry, &embedder, CrossModalSearchConfig::default())
                .with_clip_text_encoder(&clip_encoder);

        let response = searcher.text_to_image("no images here").unwrap();
        assert!(response.results.is_empty());
        assert_eq!(response.metrics.visual_result_count, 0);
        assert_eq!(response.metrics.final_result_count, 0);
    }

    #[test]
    fn test_text_to_image_results_sorted_by_similarity() {
        // Create visual data where we know the exact ordering
        let registry = build_registry_dual(
            vec![],
            vec![
                (10, normalized_vec(512, 0)), // score = 1.0 (exact match)
                (11, normalized_vec(512, 1)), // score = 0.0 (orthogonal)
                (12, {
                    let mut v = vec![0.0; 512];
                    v[0] = 0.7;
                    v[1] = 0.7;
                    v
                }), // score ≈ 0.7 (partial match)
            ],
        );

        let embedder = MockTextEmbedder::new();
        let clip_encoder = MockClipTextEncoder::new(normalized_vec(512, 0));

        let searcher = CrossModalSearch::new(
            &registry,
            &embedder,
            CrossModalSearchConfig {
                top_k: 10,
                ..Default::default()
            },
        )
        .with_clip_text_encoder(&clip_encoder);

        let response = searcher.text_to_image("sorted results test").unwrap();

        assert_eq!(response.results.len(), 3);
        // Results should be sorted by descending score
        assert_eq!(response.results[0].node_id, 10); // score = 1.0
        assert_eq!(response.results[1].node_id, 12); // score ≈ 0.7
        assert_eq!(response.results[2].node_id, 11); // score = 0.0
        assert!(response.results[0].score >= response.results[1].score);
        assert!(response.results[1].score >= response.results[2].score);
    }

    #[test]
    fn test_text_to_image_all_results_from_visual_index() {
        let registry = build_registry_dual(
            vec![(1, normalized_vec(384, 0))], // Text index has data
            vec![(10, normalized_vec(512, 0)), (11, normalized_vec(512, 1))],
        );

        let embedder = MockTextEmbedder::new();
        let clip_encoder = MockClipTextEncoder::new(normalized_vec(512, 0));

        let searcher =
            CrossModalSearch::new(&registry, &embedder, CrossModalSearchConfig::default())
                .with_clip_text_encoder(&clip_encoder);

        let response = searcher.text_to_image("test").unwrap();

        // All results should come from the visual index, not the text index
        for result in &response.results {
            assert_eq!(result.source, ResultSource::VisualIndex);
        }
        // Text index node (id=1) should NOT appear in results
        assert!(response.results.iter().all(|r| r.node_id != 1));
    }

    #[test]
    fn test_text_to_image_metrics_complete() {
        let registry = build_registry_dual(vec![], vec![(10, normalized_vec(512, 0))]);

        let embedder = MockTextEmbedder::new();
        let clip_encoder = MockClipTextEncoder::new(normalized_vec(512, 0));

        let searcher =
            CrossModalSearch::new(&registry, &embedder, CrossModalSearchConfig::default())
                .with_clip_text_encoder(&clip_encoder);

        let response = searcher.text_to_image("metrics test").unwrap();

        // Verify metrics are populated correctly
        assert!(response.metrics.total_us > 0);
        assert_eq!(response.metrics.visual_result_count, 1);
        assert_eq!(response.metrics.final_result_count, 1);
        // Text search was NOT performed
        assert_eq!(response.metrics.text_search_us, 0);
        assert_eq!(response.metrics.text_result_count, 0);
    }

    #[test]
    fn test_text_to_image_convenience_without_encoder_fails() {
        let registry = build_registry_dual(vec![], vec![(10, normalized_vec(512, 0))]);

        let embedder = MockTextEmbedder::new();

        let searcher =
            CrossModalSearch::new(&registry, &embedder, CrossModalSearchConfig::default());
        // No CLIP text encoder set

        let result = searcher.text_to_image("should fail");
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("CLIP text encoder required"));
    }

    #[test]
    fn test_text_to_image_without_visual_backend_fails() {
        // Only text backend, no visual backend
        let registry = build_registry_text_only(vec![(1, normalized_vec(384, 0))]);

        let embedder = MockTextEmbedder::new();
        let clip_encoder = MockClipTextEncoder::new(normalized_vec(512, 0));

        let searcher =
            CrossModalSearch::new(&registry, &embedder, CrossModalSearchConfig::default())
                .with_clip_text_encoder(&clip_encoder);

        let result = searcher.text_to_image("no visual backend");
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("Visual vector backend required"));
    }

    #[test]
    fn test_standalone_text_to_image_search_function() {
        let registry = build_registry_dual(
            vec![(1, normalized_vec(384, 0))],
            vec![(10, normalized_vec(512, 0)), (11, normalized_vec(512, 1))],
        );

        let embedder = MockTextEmbedder::new();
        let clip_encoder = MockClipTextEncoder::new(normalized_vec(512, 0));

        let results = text_to_image_search(
            &registry,
            &clip_encoder,
            &embedder,
            "find me images",
            5,
            0.0,
        )
        .unwrap();

        assert!(!results.is_empty());
        assert_eq!(results[0].0, 10); // Best match node_id
        assert!(results[0].1 > 0.0); // Positive similarity
    }

    #[test]
    fn test_standalone_text_to_image_with_min_similarity() {
        let registry = build_registry_dual(
            vec![],
            vec![
                (10, normalized_vec(512, 0)), // Will have score 1.0
                (11, vec![0.001; 512]),       // Very low similarity
            ],
        );

        let embedder = MockTextEmbedder::new();
        let clip_encoder = MockClipTextEncoder::new(normalized_vec(512, 0));

        let results = text_to_image_search(
            &registry,
            &clip_encoder,
            &embedder,
            "high quality only",
            10,
            0.5, // Only results above 0.5
        )
        .unwrap();

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, 10);
        assert!(results[0].1 >= 0.5);
    }

    #[test]
    fn test_text_to_image_top_k_limits_results() {
        let registry = build_registry_dual(
            vec![],
            vec![
                (10, normalized_vec(512, 0)),
                (11, normalized_vec(512, 1)),
                (12, normalized_vec(512, 2)),
                (13, normalized_vec(512, 3)),
                (14, normalized_vec(512, 4)),
            ],
        );

        let embedder = MockTextEmbedder::new();
        let clip_encoder = MockClipTextEncoder::new(normalized_vec(512, 0));

        let searcher = CrossModalSearch::new(
            &registry,
            &embedder,
            CrossModalSearchConfig {
                top_k: 2,
                ..Default::default()
            },
        )
        .with_clip_text_encoder(&clip_encoder);

        let response = searcher.text_to_image("limited results").unwrap();
        assert!(response.results.len() <= 2);
    }

    #[test]
    fn test_text_to_image_different_queries_same_results_order() {
        // All queries go through the same mock encoder, so results should be identical
        let registry = build_registry_dual(
            vec![],
            vec![(10, normalized_vec(512, 0)), (11, normalized_vec(512, 1))],
        );

        let embedder = MockTextEmbedder::new();
        let clip_encoder = MockClipTextEncoder::new(normalized_vec(512, 0));

        let searcher =
            CrossModalSearch::new(&registry, &embedder, CrossModalSearchConfig::default())
                .with_clip_text_encoder(&clip_encoder);

        let response1 = searcher.text_to_image("query one").unwrap();
        let response2 = searcher.text_to_image("query two").unwrap();

        // Same mock encoder produces same embedding, so same results
        assert_eq!(response1.results.len(), response2.results.len());
        for (r1, r2) in response1.results.iter().zip(response2.results.iter()) {
            assert_eq!(r1.node_id, r2.node_id);
            assert!((r1.score - r2.score).abs() < f32::EPSILON);
        }
    }

    // --- US-33.16: Image-to-text search tests ---

    #[test]
    fn test_image_to_text_convenience_method() {
        let registry = build_registry_text_only(vec![
            (1, normalized_vec(384, 0)),
            (2, normalized_vec(384, 1)),
            (3, normalized_vec(384, 2)),
        ]);

        let embedder = MockTextEmbedder::new();
        let image_embedder = MockImageEmbedder::new(vec![0.1; 512]);
        let projection = MockProjection::new(normalized_vec(384, 0));

        let searcher = CrossModalSearch::new(
            &registry,
            &embedder,
            CrossModalSearchConfig {
                top_k: 5,
                ..Default::default()
            },
        )
        .with_image_embedder(&image_embedder)
        .with_projection(&projection);

        let response = searcher.image_to_text(b"fake_image_bytes").unwrap();

        assert!(!response.results.is_empty());
        assert_eq!(response.results[0].node_id, 1); // Best text match after projection
        assert_eq!(response.results[0].source, ResultSource::TextIndex);
        assert!(response.results[0].score > 0.0);
    }

    #[test]
    fn test_image_to_text_top_k_override() {
        let registry = build_registry_text_only(vec![
            (1, normalized_vec(384, 0)),
            (2, normalized_vec(384, 1)),
            (3, normalized_vec(384, 2)),
            (4, normalized_vec(384, 3)),
            (5, normalized_vec(384, 4)),
        ]);

        let embedder = MockTextEmbedder::new();
        let image_embedder = MockImageEmbedder::new(vec![0.1; 512]);
        let projection = MockProjection::new(normalized_vec(384, 0));

        // Config says top_k=2, but we override to 3
        let searcher = CrossModalSearch::new(
            &registry,
            &embedder,
            CrossModalSearchConfig {
                top_k: 2,
                ..Default::default()
            },
        )
        .with_image_embedder(&image_embedder)
        .with_projection(&projection);

        let response = searcher.image_to_text_top_k(b"fake_image", 3).unwrap();
        assert_eq!(response.results.len(), 3);
        assert_eq!(response.metrics.final_result_count, 3);
    }

    #[test]
    fn test_image_to_text_with_min_similarity_filter() {
        let registry = build_registry_text_only(vec![
            (1, normalized_vec(384, 0)), // Will have score 1.0 via projection
            (2, vec![0.001; 384]),       // Very low similarity
        ]);

        let embedder = MockTextEmbedder::new();
        let image_embedder = MockImageEmbedder::new(vec![0.1; 512]);
        let projection = MockProjection::new(normalized_vec(384, 0));

        let searcher = CrossModalSearch::new(
            &registry,
            &embedder,
            CrossModalSearchConfig {
                top_k: 10,
                min_similarity: 0.5,
                ..Default::default()
            },
        )
        .with_image_embedder(&image_embedder)
        .with_projection(&projection);

        let response = searcher.image_to_text(b"test_image").unwrap();

        // Only node 1 should pass the 0.5 threshold
        assert_eq!(response.results.len(), 1);
        assert_eq!(response.results[0].node_id, 1);
        assert!(response.results[0].score >= 0.5);
    }

    #[test]
    fn test_image_to_text_empty_text_index() {
        let registry = build_registry_text_only(vec![]); // Empty text index

        let embedder = MockTextEmbedder::new();
        let image_embedder = MockImageEmbedder::new(vec![0.1; 512]);
        let projection = MockProjection::new(normalized_vec(384, 0));

        let searcher =
            CrossModalSearch::new(&registry, &embedder, CrossModalSearchConfig::default())
                .with_image_embedder(&image_embedder)
                .with_projection(&projection);

        let response = searcher.image_to_text(b"no_text_here").unwrap();
        assert!(response.results.is_empty());
        assert_eq!(response.metrics.text_result_count, 0);
        assert_eq!(response.metrics.final_result_count, 0);
    }

    #[test]
    fn test_image_to_text_results_sorted_by_similarity() {
        let registry = build_registry_text_only(vec![
            (1, normalized_vec(384, 0)), // score = 1.0 (exact match with projection output)
            (2, normalized_vec(384, 1)), // score = 0.0 (orthogonal)
            (3, {
                let mut v = vec![0.0; 384];
                v[0] = 0.7;
                v[1] = 0.7;
                v
            }), // score ≈ 0.7 (partial match)
        ]);

        let embedder = MockTextEmbedder::new();
        let image_embedder = MockImageEmbedder::new(vec![0.1; 512]);
        let projection = MockProjection::new(normalized_vec(384, 0));

        let searcher = CrossModalSearch::new(
            &registry,
            &embedder,
            CrossModalSearchConfig {
                top_k: 10,
                ..Default::default()
            },
        )
        .with_image_embedder(&image_embedder)
        .with_projection(&projection);

        let response = searcher.image_to_text(b"sorted_test").unwrap();

        assert_eq!(response.results.len(), 3);
        assert_eq!(response.results[0].node_id, 1); // score = 1.0
        assert_eq!(response.results[1].node_id, 3); // score ≈ 0.7
        assert_eq!(response.results[2].node_id, 2); // score = 0.0
        assert!(response.results[0].score >= response.results[1].score);
        assert!(response.results[1].score >= response.results[2].score);
    }

    #[test]
    fn test_image_to_text_all_results_from_text_index() {
        let registry = build_registry_dual(
            vec![(1, normalized_vec(384, 0)), (2, normalized_vec(384, 1))],
            vec![(10, normalized_vec(512, 0))], // Visual index has data too
        );

        let embedder = MockTextEmbedder::new();
        let image_embedder = MockImageEmbedder::new(vec![0.1; 512]);
        let projection = MockProjection::new(normalized_vec(384, 0));

        let searcher =
            CrossModalSearch::new(&registry, &embedder, CrossModalSearchConfig::default())
                .with_image_embedder(&image_embedder)
                .with_projection(&projection);

        let response = searcher.image_to_text(b"test").unwrap();

        // All results should come from the text index, not the visual index
        for result in &response.results {
            assert_eq!(result.source, ResultSource::TextIndex);
        }
        // Visual index node (id=10) should NOT appear in results
        assert!(response.results.iter().all(|r| r.node_id != 10));
    }

    #[test]
    fn test_image_to_text_metrics_complete() {
        let registry = build_registry_text_only(vec![(1, normalized_vec(384, 0))]);

        let embedder = MockTextEmbedder::new();
        let image_embedder = MockImageEmbedder::new(vec![0.1; 512]);
        let projection = MockProjection::new(normalized_vec(384, 0));

        let searcher =
            CrossModalSearch::new(&registry, &embedder, CrossModalSearchConfig::default())
                .with_image_embedder(&image_embedder)
                .with_projection(&projection);

        let response = searcher.image_to_text(b"metrics_test").unwrap();

        // Verify metrics are populated correctly
        assert!(response.metrics.total_us > 0);
        assert_eq!(response.metrics.text_result_count, 1);
        assert_eq!(response.metrics.final_result_count, 1);
        // Visual search was NOT performed (image-to-text goes through projection to text index)
        assert_eq!(response.metrics.visual_search_us, 0);
        assert_eq!(response.metrics.visual_result_count, 0);
    }

    #[test]
    fn test_image_to_text_convenience_without_embedder_fails() {
        let registry = build_registry_text_only(vec![(1, normalized_vec(384, 0))]);
        let embedder = MockTextEmbedder::new();
        let projection = MockProjection::new(normalized_vec(384, 0));

        let searcher =
            CrossModalSearch::new(&registry, &embedder, CrossModalSearchConfig::default())
                .with_projection(&projection);
        // No CLIP image embedder set

        let result = searcher.image_to_text(b"should_fail");
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("CLIP image embedder required"));
    }

    #[test]
    fn test_image_to_text_convenience_without_projection_fails() {
        let registry = build_registry_text_only(vec![(1, normalized_vec(384, 0))]);
        let embedder = MockTextEmbedder::new();
        let image_embedder = MockImageEmbedder::new(vec![0.1; 512]);

        let searcher =
            CrossModalSearch::new(&registry, &embedder, CrossModalSearchConfig::default())
                .with_image_embedder(&image_embedder);
        // No projection layer set

        let result = searcher.image_to_text(b"no_projection");
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("Projection layer required"));
    }

    #[test]
    fn test_standalone_image_to_text_search_function() {
        let registry = build_registry_text_only(vec![
            (1, normalized_vec(384, 0)),
            (2, normalized_vec(384, 1)),
        ]);

        let embedder = MockTextEmbedder::new();
        let image_embedder = MockImageEmbedder::new(vec![0.1; 512]);
        let projection = MockProjection::new(normalized_vec(384, 0));

        let results = image_to_text_search(
            &registry,
            &image_embedder,
            &projection,
            &embedder,
            b"find_text_from_image",
            5,
            0.0,
        )
        .unwrap();

        assert!(!results.is_empty());
        assert_eq!(results[0].0, 1); // Best match node_id
        assert!(results[0].1 > 0.0); // Positive similarity
    }

    #[test]
    fn test_standalone_image_to_text_with_min_similarity() {
        let registry = build_registry_text_only(vec![
            (1, normalized_vec(384, 0)), // Will have score 1.0
            (2, vec![0.001; 384]),       // Very low similarity
        ]);

        let embedder = MockTextEmbedder::new();
        let image_embedder = MockImageEmbedder::new(vec![0.1; 512]);
        let projection = MockProjection::new(normalized_vec(384, 0));

        let results = image_to_text_search(
            &registry,
            &image_embedder,
            &projection,
            &embedder,
            b"high_quality_only",
            10,
            0.5,
        )
        .unwrap();

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, 1);
        assert!(results[0].1 >= 0.5);
    }

    #[test]
    fn test_image_to_text_top_k_limits_results() {
        let registry = build_registry_text_only(vec![
            (1, normalized_vec(384, 0)),
            (2, normalized_vec(384, 1)),
            (3, normalized_vec(384, 2)),
            (4, normalized_vec(384, 3)),
            (5, normalized_vec(384, 4)),
        ]);

        let embedder = MockTextEmbedder::new();
        let image_embedder = MockImageEmbedder::new(vec![0.1; 512]);
        let projection = MockProjection::new(normalized_vec(384, 0));

        let searcher = CrossModalSearch::new(
            &registry,
            &embedder,
            CrossModalSearchConfig {
                top_k: 2,
                ..Default::default()
            },
        )
        .with_image_embedder(&image_embedder)
        .with_projection(&projection);

        let response = searcher.image_to_text(b"limited_results").unwrap();
        assert!(response.results.len() <= 2);
    }

    // -----------------------------------------------------------------------
    // US-33.21: Cross-modal retrieval accuracy benchmark
    // -----------------------------------------------------------------------

    /// Generate a deterministic L2-normalized vector with a dominant cluster direction
    /// plus controlled noise. Vectors in the same cluster will have high cosine
    /// similarity; vectors in different clusters will have low similarity.
    fn cluster_vec(dim: usize, cluster_idx: usize, item_offset: f32, seed: u64) -> Vec<f32> {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut v = vec![0.0f32; dim];
        // Primary cluster signal: strong component in the cluster's "direction"
        let base = cluster_idx * 10;
        for i in 0..10 {
            let idx = (base + i) % dim;
            v[idx] = 0.8 + item_offset * 0.02;
        }
        // Add deterministic noise per seed
        let mut hasher = DefaultHasher::new();
        seed.hash(&mut hasher);
        cluster_idx.hash(&mut hasher);
        let h = hasher.finish();
        #[allow(clippy::needless_range_loop)]
        for i in 0..dim {
            let noise_seed = h.wrapping_add(i as u64);
            let noise = ((noise_seed % 1000) as f32 / 1000.0 - 0.5) * 0.1;
            v[i] += noise;
        }
        // L2 normalize
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for x in &mut v {
                *x /= norm;
            }
        }
        v
    }

    /// Compute Recall@k: fraction of queries where the ground-truth item appears in top-k.
    fn recall_at_k(results: &[(NodeId, Vec<NodeId>)], k: usize) -> f64 {
        if results.is_empty() {
            return 0.0;
        }
        let hits: usize = results
            .iter()
            .filter(|(gt_id, top_k_ids)| top_k_ids.iter().take(k).any(|id| id == gt_id))
            .count();
        hits as f64 / results.len() as f64
    }

    /// Compute MRR (Mean Reciprocal Rank).
    fn mrr(results: &[(NodeId, Vec<NodeId>)]) -> f64 {
        if results.is_empty() {
            return 0.0;
        }
        let sum: f64 = results
            .iter()
            .map(
                |(gt_id, top_k_ids)| match top_k_ids.iter().position(|id| id == gt_id) {
                    Some(rank) => 1.0 / (rank as f64 + 1.0),
                    None => 0.0,
                },
            )
            .sum();
        sum / results.len() as f64
    }

    /// Compute NDCG@k (Normalized Discounted Cumulative Gain) for binary relevance.
    fn ndcg_at_k(results: &[(NodeId, Vec<NodeId>)], k: usize) -> f64 {
        if results.is_empty() {
            return 0.0;
        }
        let sum: f64 = results
            .iter()
            .map(|(gt_id, top_k_ids)| {
                // DCG: gain / log2(rank + 2) for the first hit
                let dcg: f64 = top_k_ids
                    .iter()
                    .take(k)
                    .enumerate()
                    .filter(|(_, id)| *id == gt_id)
                    .map(|(rank, _)| 1.0 / (rank as f64 + 2.0).log2())
                    .sum();
                // Ideal DCG: 1 relevant item at rank 0 → 1 / log2(2) = 1.0
                let idcg = 1.0;
                dcg / idcg
            })
            .sum();
        sum / results.len() as f64
    }

    /// Configurable mock CLIP text encoder that maps query index to a cluster vector.
    struct BenchClipTextEncoder {
        /// Maps query text → 512-dim CLIP embedding
        embeddings: HashMap<String, Vec<f32>>,
    }

    impl BenchClipTextEncoder {
        fn new(embeddings: HashMap<String, Vec<f32>>) -> Self {
            Self { embeddings }
        }
    }

    impl CrossModalTextEncoder for BenchClipTextEncoder {
        fn embed_text(&self, text: &str) -> Result<Vec<f32>> {
            Ok(self
                .embeddings
                .get(text)
                .cloned()
                .unwrap_or_else(|| vec![0.0; 512]))
        }
    }

    /// Configurable mock image embedder that maps byte patterns to cluster vectors.
    struct BenchImageEmbedder {
        embeddings: HashMap<Vec<u8>, Vec<f32>>,
    }

    impl BenchImageEmbedder {
        fn new(embeddings: HashMap<Vec<u8>, Vec<f32>>) -> Self {
            Self { embeddings }
        }
    }

    impl ImageEmbeddingPipeline for BenchImageEmbedder {
        fn embed_image_bytes(&self, bytes: &[u8]) -> Result<Vec<f32>> {
            Ok(self
                .embeddings
                .get(bytes)
                .cloned()
                .unwrap_or_else(|| vec![0.0; 512]))
        }

        fn embed_image_file(&self, _path: &std::path::Path) -> Result<Vec<f32>> {
            Ok(vec![0.0; 512])
        }
    }

    /// Configurable mock projection that maps 512-dim CLIP → 384-dim MiniLM
    /// using a cluster-aware linear mapping.
    struct BenchProjection {
        /// Maps input embedding hash → 384-dim output
        mappings: HashMap<u64, Vec<f32>>,
    }

    impl BenchProjection {
        fn new(mappings: HashMap<u64, Vec<f32>>) -> Self {
            Self { mappings }
        }

        fn embedding_hash(v: &[f32]) -> u64 {
            use std::collections::hash_map::DefaultHasher;
            use std::hash::{Hash, Hasher};
            let mut hasher = DefaultHasher::new();
            for &x in v.iter().take(10) {
                x.to_bits().hash(&mut hasher);
            }
            hasher.finish()
        }
    }

    impl CrossModalProjection for BenchProjection {
        fn project(&self, embedding: &[f32]) -> Result<Vec<f32>> {
            let hash = Self::embedding_hash(embedding);
            Ok(self
                .mappings
                .get(&hash)
                .cloned()
                .unwrap_or_else(|| vec![0.0; 384]))
        }

        fn project_batch(&self, embeddings: &[Vec<f32>]) -> Result<Vec<Vec<f32>>> {
            embeddings.iter().map(|e| self.project(e)).collect()
        }

        fn input_dim(&self) -> usize {
            512
        }

        fn output_dim(&self) -> usize {
            384
        }
    }

    /// US-33.21: Benchmark cross-modal retrieval accuracy.
    ///
    /// Creates a synthetic dataset of 100 text-image pairs across 10 semantic
    /// clusters (10 pairs each). Each pair shares a cluster-aligned embedding
    /// vector so that the matching text/image items have high cosine similarity.
    ///
    /// Measures:
    /// - Text→Image Recall@1, @5, @10, MRR, NDCG@10
    /// - Image→Text Recall@1, @5, @10, MRR, NDCG@10
    ///
    /// Acceptance: Recall@10 > 0.6 for both directions.
    #[test]
    fn test_benchmark_cross_modal_retrieval_accuracy() {
        let num_clusters = 10;
        let items_per_cluster = 10;
        let total_items = num_clusters * items_per_cluster;
        let top_k = 10;

        // ---------------------------------------------------------------
        // Step 1: Generate paired text-image dataset
        // ---------------------------------------------------------------
        // Each item i has:
        //   - A text node (NodeId = i) with a 384-dim cluster-aligned embedding
        //   - An image node (NodeId = total_items + i) with a 512-dim cluster-aligned embedding
        //   - The text and image for item i share the same cluster (i / items_per_cluster)

        let mut text_data: Vec<(NodeId, Vec<f32>)> = Vec::new();
        let mut visual_data: Vec<(NodeId, Vec<f32>)> = Vec::new();
        let mut clip_text_embeddings: HashMap<String, Vec<f32>> = HashMap::new();
        let mut image_embeddings: HashMap<Vec<u8>, Vec<f32>> = HashMap::new();
        let mut projection_mappings: HashMap<u64, Vec<f32>> = HashMap::new();

        // Ground truth: item i's image should match text node i, and vice versa
        let mut text_to_image_gt: Vec<(NodeId, NodeId)> = Vec::new(); // (image_node_id, text_query_index)
        let mut image_to_text_gt: Vec<(NodeId, NodeId)> = Vec::new(); // (text_node_id, image_query_index)

        for cluster in 0..num_clusters {
            for item in 0..items_per_cluster {
                let idx = cluster * items_per_cluster + item;
                let text_node_id = idx as NodeId;
                let image_node_id = (total_items + idx) as NodeId;
                let offset = item as f32;

                // Text embedding (384-dim) in cluster's region
                let text_emb = cluster_vec(384, cluster, offset, 42);
                text_data.push((text_node_id, text_emb.clone()));

                // Visual embedding (512-dim) in the SAME cluster's region
                let visual_emb = cluster_vec(512, cluster, offset, 43);
                visual_data.push((image_node_id, visual_emb.clone()));

                // For text→image queries: CLIP text encoder maps "query_N" → 512-dim cluster vec
                let query_key = format!("query_{}", idx);
                let clip_query_emb = cluster_vec(512, cluster, offset, 44);
                clip_text_embeddings.insert(query_key, clip_query_emb);

                // For image→text queries: image bytes → 512-dim CLIP embedding
                let image_bytes = format!("image_{}", idx).into_bytes();
                let img_clip_emb = cluster_vec(512, cluster, offset, 45);
                image_embeddings.insert(image_bytes, img_clip_emb.clone());

                // Projection: CLIP 512-dim → MiniLM 384-dim (preserving cluster alignment)
                let projected = cluster_vec(384, cluster, offset, 46);
                let hash = BenchProjection::embedding_hash(&img_clip_emb);
                projection_mappings.insert(hash, projected);

                // Ground truth mapping
                text_to_image_gt.push((image_node_id, text_node_id));
                image_to_text_gt.push((text_node_id, image_node_id));
            }
        }

        // ---------------------------------------------------------------
        // Step 2: Build registry and search orchestrator
        // ---------------------------------------------------------------
        let registry = build_registry_dual(text_data, visual_data);
        let text_embedder = MockTextEmbedder::new(); // Not used for cross-modal
        let clip_encoder = BenchClipTextEncoder::new(clip_text_embeddings.clone());
        let image_embedder = BenchImageEmbedder::new(image_embeddings.clone());
        let projection = BenchProjection::new(projection_mappings);

        let config = CrossModalSearchConfig {
            top_k,
            text_weight: 0.5,
            visual_weight: 0.5,
            min_similarity: 0.0,
        };

        let searcher = CrossModalSearch::new(&registry, &text_embedder, config)
            .with_clip_text_encoder(&clip_encoder)
            .with_image_embedder(&image_embedder)
            .with_projection(&projection);

        // ---------------------------------------------------------------
        // Step 3: Run text→image retrieval (100 queries)
        // ---------------------------------------------------------------
        let mut t2i_results: Vec<(NodeId, Vec<NodeId>)> = Vec::new();
        let mut t2i_latencies: Vec<u64> = Vec::new();

        for idx in 0..total_items {
            let query = format!("query_{}", idx);
            let expected_image_id = (total_items + idx) as NodeId;

            let start = std::time::Instant::now();
            let response = searcher
                .search(&CrossModalQuery::TextToImage { text: &query })
                .unwrap();
            t2i_latencies.push(start.elapsed().as_micros() as u64);

            let result_ids: Vec<NodeId> = response.results.iter().map(|r| r.node_id).collect();
            t2i_results.push((expected_image_id, result_ids));
        }

        // ---------------------------------------------------------------
        // Step 4: Run image→text retrieval (100 queries)
        // ---------------------------------------------------------------
        let mut i2t_results: Vec<(NodeId, Vec<NodeId>)> = Vec::new();
        let mut i2t_latencies: Vec<u64> = Vec::new();

        for idx in 0..total_items {
            let image_bytes = format!("image_{}", idx).into_bytes();
            let expected_text_id = idx as NodeId;

            let start = std::time::Instant::now();
            let response = searcher
                .search(&CrossModalQuery::ImageToText {
                    image_bytes: &image_bytes,
                })
                .unwrap();
            i2t_latencies.push(start.elapsed().as_micros() as u64);

            let result_ids: Vec<NodeId> = response.results.iter().map(|r| r.node_id).collect();
            i2t_results.push((expected_text_id, result_ids));
        }

        // ---------------------------------------------------------------
        // Step 5: Compute metrics
        // ---------------------------------------------------------------
        let t2i_r1 = recall_at_k(&t2i_results, 1);
        let t2i_r5 = recall_at_k(&t2i_results, 5);
        let t2i_r10 = recall_at_k(&t2i_results, 10);
        let t2i_mrr = mrr(&t2i_results);
        let t2i_ndcg10 = ndcg_at_k(&t2i_results, 10);

        let i2t_r1 = recall_at_k(&i2t_results, 1);
        let i2t_r5 = recall_at_k(&i2t_results, 5);
        let i2t_r10 = recall_at_k(&i2t_results, 10);
        let i2t_mrr = mrr(&i2t_results);
        let i2t_ndcg10 = ndcg_at_k(&i2t_results, 10);

        // Latency stats
        t2i_latencies.sort_unstable();
        i2t_latencies.sort_unstable();
        let t2i_p50 = if t2i_latencies.is_empty() {
            0
        } else {
            t2i_latencies[t2i_latencies.len() / 2]
        };
        let t2i_p95 = if t2i_latencies.is_empty() {
            0
        } else {
            t2i_latencies[(t2i_latencies.len() as f64 * 0.95) as usize]
        };
        let i2t_p50 = if i2t_latencies.is_empty() {
            0
        } else {
            i2t_latencies[i2t_latencies.len() / 2]
        };
        let i2t_p95 = if i2t_latencies.is_empty() {
            0
        } else {
            i2t_latencies[(i2t_latencies.len() as f64 * 0.95) as usize]
        };

        // ---------------------------------------------------------------
        // Step 6: Print benchmark report
        // ---------------------------------------------------------------
        println!();
        println!("=== US-33.21: Cross-Modal Retrieval Accuracy Benchmark ===");
        println!();
        println!(
            "Dataset: {} text-image pairs, {} clusters × {} items/cluster",
            total_items, num_clusters, items_per_cluster
        );
        println!("Top-k: {}", top_k);
        println!();
        println!("### Text→Image Retrieval");
        println!();
        println!("| Metric    | Value  |");
        println!("|-----------|--------|");
        println!("| Recall@1  | {:.4} |", t2i_r1);
        println!("| Recall@5  | {:.4} |", t2i_r5);
        println!("| Recall@10 | {:.4} |", t2i_r10);
        println!("| MRR       | {:.4} |", t2i_mrr);
        println!("| NDCG@10   | {:.4} |", t2i_ndcg10);
        println!("| Latency P50 | {}μs |", t2i_p50);
        println!("| Latency P95 | {}μs |", t2i_p95);
        println!();
        println!("### Image→Text Retrieval");
        println!();
        println!("| Metric    | Value  |");
        println!("|-----------|--------|");
        println!("| Recall@1  | {:.4} |", i2t_r1);
        println!("| Recall@5  | {:.4} |", i2t_r5);
        println!("| Recall@10 | {:.4} |", i2t_r10);
        println!("| MRR       | {:.4} |", i2t_mrr);
        println!("| NDCG@10   | {:.4} |", i2t_ndcg10);
        println!("| Latency P50 | {}μs |", i2t_p50);
        println!("| Latency P95 | {}μs |", i2t_p95);
        println!();

        // ---------------------------------------------------------------
        // Step 7: Assert acceptance criteria: Recall@10 > 0.6
        // ---------------------------------------------------------------
        assert!(
            t2i_r10 > 0.6,
            "Text→Image Recall@10 = {:.4}, expected > 0.6",
            t2i_r10
        );
        assert!(
            i2t_r10 > 0.6,
            "Image→Text Recall@10 = {:.4}, expected > 0.6",
            i2t_r10
        );

        // Additional quality assertions
        assert!(t2i_r10 >= t2i_r5, "Recall@10 should be >= Recall@5");
        assert!(t2i_r5 >= t2i_r1, "Recall@5 should be >= Recall@1");
        assert!(i2t_r10 >= i2t_r5, "Recall@10 should be >= Recall@5");
        assert!(i2t_r5 >= i2t_r1, "Recall@5 should be >= Recall@1");
        assert!(t2i_mrr > 0.0, "MRR should be positive");
        assert!(i2t_mrr > 0.0, "MRR should be positive");
        assert!(t2i_ndcg10 > 0.0, "NDCG@10 should be positive");
        assert!(i2t_ndcg10 > 0.0, "NDCG@10 should be positive");
    }

    #[test]
    fn test_recall_at_k_metric() {
        // 3 queries, ground truth = first element, results = second element
        let results = vec![
            (1_u64, vec![1_u64, 2, 3]), // Hit at rank 0
            (4, vec![2, 3, 4]),         // Hit at rank 2
            (7, vec![1, 2, 3]),         // Miss
        ];
        assert!((recall_at_k(&results, 1) - 1.0 / 3.0).abs() < 1e-6);
        assert!((recall_at_k(&results, 3) - 2.0 / 3.0).abs() < 1e-6);
        assert!((recall_at_k(&results, 10) - 2.0 / 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_mrr_metric() {
        let results = vec![
            (1_u64, vec![1_u64, 2, 3]), // RR = 1/1 = 1.0
            (4, vec![2, 3, 4]),         // RR = 1/3
            (7, vec![1, 2, 3]),         // RR = 0
        ];
        let expected = (1.0 + 1.0 / 3.0 + 0.0) / 3.0;
        assert!((mrr(&results) - expected).abs() < 1e-6);
    }

    #[test]
    fn test_ndcg_at_k_metric() {
        let results = vec![
            (1_u64, vec![1_u64, 2, 3]), // DCG = 1/log2(2) = 1.0, NDCG = 1.0
            (2, vec![3, 2, 1]),         // DCG = 1/log2(3) ≈ 0.631, NDCG ≈ 0.631
            (7, vec![1, 2, 3]),         // DCG = 0, NDCG = 0
        ];
        let expected = (1.0 + 1.0 / 3.0_f64.log2() + 0.0) / 3.0;
        assert!((ndcg_at_k(&results, 10) - expected).abs() < 1e-3);
    }

    #[test]
    fn test_recall_at_k_empty() {
        let results: Vec<(NodeId, Vec<NodeId>)> = vec![];
        assert!((recall_at_k(&results, 10) - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_mrr_empty() {
        let results: Vec<(NodeId, Vec<NodeId>)> = vec![];
        assert!((mrr(&results) - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_ndcg_at_k_empty() {
        let results: Vec<(NodeId, Vec<NodeId>)> = vec![];
        assert!((ndcg_at_k(&results, 10) - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_cluster_vec_normalized() {
        for cluster in 0..10 {
            let v = cluster_vec(384, cluster, 0.0, 42);
            let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            assert!((norm - 1.0).abs() < 1e-4, "Vector not normalized: {}", norm);
        }
    }

    #[test]
    fn test_cluster_vec_intra_similarity_high() {
        // Vectors in the same cluster should have high cosine similarity
        let v1 = cluster_vec(512, 3, 0.0, 42);
        let v2 = cluster_vec(512, 3, 1.0, 43);
        let sim: f32 = v1.iter().zip(v2.iter()).map(|(a, b)| a * b).sum();
        assert!(sim > 0.7, "Intra-cluster similarity too low: {}", sim);
    }

    #[test]
    fn test_cluster_vec_inter_similarity_low() {
        // Vectors in different clusters should have low cosine similarity
        let v1 = cluster_vec(512, 0, 0.0, 42);
        let v2 = cluster_vec(512, 5, 0.0, 42);
        let sim: f32 = v1.iter().zip(v2.iter()).map(|(a, b)| a * b).sum();
        assert!(sim < 0.5, "Inter-cluster similarity too high: {}", sim);
    }
}
