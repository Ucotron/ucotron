//! Multimodal Node Builders — helpers for constructing typed memory nodes.
//!
//! Provides [`MultimodalNodeBuilder`] for creating nodes with media-type-specific
//! validation. Each [`MediaType`] has required and optional fields:
//!
//! | MediaType | Required | Optional |
//! |-----------|----------|----------|
//! | Text | content, embedding | media_uri |
//! | Audio | content, embedding, media_uri | timestamp_range |
//! | Image | embedding_visual, media_uri | content, embedding |
//! | VideoSegment | embedding_visual, media_uri, parent_video_id | content, embedding, timestamp_range |
//!
//! # Example
//!
//! ```
//! use ucotron_core::multimodal::MultimodalNodeBuilder;
//! use ucotron_core::{MediaType, NodeType};
//!
//! // Text node (simplest case)
//! let node = MultimodalNodeBuilder::text(1, "Hello world", vec![0.1f32; 384])
//!     .node_type(NodeType::Fact)
//!     .timestamp(1_700_000_000)
//!     .build()
//!     .unwrap();
//! assert_eq!(node.media_type, Some(MediaType::Text));
//!
//! // Image node
//! let node = MultimodalNodeBuilder::image(2, "file:///photo.jpg", vec![0.2f32; 512])
//!     .content("A photo of a sunset".to_string())
//!     .embedding(vec![0.1f32; 384])
//!     .build()
//!     .unwrap();
//! assert_eq!(node.media_type, Some(MediaType::Image));
//! ```

use crate::{MediaType, Node, NodeId, NodeType, Value};
use std::collections::HashMap;

/// Validation error for multimodal node construction.
#[derive(Debug, Clone, PartialEq)]
pub enum MultimodalValidationError {
    /// A required field is missing for the given media type.
    MissingField {
        media_type: MediaType,
        field: &'static str,
    },
    /// The visual embedding has the wrong dimensionality (expected 512).
    InvalidVisualEmbeddingDim { expected: usize, actual: usize },
    /// The timestamp range is invalid (start >= end).
    InvalidTimestampRange { start: u64, end: u64 },
}

impl std::fmt::Display for MultimodalValidationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MissingField { media_type, field } => {
                write!(f, "{media_type:?} node requires field: {field}")
            }
            Self::InvalidVisualEmbeddingDim { expected, actual } => {
                write!(
                    f,
                    "visual embedding dimension mismatch: expected {expected}, got {actual}"
                )
            }
            Self::InvalidTimestampRange { start, end } => {
                write!(f, "invalid timestamp range: start ({start}) >= end ({end})")
            }
        }
    }
}

impl std::error::Error for MultimodalValidationError {}

/// Builder for constructing multimodal memory nodes with per-type validation.
///
/// Use the convenience constructors ([`text`](Self::text), [`audio`](Self::audio),
/// [`image`](Self::image), [`video_segment`](Self::video_segment)) to start building
/// a node of a specific media type, then chain optional setters before calling
/// [`build`](Self::build).
pub struct MultimodalNodeBuilder {
    id: NodeId,
    media_type: MediaType,
    content: Option<String>,
    embedding: Option<Vec<f32>>,
    metadata: HashMap<String, Value>,
    node_type: NodeType,
    timestamp: u64,
    media_uri: Option<String>,
    embedding_visual: Option<Vec<f32>>,
    timestamp_range: Option<(u64, u64)>,
    parent_video_id: Option<NodeId>,
}

impl MultimodalNodeBuilder {
    /// Create a builder for a **Text** node.
    ///
    /// Text nodes require `content` and a 384-dim text `embedding`.
    pub fn text(id: NodeId, content: impl Into<String>, embedding: Vec<f32>) -> Self {
        Self {
            id,
            media_type: MediaType::Text,
            content: Some(content.into()),
            embedding: Some(embedding),
            metadata: HashMap::new(),
            node_type: NodeType::Entity,
            timestamp: 0,
            media_uri: None,
            embedding_visual: None,
            timestamp_range: None,
            parent_video_id: None,
        }
    }

    /// Create a builder for an **Audio** node.
    ///
    /// Audio nodes require `content` (transcript), a 384-dim text `embedding`,
    /// and `media_uri` pointing to the audio file.
    pub fn audio(
        id: NodeId,
        content: impl Into<String>,
        embedding: Vec<f32>,
        media_uri: impl Into<String>,
    ) -> Self {
        Self {
            id,
            media_type: MediaType::Audio,
            content: Some(content.into()),
            embedding: Some(embedding),
            metadata: HashMap::new(),
            node_type: NodeType::Entity,
            timestamp: 0,
            media_uri: Some(media_uri.into()),
            embedding_visual: None,
            timestamp_range: None,
            parent_video_id: None,
        }
    }

    /// Create a builder for an **Image** node.
    ///
    /// Image nodes require a 512-dim `embedding_visual` (CLIP) and `media_uri`.
    /// Optionally, `content` (description) and `embedding` (text) can be set.
    pub fn image(
        id: NodeId,
        media_uri: impl Into<String>,
        embedding_visual: Vec<f32>,
    ) -> Self {
        Self {
            id,
            media_type: MediaType::Image,
            content: None,
            embedding: None,
            metadata: HashMap::new(),
            node_type: NodeType::Entity,
            timestamp: 0,
            media_uri: Some(media_uri.into()),
            embedding_visual: Some(embedding_visual),
            timestamp_range: None,
            parent_video_id: None,
        }
    }

    /// Create a builder for a **VideoSegment** node.
    ///
    /// Video segment nodes require a 512-dim `embedding_visual`, `media_uri`,
    /// and `parent_video_id` linking to the parent video node.
    pub fn video_segment(
        id: NodeId,
        media_uri: impl Into<String>,
        embedding_visual: Vec<f32>,
        parent_video_id: NodeId,
    ) -> Self {
        Self {
            id,
            media_type: MediaType::VideoSegment,
            content: None,
            embedding: None,
            metadata: HashMap::new(),
            node_type: NodeType::Event,
            timestamp: 0,
            media_uri: Some(media_uri.into()),
            embedding_visual: Some(embedding_visual),
            timestamp_range: None,
            parent_video_id: Some(parent_video_id),
        }
    }

    /// Set the text content (optional for Image/VideoSegment, required for Text/Audio).
    pub fn content(mut self, content: String) -> Self {
        self.content = Some(content);
        self
    }

    /// Set the 384-dim text embedding (optional for Image/VideoSegment).
    pub fn embedding(mut self, embedding: Vec<f32>) -> Self {
        self.embedding = Some(embedding);
        self
    }

    /// Set the node type (defaults to Entity for most, Event for VideoSegment).
    pub fn node_type(mut self, node_type: NodeType) -> Self {
        self.node_type = node_type;
        self
    }

    /// Set the timestamp (defaults to 0).
    pub fn timestamp(mut self, timestamp: u64) -> Self {
        self.timestamp = timestamp;
        self
    }

    /// Set the media URI (optional for Text nodes).
    pub fn media_uri(mut self, uri: String) -> Self {
        self.media_uri = Some(uri);
        self
    }

    /// Set the 512-dim visual embedding (optional for Text/Audio nodes).
    pub fn embedding_visual(mut self, embedding: Vec<f32>) -> Self {
        self.embedding_visual = Some(embedding);
        self
    }

    /// Set the temporal range as (start_ms, end_ms).
    pub fn timestamp_range(mut self, start_ms: u64, end_ms: u64) -> Self {
        self.timestamp_range = Some((start_ms, end_ms));
        self
    }

    /// Add a metadata key-value pair.
    pub fn metadata(mut self, key: impl Into<String>, value: Value) -> Self {
        self.metadata.insert(key.into(), value);
        self
    }

    /// Validate and build the node.
    ///
    /// Returns `Err` if required fields for the media type are missing or invalid.
    pub fn build(self) -> Result<Node, MultimodalValidationError> {
        self.validate()?;

        Ok(Node {
            id: self.id,
            content: self.content.unwrap_or_default(),
            embedding: self.embedding.unwrap_or_default(),
            metadata: self.metadata,
            node_type: self.node_type,
            timestamp: self.timestamp,
            media_type: Some(self.media_type),
            media_uri: self.media_uri,
            embedding_visual: self.embedding_visual,
            timestamp_range: self.timestamp_range,
            parent_video_id: self.parent_video_id,
        })
    }

    /// Validate required fields based on media type.
    fn validate(&self) -> Result<(), MultimodalValidationError> {
        match self.media_type {
            MediaType::Text => {
                if self.content.is_none() {
                    return Err(MultimodalValidationError::MissingField {
                        media_type: MediaType::Text,
                        field: "content",
                    });
                }
                if self.embedding.is_none() {
                    return Err(MultimodalValidationError::MissingField {
                        media_type: MediaType::Text,
                        field: "embedding",
                    });
                }
            }
            MediaType::Audio => {
                if self.content.is_none() {
                    return Err(MultimodalValidationError::MissingField {
                        media_type: MediaType::Audio,
                        field: "content",
                    });
                }
                if self.embedding.is_none() {
                    return Err(MultimodalValidationError::MissingField {
                        media_type: MediaType::Audio,
                        field: "embedding",
                    });
                }
                if self.media_uri.is_none() {
                    return Err(MultimodalValidationError::MissingField {
                        media_type: MediaType::Audio,
                        field: "media_uri",
                    });
                }
            }
            MediaType::Image => {
                if self.embedding_visual.is_none() {
                    return Err(MultimodalValidationError::MissingField {
                        media_type: MediaType::Image,
                        field: "embedding_visual",
                    });
                }
                if self.media_uri.is_none() {
                    return Err(MultimodalValidationError::MissingField {
                        media_type: MediaType::Image,
                        field: "media_uri",
                    });
                }
            }
            MediaType::VideoSegment => {
                if self.embedding_visual.is_none() {
                    return Err(MultimodalValidationError::MissingField {
                        media_type: MediaType::VideoSegment,
                        field: "embedding_visual",
                    });
                }
                if self.media_uri.is_none() {
                    return Err(MultimodalValidationError::MissingField {
                        media_type: MediaType::VideoSegment,
                        field: "media_uri",
                    });
                }
                if self.parent_video_id.is_none() {
                    return Err(MultimodalValidationError::MissingField {
                        media_type: MediaType::VideoSegment,
                        field: "parent_video_id",
                    });
                }
            }
        }

        // Validate visual embedding dimension if present
        if let Some(ref vis) = self.embedding_visual {
            if vis.len() != 512 {
                return Err(MultimodalValidationError::InvalidVisualEmbeddingDim {
                    expected: 512,
                    actual: vis.len(),
                });
            }
        }

        // Validate timestamp range
        if let Some((start, end)) = self.timestamp_range {
            if start >= end {
                return Err(MultimodalValidationError::InvalidTimestampRange {
                    start,
                    end,
                });
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- Text node tests ---

    #[test]
    fn test_text_node_basic() {
        let node = MultimodalNodeBuilder::text(1, "Hello world", vec![0.1f32; 384])
            .timestamp(1_000_000)
            .build()
            .unwrap();

        assert_eq!(node.id, 1);
        assert_eq!(node.content, "Hello world");
        assert_eq!(node.embedding.len(), 384);
        assert_eq!(node.media_type, Some(MediaType::Text));
        assert_eq!(node.node_type, NodeType::Entity);
        assert_eq!(node.timestamp, 1_000_000);
        assert!(node.media_uri.is_none());
        assert!(node.embedding_visual.is_none());
        assert!(node.timestamp_range.is_none());
        assert!(node.parent_video_id.is_none());
    }

    #[test]
    fn test_text_node_with_optional_fields() {
        let node = MultimodalNodeBuilder::text(2, "Note", vec![0.1f32; 384])
            .node_type(NodeType::Fact)
            .media_uri("file:///notes/note1.txt".to_string())
            .metadata("source", Value::String("user_input".to_string()))
            .build()
            .unwrap();

        assert_eq!(node.node_type, NodeType::Fact);
        assert_eq!(
            node.media_uri,
            Some("file:///notes/note1.txt".to_string())
        );
        assert_eq!(
            node.metadata.get("source"),
            Some(&Value::String("user_input".to_string()))
        );
    }

    // --- Audio node tests ---

    #[test]
    fn test_audio_node_basic() {
        let node = MultimodalNodeBuilder::audio(
            10,
            "Transcript of meeting",
            vec![0.2f32; 384],
            "file:///audio/meeting.wav",
        )
        .timestamp(2_000_000)
        .build()
        .unwrap();

        assert_eq!(node.id, 10);
        assert_eq!(node.content, "Transcript of meeting");
        assert_eq!(node.embedding.len(), 384);
        assert_eq!(node.media_type, Some(MediaType::Audio));
        assert_eq!(
            node.media_uri,
            Some("file:///audio/meeting.wav".to_string())
        );
    }

    #[test]
    fn test_audio_node_with_timestamp_range() {
        let node = MultimodalNodeBuilder::audio(
            11,
            "Segment transcript",
            vec![0.2f32; 384],
            "file:///audio/podcast.mp3",
        )
        .timestamp_range(30_000, 60_000)
        .build()
        .unwrap();

        assert_eq!(node.timestamp_range, Some((30_000, 60_000)));
    }

    #[test]
    fn test_audio_node_missing_media_uri() {
        // Construct manually to bypass the constructor
        let builder = MultimodalNodeBuilder {
            id: 12,
            media_type: MediaType::Audio,
            content: Some("transcript".to_string()),
            embedding: Some(vec![0.1f32; 384]),
            metadata: HashMap::new(),
            node_type: NodeType::Entity,
            timestamp: 0,
            media_uri: None, // missing!
            embedding_visual: None,
            timestamp_range: None,
            parent_video_id: None,
        };
        let err = builder.build().unwrap_err();
        assert_eq!(
            err,
            MultimodalValidationError::MissingField {
                media_type: MediaType::Audio,
                field: "media_uri"
            }
        );
    }

    // --- Image node tests ---

    #[test]
    fn test_image_node_basic() {
        let node =
            MultimodalNodeBuilder::image(20, "file:///images/photo.jpg", vec![0.3f32; 512])
                .timestamp(3_000_000)
                .build()
                .unwrap();

        assert_eq!(node.id, 20);
        assert_eq!(node.content, ""); // no content by default
        assert!(node.embedding.is_empty()); // no text embedding by default
        assert_eq!(node.media_type, Some(MediaType::Image));
        assert_eq!(
            node.media_uri,
            Some("file:///images/photo.jpg".to_string())
        );
        assert_eq!(node.embedding_visual.as_ref().unwrap().len(), 512);
    }

    #[test]
    fn test_image_node_with_description() {
        let node =
            MultimodalNodeBuilder::image(21, "file:///images/sunset.png", vec![0.3f32; 512])
                .content("A beautiful sunset over the ocean".to_string())
                .embedding(vec![0.1f32; 384])
                .build()
                .unwrap();

        assert_eq!(node.content, "A beautiful sunset over the ocean");
        assert_eq!(node.embedding.len(), 384);
        assert_eq!(node.embedding_visual.as_ref().unwrap().len(), 512);
    }

    #[test]
    fn test_image_node_invalid_visual_dim() {
        let err =
            MultimodalNodeBuilder::image(22, "file:///img.jpg", vec![0.3f32; 256]) // wrong dim!
                .build()
                .unwrap_err();
        assert_eq!(
            err,
            MultimodalValidationError::InvalidVisualEmbeddingDim {
                expected: 512,
                actual: 256
            }
        );
    }

    // --- VideoSegment node tests ---

    #[test]
    fn test_video_segment_basic() {
        let node = MultimodalNodeBuilder::video_segment(
            30,
            "file:///video/clip.mp4",
            vec![0.4f32; 512],
            1000, // parent_video_id
        )
        .timestamp(4_000_000)
        .build()
        .unwrap();

        assert_eq!(node.id, 30);
        assert_eq!(node.media_type, Some(MediaType::VideoSegment));
        assert_eq!(node.node_type, NodeType::Event);
        assert_eq!(
            node.media_uri,
            Some("file:///video/clip.mp4".to_string())
        );
        assert_eq!(node.embedding_visual.as_ref().unwrap().len(), 512);
        assert_eq!(node.parent_video_id, Some(1000));
    }

    #[test]
    fn test_video_segment_with_transcript() {
        let node = MultimodalNodeBuilder::video_segment(
            31,
            "file:///video/lecture.mp4",
            vec![0.4f32; 512],
            1000,
        )
        .content("Speaker discusses neural networks".to_string())
        .embedding(vec![0.1f32; 384])
        .timestamp_range(0, 15_000)
        .build()
        .unwrap();

        assert_eq!(node.content, "Speaker discusses neural networks");
        assert_eq!(node.embedding.len(), 384);
        assert_eq!(node.timestamp_range, Some((0, 15_000)));
    }

    #[test]
    fn test_video_segment_missing_parent() {
        let builder = MultimodalNodeBuilder {
            id: 32,
            media_type: MediaType::VideoSegment,
            content: None,
            embedding: None,
            metadata: HashMap::new(),
            node_type: NodeType::Event,
            timestamp: 0,
            media_uri: Some("file:///video.mp4".to_string()),
            embedding_visual: Some(vec![0.4f32; 512]),
            timestamp_range: None,
            parent_video_id: None, // missing!
        };
        let err = builder.build().unwrap_err();
        assert_eq!(
            err,
            MultimodalValidationError::MissingField {
                media_type: MediaType::VideoSegment,
                field: "parent_video_id"
            }
        );
    }

    // --- Cross-cutting validation tests ---

    #[test]
    fn test_invalid_timestamp_range() {
        let err = MultimodalNodeBuilder::audio(
            40,
            "transcript",
            vec![0.1f32; 384],
            "file:///audio.wav",
        )
        .timestamp_range(60_000, 30_000) // start > end
        .build()
        .unwrap_err();

        assert_eq!(
            err,
            MultimodalValidationError::InvalidTimestampRange {
                start: 60_000,
                end: 30_000
            }
        );
    }

    #[test]
    fn test_timestamp_range_equal_invalid() {
        let err = MultimodalNodeBuilder::audio(
            41,
            "transcript",
            vec![0.1f32; 384],
            "file:///audio.wav",
        )
        .timestamp_range(5_000, 5_000) // start == end
        .build()
        .unwrap_err();

        assert_eq!(
            err,
            MultimodalValidationError::InvalidTimestampRange {
                start: 5_000,
                end: 5_000
            }
        );
    }

    #[test]
    fn test_error_display() {
        let err = MultimodalValidationError::MissingField {
            media_type: MediaType::Image,
            field: "embedding_visual",
        };
        assert_eq!(
            err.to_string(),
            "Image node requires field: embedding_visual"
        );

        let err = MultimodalValidationError::InvalidVisualEmbeddingDim {
            expected: 512,
            actual: 384,
        };
        assert_eq!(
            err.to_string(),
            "visual embedding dimension mismatch: expected 512, got 384"
        );

        let err = MultimodalValidationError::InvalidTimestampRange {
            start: 100,
            end: 50,
        };
        assert_eq!(
            err.to_string(),
            "invalid timestamp range: start (100) >= end (50)"
        );
    }

    #[test]
    fn test_metadata_chaining() {
        let node = MultimodalNodeBuilder::text(50, "test", vec![0.1f32; 384])
            .metadata("key1", Value::String("val1".to_string()))
            .metadata("key2", Value::Integer(42))
            .metadata("key3", Value::Bool(true))
            .build()
            .unwrap();

        assert_eq!(node.metadata.len(), 3);
        assert_eq!(
            node.metadata.get("key1"),
            Some(&Value::String("val1".to_string()))
        );
        assert_eq!(node.metadata.get("key2"), Some(&Value::Integer(42)));
        assert_eq!(node.metadata.get("key3"), Some(&Value::Bool(true)));
    }
}
