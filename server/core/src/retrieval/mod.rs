//! Retrieval scoring and ranking modules.
//!
//! Provides mindset-aware scoring for retrieval candidates via
//! [`MindsetScorer`], which adjusts ranking weights based on the
//! cognitive processing mode ([`MindsetTag`](crate::MindsetTag)).

mod mindset_scorer;

pub use mindset_scorer::{MindsetDetector, MindsetKeyword, MindsetScorer, MindsetWeights};
