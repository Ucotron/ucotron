//! Mindset-aware scoring for retrieval results.
//!
//! Scores retrieval candidates based on cognitive processing mode (Chain of Mindset).
//! Each [`MindsetTag`] represents a different retrieval strategy:
//!
//! - **Convergent**: Prefers consensus, high-confidence, well-connected nodes.
//!   Supports graph-aware scoring via [`MindsetScorer::score_convergent`] which
//!   boosts memories with multiple independent corroborating paths.
//! - **Divergent**: Prefers diversity, alternative viewpoints, and contradictions.
//!   Supports graph-aware scoring via [`MindsetScorer::score_divergent`] which
//!   boosts memories with rare predicate combinations and unique connections.
//! - **Algorithmic**: Prefers precision, recency, and strict logical consistency.
//!   Supports graph-aware scoring via [`MindsetScorer::score_algorithmic`] which
//!   boosts verified, contradiction-free memories.
//!
//! The scorer adjusts retrieval rankings by weighting confidence, recency, diversity,
//! and connectivity differently based on the active mindset context.

use crate::types::{Fact, MindsetTag, Node, ResolutionState};

/// Configuration weights for a single mindset mode.
///
/// Each weight controls how much a particular signal contributes to the
/// final score under this mindset. Weights are normalized during scoring
/// so their absolute values define relative importance.
#[derive(Debug, Clone)]
pub struct MindsetWeights {
    /// Weight for source confidence (higher = prefer confident facts).
    pub confidence: f32,
    /// Weight for temporal recency (higher = prefer recent facts).
    pub recency: f32,
    /// Weight for viewpoint diversity (higher = prefer contradictions/alternatives).
    pub diversity: f32,
    /// Weight for graph connectivity (higher = prefer well-connected nodes).
    pub connectivity: f32,
}

/// Scores retrieval candidates based on the active cognitive mindset.
///
/// Different retrieval contexts require different scoring strategies. A convergent
/// context (summarization, fact-checking) should prefer high-confidence consensus.
/// A divergent context (brainstorming, exploration) should surface alternatives.
/// An algorithmic context (verification, computation) should prefer precise, recent data.
///
/// # Example
///
/// ```
/// use ucotron_core::retrieval::{MindsetScorer, MindsetWeights};
/// use ucotron_core::MindsetTag;
///
/// let scorer = MindsetScorer::default();
///
/// // Score a high-confidence, recent fact under convergent mindset
/// let score = scorer.score(
///     MindsetTag::Convergent,
///     0.95,  // confidence
///     0.8,   // recency (normalized 0-1)
///     0.1,   // diversity (low = consensus)
///     0.7,   // connectivity
/// );
/// assert!(score > 0.5);
/// ```
#[derive(Debug, Clone)]
pub struct MindsetScorer {
    /// Weights applied when the mindset is Convergent.
    pub convergent: MindsetWeights,
    /// Weights applied when the mindset is Divergent.
    pub divergent: MindsetWeights,
    /// Weights applied when the mindset is Algorithmic.
    pub algorithmic: MindsetWeights,
    /// Weights applied when the mindset is Spatial (graph traversal optimization).
    pub spatial: MindsetWeights,
}

impl Default for MindsetScorer {
    fn default() -> Self {
        Self {
            convergent: MindsetWeights {
                confidence: 0.5,
                recency: 0.2,
                diversity: 0.0,
                connectivity: 0.3,
            },
            divergent: MindsetWeights {
                confidence: 0.1,
                recency: 0.1,
                diversity: 0.6,
                connectivity: 0.2,
            },
            algorithmic: MindsetWeights {
                confidence: 0.3,
                recency: 0.4,
                diversity: 0.0,
                connectivity: 0.3,
            },
            spatial: MindsetWeights {
                confidence: 0.2,
                recency: 0.1,
                diversity: 0.2,
                connectivity: 0.5,
            },
        }
    }
}

impl MindsetScorer {
    /// Creates a scorer with custom weights for each mindset.
    pub fn new(
        convergent: MindsetWeights,
        divergent: MindsetWeights,
        algorithmic: MindsetWeights,
        spatial: MindsetWeights,
    ) -> Self {
        Self {
            convergent,
            divergent,
            algorithmic,
            spatial,
        }
    }

    /// Returns the weight configuration for the given mindset.
    pub fn weights_for(&self, tag: MindsetTag) -> &MindsetWeights {
        match tag {
            MindsetTag::Convergent => &self.convergent,
            MindsetTag::Divergent => &self.divergent,
            MindsetTag::Algorithmic => &self.algorithmic,
            MindsetTag::Spatial => &self.spatial,
        }
    }

    /// Computes a weighted score for a retrieval candidate under the given mindset.
    ///
    /// All input signals should be normalized to the range [0.0, 1.0]:
    /// - `confidence`: Source confidence of the fact (from `Fact::source_confidence`).
    /// - `recency`: Temporal recency, where 1.0 = most recent, 0.0 = oldest.
    /// - `diversity`: Degree of divergence from consensus (1.0 = highly contradictory).
    /// - `connectivity`: Normalized graph degree (1.0 = hub node, 0.0 = isolated).
    ///
    /// Returns a score in [0.0, 1.0].
    pub fn score(
        &self,
        tag: MindsetTag,
        confidence: f32,
        recency: f32,
        diversity: f32,
        connectivity: f32,
    ) -> f32 {
        let w = self.weights_for(tag);
        let weight_sum = w.confidence + w.recency + w.diversity + w.connectivity;

        if weight_sum == 0.0 {
            return 0.0;
        }

        let raw = w.confidence * confidence.clamp(0.0, 1.0)
            + w.recency * recency.clamp(0.0, 1.0)
            + w.diversity * diversity.clamp(0.0, 1.0)
            + w.connectivity * connectivity.clamp(0.0, 1.0);

        (raw / weight_sum).clamp(0.0, 1.0)
    }

    /// Scores a [`Fact`] under the given mindset context.
    ///
    /// Derives signals from the fact's fields:
    /// - `confidence` from `fact.source_confidence`
    /// - `recency` from normalized timestamp (requires `now` and `time_range` for normalization)
    /// - `diversity` from resolution state (contradictions/ambiguous score higher)
    /// - `connectivity` must be provided externally (graph degree info)
    pub fn score_fact(
        &self,
        tag: MindsetTag,
        fact: &Fact,
        now: u64,
        time_range: u64,
        connectivity: f32,
    ) -> f32 {
        let confidence = fact.source_confidence;

        // Recency: 1.0 for now, 0.0 for oldest
        let recency = if time_range > 0 {
            let age = now.saturating_sub(fact.timestamp);
            1.0 - (age as f32 / time_range as f32).clamp(0.0, 1.0)
        } else {
            1.0
        };

        // Diversity: contradictions and ambiguous states indicate alternative viewpoints
        let diversity = match fact.resolution_state {
            ResolutionState::Accepted => 0.0,
            ResolutionState::Superseded => 0.2,
            ResolutionState::Contradiction => 0.8,
            ResolutionState::Ambiguous => 1.0,
        };

        self.score(tag, confidence, recency, diversity, connectivity)
    }

    /// Scores a [`Node`] under the given mindset with minimal context.
    ///
    /// Uses the node's timestamp for recency and requires external connectivity.
    /// Since nodes don't carry confidence or resolution state, those default to neutral.
    pub fn score_node(
        &self,
        tag: MindsetTag,
        node: &Node,
        now: u64,
        time_range: u64,
        connectivity: f32,
    ) -> f32 {
        let confidence = 0.5; // Neutral — nodes don't have confidence

        let recency = if time_range > 0 {
            let age = now.saturating_sub(node.timestamp);
            1.0 - (age as f32 / time_range as f32).clamp(0.0, 1.0)
        } else {
            1.0
        };

        let diversity = 0.0; // Neutral — nodes don't have resolution state

        self.score(tag, confidence, recency, diversity, connectivity)
    }

    // --- Graph-aware mindset scoring methods ---

    /// Scores a memory under **Convergent** mindset using graph evidence.
    ///
    /// Convergent scoring boosts memories that have multiple independent
    /// corroborating paths. The more independent paths point to or support
    /// a memory, the stronger its consensus signal.
    ///
    /// - `base_confidence`: The base confidence of the memory (0.0-1.0).
    /// - `supporting_path_count`: Number of independent paths that corroborate this memory.
    /// - `recency`: Normalized temporal recency (0.0=oldest, 1.0=most recent).
    /// - `connectivity`: Normalized graph connectivity (0.0=isolated, 1.0=hub).
    ///
    /// The path count is converted to a corroboration signal using a saturating
    /// logarithmic curve: `min(1.0, ln(1 + count) / ln(1 + saturation))`.
    /// By default, saturation occurs at 5 paths (5+ paths ≈ 1.0 corroboration).
    ///
    /// Returns a score in [0.0, 1.0] where higher = stronger convergent evidence.
    pub fn score_convergent(
        &self,
        base_confidence: f32,
        supporting_path_count: usize,
        recency: f32,
        connectivity: f32,
    ) -> f32 {
        // Convert path count to a saturating corroboration signal
        let corroboration = Self::path_count_to_signal(supporting_path_count, 5);

        // Blend corroboration into confidence: more paths = boosted confidence
        let boosted_confidence = (base_confidence * 0.6 + corroboration * 0.4).clamp(0.0, 1.0);

        self.score(
            MindsetTag::Convergent,
            boosted_confidence,
            recency,
            0.0,
            connectivity,
        )
    }

    /// Scores a [`Fact`] under **Convergent** mindset with graph corroboration.
    ///
    /// Combines the fact's inherent confidence with the number of independent
    /// supporting paths in the graph. Facts corroborated by multiple sources
    /// receive boosted scores.
    pub fn score_fact_convergent(
        &self,
        fact: &Fact,
        supporting_path_count: usize,
        now: u64,
        time_range: u64,
        connectivity: f32,
    ) -> f32 {
        let recency = if time_range > 0 {
            let age = now.saturating_sub(fact.timestamp);
            1.0 - (age as f32 / time_range as f32).clamp(0.0, 1.0)
        } else {
            1.0
        };

        self.score_convergent(
            fact.source_confidence,
            supporting_path_count,
            recency,
            connectivity,
        )
    }

    /// Scores a memory under **Divergent** mindset using graph novelty.
    ///
    /// Divergent scoring boosts memories that have unusual, rare, or unexpected
    /// connections. Memories connected via rare predicate types or with few
    /// corroborating paths (indicating uniqueness) score higher.
    ///
    /// - `base_confidence`: The base confidence of the memory (0.0-1.0).
    /// - `predicate_rarity`: How rare the connecting predicate is (0.0=common, 1.0=unique).
    ///   Computed externally as `1.0 - (predicate_frequency / max_frequency)`.
    /// - `supporting_path_count`: Number of supporting paths (fewer = more novel).
    /// - `recency`: Normalized temporal recency.
    /// - `connectivity`: Normalized graph connectivity.
    ///
    /// Returns a score in [0.0, 1.0] where higher = more novel/divergent.
    pub fn score_divergent(
        &self,
        base_confidence: f32,
        predicate_rarity: f32,
        supporting_path_count: usize,
        recency: f32,
        connectivity: f32,
    ) -> f32 {
        // Novelty is inversely related to corroboration: fewer paths = more unique
        let corroboration = Self::path_count_to_signal(supporting_path_count, 5);
        let uniqueness = 1.0 - corroboration;

        // Diversity signal combines predicate rarity and path uniqueness
        let diversity = (predicate_rarity * 0.6 + uniqueness * 0.4).clamp(0.0, 1.0);

        self.score(
            MindsetTag::Divergent,
            base_confidence,
            recency,
            diversity,
            connectivity,
        )
    }

    /// Scores a [`Fact`] under **Divergent** mindset with graph novelty.
    pub fn score_fact_divergent(
        &self,
        fact: &Fact,
        predicate_rarity: f32,
        supporting_path_count: usize,
        now: u64,
        time_range: u64,
        connectivity: f32,
    ) -> f32 {
        let recency = if time_range > 0 {
            let age = now.saturating_sub(fact.timestamp);
            1.0 - (age as f32 / time_range as f32).clamp(0.0, 1.0)
        } else {
            1.0
        };

        self.score_divergent(
            fact.source_confidence,
            predicate_rarity,
            supporting_path_count,
            recency,
            connectivity,
        )
    }

    /// Scores a memory under **Algorithmic** mindset using verification status.
    ///
    /// Algorithmic scoring boosts memories that are verified and contradiction-free.
    /// The resolution state directly affects the score: Accepted facts score highest,
    /// while Contradictions score lowest.
    ///
    /// - `base_confidence`: The base confidence of the memory (0.0-1.0).
    /// - `resolution_state`: Current state in the conflict resolution lifecycle.
    /// - `contradiction_count`: Number of contradicting facts for this memory.
    /// - `recency`: Normalized temporal recency.
    /// - `connectivity`: Normalized graph connectivity.
    ///
    /// Returns a score in [0.0, 1.0] where higher = more verified/reliable.
    pub fn score_algorithmic(
        &self,
        base_confidence: f32,
        resolution_state: ResolutionState,
        contradiction_count: usize,
        recency: f32,
        connectivity: f32,
    ) -> f32 {
        // Resolution state maps to a verification signal
        let verification = match resolution_state {
            ResolutionState::Accepted => 1.0,
            ResolutionState::Superseded => 0.4,
            ResolutionState::Ambiguous => 0.2,
            ResolutionState::Contradiction => 0.0,
        };

        // Contradiction-free bonus: penalize memories with contradictions
        let contradiction_penalty = Self::path_count_to_signal(contradiction_count, 3);
        let clean_bonus = 1.0 - contradiction_penalty;

        // Boost confidence based on verification and clean status
        let boosted_confidence =
            (base_confidence * 0.4 + verification * 0.4 + clean_bonus * 0.2).clamp(0.0, 1.0);

        // Diversity is 0 for algorithmic (we want verified facts, not alternatives)
        self.score(
            MindsetTag::Algorithmic,
            boosted_confidence,
            recency,
            0.0,
            connectivity,
        )
    }

    /// Scores a [`Fact`] under **Algorithmic** mindset with verification status.
    pub fn score_fact_algorithmic(
        &self,
        fact: &Fact,
        contradiction_count: usize,
        now: u64,
        time_range: u64,
        connectivity: f32,
    ) -> f32 {
        let recency = if time_range > 0 {
            let age = now.saturating_sub(fact.timestamp);
            1.0 - (age as f32 / time_range as f32).clamp(0.0, 1.0)
        } else {
            1.0
        };

        self.score_algorithmic(
            fact.source_confidence,
            fact.resolution_state,
            contradiction_count,
            recency,
            connectivity,
        )
    }

    /// Scores a memory under **Spatial** mindset using graph traversal optimization.
    ///
    /// Spatial scoring optimizes for path-based reasoning and graph traversal. It boosts
    /// nodes that serve as bridges between communities, hub nodes with high connectivity,
    /// and paths with high centrality. This mindset is ideal for queries about:
    /// - "How is X connected to Y?"
    /// - "What are the shortest paths between entities?"
    /// - "Which entities bridge multiple communities?"
    ///
    /// - `base_confidence`: The base confidence of the memory (0.0-1.0).
    /// - `path_length`: Number of hops from source (shorter = higher score).
    /// - `betweenness_centrality`: Normalized betweenness centrality (0.0-1.0).
    /// - `hub_score`: Hub centrality score (0.0-1.0), higher = more influential.
    /// - `connectivity`: Normalized graph connectivity (0.0-1.0).
    ///
    /// Returns a score in [0.0, 1.0] where higher = better for path-based reasoning.
    pub fn score_spatial(
        &self,
        base_confidence: f32,
        path_length: usize,
        betweenness_centrality: f32,
        hub_score: f32,
        connectivity: f32,
    ) -> f32 {
        // Shorter paths score higher (inverse relationship)
        let path_score = if path_length == 0 {
            1.0
        } else {
            (1.0 / path_length as f32).clamp(0.0, 1.0)
        };

        // Combine centrality metrics: betweenness and hub score indicate traversal importance
        let centrality_score = (betweenness_centrality * 0.5 + hub_score * 0.5).clamp(0.0, 1.0);

        // Hub nodes and bridge nodes are valuable for traversal
        let traversal_value =
            (connectivity * 0.4 + centrality_score * 0.4 + path_score * 0.2).clamp(0.0, 1.0);

        // Spatial prioritizes connectivity over confidence
        let adjusted_confidence = (base_confidence * 0.4 + traversal_value * 0.6).clamp(0.0, 1.0);

        self.score(
            MindsetTag::Spatial,
            adjusted_confidence,
            path_score,
            centrality_score,
            connectivity,
        )
    }

    /// Scores a [`Fact`] under **Spatial** mindset with graph traversal metrics.
    pub fn score_fact_spatial(
        &self,
        fact: &Fact,
        path_length: usize,
        betweenness_centrality: f32,
        hub_score: f32,
        connectivity: f32,
    ) -> f32 {
        self.score_spatial(
            fact.source_confidence,
            path_length,
            betweenness_centrality,
            hub_score,
            connectivity,
        )
    }

    /// Converts a count (e.g., path count, contradiction count) to a
    /// saturating signal in [0.0, 1.0] using a logarithmic curve.
    ///
    /// `saturation_point` defines the count at which the signal reaches ~1.0.
    /// For example, `path_count_to_signal(5, 5) ≈ 1.0`.
    fn path_count_to_signal(count: usize, saturation_point: usize) -> f32 {
        if saturation_point == 0 || count == 0 {
            return 0.0;
        }
        let signal = (1.0 + count as f64).ln() / (1.0 + saturation_point as f64).ln();
        (signal as f32).clamp(0.0, 1.0)
    }
}

// ---------------------------------------------------------------------------
// Mindset Auto-Detection
// ---------------------------------------------------------------------------

/// Keyword-to-mindset mapping entry.
#[derive(Debug, Clone)]
pub struct MindsetKeyword {
    /// The keyword or phrase to match (case-insensitive).
    pub pattern: String,
    /// The mindset tag this keyword maps to.
    pub mindset: MindsetTag,
}

/// Automatic mindset detection from query text using configurable keyword patterns.
///
/// Scans query text for keyword patterns that indicate the user's cognitive intent,
/// then returns the matching [`MindsetTag`]. If no keywords match, returns `None`.
///
/// # Default Keywords
///
/// | Keywords | Mindset |
/// |----------|---------|
/// | verify, confirm, check, validate, prove, correct | Algorithmic |
/// | explore, what if, brainstorm, alternative, imagine, creative | Divergent |
/// | summarize, consensus, agree, common, overview, conclude | Convergent |
///
/// # Example
///
/// ```
/// use ucotron_core::retrieval::MindsetDetector;
/// use ucotron_core::MindsetTag;
///
/// let detector = MindsetDetector::default();
/// assert_eq!(detector.detect("Can you verify this claim?"), Some(MindsetTag::Algorithmic));
/// assert_eq!(detector.detect("What if we tried something different?"), Some(MindsetTag::Divergent));
/// assert_eq!(detector.detect("Please summarize the discussion"), Some(MindsetTag::Convergent));
/// assert_eq!(detector.detect("Tell me about cats"), None);
/// ```
#[derive(Debug, Clone)]
pub struct MindsetDetector {
    /// Ordered list of keyword patterns. Earlier entries take priority.
    keywords: Vec<MindsetKeyword>,
}

impl Default for MindsetDetector {
    fn default() -> Self {
        Self {
            keywords: vec![
                // Algorithmic — verification and logical checking
                MindsetKeyword {
                    pattern: "verify".into(),
                    mindset: MindsetTag::Algorithmic,
                },
                MindsetKeyword {
                    pattern: "confirm".into(),
                    mindset: MindsetTag::Algorithmic,
                },
                MindsetKeyword {
                    pattern: "check".into(),
                    mindset: MindsetTag::Algorithmic,
                },
                MindsetKeyword {
                    pattern: "validate".into(),
                    mindset: MindsetTag::Algorithmic,
                },
                MindsetKeyword {
                    pattern: "prove".into(),
                    mindset: MindsetTag::Algorithmic,
                },
                MindsetKeyword {
                    pattern: "correct".into(),
                    mindset: MindsetTag::Algorithmic,
                },
                // Divergent — exploration and creative thinking
                MindsetKeyword {
                    pattern: "what if".into(),
                    mindset: MindsetTag::Divergent,
                },
                MindsetKeyword {
                    pattern: "explore".into(),
                    mindset: MindsetTag::Divergent,
                },
                MindsetKeyword {
                    pattern: "brainstorm".into(),
                    mindset: MindsetTag::Divergent,
                },
                MindsetKeyword {
                    pattern: "alternative".into(),
                    mindset: MindsetTag::Divergent,
                },
                MindsetKeyword {
                    pattern: "imagine".into(),
                    mindset: MindsetTag::Divergent,
                },
                MindsetKeyword {
                    pattern: "creative".into(),
                    mindset: MindsetTag::Divergent,
                },
                // Convergent — synthesis and consensus
                MindsetKeyword {
                    pattern: "summarize".into(),
                    mindset: MindsetTag::Convergent,
                },
                MindsetKeyword {
                    pattern: "consensus".into(),
                    mindset: MindsetTag::Convergent,
                },
                MindsetKeyword {
                    pattern: "agree".into(),
                    mindset: MindsetTag::Convergent,
                },
                MindsetKeyword {
                    pattern: "common".into(),
                    mindset: MindsetTag::Convergent,
                },
                MindsetKeyword {
                    pattern: "overview".into(),
                    mindset: MindsetTag::Convergent,
                },
                MindsetKeyword {
                    pattern: "conclude".into(),
                    mindset: MindsetTag::Convergent,
                },
                // Spatial — graph traversal and path-based reasoning
                MindsetKeyword {
                    pattern: "connected".into(),
                    mindset: MindsetTag::Spatial,
                },
                MindsetKeyword {
                    pattern: "path".into(),
                    mindset: MindsetTag::Spatial,
                },
                MindsetKeyword {
                    pattern: "route".into(),
                    mindset: MindsetTag::Spatial,
                },
                MindsetKeyword {
                    pattern: "bridge".into(),
                    mindset: MindsetTag::Spatial,
                },
                MindsetKeyword {
                    pattern: "relationship".into(),
                    mindset: MindsetTag::Spatial,
                },
                MindsetKeyword {
                    pattern: "link".into(),
                    mindset: MindsetTag::Spatial,
                },
                MindsetKeyword {
                    pattern: "network".into(),
                    mindset: MindsetTag::Spatial,
                },
                MindsetKeyword {
                    pattern: "graph".into(),
                    mindset: MindsetTag::Spatial,
                },
            ],
        }
    }
}

impl MindsetDetector {
    /// Creates a detector with custom keyword mappings.
    ///
    /// The keywords list is ordered by priority — the first matching keyword
    /// determines the result.
    pub fn new(keywords: Vec<MindsetKeyword>) -> Self {
        Self { keywords }
    }

    /// Creates a detector from parallel lists of keywords per mindset.
    ///
    /// Keywords are ordered: Algorithmic first, then Divergent, then Convergent, then Spatial.
    pub fn from_keyword_lists(
        algorithmic: &[&str],
        divergent: &[&str],
        convergent: &[&str],
        spatial: &[&str],
    ) -> Self {
        let mut keywords = Vec::new();
        for &kw in algorithmic {
            keywords.push(MindsetKeyword {
                pattern: kw.to_lowercase(),
                mindset: MindsetTag::Algorithmic,
            });
        }
        for &kw in divergent {
            keywords.push(MindsetKeyword {
                pattern: kw.to_lowercase(),
                mindset: MindsetTag::Divergent,
            });
        }
        for &kw in convergent {
            keywords.push(MindsetKeyword {
                pattern: kw.to_lowercase(),
                mindset: MindsetTag::Convergent,
            });
        }
        for &kw in spatial {
            keywords.push(MindsetKeyword {
                pattern: kw.to_lowercase(),
                mindset: MindsetTag::Spatial,
            });
        }
        Self { keywords }
    }

    /// Analyzes query text to determine the most appropriate mindset.
    ///
    /// This method performs a more sophisticated analysis than simple keyword matching.
    /// It considers:
    /// - Explicit keywords (like `detect`)
    /// - Query structure patterns (questions, commands)
    /// - Entity relationships implied by the query
    ///
    /// Returns the most appropriate [`MindsetTag`] based on the analysis.
    pub fn analyze(&self, query: &str) -> Option<MindsetTag> {
        // First try explicit keyword matching
        let keyword_match = self.detect(query);
        if keyword_match.is_some() {
            return keyword_match;
        }

        // Analyze query structure for implicit intent
        let query_lower = query.to_lowercase();

        // Path/relationship queries suggest Spatial mindset
        if query_lower.contains("how")
            && (query_lower.contains("connect")
                || query_lower.contains("relate")
                || query_lower.contains("link to"))
        {
            return Some(MindsetTag::Spatial);
        }

        // "What's the difference" or comparison queries suggest Divergent
        if query_lower.contains("different")
            || query_lower.contains("compare")
            || query_lower.contains("versus")
            || query_lower.contains("vs ")
        {
            return Some(MindsetTag::Divergent);
        }

        // "Is it true" or "should I" queries suggest Algorithmic
        if query_lower.starts_with("is ")
            || query_lower.starts_with("are ")
            || query_lower.starts_with("should ")
            || query_lower.starts_with("does ")
        {
            return Some(MindsetTag::Algorithmic);
        }

        // "What is" / "Tell me about" with singular entities suggests Convergent
        if query_lower.starts_with("what is")
            || query_lower.starts_with("what's")
            || query_lower.starts_with("tell me about")
        {
            return Some(MindsetTag::Convergent);
        }

        None
    }

    /// Detects the cognitive mindset from query text.
    ///
    /// Performs case-insensitive word-boundary matching against the configured
    /// keyword patterns. Returns the mindset of the first matching keyword,
    /// or `None` if no keywords match.
    pub fn detect(&self, query: &str) -> Option<MindsetTag> {
        let query_lower = query.to_lowercase();
        for kw in &self.keywords {
            if contains_keyword(&query_lower, &kw.pattern) {
                return Some(kw.mindset);
            }
        }
        None
    }

    /// Returns a reference to the configured keyword list.
    pub fn keywords(&self) -> &[MindsetKeyword] {
        &self.keywords
    }
}

/// Checks if `text` contains `keyword` as a whole word or phrase.
///
/// Uses simple word-boundary detection: the keyword must be preceded and
/// followed by a non-alphanumeric character (or string boundary).
fn contains_keyword(text: &str, keyword: &str) -> bool {
    let mut start = 0;
    while let Some(pos) = text[start..].find(keyword) {
        let abs_pos = start + pos;
        let end_pos = abs_pos + keyword.len();

        // Check left boundary: start of string or non-alphanumeric
        let left_ok = abs_pos == 0 || !text.as_bytes()[abs_pos - 1].is_ascii_alphanumeric();

        // Check right boundary: end of string or non-alphanumeric
        let right_ok = end_pos >= text.len() || !text.as_bytes()[end_pos].is_ascii_alphanumeric();

        if left_ok && right_ok {
            return true;
        }

        // Move past current match to avoid infinite loop
        start = abs_pos + 1;
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{FactId, NodeType};
    use std::collections::HashMap;

    fn make_fact(confidence: f32, timestamp: u64, state: ResolutionState) -> Fact {
        Fact {
            id: 1 as FactId,
            subject: 1,
            predicate: "lives_in".to_string(),
            object: "Madrid".to_string(),
            source_confidence: confidence,
            timestamp,
            mindset_tag: MindsetTag::Convergent,
            resolution_state: state,
        }
    }

    fn make_node(timestamp: u64) -> Node {
        Node {
            id: 1,
            content: "test".to_string(),
            embedding: vec![0.0; 384],
            metadata: HashMap::new(),
            node_type: NodeType::Entity,
            timestamp,
            media_type: None,
            media_uri: None,
            embedding_visual: None,
            timestamp_range: None,
            parent_video_id: None,
        }
    }

    #[test]
    fn test_default_weights() {
        let scorer = MindsetScorer::default();
        // Convergent prioritizes confidence
        assert_eq!(scorer.convergent.confidence, 0.5);
        assert_eq!(scorer.convergent.diversity, 0.0);
        // Divergent prioritizes diversity
        assert_eq!(scorer.divergent.diversity, 0.6);
        assert_eq!(scorer.divergent.confidence, 0.1);
        // Algorithmic prioritizes recency
        assert_eq!(scorer.algorithmic.recency, 0.4);
        assert_eq!(scorer.algorithmic.diversity, 0.0);
    }

    #[test]
    fn test_convergent_prefers_high_confidence() {
        let scorer = MindsetScorer::default();
        let high_conf = scorer.score(MindsetTag::Convergent, 0.95, 0.5, 0.1, 0.5);
        let low_conf = scorer.score(MindsetTag::Convergent, 0.2, 0.5, 0.1, 0.5);
        assert!(
            high_conf > low_conf,
            "Convergent should prefer high confidence"
        );
    }

    #[test]
    fn test_divergent_prefers_diversity() {
        let scorer = MindsetScorer::default();
        let high_div = scorer.score(MindsetTag::Divergent, 0.5, 0.5, 0.9, 0.5);
        let low_div = scorer.score(MindsetTag::Divergent, 0.5, 0.5, 0.1, 0.5);
        assert!(high_div > low_div, "Divergent should prefer high diversity");
    }

    #[test]
    fn test_algorithmic_prefers_recency() {
        let scorer = MindsetScorer::default();
        let recent = scorer.score(MindsetTag::Algorithmic, 0.5, 0.95, 0.0, 0.5);
        let old = scorer.score(MindsetTag::Algorithmic, 0.5, 0.1, 0.0, 0.5);
        assert!(recent > old, "Algorithmic should prefer recent facts");
    }

    #[test]
    fn test_score_range_zero_to_one() {
        let scorer = MindsetScorer::default();
        for tag in [
            MindsetTag::Convergent,
            MindsetTag::Divergent,
            MindsetTag::Algorithmic,
        ] {
            let min = scorer.score(tag, 0.0, 0.0, 0.0, 0.0);
            let max = scorer.score(tag, 1.0, 1.0, 1.0, 1.0);
            assert!(min >= 0.0 && min <= 1.0, "Min score out of range: {min}");
            assert!(max >= 0.0 && max <= 1.0, "Max score out of range: {max}");
        }
    }

    #[test]
    fn test_score_fact_uses_confidence() {
        let scorer = MindsetScorer::default();
        let now = 1_000_000u64;
        let time_range = 500_000u64;

        let high = make_fact(0.95, now, ResolutionState::Accepted);
        let low = make_fact(0.2, now, ResolutionState::Accepted);

        let s_high = scorer.score_fact(MindsetTag::Convergent, &high, now, time_range, 0.5);
        let s_low = scorer.score_fact(MindsetTag::Convergent, &low, now, time_range, 0.5);

        assert!(s_high > s_low);
    }

    #[test]
    fn test_score_fact_recency() {
        let scorer = MindsetScorer::default();
        let now = 1_000_000u64;
        let time_range = 500_000u64;

        let recent = make_fact(0.5, now, ResolutionState::Accepted);
        let old = make_fact(0.5, now - time_range, ResolutionState::Accepted);

        let s_recent = scorer.score_fact(MindsetTag::Algorithmic, &recent, now, time_range, 0.5);
        let s_old = scorer.score_fact(MindsetTag::Algorithmic, &old, now, time_range, 0.5);

        assert!(
            s_recent > s_old,
            "Recent facts should score higher under Algorithmic"
        );
    }

    #[test]
    fn test_score_fact_diversity_from_resolution_state() {
        let scorer = MindsetScorer::default();
        let now = 1_000_000u64;
        let time_range = 500_000u64;

        let accepted = make_fact(0.5, now, ResolutionState::Accepted);
        let contradiction = make_fact(0.5, now, ResolutionState::Contradiction);
        let ambiguous = make_fact(0.5, now, ResolutionState::Ambiguous);

        let s_acc = scorer.score_fact(MindsetTag::Divergent, &accepted, now, time_range, 0.5);
        let s_con = scorer.score_fact(MindsetTag::Divergent, &contradiction, now, time_range, 0.5);
        let s_amb = scorer.score_fact(MindsetTag::Divergent, &ambiguous, now, time_range, 0.5);

        assert!(
            s_con > s_acc,
            "Contradictions should score higher under Divergent"
        );
        assert!(
            s_amb > s_con,
            "Ambiguous should score highest under Divergent"
        );
    }

    #[test]
    fn test_score_node_recency() {
        let scorer = MindsetScorer::default();
        let now = 1_000_000u64;
        let time_range = 500_000u64;

        let recent = make_node(now);
        let old = make_node(now - time_range);

        let s_recent = scorer.score_node(MindsetTag::Algorithmic, &recent, now, time_range, 0.5);
        let s_old = scorer.score_node(MindsetTag::Algorithmic, &old, now, time_range, 0.5);

        assert!(s_recent > s_old);
    }

    #[test]
    fn test_zero_time_range_gives_full_recency() {
        let scorer = MindsetScorer::default();
        let fact = make_fact(0.5, 100, ResolutionState::Accepted);
        let score = scorer.score_fact(MindsetTag::Convergent, &fact, 1000, 0, 0.5);
        // With zero time_range, recency = 1.0
        assert!(score > 0.0);
    }

    #[test]
    fn test_custom_weights() {
        let scorer = MindsetScorer::new(
            MindsetWeights {
                confidence: 1.0,
                recency: 0.0,
                diversity: 0.0,
                connectivity: 0.0,
            },
            MindsetWeights {
                confidence: 0.0,
                recency: 0.0,
                diversity: 1.0,
                connectivity: 0.0,
            },
            MindsetWeights {
                confidence: 0.0,
                recency: 1.0,
                diversity: 0.0,
                connectivity: 0.0,
            },
            MindsetWeights {
                confidence: 0.0,
                recency: 0.0,
                diversity: 0.0,
                connectivity: 1.0,
            },
        );

        // Convergent: only confidence matters
        let score = scorer.score(MindsetTag::Convergent, 0.9, 0.1, 0.1, 0.1);
        assert!((score - 0.9).abs() < 0.01);

        // Divergent: only diversity matters
        let score = scorer.score(MindsetTag::Divergent, 0.1, 0.1, 0.9, 0.1);
        assert!((score - 0.9).abs() < 0.01);

        // Algorithmic: only recency matters
        let score = scorer.score(MindsetTag::Algorithmic, 0.1, 0.9, 0.1, 0.1);
        assert!((score - 0.9).abs() < 0.01);

        // Spatial: only connectivity matters
        let score = scorer.score(MindsetTag::Spatial, 0.1, 0.1, 0.1, 0.9);
        assert!((score - 0.9).abs() < 0.01);
    }

    #[test]
    fn test_all_zero_weights_returns_zero() {
        let scorer = MindsetScorer::new(
            MindsetWeights {
                confidence: 0.0,
                recency: 0.0,
                diversity: 0.0,
                connectivity: 0.0,
            },
            MindsetWeights {
                confidence: 0.0,
                recency: 0.0,
                diversity: 0.0,
                connectivity: 0.0,
            },
            MindsetWeights {
                confidence: 0.0,
                recency: 0.0,
                diversity: 0.0,
                connectivity: 0.0,
            },
            MindsetWeights {
                confidence: 0.0,
                recency: 0.0,
                diversity: 0.0,
                connectivity: 0.0,
            },
        );
        let score = scorer.score(MindsetTag::Convergent, 1.0, 1.0, 1.0, 1.0);
        assert_eq!(score, 0.0);
    }

    #[test]
    fn test_input_clamping() {
        let scorer = MindsetScorer::default();
        // Inputs outside [0, 1] should be clamped
        let score = scorer.score(MindsetTag::Convergent, 2.0, -1.0, 5.0, -0.5);
        assert!(score >= 0.0 && score <= 1.0);
    }

    #[test]
    fn test_weights_for_returns_correct_variant() {
        let scorer = MindsetScorer::default();
        let w = scorer.weights_for(MindsetTag::Divergent);
        assert_eq!(w.diversity, 0.6);
        assert_eq!(w.confidence, 0.1);
    }

    // --- Convergent scoring tests (US-27.6) ---

    #[test]
    fn test_convergent_more_paths_higher_score() {
        let scorer = MindsetScorer::default();
        // Fact with 3 supporting paths should score higher than isolated fact (0 paths)
        let many_paths = scorer.score_convergent(0.5, 3, 0.5, 0.5);
        let no_paths = scorer.score_convergent(0.5, 0, 0.5, 0.5);
        assert!(
            many_paths > no_paths,
            "Fact with 3 supporting paths ({many_paths}) should score higher than isolated ({no_paths})"
        );
    }

    #[test]
    fn test_convergent_five_paths_higher_than_one() {
        let scorer = MindsetScorer::default();
        let five = scorer.score_convergent(0.5, 5, 0.5, 0.5);
        let one = scorer.score_convergent(0.5, 1, 0.5, 0.5);
        assert!(five > one, "5 paths ({five}) > 1 path ({one})");
    }

    #[test]
    fn test_convergent_saturates_after_many_paths() {
        let scorer = MindsetScorer::default();
        let five = scorer.score_convergent(0.5, 5, 0.5, 0.5);
        let hundred = scorer.score_convergent(0.5, 100, 0.5, 0.5);
        // Should saturate: 100 paths not much better than 5
        let diff = (hundred - five).abs();
        assert!(diff < 0.1, "Score should saturate: diff={diff}");
    }

    #[test]
    fn test_convergent_fact_with_paths() {
        let scorer = MindsetScorer::default();
        let now = 1_000_000u64;
        let time_range = 500_000u64;

        let corroborated = make_fact(0.7, now, ResolutionState::Accepted);
        let isolated = make_fact(0.7, now, ResolutionState::Accepted);

        let s_corr = scorer.score_fact_convergent(&corroborated, 3, now, time_range, 0.5);
        let s_iso = scorer.score_fact_convergent(&isolated, 0, now, time_range, 0.5);

        assert!(
            s_corr > s_iso,
            "Corroborated fact ({s_corr}) > isolated ({s_iso})"
        );
    }

    #[test]
    fn test_path_count_to_signal() {
        // 0 paths = 0.0
        assert_eq!(MindsetScorer::path_count_to_signal(0, 5), 0.0);
        // saturation_point 0 = 0.0
        assert_eq!(MindsetScorer::path_count_to_signal(3, 0), 0.0);
        // At saturation point, signal should be ~1.0
        let at_sat = MindsetScorer::path_count_to_signal(5, 5);
        assert!((at_sat - 1.0).abs() < 0.01, "At saturation: {at_sat}");
        // Monotonically increasing
        let s1 = MindsetScorer::path_count_to_signal(1, 5);
        let s3 = MindsetScorer::path_count_to_signal(3, 5);
        assert!(s3 > s1, "3 paths ({s3}) > 1 path ({s1})");
    }

    // --- Divergent scoring tests (US-27.7) ---

    #[test]
    fn test_divergent_rare_predicate_scores_higher() {
        let scorer = MindsetScorer::default();
        let rare = scorer.score_divergent(0.5, 0.9, 1, 0.5, 0.5);
        let common = scorer.score_divergent(0.5, 0.1, 1, 0.5, 0.5);
        assert!(rare > common, "Rare predicate ({rare}) > common ({common})");
    }

    #[test]
    fn test_divergent_unique_connection_scores_higher() {
        let scorer = MindsetScorer::default();
        // Fewer supporting paths = more unique/novel
        let unique = scorer.score_divergent(0.5, 0.5, 0, 0.5, 0.5);
        let well_known = scorer.score_divergent(0.5, 0.5, 10, 0.5, 0.5);
        assert!(
            unique > well_known,
            "Unique ({unique}) > well-known ({well_known})"
        );
    }

    #[test]
    fn test_divergent_fact_with_rarity() {
        let scorer = MindsetScorer::default();
        let now = 1_000_000u64;
        let time_range = 500_000u64;

        let unusual = make_fact(0.5, now, ResolutionState::Accepted);
        let normal = make_fact(0.5, now, ResolutionState::Accepted);

        let s_unusual = scorer.score_fact_divergent(&unusual, 0.9, 0, now, time_range, 0.5);
        let s_normal = scorer.score_fact_divergent(&normal, 0.1, 5, now, time_range, 0.5);

        assert!(
            s_unusual > s_normal,
            "Unusual ({s_unusual}) > normal ({s_normal})"
        );
    }

    // --- Algorithmic scoring tests (US-27.8) ---

    #[test]
    fn test_algorithmic_accepted_scores_higher_than_contradiction() {
        let scorer = MindsetScorer::default();
        let verified = scorer.score_algorithmic(0.8, ResolutionState::Accepted, 0, 0.5, 0.5);
        let contested = scorer.score_algorithmic(0.8, ResolutionState::Contradiction, 3, 0.5, 0.5);
        assert!(
            verified > contested,
            "Verified ({verified}) > contested ({contested})"
        );
    }

    #[test]
    fn test_algorithmic_no_contradictions_better() {
        let scorer = MindsetScorer::default();
        let clean = scorer.score_algorithmic(0.7, ResolutionState::Accepted, 0, 0.5, 0.5);
        let dirty = scorer.score_algorithmic(0.7, ResolutionState::Accepted, 5, 0.5, 0.5);
        assert!(clean > dirty, "Clean ({clean}) > dirty ({dirty})");
    }

    #[test]
    fn test_algorithmic_resolution_ordering() {
        let scorer = MindsetScorer::default();
        let accepted = scorer.score_algorithmic(0.5, ResolutionState::Accepted, 0, 0.5, 0.5);
        let superseded = scorer.score_algorithmic(0.5, ResolutionState::Superseded, 0, 0.5, 0.5);
        let ambiguous = scorer.score_algorithmic(0.5, ResolutionState::Ambiguous, 0, 0.5, 0.5);
        let contradiction =
            scorer.score_algorithmic(0.5, ResolutionState::Contradiction, 0, 0.5, 0.5);

        assert!(accepted > superseded, "Accepted > Superseded");
        assert!(superseded > ambiguous, "Superseded > Ambiguous");
        assert!(ambiguous > contradiction, "Ambiguous > Contradiction");
    }

    #[test]
    fn test_algorithmic_fact_scoring() {
        let scorer = MindsetScorer::default();
        let now = 1_000_000u64;
        let time_range = 500_000u64;

        let verified = make_fact(0.9, now, ResolutionState::Accepted);
        let contested = make_fact(0.9, now, ResolutionState::Contradiction);

        let s_ver = scorer.score_fact_algorithmic(&verified, 0, now, time_range, 0.5);
        let s_con = scorer.score_fact_algorithmic(&contested, 3, now, time_range, 0.5);

        assert!(
            s_ver > s_con,
            "Verified fact ({s_ver}) > contested ({s_con})"
        );
    }

    // --- MindsetDetector tests (US-27.10) ---

    #[test]
    fn test_detector_default_algorithmic_keywords() {
        let detector = MindsetDetector::default();
        assert_eq!(
            detector.detect("Can you verify this claim?"),
            Some(MindsetTag::Algorithmic)
        );
        assert_eq!(
            detector.detect("Please confirm the data"),
            Some(MindsetTag::Algorithmic)
        );
        assert_eq!(
            detector.detect("Check if this is right"),
            Some(MindsetTag::Algorithmic)
        );
        assert_eq!(
            detector.detect("Validate the result"),
            Some(MindsetTag::Algorithmic)
        );
        assert_eq!(
            detector.detect("Prove this hypothesis"),
            Some(MindsetTag::Algorithmic)
        );
        assert_eq!(
            detector.detect("Is this correct?"),
            Some(MindsetTag::Algorithmic)
        );
    }

    #[test]
    fn test_detector_default_divergent_keywords() {
        let detector = MindsetDetector::default();
        assert_eq!(
            detector.detect("What if we changed the approach?"),
            Some(MindsetTag::Divergent)
        );
        assert_eq!(
            detector.detect("Let's explore new ideas"),
            Some(MindsetTag::Divergent)
        );
        assert_eq!(
            detector.detect("Brainstorm solutions for this"),
            Some(MindsetTag::Divergent)
        );
        assert_eq!(
            detector.detect("Any alternative to this?"),
            Some(MindsetTag::Divergent)
        );
        assert_eq!(
            detector.detect("Imagine a world without gravity"),
            Some(MindsetTag::Divergent)
        );
        assert_eq!(
            detector.detect("Be creative with this"),
            Some(MindsetTag::Divergent)
        );
    }

    #[test]
    fn test_detector_default_convergent_keywords() {
        let detector = MindsetDetector::default();
        assert_eq!(
            detector.detect("Summarize the discussion"),
            Some(MindsetTag::Convergent)
        );
        assert_eq!(
            detector.detect("What's the consensus?"),
            Some(MindsetTag::Convergent)
        );
        assert_eq!(
            detector.detect("Do sources agree on this?"),
            Some(MindsetTag::Convergent)
        );
        assert_eq!(
            detector.detect("What's common across these?"),
            Some(MindsetTag::Convergent)
        );
        assert_eq!(
            detector.detect("Give me an overview"),
            Some(MindsetTag::Convergent)
        );
        assert_eq!(
            detector.detect("Let's conclude this topic"),
            Some(MindsetTag::Convergent)
        );
    }

    #[test]
    fn test_detector_no_match_returns_none() {
        let detector = MindsetDetector::default();
        assert_eq!(detector.detect("Tell me about cats"), None);
        assert_eq!(detector.detect("What is the weather today?"), None);
        assert_eq!(detector.detect("Hello world"), None);
        assert_eq!(detector.detect(""), None);
    }

    #[test]
    fn test_detector_case_insensitive() {
        let detector = MindsetDetector::default();
        assert_eq!(
            detector.detect("VERIFY this claim"),
            Some(MindsetTag::Algorithmic)
        );
        assert_eq!(
            detector.detect("Verify This Claim"),
            Some(MindsetTag::Algorithmic)
        );
        assert_eq!(
            detector.detect("EXPLORE ideas"),
            Some(MindsetTag::Divergent)
        );
        assert_eq!(
            detector.detect("SUMMARIZE the data"),
            Some(MindsetTag::Convergent)
        );
    }

    #[test]
    fn test_detector_word_boundary() {
        let detector = MindsetDetector::default();
        // "verify" should match as a standalone word
        assert_eq!(detector.detect("verify it"), Some(MindsetTag::Algorithmic));
        // "verified" should NOT match "verify" (not a word boundary)
        assert_eq!(detector.detect("I verified it"), None);
        // "unverified" should NOT match
        assert_eq!(detector.detect("unverified claim"), None);
    }

    #[test]
    fn test_detector_multi_word_pattern() {
        let detector = MindsetDetector::default();
        // "what if" is a multi-word pattern
        assert_eq!(
            detector.detect("what if we tried?"),
            Some(MindsetTag::Divergent)
        );
        // Only "what" alone shouldn't match
        assert_eq!(detector.detect("what is this?"), None);
    }

    #[test]
    fn test_detector_first_match_wins() {
        // Algorithmic keywords come before divergent in default order
        let detector = MindsetDetector::default();
        // Query with both verify (algorithmic) and explore (divergent)
        // "verify" should win since it appears first in the keyword list
        assert_eq!(
            detector.detect("verify and explore this topic"),
            Some(MindsetTag::Algorithmic)
        );
    }

    #[test]
    fn test_detector_custom_keywords() {
        let detector = MindsetDetector::from_keyword_lists(
            &["fact-check", "audit"],
            &["brainstorm", "hypothesize"],
            &["summarize", "wrap up"],
            &["path", "network"],
        );
        assert_eq!(
            detector.detect("fact-check this"),
            Some(MindsetTag::Algorithmic)
        );
        assert_eq!(
            detector.detect("let's hypothesize"),
            Some(MindsetTag::Divergent)
        );
        assert_eq!(
            detector.detect("wrap up the session"),
            Some(MindsetTag::Convergent)
        );
        assert_eq!(detector.detect("verify this"), None); // not in custom set
    }

    #[test]
    fn test_detector_empty_keywords() {
        let detector = MindsetDetector::new(vec![]);
        assert_eq!(detector.detect("verify something"), None);
        assert_eq!(detector.detect("explore ideas"), None);
    }

    #[test]
    fn test_contains_keyword_boundary() {
        // Test the boundary detection directly
        assert!(contains_keyword("please verify this", "verify"));
        assert!(contains_keyword("verify this", "verify"));
        assert!(contains_keyword("this is verify", "verify"));
        assert!(!contains_keyword("unverify this", "verify"));
        assert!(!contains_keyword("verifying this", "verify"));
        // Multi-word
        assert!(contains_keyword("and what if we", "what if"));
        assert!(!contains_keyword("somewhat iffy", "what if"));
    }
}
