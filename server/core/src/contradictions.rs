//! Contradiction detection and resolution for the cognitive data model.
//!
//! Implements the Divergent/Evaluative mindset mode from Chain of Mindset (CoM).
//! When a new fact conflicts with existing facts (same subject + predicate,
//! different object), the system detects the conflict and applies ordered
//! resolution rules.
//!
//! # Resolution Rules (applied in order)
//!
//! 1. **Temporal**: If timestamps differ by more than 1 year, the newer fact wins.
//!    The older fact is marked [`ResolutionState::Superseded`].
//! 2. **Confidence**: If timestamps are close but confidence difference > 0.3,
//!    the higher-confidence fact wins.
//! 3. **Ambiguous**: If neither rule applies, both facts are marked as
//!    [`ResolutionState::Contradiction`] (requires external intervention).
//!
//! # Invariant
//!
//! Old facts are **never** deleted. All conflicts produce
//! [`EdgeType::ConflictsWith`] edges between the conflicting facts.

use crate::types::{
    Conflict, ConflictConfig, Edge, Fact, Resolution, ResolutionState, ResolutionStrategy,
};

/// Detect a conflict between a new fact and a set of existing facts.
///
/// Searches `existing_facts` for any fact that shares the same `subject` and
/// `predicate` as `new_fact` but has a different `object`. Returns the first
/// conflict found (if any).
///
/// The returned [`Conflict`] contains both facts and a preliminary resolution
/// strategy computed via [`resolve_conflict`].
///
/// # Arguments
/// * `new_fact` - The incoming fact to check for conflicts.
/// * `existing_facts` - The set of existing facts to check against.
/// * `config` - Thresholds for temporal and confidence resolution rules.
///
/// # Returns
/// `Some(Conflict)` if a conflicting fact is found, `None` otherwise.
pub fn detect_conflict(
    new_fact: &Fact,
    existing_facts: &[Fact],
    config: &ConflictConfig,
) -> Option<Conflict> {
    for existing in existing_facts {
        // Same subject + predicate but different object = conflict
        if existing.subject == new_fact.subject
            && existing.predicate == new_fact.predicate
            && existing.object != new_fact.object
            // Only conflict with facts that are still Accepted
            && existing.resolution_state == ResolutionState::Accepted
        {
            let resolution = resolve_conflict(existing, new_fact, config);
            return Some(Conflict {
                existing: resolution.winner.clone(),
                incoming: resolution.loser.clone(),
                strategy: resolution.strategy,
            });
        }
    }
    None
}

/// Resolve a conflict between two facts using ordered rules.
///
/// Applies the resolution rules in order:
///
/// 1. **Temporal**: If `|timestamp_a - timestamp_b| > temporal_threshold_secs`,
///    the newer fact wins. The older fact is marked [`ResolutionState::Superseded`].
/// 2. **Confidence**: If timestamps are within the temporal threshold but
///    `|confidence_a - confidence_b| > confidence_threshold`, the
///    higher-confidence fact wins.
/// 3. **Ambiguous**: If neither rule applies, both facts are marked as
///    [`ResolutionState::Contradiction`].
///
/// # Arguments
/// * `fact_a` - The first fact (typically the existing/older fact).
/// * `fact_b` - The second fact (typically the incoming/newer fact).
/// * `config` - Thresholds for resolution rules.
///
/// # Returns
/// A [`Resolution`] containing the winner, loser, and strategy applied.
pub fn resolve_conflict(fact_a: &Fact, fact_b: &Fact, config: &ConflictConfig) -> Resolution {
    let time_diff = fact_a.timestamp.abs_diff(fact_b.timestamp);

    // Rule 1: Temporal — if timestamps differ by more than threshold, newer wins
    if time_diff > config.temporal_threshold_secs {
        let (newer, older) = if fact_a.timestamp >= fact_b.timestamp {
            (fact_a, fact_b)
        } else {
            (fact_b, fact_a)
        };
        let mut winner = newer.clone();
        let mut loser = older.clone();
        winner.resolution_state = ResolutionState::Accepted;
        loser.resolution_state = ResolutionState::Superseded;
        return Resolution {
            winner,
            loser,
            strategy: ResolutionStrategy::Temporal,
        };
    }

    // Rule 2: Confidence — if confidence gap exceeds threshold, higher wins
    let confidence_diff = (fact_a.source_confidence - fact_b.source_confidence).abs();
    if confidence_diff > config.confidence_threshold {
        let (higher, lower) = if fact_a.source_confidence >= fact_b.source_confidence {
            (fact_a, fact_b)
        } else {
            (fact_b, fact_a)
        };
        let mut winner = higher.clone();
        let mut loser = lower.clone();
        winner.resolution_state = ResolutionState::Accepted;
        loser.resolution_state = ResolutionState::Superseded;
        return Resolution {
            winner,
            loser,
            strategy: ResolutionStrategy::Confidence,
        };
    }

    // Rule 3: Ambiguous — cannot auto-resolve
    let mut fact_a_out = fact_a.clone();
    let mut fact_b_out = fact_b.clone();
    fact_a_out.resolution_state = ResolutionState::Contradiction;
    fact_b_out.resolution_state = ResolutionState::Contradiction;
    Resolution {
        winner: fact_a_out,
        loser: fact_b_out,
        strategy: ResolutionStrategy::Ambiguous,
    }
}

/// Build the edges that should be created as a result of a conflict resolution.
///
/// Always creates a `CONFLICTS_WITH` edge. If one fact supersedes the other,
/// also creates a `SUPERSEDES` edge from the winner to the loser.
///
/// # Arguments
/// * `resolution` - The resolution result from [`resolve_conflict`].
/// * `detected_at` - Timestamp when the conflict was detected.
///
/// # Returns
/// A vector of edges to insert into the storage engine.
pub fn build_conflict_edges(resolution: &Resolution, detected_at: u64) -> Vec<Edge> {
    let mut edges = vec![Edge::conflict(
        resolution.winner.id,
        resolution.loser.id,
        detected_at,
        resolution.strategy,
    )];

    // If one fact supersedes the other, also create a SUPERSEDES edge
    if resolution.strategy != ResolutionStrategy::Ambiguous {
        edges.push(Edge::supersedes(resolution.winner.id, resolution.loser.id));
    }

    edges
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{EdgeType, Value};

    // Helper: one year in seconds (≈365.25 days)
    const ONE_YEAR: u64 = 365 * 24 * 3600 + 6 * 3600;

    // -----------------------------------------------------------------------
    // PRD mandatory test case 1:
    // "El cielo es azul" (t=1, conf=0.9) → "El cielo es verde" (t=2, conf=0.9)
    // → Verde wins (temporal if >1yr apart), Azul marked Superseded
    // -----------------------------------------------------------------------
    #[test]
    fn test_prd_case1_sky_color_temporal() {
        let config = ConflictConfig::default();
        let t_base = 1_700_000_000u64;

        // "Sky is blue" — old fact
        let fact_blue = Fact::new(1, 100, "color_is", "blue", 0.9, t_base);
        // "Sky is green" — new fact, more than 1 year later
        let fact_green = Fact::new(2, 100, "color_is", "green", 0.9, t_base + ONE_YEAR + 1);

        let resolution = resolve_conflict(&fact_blue, &fact_green, &config);

        assert_eq!(resolution.strategy, ResolutionStrategy::Temporal);
        assert_eq!(resolution.winner.object, "green");
        assert_eq!(resolution.winner.resolution_state, ResolutionState::Accepted);
        assert_eq!(resolution.loser.object, "blue");
        assert_eq!(resolution.loser.resolution_state, ResolutionState::Superseded);
    }

    // -----------------------------------------------------------------------
    // PRD mandatory test case 2:
    // "El cielo es rojo" (t=2.1, conf=0.3) after the green fact
    // → Conflict detected; confidence rule: green (0.9) >> red (0.3)
    // -----------------------------------------------------------------------
    #[test]
    fn test_prd_case2_sky_color_low_confidence() {
        let config = ConflictConfig::default();
        let t_base = 1_700_000_000u64;
        let t_green = t_base + ONE_YEAR + 1;

        // "Sky is green" — accepted winner from case 1
        let fact_green = Fact::new(2, 100, "color_is", "green", 0.9, t_green);
        // "Sky is red" — shortly after green, very low confidence
        let fact_red = Fact::new(3, 100, "color_is", "red", 0.3, t_green + 100);

        // Timestamps are close (100s apart < 1 year), but confidence gap = 0.6 > 0.3
        let resolution = resolve_conflict(&fact_green, &fact_red, &config);

        assert_eq!(resolution.strategy, ResolutionStrategy::Confidence);
        assert_eq!(resolution.winner.object, "green");
        assert_eq!(resolution.winner.resolution_state, ResolutionState::Accepted);
        assert_eq!(resolution.loser.object, "red");
        assert_eq!(resolution.loser.resolution_state, ResolutionState::Superseded);
    }

    // -----------------------------------------------------------------------
    // PRD mandatory test case 3:
    // "Juan vive en Madrid" (t=1) → "Juan vive en Berlín" (t=1.5)
    // → Berlín wins (temporal, if >1yr apart), Madrid marked Superseded,
    //   CONFLICTS_WITH edge created
    // -----------------------------------------------------------------------
    #[test]
    fn test_prd_case3_juan_location_change() {
        let config = ConflictConfig::default();
        let t_base = 1_700_000_000u64;

        // "Juan lives in Madrid"
        let fact_madrid = Fact::new(10, 200, "lives_in", "Madrid", 0.9, t_base);
        // "Juan lives in Berlin" — more than 1 year later
        let fact_berlin = Fact::new(11, 200, "lives_in", "Berlin", 0.9, t_base + ONE_YEAR + 1);

        let resolution = resolve_conflict(&fact_madrid, &fact_berlin, &config);

        assert_eq!(resolution.strategy, ResolutionStrategy::Temporal);
        assert_eq!(resolution.winner.object, "Berlin");
        assert_eq!(resolution.loser.object, "Madrid");
        assert_eq!(
            resolution.loser.resolution_state,
            ResolutionState::Superseded
        );

        // Verify CONFLICTS_WITH edge is created
        let edges = build_conflict_edges(&resolution, t_base + ONE_YEAR + 1);
        assert_eq!(edges.len(), 2); // CONFLICTS_WITH + SUPERSEDES

        let conflict_edge = &edges[0];
        assert_eq!(conflict_edge.edge_type, EdgeType::ConflictsWith);
        assert_eq!(conflict_edge.source, 11); // Berlin (winner)
        assert_eq!(conflict_edge.target, 10); // Madrid (loser)
        assert_eq!(
            conflict_edge.metadata.get("resolution_strategy"),
            Some(&Value::String("temporal".to_string()))
        );

        let supersedes_edge = &edges[1];
        assert_eq!(supersedes_edge.edge_type, EdgeType::Supersedes);
        assert_eq!(supersedes_edge.source, 11); // Berlin (newer)
        assert_eq!(supersedes_edge.target, 10); // Madrid (older)
    }

    // -----------------------------------------------------------------------
    // detect_conflict integration test: finds conflict in list of existing facts
    // -----------------------------------------------------------------------
    #[test]
    fn test_detect_conflict_finds_match() {
        let config = ConflictConfig::default();
        let t_base = 1_700_000_000u64;

        let existing_facts = vec![
            Fact::new(1, 100, "color_is", "blue", 0.9, t_base),
            Fact::new(2, 200, "lives_in", "Madrid", 0.85, t_base),
            Fact::new(3, 300, "works_at", "Google", 0.7, t_base),
        ];

        // New fact conflicts with fact #2 (same subject 200, same predicate "lives_in")
        let new_fact = Fact::new(4, 200, "lives_in", "Berlin", 0.9, t_base + ONE_YEAR + 1);

        let conflict = detect_conflict(&new_fact, &existing_facts, &config);
        assert!(conflict.is_some());
        let conflict = conflict.unwrap();
        assert_eq!(conflict.strategy, ResolutionStrategy::Temporal);
    }

    #[test]
    fn test_detect_conflict_no_match() {
        let config = ConflictConfig::default();
        let t_base = 1_700_000_000u64;

        let existing_facts = vec![
            Fact::new(1, 100, "color_is", "blue", 0.9, t_base),
        ];

        // Different subject — no conflict
        let new_fact = Fact::new(2, 200, "color_is", "green", 0.9, t_base + 100);
        assert!(detect_conflict(&new_fact, &existing_facts, &config).is_none());

        // Same subject, different predicate — no conflict
        let new_fact2 = Fact::new(3, 100, "shape_is", "round", 0.9, t_base + 100);
        assert!(detect_conflict(&new_fact2, &existing_facts, &config).is_none());

        // Same subject + predicate + same object — not a conflict (agreement)
        let new_fact3 = Fact::new(4, 100, "color_is", "blue", 0.9, t_base + 100);
        assert!(detect_conflict(&new_fact3, &existing_facts, &config).is_none());
    }

    #[test]
    fn test_detect_conflict_skips_already_superseded() {
        let config = ConflictConfig::default();
        let t_base = 1_700_000_000u64;

        let mut old_fact = Fact::new(1, 100, "color_is", "blue", 0.9, t_base);
        old_fact.resolution_state = ResolutionState::Superseded;

        let existing_facts = vec![old_fact];

        // Should NOT conflict with a superseded fact
        let new_fact = Fact::new(2, 100, "color_is", "green", 0.9, t_base + 100);
        assert!(detect_conflict(&new_fact, &existing_facts, &config).is_none());
    }

    // -----------------------------------------------------------------------
    // Ambiguous resolution: close timestamps AND close confidence
    // -----------------------------------------------------------------------
    #[test]
    fn test_resolve_conflict_ambiguous() {
        let config = ConflictConfig::default();
        let t_base = 1_700_000_000u64;

        // Close timestamps (100s apart) AND close confidence (0.1 diff, < 0.3 threshold)
        let fact_a = Fact::new(1, 100, "color_is", "blue", 0.8, t_base);
        let fact_b = Fact::new(2, 100, "color_is", "green", 0.7, t_base + 100);

        let resolution = resolve_conflict(&fact_a, &fact_b, &config);

        assert_eq!(resolution.strategy, ResolutionStrategy::Ambiguous);
        assert_eq!(
            resolution.winner.resolution_state,
            ResolutionState::Contradiction
        );
        assert_eq!(
            resolution.loser.resolution_state,
            ResolutionState::Contradiction
        );
    }

    #[test]
    fn test_build_conflict_edges_temporal() {
        let config = ConflictConfig::default();
        let t_base = 1_700_000_000u64;

        let fact_old = Fact::new(1, 100, "p", "a", 0.9, t_base);
        let fact_new = Fact::new(2, 100, "p", "b", 0.9, t_base + ONE_YEAR + 1);

        let resolution = resolve_conflict(&fact_old, &fact_new, &config);
        let edges = build_conflict_edges(&resolution, t_base + ONE_YEAR + 1);

        // Temporal: should produce CONFLICTS_WITH + SUPERSEDES
        assert_eq!(edges.len(), 2);
        assert_eq!(edges[0].edge_type, EdgeType::ConflictsWith);
        assert_eq!(edges[1].edge_type, EdgeType::Supersedes);
    }

    #[test]
    fn test_build_conflict_edges_ambiguous() {
        let config = ConflictConfig::default();
        let t_base = 1_700_000_000u64;

        let fact_a = Fact::new(1, 100, "p", "a", 0.8, t_base);
        let fact_b = Fact::new(2, 100, "p", "b", 0.7, t_base + 100);

        let resolution = resolve_conflict(&fact_a, &fact_b, &config);
        let edges = build_conflict_edges(&resolution, t_base + 100);

        // Ambiguous: should produce ONLY CONFLICTS_WITH (no SUPERSEDES)
        assert_eq!(edges.len(), 1);
        assert_eq!(edges[0].edge_type, EdgeType::ConflictsWith);
    }

    #[test]
    fn test_resolve_conflict_confidence_rule() {
        let config = ConflictConfig::default();
        let t_base = 1_700_000_000u64;

        // Close timestamps (1 day apart) but large confidence gap (0.5)
        let fact_high = Fact::new(1, 100, "p", "correct", 0.95, t_base);
        let fact_low = Fact::new(2, 100, "p", "wrong", 0.45, t_base + 86400);

        let resolution = resolve_conflict(&fact_high, &fact_low, &config);

        assert_eq!(resolution.strategy, ResolutionStrategy::Confidence);
        assert_eq!(resolution.winner.object, "correct");
        assert_eq!(resolution.winner.source_confidence, 0.95);
        assert_eq!(resolution.loser.object, "wrong");
        assert_eq!(resolution.loser.resolution_state, ResolutionState::Superseded);
    }

    #[test]
    fn test_resolve_conflict_custom_config() {
        // Custom config: very short temporal threshold, very high confidence threshold
        let config = ConflictConfig {
            temporal_threshold_secs: 3600, // 1 hour
            confidence_threshold: 0.9,     // very high
        };
        let t_base = 1_700_000_000u64;

        // 2 hours apart (> 1hr threshold) → temporal wins even with close timestamps
        let fact_a = Fact::new(1, 100, "p", "a", 0.8, t_base);
        let fact_b = Fact::new(2, 100, "p", "b", 0.8, t_base + 7200);

        let resolution = resolve_conflict(&fact_a, &fact_b, &config);
        assert_eq!(resolution.strategy, ResolutionStrategy::Temporal);
        assert_eq!(resolution.winner.object, "b"); // newer wins
    }

    #[test]
    fn test_facts_never_deleted() {
        let config = ConflictConfig::default();
        let t_base = 1_700_000_000u64;

        let fact_old = Fact::new(1, 100, "lives_in", "Madrid", 0.9, t_base);
        let fact_new = Fact::new(2, 100, "lives_in", "Berlin", 0.9, t_base + ONE_YEAR + 1);

        let resolution = resolve_conflict(&fact_old, &fact_new, &config);

        // Both facts still exist in the resolution — nothing deleted
        assert_eq!(resolution.winner.id, 2);
        assert_eq!(resolution.loser.id, 1);
        // Loser is Superseded, not removed
        assert_eq!(resolution.loser.resolution_state, ResolutionState::Superseded);
        // Winner retains original data
        assert_eq!(resolution.winner.predicate, "lives_in");
        assert_eq!(resolution.winner.object, "Berlin");
    }

    #[test]
    fn test_detect_conflict_same_object_is_agreement() {
        // Same (subject, predicate, object) should NOT be detected as conflict
        let config = ConflictConfig::default();
        let existing = vec![Fact::new(1, 100, "lives_in", "Madrid", 0.9, 1000)];
        let new_fact = Fact::new(2, 100, "lives_in", "Madrid", 0.8, 2000);
        let conflict = detect_conflict(&new_fact, &existing, &config);
        assert!(conflict.is_none(), "Same object value is agreement, not conflict");
    }

    #[test]
    fn test_detect_conflict_different_predicate() {
        // Different predicate should NOT conflict even with same subject
        let config = ConflictConfig::default();
        let existing = vec![Fact::new(1, 100, "lives_in", "Madrid", 0.9, 1000)];
        let new_fact = Fact::new(2, 100, "works_at", "Google", 0.8, 2000);
        let conflict = detect_conflict(&new_fact, &existing, &config);
        assert!(conflict.is_none());
    }

    #[test]
    fn test_resolve_conflict_identical_timestamps() {
        // Identical timestamps (diff=0) should NOT trigger temporal rule
        let config = ConflictConfig::default();
        let fact_a = Fact::new(1, 100, "color_is", "Blue", 0.5, 1000);
        let fact_b = Fact::new(2, 100, "color_is", "Red", 0.5, 1000);

        let resolution = resolve_conflict(&fact_a, &fact_b, &config);
        // Same timestamp AND same confidence → should be Ambiguous
        assert_eq!(resolution.strategy, ResolutionStrategy::Ambiguous);
    }

    #[test]
    fn test_resolve_conflict_confidence_at_exact_threshold_is_ambiguous() {
        // Confidence diff exactly at threshold (0.3) uses strict > comparison,
        // so diff=0.3 falls through to ambiguous
        let config = ConflictConfig::default();
        let t = 1_000_000u64;
        let fact_high = Fact::new(1, 100, "color_is", "Blue", 0.8, t);
        let fact_low = Fact::new(2, 100, "color_is", "Red", 0.5, t); // diff = 0.3 exactly

        let resolution = resolve_conflict(&fact_high, &fact_low, &config);
        // 0.3 NOT > 0.3 → ambiguous
        assert_eq!(resolution.strategy, ResolutionStrategy::Ambiguous);
    }

    #[test]
    fn test_resolve_conflict_confidence_above_threshold() {
        // Confidence diff just above threshold should trigger confidence rule
        let config = ConflictConfig::default();
        let t = 1_000_000u64;
        let fact_high = Fact::new(1, 100, "color_is", "Blue", 0.81, t);
        let fact_low = Fact::new(2, 100, "color_is", "Red", 0.5, t); // diff = 0.31

        let resolution = resolve_conflict(&fact_high, &fact_low, &config);
        assert_eq!(resolution.strategy, ResolutionStrategy::Confidence);
        assert_eq!(resolution.winner.id, 1, "Higher confidence should win");
    }

    #[test]
    fn test_temporal_rule_checked_before_confidence() {
        // When both rules would apply, temporal should win (it's checked first)
        let config = ConflictConfig::default();
        let t_old = 1_000_000u64;
        let t_new = t_old + ONE_YEAR + 100;

        // Big temporal gap AND big confidence gap
        let fact_old = Fact::new(1, 100, "color_is", "Blue", 1.0, t_old);
        let fact_new = Fact::new(2, 100, "color_is", "Red", 0.1, t_new);

        let resolution = resolve_conflict(&fact_old, &fact_new, &config);
        // Temporal rule should trigger (>1yr gap), even though confidence says old is better
        assert_eq!(resolution.strategy, ResolutionStrategy::Temporal);
        assert_eq!(resolution.winner.id, 2, "Newer fact wins by temporal rule");
    }
}
