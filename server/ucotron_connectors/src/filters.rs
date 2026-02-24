//! Content filtering for connectors.
//!
//! Filters control which content items are included or excluded during
//! connector sync operations. They can filter by date range, content patterns,
//! source-specific attributes (e.g., Slack channels, GitHub labels).

use serde::{Deserialize, Serialize};

/// Filter criteria for connector content.
///
/// Multiple filters are combined with AND logic — a content item must
/// pass all active filters to be included.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ContentFilter {
    /// Only include content created after this timestamp (Unix seconds).
    pub after: Option<u64>,
    /// Only include content created before this timestamp (Unix seconds).
    pub before: Option<u64>,
    /// Only include content matching these patterns (substring match).
    pub include_patterns: Vec<String>,
    /// Exclude content matching these patterns (substring match).
    pub exclude_patterns: Vec<String>,
    /// Only include content from these authors/users.
    pub authors: Vec<String>,
    /// Source-specific filter (e.g., channel names for Slack, repo names for GitHub).
    pub source_filter: Option<SourceFilter>,
    /// Maximum number of items to return.
    pub limit: Option<usize>,
}

/// Source-specific filtering criteria.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum SourceFilter {
    /// Filter Slack messages by channel.
    Slack { channels: Vec<String> },
    /// Filter GitHub content by repository and type.
    GitHub {
        repos: Vec<String>,
        include_issues: bool,
        include_prs: bool,
        labels: Vec<String>,
    },
    /// Filter Notion pages by database ID.
    Notion { database_ids: Vec<String> },
    /// Filter Discord messages by channel.
    Discord { channels: Vec<String> },
    /// Filter by file path patterns (Obsidian, local filesystems).
    FileSystem { path_patterns: Vec<String> },
    /// Generic key-value filter for other connectors.
    Custom {
        filters: std::collections::HashMap<String, String>,
    },
}

impl ContentFilter {
    /// Creates an empty filter that matches everything.
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the minimum timestamp.
    pub fn after(mut self, timestamp: u64) -> Self {
        self.after = Some(timestamp);
        self
    }

    /// Sets the maximum timestamp.
    pub fn before(mut self, timestamp: u64) -> Self {
        self.before = Some(timestamp);
        self
    }

    /// Adds an include pattern.
    pub fn include_pattern(mut self, pattern: impl Into<String>) -> Self {
        self.include_patterns.push(pattern.into());
        self
    }

    /// Adds an exclude pattern.
    pub fn exclude_pattern(mut self, pattern: impl Into<String>) -> Self {
        self.exclude_patterns.push(pattern.into());
        self
    }

    /// Adds an author filter.
    pub fn author(mut self, author: impl Into<String>) -> Self {
        self.authors.push(author.into());
        self
    }

    /// Sets the source-specific filter.
    pub fn source(mut self, filter: SourceFilter) -> Self {
        self.source_filter = Some(filter);
        self
    }

    /// Sets the maximum number of items.
    pub fn limit(mut self, limit: usize) -> Self {
        self.limit = Some(limit);
        self
    }

    /// Checks if a content string passes the include/exclude pattern filters.
    pub fn matches_content(&self, content: &str) -> bool {
        // If include patterns are set, content must match at least one
        if !self.include_patterns.is_empty()
            && !self.include_patterns.iter().any(|p| content.contains(p))
        {
            return false;
        }
        // Content must not match any exclude pattern
        if self.exclude_patterns.iter().any(|p| content.contains(p)) {
            return false;
        }
        true
    }

    /// Checks if a timestamp falls within the configured range.
    pub fn matches_timestamp(&self, timestamp: u64) -> bool {
        if let Some(after) = self.after {
            if timestamp < after {
                return false;
            }
        }
        if let Some(before) = self.before {
            if timestamp > before {
                return false;
            }
        }
        true
    }

    /// Checks if an author passes the author filter.
    pub fn matches_author(&self, author: &str) -> bool {
        if self.authors.is_empty() {
            return true;
        }
        self.authors.iter().any(|a| a == author)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_filter_matches_everything() {
        let filter = ContentFilter::new();
        assert!(filter.matches_content("anything"));
        assert!(filter.matches_timestamp(1700000000));
        assert!(filter.matches_author("anyone"));
    }

    #[test]
    fn test_include_pattern() {
        let filter = ContentFilter::new().include_pattern("important");
        assert!(filter.matches_content("this is important"));
        assert!(!filter.matches_content("this is trivial"));
    }

    #[test]
    fn test_exclude_pattern() {
        let filter = ContentFilter::new().exclude_pattern("spam");
        assert!(filter.matches_content("good message"));
        assert!(!filter.matches_content("this is spam"));
    }

    #[test]
    fn test_timestamp_range() {
        let filter = ContentFilter::new().after(100).before(200);
        assert!(!filter.matches_timestamp(50));
        assert!(filter.matches_timestamp(150));
        assert!(!filter.matches_timestamp(250));
    }

    #[test]
    fn test_author_filter() {
        let filter = ContentFilter::new().author("alice").author("bob");
        assert!(filter.matches_author("alice"));
        assert!(filter.matches_author("bob"));
        assert!(!filter.matches_author("charlie"));
    }

    #[test]
    fn test_builder_chain() {
        let filter = ContentFilter::new()
            .after(1000)
            .before(2000)
            .include_pattern("rust")
            .exclude_pattern("deprecated")
            .author("alice")
            .limit(50);

        assert_eq!(filter.after, Some(1000));
        assert_eq!(filter.before, Some(2000));
        assert_eq!(filter.include_patterns, vec!["rust"]);
        assert_eq!(filter.exclude_patterns, vec!["deprecated"]);
        assert_eq!(filter.authors, vec!["alice"]);
        assert_eq!(filter.limit, Some(50));
    }

    #[test]
    fn test_filter_serialization() {
        let filter = ContentFilter::new()
            .after(1000)
            .include_pattern("test")
            .source(SourceFilter::Slack {
                channels: vec!["general".to_string()],
            });
        let json = serde_json::to_string(&filter).unwrap();
        let deserialized: ContentFilter = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.after, Some(1000));
        assert_eq!(deserialized.include_patterns, vec!["test"]);
    }
}
