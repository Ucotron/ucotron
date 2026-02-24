//! Chunker number-handling test suite (TEST-2).
//!
//! Regression tests for BUG-2: the chunker must NOT split on decimal points
//! within numbers. Covers prices, version strings, percentages, and ranges
//! across both chunking paths:
//!
//! - `ingestion::chunk_text()` — the primary sentence-level chunker
//! - `ocr::chunk_document_text()` → internal `split_into_sentences()` — the OCR path

use ucotron_extraction::ingestion::chunk_text;
use ucotron_extraction::ocr::{chunk_document_text, PageExtraction};

// =========================================================================
// ingestion::chunk_text — decimal number preservation
// =========================================================================

#[test]
fn test_decimal_numbers_preserved() {
    // BUG-2 regression: "$99.99" must not be split into "$99." and "99"
    let chunks = chunk_text("The price is $99.99 for this item.");
    assert_eq!(chunks.len(), 1, "Should be one chunk: {:?}", chunks);
    assert!(
        chunks[0].contains("$99.99"),
        "Decimal price $99.99 should not be split: {:?}",
        chunks
    );
}

#[test]
fn test_version_numbers_preserved() {
    // Version numbers like 2.0.1 contain two decimal points
    let chunks = chunk_text("Using version 2.0.1 of the library.");
    assert_eq!(chunks.len(), 1, "Should be one chunk: {:?}", chunks);
    assert!(
        chunks[0].contains("2.0.1"),
        "Version number 2.0.1 should stay together: {:?}",
        chunks
    );
}

#[test]
fn test_percentage_preserved() {
    // "15.5 percent" must keep the number intact
    let chunks = chunk_text("Inflation was 15.5 percent last year. It may decrease.");
    // Two sentences separated by a real period after "year"
    assert_eq!(chunks.len(), 2, "Should be two sentences: {:?}", chunks);
    assert!(
        chunks[0].contains("15.5"),
        "Percentage 15.5 should not be split: {:?}",
        chunks
    );
}

#[test]
fn test_range_preserved() {
    // "From 8.2 to 6.5" — both numbers must survive intact
    let chunks = chunk_text("Growth from 8.2 to 6.5 percent.");
    assert_eq!(chunks.len(), 1, "Should be one chunk: {:?}", chunks);
    assert!(
        chunks[0].contains("8.2"),
        "First range number 8.2 should be intact: {:?}",
        chunks
    );
    assert!(
        chunks[0].contains("6.5"),
        "Second range number 6.5 should be intact: {:?}",
        chunks
    );
}

// =========================================================================
// ocr::chunk_document_text — decimal preservation via split_into_sentences
// =========================================================================

/// Helper: build a single-page document with the given text.
fn one_page(text: &str) -> Vec<PageExtraction> {
    vec![PageExtraction {
        page_number: 1,
        text: text.to_string(),
    }]
}

#[test]
fn test_ocr_chunker_decimal_price_preserved() {
    // Force sentence splitting by making the page exceed max_chunk_size
    // so chunk_document_text calls split_into_sentences internally.
    let text = "The total was $99.99 for the premium plan. \
                Customers paid promptly. \
                Revenue reached $1000.50 by end of quarter. \
                The CFO was satisfied with results.";
    let pages = one_page(text);
    let chunks = chunk_document_text(&pages, 60);

    // Verify no chunk splits a decimal number
    for chunk in &chunks {
        // If a chunk contains "99." it must also contain "99.99"
        if chunk.contains("99.") {
            assert!(
                chunk.contains("99.99"),
                "Price $99.99 was split across chunks: {:?}",
                chunks
            );
        }
        if chunk.contains("1000.") {
            assert!(
                chunk.contains("1000.50"),
                "Price $1000.50 was split across chunks: {:?}",
                chunks
            );
        }
    }
}

#[test]
fn test_ocr_chunker_version_number_preserved() {
    let text = "We upgraded to version 2.0.1 of the framework. \
                It improved performance significantly. \
                The previous version 1.9.3 had critical bugs. \
                All issues were resolved in the update.";
    let pages = one_page(text);
    let chunks = chunk_document_text(&pages, 70);

    let all_text: String = chunks.join(" ");
    assert!(
        all_text.contains("2.0.1"),
        "Version 2.0.1 should be preserved: {:?}",
        chunks
    );
    assert!(
        all_text.contains("1.9.3"),
        "Version 1.9.3 should be preserved: {:?}",
        chunks
    );
}

#[test]
fn test_ocr_chunker_range_numbers_preserved() {
    let text = "Temperature dropped from 8.2 to 6.5 degrees overnight. \
                The forecast predicted further decline. \
                By morning it reached 3.1 degrees celsius. \
                Residents were advised to stay warm.";
    let pages = one_page(text);
    let chunks = chunk_document_text(&pages, 70);

    let all_text: String = chunks.join(" ");
    assert!(
        all_text.contains("8.2"),
        "Number 8.2 should be intact: {:?}",
        chunks
    );
    assert!(
        all_text.contains("6.5"),
        "Number 6.5 should be intact: {:?}",
        chunks
    );
    assert!(
        all_text.contains("3.1"),
        "Number 3.1 should be intact: {:?}",
        chunks
    );
}
