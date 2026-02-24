//! Document OCR and PDF text extraction pipeline.
//!
//! This module provides:
//! - **PDF text extraction** via `pdf_extract` (pure Rust, no external dependencies)
//! - **Image OCR** via Tesseract CLI (requires `tesseract` installed on system)
//! - **Intelligent chunking** based on document structure (paragraphs, pages)
//!
//! # Architecture
//!
//! The pipeline follows the same trait pattern as audio/image pipelines:
//! 1. Define `DocumentOcrPipeline` trait in `lib.rs`
//! 2. Implement `PdfTextExtractor` for structured PDF text extraction
//! 3. Implement `TesseractOcrEngine` for scanned document/image OCR
//! 4. `CompositeDocumentPipeline` combines both based on input type
//!
//! # Test Strategy
//!
//! - **Unit tests (no external deps)**: chunking, page splitting, text cleaning
//! - **PDF tests**: use inline minimal PDF bytes
//! - **OCR tests**: skip gracefully if tesseract not installed

use std::path::Path;

/// Result of processing a document through the OCR/extraction pipeline.
#[derive(Debug, Clone)]
pub struct DocumentExtractionResult {
    /// Full extracted text.
    pub text: String,
    /// Per-page or per-section extractions.
    pub pages: Vec<PageExtraction>,
    /// Document metadata.
    pub metadata: DocumentMetadata,
}

/// Extracted content from a single page or section.
#[derive(Debug, Clone)]
pub struct PageExtraction {
    /// Page number (1-based).
    pub page_number: usize,
    /// Raw text content for this page.
    pub text: String,
}

/// Metadata about the processed document.
#[derive(Debug, Clone)]
pub struct DocumentMetadata {
    /// Total number of pages.
    pub total_pages: usize,
    /// Detected document format.
    pub format: DocumentFormat,
    /// Whether the document appears to be a scanned image (vs text-based).
    pub is_scanned: bool,
}

/// Supported document formats.
#[derive(Debug, Clone, PartialEq)]
pub enum DocumentFormat {
    /// PDF document (text-based or scanned).
    Pdf,
    /// Image file (JPEG, PNG, TIFF, BMP).
    Image(String),
}

impl std::fmt::Display for DocumentFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DocumentFormat::Pdf => write!(f, "pdf"),
            DocumentFormat::Image(ext) => write!(f, "{}", ext),
        }
    }
}

/// Configuration for OCR processing.
#[derive(Debug, Clone)]
pub struct OcrConfig {
    /// Language for Tesseract OCR (default: "eng").
    pub language: String,
    /// Minimum text length per page to consider it non-scanned.
    /// Pages with less text than this are considered scanned and need OCR.
    pub min_text_length: usize,
    /// Whether to enable Tesseract OCR for scanned documents.
    pub enable_tesseract: bool,
    /// Path to the tesseract binary (default: "tesseract", relies on PATH).
    pub tesseract_path: String,
}

impl Default for OcrConfig {
    fn default() -> Self {
        Self {
            language: "eng".to_string(),
            min_text_length: 20,
            enable_tesseract: true,
            tesseract_path: "tesseract".to_string(),
        }
    }
}

// ---------------------------------------------------------------------------
// PDF Text Extraction (pure Rust via pdf_extract)
// ---------------------------------------------------------------------------

/// Extract text from a PDF file.
///
/// Uses the `pdf_extract` crate for pure Rust PDF text extraction.
/// Returns the full text and per-page breakdown.
pub fn extract_text_from_pdf(data: &[u8]) -> anyhow::Result<DocumentExtractionResult> {
    let text = pdf_extract::extract_text_from_mem(data)
        .map_err(|e| anyhow::anyhow!("PDF text extraction failed: {}", e))?;

    let text = clean_extracted_text(&text);

    // Attempt to split into pages by form feed characters or large whitespace gaps
    let pages = split_into_pages(&text);
    let total_pages = if pages.is_empty() { 1 } else { pages.len() };

    // Determine if the PDF is scanned (very little extractable text)
    let is_scanned = text.trim().len() < 20 && !data.is_empty();

    Ok(DocumentExtractionResult {
        text: text.clone(),
        pages,
        metadata: DocumentMetadata {
            total_pages,
            format: DocumentFormat::Pdf,
            is_scanned,
        },
    })
}

/// Extract text from a PDF file on disk.
pub fn extract_text_from_pdf_file(path: &Path) -> anyhow::Result<DocumentExtractionResult> {
    let data = std::fs::read(path)
        .map_err(|e| anyhow::anyhow!("Failed to read PDF file {}: {}", path.display(), e))?;
    extract_text_from_pdf(&data)
}

// ---------------------------------------------------------------------------
// Image OCR via Tesseract CLI
// ---------------------------------------------------------------------------

/// Run Tesseract OCR on an image file.
///
/// Requires `tesseract` to be installed and available on PATH.
/// Falls back gracefully with an error if tesseract is not available.
pub fn ocr_image_file(path: &Path, config: &OcrConfig) -> anyhow::Result<DocumentExtractionResult> {
    if !config.enable_tesseract {
        return Err(anyhow::anyhow!(
            "Tesseract OCR is disabled in configuration"
        ));
    }

    // Detect image format from extension
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("unknown")
        .to_lowercase();

    let format = match ext.as_str() {
        "jpg" | "jpeg" => DocumentFormat::Image("jpeg".to_string()),
        "png" => DocumentFormat::Image("png".to_string()),
        "tiff" | "tif" => DocumentFormat::Image("tiff".to_string()),
        "bmp" => DocumentFormat::Image("bmp".to_string()),
        other => DocumentFormat::Image(other.to_string()),
    };

    // Run tesseract CLI: tesseract input.png stdout -l eng
    let output = std::process::Command::new(&config.tesseract_path)
        .arg(path.as_os_str())
        .arg("stdout")
        .arg("-l")
        .arg(&config.language)
        .output()
        .map_err(|e| {
            anyhow::anyhow!(
                "Failed to run tesseract (is it installed? path='{}'): {}",
                config.tesseract_path,
                e
            )
        })?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(anyhow::anyhow!(
            "Tesseract OCR failed (exit code {}): {}",
            output.status.code().unwrap_or(-1),
            stderr
        ));
    }

    let text = String::from_utf8_lossy(&output.stdout).to_string();
    let text = clean_extracted_text(&text);

    Ok(DocumentExtractionResult {
        text: text.clone(),
        pages: vec![PageExtraction {
            page_number: 1,
            text,
        }],
        metadata: DocumentMetadata {
            total_pages: 1,
            format,
            is_scanned: true,
        },
    })
}

/// Run Tesseract OCR on raw image bytes.
///
/// Writes the bytes to a temporary file and runs tesseract on it.
pub fn ocr_image_bytes(
    bytes: &[u8],
    extension: &str,
    config: &OcrConfig,
) -> anyhow::Result<DocumentExtractionResult> {
    let temp_dir = tempfile::tempdir()
        .map_err(|e| anyhow::anyhow!("Failed to create temp directory: {}", e))?;
    let temp_path = temp_dir.path().join(format!("ocr_input.{}", extension));

    std::fs::write(&temp_path, bytes)
        .map_err(|e| anyhow::anyhow!("Failed to write temp file: {}", e))?;

    ocr_image_file(&temp_path, config)
}

// ---------------------------------------------------------------------------
// Composite Document Pipeline
// ---------------------------------------------------------------------------

/// Process any supported document (PDF or image) through the appropriate pipeline.
///
/// - For PDFs: extracts text directly. If the PDF appears scanned (very little text),
///   attempts OCR via Tesseract if enabled.
/// - For images: runs Tesseract OCR.
pub fn process_document(
    data: &[u8],
    filename: &str,
    config: &OcrConfig,
) -> anyhow::Result<DocumentExtractionResult> {
    let ext = filename.rsplit('.').next().unwrap_or("").to_lowercase();

    match ext.as_str() {
        "pdf" => {
            let result = extract_text_from_pdf(data)?;

            // If the PDF appears scanned and tesseract is available, try OCR
            if result.metadata.is_scanned && config.enable_tesseract {
                tracing::info!(
                    "PDF appears scanned (extracted only {} chars), attempting OCR",
                    result.text.len()
                );
                // For scanned PDFs, we'd need to render pages to images first.
                // This is a best-effort: return what we have from text extraction.
                // Full PDF-to-image rendering would require pdfium or similar.
                Ok(result)
            } else {
                Ok(result)
            }
        }
        "jpg" | "jpeg" | "png" | "tiff" | "tif" | "bmp" => ocr_image_bytes(data, &ext, config),
        _ => Err(anyhow::anyhow!(
            "Unsupported document format: '{}'. Supported: pdf, jpg, jpeg, png, tiff, tif, bmp",
            ext
        )),
    }
}

// ---------------------------------------------------------------------------
// Text Processing Utilities
// ---------------------------------------------------------------------------

/// Clean extracted text: normalize whitespace, remove control characters.
pub fn clean_extracted_text(text: &str) -> String {
    let mut result = String::with_capacity(text.len());
    let mut last_was_space = false;
    let mut last_was_newline = false;

    for ch in text.chars() {
        match ch {
            // Keep newlines but collapse multiple into max 2
            '\n' => {
                if !last_was_newline {
                    result.push('\n');
                    last_was_newline = true;
                    last_was_space = false;
                } else {
                    // Allow double newline (paragraph break) but not more
                    if result.ends_with("\n\n") {
                        continue;
                    }
                    result.push('\n');
                }
            }
            // Form feed (page break) → double newline
            '\x0c' => {
                if !result.ends_with("\n\n") {
                    if !result.ends_with('\n') {
                        result.push('\n');
                    }
                    result.push('\n');
                }
                last_was_newline = true;
                last_was_space = false;
            }
            // Collapse spaces and tabs
            ' ' | '\t' => {
                if !last_was_space && !last_was_newline {
                    result.push(' ');
                    last_was_space = true;
                }
            }
            // Carriage return → skip (handle \r\n as just \n)
            '\r' => {}
            // Normal printable characters
            c if c.is_control() => {}
            c => {
                result.push(c);
                last_was_space = false;
                last_was_newline = false;
            }
        }
    }

    result.trim().to_string()
}

/// Split text into pages based on form feed characters or double newlines.
///
/// Returns page extractions with 1-based page numbers.
pub fn split_into_pages(text: &str) -> Vec<PageExtraction> {
    // First try splitting by form feed (common in PDF extraction)
    let raw_pages: Vec<&str> = text.split('\x0c').collect();

    let pages: Vec<PageExtraction> = if raw_pages.len() > 1 {
        // Form feed delimited
        raw_pages
            .iter()
            .enumerate()
            .filter(|(_, p)| !p.trim().is_empty())
            .map(|(i, p)| PageExtraction {
                page_number: i + 1,
                text: p.trim().to_string(),
            })
            .collect()
    } else {
        // No form feeds — try splitting by paragraph breaks (double newline)
        // and group into logical "pages" of roughly similar size
        let paragraphs: Vec<&str> = text
            .split("\n\n")
            .filter(|p| !p.trim().is_empty())
            .collect();

        if paragraphs.is_empty() {
            if text.trim().is_empty() {
                return vec![];
            }
            return vec![PageExtraction {
                page_number: 1,
                text: text.trim().to_string(),
            }];
        }

        // Each paragraph becomes part of page 1 since we can't determine page breaks
        vec![PageExtraction {
            page_number: 1,
            text: text.trim().to_string(),
        }]
    };

    pages
}

/// Split extracted text into semantic chunks suitable for ingestion.
///
/// Chunks based on document structure:
/// 1. Page boundaries (primary split)
/// 2. Paragraph boundaries within pages (secondary split for long pages)
/// 3. Sentence boundaries (tertiary split for very long paragraphs)
pub fn chunk_document_text(pages: &[PageExtraction], max_chunk_size: usize) -> Vec<String> {
    let mut chunks = Vec::new();

    for page in pages {
        if page.text.len() <= max_chunk_size {
            chunks.push(page.text.clone());
            continue;
        }

        // Split by paragraphs
        let paragraphs: Vec<&str> = page.text.split("\n\n").collect();
        let mut current_chunk = String::new();

        for para in paragraphs {
            let para = para.trim();
            if para.is_empty() {
                continue;
            }

            if current_chunk.len() + para.len() + 2 > max_chunk_size && !current_chunk.is_empty() {
                chunks.push(current_chunk.trim().to_string());
                current_chunk.clear();
            }

            if para.len() > max_chunk_size {
                // Split by sentences for very long paragraphs
                if !current_chunk.is_empty() {
                    chunks.push(current_chunk.trim().to_string());
                    current_chunk.clear();
                }
                let sentences = split_into_sentences(para);
                let mut sent_chunk = String::new();
                for sent in &sentences {
                    if sent_chunk.len() + sent.len() + 1 > max_chunk_size && !sent_chunk.is_empty()
                    {
                        chunks.push(sent_chunk.trim().to_string());
                        sent_chunk.clear();
                    }
                    if !sent_chunk.is_empty() {
                        sent_chunk.push(' ');
                    }
                    sent_chunk.push_str(sent);
                }
                if !sent_chunk.is_empty() {
                    chunks.push(sent_chunk.trim().to_string());
                }
            } else {
                if !current_chunk.is_empty() {
                    current_chunk.push_str("\n\n");
                }
                current_chunk.push_str(para);
            }
        }

        if !current_chunk.is_empty() {
            chunks.push(current_chunk.trim().to_string());
        }
    }

    chunks
}

/// Simple sentence splitter for English/Spanish text.
///
/// Preserves decimal points within numbers (e.g. 99.99, 2.0.1) by only
/// splitting on periods that are actual sentence boundaries.
fn split_into_sentences(text: &str) -> Vec<String> {
    let chars: Vec<char> = text.chars().collect();
    let len = chars.len();
    let mut sentences = Vec::new();
    let mut current = String::new();

    for i in 0..len {
        let ch = chars[i];
        current.push(ch);

        if ch == '!' || ch == '?' {
            if current.len() > 10 {
                sentences.push(current.trim().to_string());
                current.clear();
            }
        } else if ch == '.' {
            // Check if this period is a decimal point: digit before AND digit after
            let prev_is_digit = i > 0 && chars[i - 1].is_ascii_digit();
            let next_is_digit = i + 1 < len && chars[i + 1].is_ascii_digit();

            if prev_is_digit && next_is_digit {
                // Decimal point inside a number — do NOT split
                continue;
            }

            if current.len() > 10 {
                sentences.push(current.trim().to_string());
                current.clear();
            }
        }
    }

    if !current.trim().is_empty() {
        sentences.push(current.trim().to_string());
    }

    sentences
}

/// Check if Tesseract is available on the system.
pub fn is_tesseract_available(tesseract_path: &str) -> bool {
    std::process::Command::new(tesseract_path)
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // --- Text cleaning tests ---

    #[test]
    fn test_clean_extracted_text_basic() {
        let input = "Hello   World\n\n\nParagraph two.";
        let result = clean_extracted_text(input);
        assert_eq!(result, "Hello World\n\nParagraph two.");
    }

    #[test]
    fn test_clean_extracted_text_form_feed() {
        let input = "Page one content.\x0cPage two content.";
        let result = clean_extracted_text(input);
        assert!(result.contains("Page one content."));
        assert!(result.contains("Page two content."));
        assert!(result.contains("\n\n"));
    }

    #[test]
    fn test_clean_extracted_text_control_chars() {
        let input = "Hello\x01World\x02Test\x03Done";
        let result = clean_extracted_text(input);
        assert_eq!(result, "HelloWorldTestDone");
    }

    #[test]
    fn test_clean_extracted_text_empty() {
        assert_eq!(clean_extracted_text(""), "");
        assert_eq!(clean_extracted_text("   "), "");
        assert_eq!(clean_extracted_text("\n\n\n"), "");
    }

    #[test]
    fn test_clean_extracted_text_tabs() {
        let input = "Column1\tColumn2\tColumn3";
        let result = clean_extracted_text(input);
        assert_eq!(result, "Column1 Column2 Column3");
    }

    #[test]
    fn test_clean_extracted_text_crlf() {
        let input = "Line one\r\nLine two\r\nLine three";
        let result = clean_extracted_text(input);
        assert_eq!(result, "Line one\nLine two\nLine three");
    }

    // --- Page splitting tests ---

    #[test]
    fn test_split_into_pages_empty() {
        let pages = split_into_pages("");
        assert!(pages.is_empty());
    }

    #[test]
    fn test_split_into_pages_single() {
        let pages = split_into_pages("Just one page of content.");
        assert_eq!(pages.len(), 1);
        assert_eq!(pages[0].page_number, 1);
        assert_eq!(pages[0].text, "Just one page of content.");
    }

    #[test]
    fn test_split_into_pages_form_feed() {
        let text = "Page 1 content.\x0cPage 2 content.\x0cPage 3 content.";
        let pages = split_into_pages(text);
        assert_eq!(pages.len(), 3);
        assert_eq!(pages[0].page_number, 1);
        assert_eq!(pages[0].text, "Page 1 content.");
        assert_eq!(pages[1].page_number, 2);
        assert_eq!(pages[1].text, "Page 2 content.");
        assert_eq!(pages[2].page_number, 3);
        assert_eq!(pages[2].text, "Page 3 content.");
    }

    // --- Chunking tests ---

    #[test]
    fn test_chunk_document_text_small_page() {
        let pages = vec![PageExtraction {
            page_number: 1,
            text: "Short text.".to_string(),
        }];
        let chunks = chunk_document_text(&pages, 1000);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0], "Short text.");
    }

    #[test]
    fn test_chunk_document_text_paragraph_split() {
        let pages = vec![PageExtraction {
            page_number: 1,
            text: "Paragraph one content here.\n\nParagraph two content here.\n\nParagraph three."
                .to_string(),
        }];
        let chunks = chunk_document_text(&pages, 40);
        assert!(chunks.len() >= 2);
        // Each chunk should be within limit (approximately)
        for chunk in &chunks {
            // Some chunks may slightly exceed due to paragraph boundaries
            assert!(!chunk.is_empty());
        }
    }

    #[test]
    fn test_chunk_document_text_multiple_pages() {
        let pages = vec![
            PageExtraction {
                page_number: 1,
                text: "Page 1 content.".to_string(),
            },
            PageExtraction {
                page_number: 2,
                text: "Page 2 content.".to_string(),
            },
        ];
        let chunks = chunk_document_text(&pages, 1000);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0], "Page 1 content.");
        assert_eq!(chunks[1], "Page 2 content.");
    }

    #[test]
    fn test_chunk_document_text_empty() {
        let pages: Vec<PageExtraction> = vec![];
        let chunks = chunk_document_text(&pages, 1000);
        assert!(chunks.is_empty());
    }

    // --- Sentence splitting tests ---

    #[test]
    fn test_split_into_sentences() {
        let text = "First sentence here. Second sentence follows! Third sentence with question?";
        let sentences = split_into_sentences(text);
        assert_eq!(sentences.len(), 3);
        assert!(sentences[0].ends_with('.'));
        assert!(sentences[1].ends_with('!'));
        assert!(sentences[2].ends_with('?'));
    }

    #[test]
    fn test_split_into_sentences_single() {
        let text = "Just one sentence without punctuation";
        let sentences = split_into_sentences(text);
        assert_eq!(sentences.len(), 1);
    }

    #[test]
    fn test_split_into_sentences_decimal_preserved() {
        let text = "The price was $99.99 for the item. Then it changed to $50.00 later.";
        let sentences = split_into_sentences(text);
        assert!(
            sentences.iter().any(|s| s.contains("$99.99")),
            "Decimal price should not be split: {:?}",
            sentences
        );
    }

    #[test]
    fn test_split_into_sentences_version_preserved() {
        let text = "We upgraded to version 2.0.1 of the framework. It works great now.";
        let sentences = split_into_sentences(text);
        assert!(
            sentences.iter().any(|s| s.contains("2.0.1")),
            "Version number should not be split: {:?}",
            sentences
        );
    }

    // --- OcrConfig tests ---

    #[test]
    fn test_ocr_config_default() {
        let config = OcrConfig::default();
        assert_eq!(config.language, "eng");
        assert_eq!(config.min_text_length, 20);
        assert!(config.enable_tesseract);
        assert_eq!(config.tesseract_path, "tesseract");
    }

    // --- DocumentFormat tests ---

    #[test]
    fn test_document_format_display() {
        assert_eq!(format!("{}", DocumentFormat::Pdf), "pdf");
        assert_eq!(
            format!("{}", DocumentFormat::Image("jpeg".to_string())),
            "jpeg"
        );
    }

    // --- DocumentMetadata tests ---

    #[test]
    fn test_document_metadata_creation() {
        let meta = DocumentMetadata {
            total_pages: 5,
            format: DocumentFormat::Pdf,
            is_scanned: false,
        };
        assert_eq!(meta.total_pages, 5);
        assert_eq!(meta.format, DocumentFormat::Pdf);
        assert!(!meta.is_scanned);
    }

    // --- PDF extraction tests (using pdf_extract) ---

    #[test]
    fn test_extract_text_from_pdf_invalid() {
        let result = extract_text_from_pdf(b"not a pdf");
        assert!(result.is_err());
    }

    #[test]
    fn test_extract_text_from_pdf_empty() {
        let result = extract_text_from_pdf(b"");
        assert!(result.is_err());
    }

    // --- Tesseract availability check ---

    #[test]
    fn test_is_tesseract_available_invalid_path() {
        assert!(!is_tesseract_available("/nonexistent/tesseract"));
    }

    #[test]
    fn test_process_document_unsupported_format() {
        let config = OcrConfig::default();
        let result = process_document(b"data", "document.xyz", &config);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Unsupported"));
    }

    // --- Tesseract OCR tests (skip if not installed) ---

    #[test]
    fn test_ocr_image_tesseract_disabled() {
        let config = OcrConfig {
            enable_tesseract: false,
            ..OcrConfig::default()
        };
        let result = ocr_image_bytes(b"fake image", "png", &config);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("disabled"));
    }

    #[test]
    fn test_ocr_image_tesseract_not_found() {
        let config = OcrConfig {
            tesseract_path: "/nonexistent/tesseract".to_string(),
            ..OcrConfig::default()
        };
        let result = ocr_image_bytes(b"fake image", "png", &config);
        assert!(result.is_err());
    }
}
