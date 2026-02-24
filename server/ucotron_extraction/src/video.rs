//! Video frame extraction pipeline using FFmpeg.
//!
//! Provides [`FfmpegVideoPipeline`] for extracting keyframes from video files
//! with configurable FPS rate and scene change detection.
//!
//! # Architecture
//!
//! Uses `ffmpeg-next` (safe FFmpeg bindings) to:
//! 1. Open video files and find the best video stream
//! 2. Decode frames at configurable intervals
//! 3. Detect keyframes and compute scene change scores via frame histogram diff
//! 4. Output RGB image bytes with timestamps
//!
//! # Tests
//!
//! Mock-based tests validate config, scoring, and histogram logic without
//! requiring actual video files. Integration tests with real videos are
//! gated behind the presence of test fixtures.

use anyhow::{Context, Result};
use std::path::Path;

/// Configuration for video frame extraction.
#[derive(Debug, Clone)]
pub struct VideoConfig {
    /// Target frames per second to extract (default: 1.0).
    /// Set lower for less frames, higher for more granularity.
    pub fps: f64,
    /// Scene change detection threshold (0.0-1.0, default: 0.3).
    /// Lower values are more sensitive to scene changes.
    pub scene_change_threshold: f64,
    /// Maximum number of frames to extract (0 = unlimited).
    pub max_frames: usize,
    /// Output image width (0 = original width).
    pub output_width: u32,
    /// Output image height (0 = original height).
    pub output_height: u32,
}

impl Default for VideoConfig {
    fn default() -> Self {
        Self {
            fps: 1.0,
            scene_change_threshold: 0.3,
            max_frames: 0,
            output_width: 0,
            output_height: 0,
        }
    }
}

/// A single extracted video frame with metadata.
#[derive(Debug, Clone)]
pub struct ExtractedFrame {
    /// Presentation timestamp in milliseconds from video start.
    pub timestamp_ms: u64,
    /// Whether this frame is a keyframe (I-frame) in the video codec.
    pub is_keyframe: bool,
    /// Scene change score (0.0 = no change, 1.0 = maximum change).
    /// Computed as normalized histogram difference from the previous frame.
    pub scene_change_score: f64,
    /// Frame width in pixels.
    pub width: u32,
    /// Frame height in pixels.
    pub height: u32,
    /// Raw RGB pixel data (3 bytes per pixel, row-major).
    pub rgb_data: Vec<u8>,
}

/// Result of frame extraction from a video file.
#[derive(Debug, Clone)]
pub struct FrameExtractionResult {
    /// All extracted frames ordered by timestamp.
    pub frames: Vec<ExtractedFrame>,
    /// Total video duration in milliseconds.
    pub duration_ms: u64,
    /// Original video width.
    pub video_width: u32,
    /// Original video height.
    pub video_height: u32,
    /// Video frame rate (fps).
    pub video_fps: f64,
    /// Total number of frames in the video (estimated).
    pub total_frames_estimated: u64,
}

/// FFmpeg-based video pipeline for frame extraction and keyframe detection.
///
/// Thread-safe: the pipeline itself is Send + Sync since it only holds config.
/// Each `extract_frames` call creates its own FFmpeg context.
pub struct FfmpegVideoPipeline {
    config: VideoConfig,
}

impl FfmpegVideoPipeline {
    /// Create a new video pipeline with the given configuration.
    pub fn new(config: VideoConfig) -> Self {
        // Initialize FFmpeg (safe to call multiple times).
        ffmpeg_next::init().ok();
        Self { config }
    }

    /// Extract frames from a video file.
    pub fn extract_frames(&self, path: &Path) -> Result<FrameExtractionResult> {
        let path_str = path
            .to_str()
            .context("Video path contains invalid UTF-8")?;

        // Open input file
        let mut ictx = ffmpeg_next::format::input(&path_str)
            .with_context(|| format!("Failed to open video file: {}", path_str))?;

        // Find best video stream
        let video_stream_index = ictx
            .streams()
            .best(ffmpeg_next::media::Type::Video)
            .context("No video stream found in file")?
            .index();

        // Get stream info
        let stream = ictx.stream(video_stream_index).unwrap();
        let time_base = stream.time_base();
        let stream_fps = f64::from(stream.avg_frame_rate());
        let duration_ticks = stream.duration();
        let duration_ms = if duration_ticks > 0 {
            (duration_ticks as f64 * f64::from(time_base) * 1000.0) as u64
        } else {
            // Fallback to container duration
            let container_dur = ictx.duration();
            if container_dur > 0 {
                (container_dur as u64) / 1000 // AV_TIME_BASE is microseconds
            } else {
                0
            }
        };

        let total_frames_estimated = if stream_fps > 0.0 && duration_ms > 0 {
            (stream_fps * duration_ms as f64 / 1000.0) as u64
        } else {
            0
        };

        // Set up decoder
        let codec_params = stream.parameters();
        let mut decoder = ffmpeg_next::codec::context::Context::from_parameters(codec_params)
            .context("Failed to create codec context")?
            .decoder()
            .video()
            .context("Failed to create video decoder")?;

        let src_width = decoder.width();
        let src_height = decoder.height();

        let out_width = if self.config.output_width > 0 {
            self.config.output_width
        } else {
            src_width
        };
        let out_height = if self.config.output_height > 0 {
            self.config.output_height
        } else {
            src_height
        };

        // Calculate frame interval based on target FPS
        let frame_interval = if self.config.fps > 0.0 && stream_fps > 0.0 {
            (stream_fps / self.config.fps).max(1.0) as u64
        } else {
            1
        };

        let mut frames = Vec::new();
        let mut frame_count: u64 = 0;
        let mut prev_histogram: Option<[u32; 256]> = None;

        // Set up pixel format converter (created lazily after first frame)
        let mut scaler: Option<ffmpeg_next::software::scaling::Context> = None;

        // Iterate packets
        for (stream_ref, packet) in ictx.packets() {
            if stream_ref.index() != video_stream_index {
                continue;
            }

            decoder.send_packet(&packet)?;

            let mut decoded_frame = ffmpeg_next::util::frame::video::Video::empty();
            while decoder.receive_frame(&mut decoded_frame).is_ok() {
                // Check if we should extract this frame based on interval
                if frame_count % frame_interval != 0 {
                    frame_count += 1;
                    continue;
                }

                // Check max_frames limit
                if self.config.max_frames > 0 && frames.len() >= self.config.max_frames {
                    return Ok(FrameExtractionResult {
                        frames,
                        duration_ms,
                        video_width: src_width,
                        video_height: src_height,
                        video_fps: stream_fps,
                        total_frames_estimated,
                    });
                }

                // Initialize scaler on first frame (now we know the pixel format)
                if scaler.is_none() {
                    scaler = Some(
                        ffmpeg_next::software::scaling::Context::get(
                            decoded_frame.format(),
                            src_width,
                            src_height,
                            ffmpeg_next::format::Pixel::RGB24,
                            out_width,
                            out_height,
                            ffmpeg_next::software::scaling::Flags::BILINEAR,
                        )
                        .context("Failed to create pixel format scaler")?,
                    );
                }

                // Convert to RGB
                let mut rgb_frame =
                    ffmpeg_next::util::frame::video::Video::empty();
                scaler.as_mut().unwrap().run(&decoded_frame, &mut rgb_frame)?;

                // Compute grayscale histogram for scene change detection
                let rgb_data = rgb_frame.data(0);
                let stride = rgb_frame.stride(0) as usize;
                let mut flat_rgb =
                    Vec::with_capacity((out_width * out_height * 3) as usize);
                for y in 0..out_height as usize {
                    let row_start = y * stride;
                    let row_end = row_start + (out_width as usize * 3);
                    if row_end <= rgb_data.len() {
                        flat_rgb.extend_from_slice(&rgb_data[row_start..row_end]);
                    }
                }

                let histogram = compute_grayscale_histogram(&flat_rgb);
                let scene_change_score = if let Some(ref prev) = prev_histogram {
                    histogram_diff(prev, &histogram)
                } else {
                    1.0 // First frame always has max score
                };
                prev_histogram = Some(histogram);

                // Calculate timestamp
                let pts = decoded_frame.pts().unwrap_or(0);
                let timestamp_ms =
                    (pts as f64 * f64::from(time_base) * 1000.0).max(0.0) as u64;

                let is_keyframe = decoded_frame.is_key();

                frames.push(ExtractedFrame {
                    timestamp_ms,
                    is_keyframe,
                    scene_change_score,
                    width: out_width,
                    height: out_height,
                    rgb_data: flat_rgb,
                });

                frame_count += 1;
            }

            // Increment even for non-decoded frames
            if frame_count == 0 {
                frame_count = 1;
            }
        }

        // Drain remaining frames
        decoder.send_eof()?;
        let mut decoded_frame = ffmpeg_next::util::frame::video::Video::empty();
        while decoder.receive_frame(&mut decoded_frame).is_ok() {
            if self.config.max_frames > 0 && frames.len() >= self.config.max_frames {
                break;
            }

            if let Some(ref mut s) = scaler {
                let mut rgb_frame = ffmpeg_next::util::frame::video::Video::empty();
                if s.run(&decoded_frame, &mut rgb_frame).is_ok() {
                    let rgb_data = rgb_frame.data(0);
                    let stride = rgb_frame.stride(0) as usize;
                    let mut flat_rgb =
                        Vec::with_capacity((out_width * out_height * 3) as usize);
                    for y in 0..out_height as usize {
                        let row_start = y * stride;
                        let row_end = row_start + (out_width as usize * 3);
                        if row_end <= rgb_data.len() {
                            flat_rgb.extend_from_slice(&rgb_data[row_start..row_end]);
                        }
                    }

                    let histogram = compute_grayscale_histogram(&flat_rgb);
                    let scene_change_score = if let Some(ref prev) = prev_histogram {
                        histogram_diff(prev, &histogram)
                    } else {
                        1.0
                    };
                    prev_histogram = Some(histogram);

                    let pts = decoded_frame.pts().unwrap_or(0);
                    let timestamp_ms =
                        (pts as f64 * f64::from(time_base) * 1000.0).max(0.0) as u64;
                    let is_keyframe = decoded_frame.is_key();

                    frames.push(ExtractedFrame {
                        timestamp_ms,
                        is_keyframe,
                        scene_change_score,
                        width: out_width,
                        height: out_height,
                        rgb_data: flat_rgb,
                    });
                }
            }
        }

        Ok(FrameExtractionResult {
            frames,
            duration_ms,
            video_width: src_width,
            video_height: src_height,
            video_fps: stream_fps,
            total_frames_estimated,
        })
    }
}

// Implement the trait from lib.rs
impl crate::VideoPipeline for FfmpegVideoPipeline {
    fn extract_frames(&self, path: &Path) -> Result<FrameExtractionResult> {
        self.extract_frames(path)
    }
}

// SAFETY: FfmpegVideoPipeline only holds VideoConfig (Clone + Send + Sync).
// Each extract_frames call creates its own FFmpeg context locally.
unsafe impl Send for FfmpegVideoPipeline {}
unsafe impl Sync for FfmpegVideoPipeline {}

/// Configuration for temporal segmentation of extracted frames.
#[derive(Debug, Clone)]
pub struct SegmentConfig {
    /// Minimum segment duration in milliseconds (default: 5000 = 5s).
    /// Segments shorter than this will be merged with the next segment.
    pub min_duration_ms: u64,
    /// Maximum segment duration in milliseconds (default: 30000 = 30s).
    /// Segments longer than this will be split at the next frame boundary.
    pub max_duration_ms: u64,
    /// Scene change threshold (0.0-1.0, default: 0.3).
    /// Frames with scene_change_score above this trigger a new segment.
    pub scene_change_threshold: f64,
}

impl Default for SegmentConfig {
    fn default() -> Self {
        Self {
            min_duration_ms: 5_000,
            max_duration_ms: 30_000,
            scene_change_threshold: 0.3,
        }
    }
}

/// A temporal segment grouping consecutive video frames.
///
/// Each segment represents a contiguous portion of the video with visually
/// similar content (no scene changes above the threshold).
#[derive(Debug, Clone)]
pub struct FrameSegment {
    /// Segment index (0-based).
    pub index: usize,
    /// Start timestamp in milliseconds.
    pub start_ms: u64,
    /// End timestamp in milliseconds.
    pub end_ms: u64,
    /// Indices into the original `FrameExtractionResult.frames` vec.
    pub frame_indices: Vec<usize>,
    /// Whether this segment starts with a detected scene change.
    pub is_scene_change: bool,
}

impl FrameSegment {
    /// Duration of this segment in milliseconds.
    pub fn duration_ms(&self) -> u64 {
        self.end_ms.saturating_sub(self.start_ms)
    }

    /// Number of frames in this segment.
    pub fn frame_count(&self) -> usize {
        self.frame_indices.len()
    }
}

/// Segment extracted frames into temporal groups based on scene changes and duration constraints.
///
/// Algorithm:
/// 1. Walk frames in timestamp order
/// 2. Start a new segment when:
///    - A frame's `scene_change_score` exceeds the threshold (and min duration is met), OR
///    - The current segment exceeds `max_duration_ms`
/// 3. After all frames are processed, merge any trailing segment shorter than `min_duration_ms`
///    into the previous segment (if one exists)
///
/// Returns an empty vec if `frames` is empty.
pub fn segment_frames(
    frames: &[ExtractedFrame],
    config: &SegmentConfig,
) -> Vec<FrameSegment> {
    if frames.is_empty() {
        return Vec::new();
    }

    let mut segments: Vec<FrameSegment> = Vec::new();
    let mut current_start_ms = frames[0].timestamp_ms;
    let mut current_indices: Vec<usize> = vec![0];
    let mut current_is_scene_change = true; // first segment always starts a "scene"

    for i in 1..frames.len() {
        let frame = &frames[i];
        let elapsed = frame.timestamp_ms.saturating_sub(current_start_ms);
        let is_scene = frame.scene_change_score >= config.scene_change_threshold;

        // Split segment if:
        // - scene change detected AND minimum duration met, OR
        // - maximum duration exceeded
        let should_split =
            (is_scene && elapsed >= config.min_duration_ms) || elapsed >= config.max_duration_ms;

        if should_split {
            // Finalize current segment — end time is this frame's timestamp
            // (the new segment starts at this frame)
            let end_ms = frame.timestamp_ms;
            segments.push(FrameSegment {
                index: segments.len(),
                start_ms: current_start_ms,
                end_ms,
                frame_indices: std::mem::take(&mut current_indices),
                is_scene_change: current_is_scene_change,
            });

            // Start new segment at this frame
            current_start_ms = frame.timestamp_ms;
            current_indices = vec![i];
            current_is_scene_change = is_scene;
        } else {
            current_indices.push(i);
        }
    }

    // Finalize the last segment
    let last_frame = &frames[frames.len() - 1];
    // End time = last frame timestamp + a small epsilon (we use the timestamp itself
    // since we don't know the frame duration; downstream can adjust if needed)
    let last_end_ms = last_frame.timestamp_ms;
    let last_segment = FrameSegment {
        index: segments.len(),
        start_ms: current_start_ms,
        end_ms: last_end_ms,
        frame_indices: current_indices,
        is_scene_change: current_is_scene_change,
    };

    // If the last segment is too short, merge it into the previous one
    if last_segment.duration_ms() < config.min_duration_ms && !segments.is_empty() {
        let prev = segments.last_mut().unwrap();
        prev.end_ms = last_segment.end_ms;
        prev.frame_indices.extend(last_segment.frame_indices);
    } else {
        segments.push(last_segment);
    }

    segments
}

/// Compute a 256-bin grayscale intensity histogram from RGB data.
///
/// Grayscale = 0.299*R + 0.587*G + 0.114*B (ITU-R BT.601).
pub(crate) fn compute_grayscale_histogram(rgb_data: &[u8]) -> [u32; 256] {
    let mut histogram = [0u32; 256];
    for pixel in rgb_data.chunks_exact(3) {
        let gray = (0.299 * pixel[0] as f64 + 0.587 * pixel[1] as f64 + 0.114 * pixel[2] as f64)
            as u8;
        histogram[gray as usize] += 1;
    }
    histogram
}

/// Compute normalized histogram difference between two frames.
///
/// Returns a value in [0.0, 1.0] where 0.0 = identical, 1.0 = maximally different.
/// Uses Chi-squared distance normalized by total pixel count.
pub(crate) fn histogram_diff(h1: &[u32; 256], h2: &[u32; 256]) -> f64 {
    let total1: u64 = h1.iter().map(|&v| v as u64).sum();
    let total2: u64 = h2.iter().map(|&v| v as u64).sum();

    if total1 == 0 || total2 == 0 {
        return 1.0;
    }

    // Normalize both histograms to probability distributions and compute
    // symmetric chi-squared divergence
    let mut chi_sq = 0.0f64;
    for i in 0..256 {
        let p = h1[i] as f64 / total1 as f64;
        let q = h2[i] as f64 / total2 as f64;
        let denom = p + q;
        if denom > 0.0 {
            chi_sq += (p - q).powi(2) / denom;
        }
    }

    // Chi-squared ranges [0, 2], normalize to [0, 1]
    (chi_sq / 2.0).clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_video_config_default() {
        let config = VideoConfig::default();
        assert!((config.fps - 1.0).abs() < f64::EPSILON);
        assert!((config.scene_change_threshold - 0.3).abs() < f64::EPSILON);
        assert_eq!(config.max_frames, 0);
        assert_eq!(config.output_width, 0);
        assert_eq!(config.output_height, 0);
    }

    #[test]
    fn test_video_config_custom() {
        let config = VideoConfig {
            fps: 2.5,
            scene_change_threshold: 0.5,
            max_frames: 100,
            output_width: 224,
            output_height: 224,
        };
        assert!((config.fps - 2.5).abs() < f64::EPSILON);
        assert_eq!(config.max_frames, 100);
        assert_eq!(config.output_width, 224);
    }

    #[test]
    fn test_histogram_identical() {
        let h = [100u32; 256];
        let diff = histogram_diff(&h, &h);
        assert!(diff.abs() < 1e-10, "Identical histograms should have diff=0, got {diff}");
    }

    #[test]
    fn test_histogram_completely_different() {
        let mut h1 = [0u32; 256];
        let mut h2 = [0u32; 256];
        h1[0] = 1000; // All pixels are black
        h2[255] = 1000; // All pixels are white
        let diff = histogram_diff(&h1, &h2);
        assert!(
            (diff - 1.0).abs() < 1e-10,
            "Maximally different histograms should have diff=1.0, got {diff}"
        );
    }

    #[test]
    fn test_histogram_partially_different() {
        let mut h1 = [0u32; 256];
        let mut h2 = [0u32; 256];
        h1[0] = 500;
        h1[128] = 500;
        h2[0] = 500;
        h2[255] = 500;
        let diff = histogram_diff(&h1, &h2);
        assert!(diff > 0.0 && diff < 1.0, "Partial diff should be in (0,1), got {diff}");
    }

    #[test]
    fn test_histogram_empty() {
        let h1 = [0u32; 256];
        let h2 = [0u32; 256];
        let diff = histogram_diff(&h1, &h2);
        assert!((diff - 1.0).abs() < 1e-10, "Empty histograms should return 1.0");
    }

    #[test]
    fn test_histogram_one_empty() {
        let h1 = [0u32; 256];
        let mut h2 = [0u32; 256];
        h2[128] = 1000;
        let diff = histogram_diff(&h1, &h2);
        assert!((diff - 1.0).abs() < 1e-10, "One empty histogram should return 1.0");
    }

    #[test]
    fn test_compute_grayscale_histogram_black() {
        // 3 black pixels (R=0, G=0, B=0)
        let rgb = vec![0u8; 9];
        let hist = compute_grayscale_histogram(&rgb);
        assert_eq!(hist[0], 3);
        for i in 1..256 {
            assert_eq!(hist[i], 0);
        }
    }

    #[test]
    fn test_compute_grayscale_histogram_white() {
        // 2 white pixels
        let rgb = vec![255u8; 6];
        let hist = compute_grayscale_histogram(&rgb);
        assert_eq!(hist[255], 2);
        assert_eq!(hist[0], 0);
    }

    #[test]
    fn test_compute_grayscale_histogram_mixed() {
        // Red pixel (255,0,0) → gray = 0.299*255 = 76
        // Green pixel (0,255,0) → gray = 0.587*255 = 149
        // Blue pixel (0,0,255) → gray = 0.114*255 = 29
        let rgb = vec![255, 0, 0, 0, 255, 0, 0, 0, 255];
        let hist = compute_grayscale_histogram(&rgb);
        assert_eq!(hist[76], 1); // Red
        assert_eq!(hist[149], 1); // Green
        assert_eq!(hist[29], 1); // Blue
    }

    #[test]
    fn test_extracted_frame_creation() {
        let frame = ExtractedFrame {
            timestamp_ms: 5000,
            is_keyframe: true,
            scene_change_score: 0.85,
            width: 640,
            height: 480,
            rgb_data: vec![0u8; 640 * 480 * 3],
        };
        assert_eq!(frame.timestamp_ms, 5000);
        assert!(frame.is_keyframe);
        assert!((frame.scene_change_score - 0.85).abs() < f64::EPSILON);
        assert_eq!(frame.rgb_data.len(), 640 * 480 * 3);
    }

    #[test]
    fn test_frame_extraction_result_creation() {
        let result = FrameExtractionResult {
            frames: vec![],
            duration_ms: 60000,
            video_width: 1920,
            video_height: 1080,
            video_fps: 30.0,
            total_frames_estimated: 1800,
        };
        assert_eq!(result.duration_ms, 60000);
        assert_eq!(result.video_width, 1920);
        assert!((result.video_fps - 30.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_pipeline_creation() {
        let pipeline = FfmpegVideoPipeline::new(VideoConfig::default());
        assert!((pipeline.config.fps - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_pipeline_with_custom_config() {
        let config = VideoConfig {
            fps: 0.5,
            scene_change_threshold: 0.2,
            max_frames: 50,
            output_width: 320,
            output_height: 240,
        };
        let pipeline = FfmpegVideoPipeline::new(config);
        assert!((pipeline.config.fps - 0.5).abs() < f64::EPSILON);
        assert_eq!(pipeline.config.max_frames, 50);
    }

    #[test]
    fn test_extract_frames_nonexistent_file() {
        let pipeline = FfmpegVideoPipeline::new(VideoConfig::default());
        let result = pipeline.extract_frames(Path::new("/nonexistent/video.mp4"));
        assert!(result.is_err());
    }

    #[test]
    fn test_histogram_diff_symmetric() {
        let mut h1 = [0u32; 256];
        let mut h2 = [0u32; 256];
        h1[0] = 100;
        h1[50] = 200;
        h2[50] = 100;
        h2[200] = 300;
        let d1 = histogram_diff(&h1, &h2);
        let d2 = histogram_diff(&h2, &h1);
        assert!(
            (d1 - d2).abs() < 1e-10,
            "histogram_diff should be symmetric: {d1} vs {d2}"
        );
    }

    #[test]
    fn test_scene_change_score_range() {
        // Generate a bunch of random-ish histograms and verify scores are in [0,1]
        for offset in 0..10 {
            let mut h1 = [0u32; 256];
            let mut h2 = [0u32; 256];
            for i in 0..256 {
                h1[i] = ((i + offset) % 50) as u32;
                h2[i] = ((i + offset * 3) % 70) as u32;
            }
            let score = histogram_diff(&h1, &h2);
            assert!(
                (0.0..=1.0).contains(&score),
                "Score {score} out of range for offset {offset}"
            );
        }
    }

    // --- Temporal segmentation tests ---

    /// Helper: create a frame with given timestamp and scene change score.
    fn make_frame(timestamp_ms: u64, scene_change_score: f64) -> ExtractedFrame {
        ExtractedFrame {
            timestamp_ms,
            is_keyframe: false,
            scene_change_score,
            width: 64,
            height: 64,
            rgb_data: vec![0u8; 64 * 64 * 3],
        }
    }

    #[test]
    fn test_segment_config_default() {
        let config = SegmentConfig::default();
        assert_eq!(config.min_duration_ms, 5_000);
        assert_eq!(config.max_duration_ms, 30_000);
        assert!((config.scene_change_threshold - 0.3).abs() < f64::EPSILON);
    }

    #[test]
    fn test_segment_empty_frames() {
        let segments = segment_frames(&[], &SegmentConfig::default());
        assert!(segments.is_empty());
    }

    #[test]
    fn test_segment_single_frame() {
        let frames = vec![make_frame(0, 1.0)];
        let segments = segment_frames(&frames, &SegmentConfig::default());
        assert_eq!(segments.len(), 1);
        assert_eq!(segments[0].start_ms, 0);
        assert_eq!(segments[0].end_ms, 0);
        assert_eq!(segments[0].frame_indices, vec![0]);
        assert!(segments[0].is_scene_change);
    }

    #[test]
    fn test_segment_no_scene_changes() {
        // 10 frames spanning 20s, no scene changes — all below threshold
        let frames: Vec<_> = (0..10)
            .map(|i| make_frame(i * 2000, 0.1))
            .collect();
        let segments = segment_frames(&frames, &SegmentConfig::default());
        // All frames should be in a single segment (no splits triggered)
        assert_eq!(segments.len(), 1);
        assert_eq!(segments[0].frame_indices.len(), 10);
        assert_eq!(segments[0].start_ms, 0);
        assert_eq!(segments[0].end_ms, 18000);
    }

    #[test]
    fn test_segment_scene_change_splits() {
        // 8 frames: 0s, 2s, 4s, 6s, 8s, 10s, 12s, 14s
        // Scene change at frame 3 (6s) — score 0.8 > threshold 0.3
        // Min duration 5s is met (6s - 0s = 6s >= 5s)
        // Second segment (6s-14s = 8s) is long enough to NOT be merged
        let frames = vec![
            make_frame(0, 1.0),      // segment 0 start
            make_frame(2000, 0.1),   // same scene
            make_frame(4000, 0.1),   // same scene
            make_frame(6000, 0.8),   // scene change! elapsed=6s >= min=5s → split
            make_frame(8000, 0.1),   // new segment
            make_frame(10000, 0.1),
            make_frame(12000, 0.1),
            make_frame(14000, 0.1),  // end — tail is 8s >= min 5s
        ];
        let config = SegmentConfig {
            min_duration_ms: 5_000,
            max_duration_ms: 30_000,
            scene_change_threshold: 0.3,
        };
        let segments = segment_frames(&frames, &config);
        assert_eq!(segments.len(), 2);
        // First segment: frames 0,1,2
        assert_eq!(segments[0].frame_indices, vec![0, 1, 2]);
        assert_eq!(segments[0].start_ms, 0);
        assert_eq!(segments[0].end_ms, 6000);
        // Second segment: frames 3,4,5,6,7
        assert_eq!(segments[1].frame_indices, vec![3, 4, 5, 6, 7]);
        assert_eq!(segments[1].start_ms, 6000);
        assert_eq!(segments[1].end_ms, 14000);
        assert!(segments[1].is_scene_change);
    }

    #[test]
    fn test_segment_scene_change_before_min_duration() {
        // Scene change at frame 1 (1s), but min duration is 5s — should NOT split
        let frames = vec![
            make_frame(0, 1.0),
            make_frame(1000, 0.9), // scene change but only 1s elapsed
            make_frame(2000, 0.1),
            make_frame(3000, 0.1),
        ];
        let config = SegmentConfig {
            min_duration_ms: 5_000,
            max_duration_ms: 30_000,
            scene_change_threshold: 0.3,
        };
        let segments = segment_frames(&frames, &config);
        // Should be 1 segment since min_duration wasn't met for the scene change
        assert_eq!(segments.len(), 1);
        assert_eq!(segments[0].frame_indices.len(), 4);
    }

    #[test]
    fn test_segment_max_duration_split() {
        // 7 frames spanning 35s — max duration is 10s
        let frames: Vec<_> = (0..7)
            .map(|i| make_frame(i * 5000, 0.05)) // no scene changes
            .collect();
        let config = SegmentConfig {
            min_duration_ms: 3_000,
            max_duration_ms: 10_000,
            scene_change_threshold: 0.3,
        };
        let segments = segment_frames(&frames, &config);
        // Should split at 10s and 20s boundaries
        assert!(segments.len() >= 3, "Expected at least 3 segments for 35s video with 10s max, got {}", segments.len());
        for seg in &segments {
            // No individual segment should exceed max_duration (except possibly last due to merge)
            if seg.index < segments.len() - 1 {
                assert!(seg.duration_ms() <= config.max_duration_ms,
                    "Segment {} duration {}ms exceeds max {}ms",
                    seg.index, seg.duration_ms(), config.max_duration_ms);
            }
        }
    }

    #[test]
    fn test_segment_short_tail_merged() {
        // 5 frames: 0s, 5s, 10s, 15s, 16s
        // Scene change at 15s → split. Remaining segment (15s-16s = 1s) < min (5s) → merge back
        let frames = vec![
            make_frame(0, 1.0),
            make_frame(5000, 0.1),
            make_frame(10000, 0.1),
            make_frame(15000, 0.8), // scene change, elapsed=15s
            make_frame(16000, 0.1), // short tail (1s)
        ];
        let config = SegmentConfig {
            min_duration_ms: 5_000,
            max_duration_ms: 30_000,
            scene_change_threshold: 0.3,
        };
        let segments = segment_frames(&frames, &config);
        // The tail segment (1s) should be merged into previous
        assert_eq!(segments.len(), 1, "Short tail should be merged back");
        assert_eq!(segments[0].frame_indices.len(), 5);
    }

    #[test]
    fn test_segment_indices_sequential() {
        let frames: Vec<_> = (0..20)
            .map(|i| {
                let score = if i % 5 == 0 && i > 0 { 0.9 } else { 0.05 };
                make_frame(i * 2000, score)
            })
            .collect();
        let config = SegmentConfig {
            min_duration_ms: 5_000,
            max_duration_ms: 60_000,
            scene_change_threshold: 0.3,
        };
        let segments = segment_frames(&frames, &config);

        // Verify: segment indices are sequential
        for (i, seg) in segments.iter().enumerate() {
            assert_eq!(seg.index, i, "Segment index mismatch");
        }

        // Verify: all frame indices covered exactly once
        let mut all_indices: Vec<usize> = segments
            .iter()
            .flat_map(|s| s.frame_indices.iter().copied())
            .collect();
        all_indices.sort();
        let expected: Vec<usize> = (0..20).collect();
        assert_eq!(all_indices, expected, "All frames should be covered exactly once");
    }

    #[test]
    fn test_frame_segment_duration() {
        let seg = FrameSegment {
            index: 0,
            start_ms: 5000,
            end_ms: 15000,
            frame_indices: vec![1, 2, 3],
            is_scene_change: false,
        };
        assert_eq!(seg.duration_ms(), 10000);
        assert_eq!(seg.frame_count(), 3);
    }

    #[test]
    fn test_segment_multiple_scene_changes() {
        // Multiple scene changes with sufficient gaps.
        // Each segment must be >= min_duration (5s) to avoid tail merge.
        let frames = vec![
            make_frame(0, 1.0),       // seg 0
            make_frame(3000, 0.05),
            make_frame(6000, 0.9),    // scene change at 6s (>= 5s min) → seg 1
            make_frame(9000, 0.05),
            make_frame(12000, 0.9),   // scene change at 12s (6s since seg1 start >= 5s) → seg 2
            make_frame(15000, 0.05),
            make_frame(18000, 0.9),   // scene change at 18s (6s since seg2 start >= 5s) → seg 3
            make_frame(21000, 0.05),
            make_frame(24000, 0.05),  // ensure last segment is 6s (18s-24s >= 5s min)
        ];
        let config = SegmentConfig {
            min_duration_ms: 5_000,
            max_duration_ms: 60_000,
            scene_change_threshold: 0.3,
        };
        let segments = segment_frames(&frames, &config);
        assert_eq!(segments.len(), 4, "Expected 4 segments from 3 scene changes + initial");
        assert!(segments[1].is_scene_change);
        assert!(segments[2].is_scene_change);
        assert!(segments[3].is_scene_change);
    }

    // Integration test: only runs if a test video exists
    #[test]
    fn test_extract_frames_with_real_video() {
        let test_video = Path::new("../models/test_video.mp4");
        if !test_video.exists() {
            eprintln!("Skipping real video test: test_video.mp4 not found");
            return;
        }

        let config = VideoConfig {
            fps: 1.0,
            max_frames: 5,
            ..Default::default()
        };
        let pipeline = FfmpegVideoPipeline::new(config);
        let result = pipeline.extract_frames(test_video).expect("Should extract frames");

        assert!(!result.frames.is_empty(), "Should extract at least one frame");
        assert!(result.video_width > 0, "Video width should be positive");
        assert!(result.video_height > 0, "Video height should be positive");

        for frame in &result.frames {
            assert!(frame.width > 0);
            assert!(frame.height > 0);
            assert!(!frame.rgb_data.is_empty());
            assert!(
                (0.0..=1.0).contains(&frame.scene_change_score),
                "Scene score out of range"
            );
        }

        // First frame should have score 1.0
        assert!(
            (result.frames[0].scene_change_score - 1.0).abs() < f64::EPSILON,
            "First frame should have scene_change_score=1.0"
        );
    }
}
