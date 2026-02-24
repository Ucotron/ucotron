//! # Audio Transcription Pipeline
//!
//! Voice-to-memory ingestion using Whisper ONNX models for speech-to-text
//! transcription. Supports WAV, and provides integration with the existing
//! ingestion pipeline to store transcribed text as memory.
//!
//! ## Architecture
//!
//! The pipeline:
//! 1. Loads audio from a WAV file (mono, 16kHz, 16-bit PCM — resamples if needed)
//! 2. Computes a log-mel spectrogram (80 mel bins, 3000 frames = 30s chunks)
//! 3. Runs the Whisper ONNX encoder to produce cross-attention KV cache
//! 4. Runs the Whisper ONNX decoder in an autoregressive loop (greedy decoding)
//! 5. Decodes generated tokens back to text
//!
//! ## Usage
//!
//! ```no_run
//! use ucotron_extraction::audio::{WhisperOnnxPipeline, WhisperConfig};
//! use ucotron_extraction::TranscriptionPipeline;
//!
//! let pipeline = WhisperOnnxPipeline::new(
//!     "models/whisper-tiny/encoder.onnx",
//!     "models/whisper-tiny/decoder.onnx",
//!     "models/whisper-tiny/tokens.txt",
//!     WhisperConfig::default(),
//! ).unwrap();
//!
//! let result = pipeline.transcribe_file(std::path::Path::new("audio.wav")).unwrap();
//! println!("{}", result.text);
//! ```

use anyhow::{Context, Result};
use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use ort::value::Tensor;
use std::collections::HashMap;
use std::path::Path;
use std::sync::Mutex;

use crate::TranscriptionPipeline;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Whisper sample rate: 16 kHz.
const WHISPER_SAMPLE_RATE: u32 = 16_000;

/// Whisper chunk length in seconds (each encoder call processes 30s of audio).
const CHUNK_SECONDS: usize = 30;

/// Number of audio samples per chunk (30s × 16000 = 480000).
const CHUNK_SAMPLES: usize = CHUNK_SECONDS * WHISPER_SAMPLE_RATE as usize;

/// FFT window size for STFT (25ms at 16kHz).
const FFT_SIZE: usize = 400;

/// Hop size for STFT (10ms at 16kHz).
const HOP_SIZE: usize = 160;

/// Number of mel filterbank bins for non-large Whisper models.
const N_MELS: usize = 80;

/// Number of mel spectrogram frames per 30-second chunk.
const N_FRAMES: usize = 3000;

/// Maximum number of decoder tokens before forced stop (safety limit).
const MAX_DECODER_TOKENS: usize = 440;

// Whisper special token IDs (multilingual tokenizer)
const SOT_TOKEN: i64 = 50258; // <|startoftranscript|>
const EOT_TOKEN: i64 = 50257; // <|endoftext|>
const TRANSLATE_TOKEN: i64 = 50359;
const TRANSCRIBE_TOKEN: i64 = 50360;
const NO_TIMESTAMPS_TOKEN: i64 = 50364;

// Language token offset: language tokens start at 50259 (English = 50259)
const EN_LANG_TOKEN: i64 = 50259;

/// Mel filterbank generation: compute triangular mel filter weights.
///
/// Produces an `[n_mels, n_fft/2+1]` matrix of filter weights.
fn mel_filterbank(n_mels: usize, n_fft: usize, sample_rate: f32) -> Vec<Vec<f32>> {
    let n_freqs = n_fft / 2 + 1;

    // Mel scale conversion helpers
    let hz_to_mel = |hz: f32| -> f32 { 2595.0 * (1.0 + hz / 700.0).log10() };
    let mel_to_hz = |mel: f32| -> f32 { 700.0 * (10.0_f32.powf(mel / 2595.0) - 1.0) };

    let mel_min = hz_to_mel(0.0);
    let mel_max = hz_to_mel(sample_rate / 2.0);

    // n_mels + 2 equally spaced mel points
    let mel_points: Vec<f32> = (0..=n_mels + 1)
        .map(|i| mel_min + (mel_max - mel_min) * i as f32 / (n_mels + 1) as f32)
        .collect();

    // Convert back to Hz, then to FFT bin indices
    let fft_freqs: Vec<f32> = mel_points.iter().map(|&m| mel_to_hz(m)).collect();
    let bin_indices: Vec<f32> = fft_freqs
        .iter()
        .map(|&f| f * n_fft as f32 / sample_rate)
        .collect();

    let mut filters = vec![vec![0.0f32; n_freqs]; n_mels];

    for m in 0..n_mels {
        let left = bin_indices[m];
        let center = bin_indices[m + 1];
        let right = bin_indices[m + 2];

        #[allow(clippy::needless_range_loop)]
        for k in 0..n_freqs {
            let freq = k as f32;
            if freq >= left && freq <= center {
                let denom = center - left;
                if denom > 0.0 {
                    filters[m][k] = (freq - left) / denom;
                }
            } else if freq > center && freq <= right {
                let denom = right - center;
                if denom > 0.0 {
                    filters[m][k] = (right - freq) / denom;
                }
            }
        }

        // Slaney-style normalization: divide by mel bandwidth
        let enorm = 2.0 / (mel_to_hz(mel_points[m + 2]) - mel_to_hz(mel_points[m]));
        #[allow(clippy::needless_range_loop)]
        for k in 0..n_freqs {
            filters[m][k] *= enorm;
        }
    }

    filters
}

/// Compute the log-mel spectrogram from audio samples.
///
/// Returns a flat `Vec<f32>` of shape `[n_mels, n_frames]` (row-major).
fn log_mel_spectrogram(samples: &[f32], n_mels: usize) -> Vec<f32> {
    let filters = mel_filterbank(n_mels, FFT_SIZE, WHISPER_SAMPLE_RATE as f32);
    let n_freqs = FFT_SIZE / 2 + 1;

    // Hann window
    let hann: Vec<f32> = (0..FFT_SIZE)
        .map(|i| 0.5 * (1.0 - (2.0 * std::f32::consts::PI * i as f32 / FFT_SIZE as f32).cos()))
        .collect();

    // Compute STFT frames
    let n_frames_actual = if samples.len() >= FFT_SIZE {
        (samples.len() - FFT_SIZE) / HOP_SIZE + 1
    } else {
        0
    };

    // Allocate output: n_mels × N_FRAMES (padded or truncated to 3000 frames)
    let mut mel_spec = vec![0.0f32; n_mels * N_FRAMES];

    for frame_idx in 0..n_frames_actual.min(N_FRAMES) {
        let start = frame_idx * HOP_SIZE;

        // Apply Hann window and compute FFT magnitudes using DFT
        // For Whisper's small FFT size (400), a direct DFT is practical
        let mut magnitudes = vec![0.0f32; n_freqs];
        #[allow(clippy::needless_range_loop)]
        for k in 0..n_freqs {
            let mut real = 0.0f32;
            let mut imag = 0.0f32;
            #[allow(clippy::needless_range_loop)]
            for n in 0..FFT_SIZE {
                let sample_idx = start + n;
                let sample = if sample_idx < samples.len() {
                    samples[sample_idx] * hann[n]
                } else {
                    0.0
                };
                let angle = -2.0 * std::f32::consts::PI * k as f32 * n as f32 / FFT_SIZE as f32;
                real += sample * angle.cos();
                imag += sample * angle.sin();
            }
            magnitudes[k] = real * real + imag * imag;
        }

        // Apply mel filterbank
        for m in 0..n_mels {
            let mut energy = 0.0f32;
            for k in 0..n_freqs {
                energy += filters[m][k] * magnitudes[k];
            }
            mel_spec[m * N_FRAMES + frame_idx] = energy;
        }
    }

    // Log transform (clamp to avoid log(0))
    let max_val = mel_spec.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let clamp_min = if max_val > 0.0 {
        max_val * 1e-10
    } else {
        1e-10
    };

    for val in mel_spec.iter_mut() {
        *val = (*val).max(clamp_min).log10();
    }

    // Normalize: scale to [-1, 1] range (Whisper convention)
    let log_max = mel_spec.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let log_min_clamp = log_max - 8.0; // 8 decades dynamic range

    for val in mel_spec.iter_mut() {
        *val = ((*val).max(log_min_clamp) - log_max) / 4.0 + 1.0;
    }

    mel_spec
}

/// Load a WAV file and return mono f32 samples at 16kHz.
///
/// If the file is stereo, channels are averaged to mono.
/// If the sample rate differs from 16kHz, a simple linear resampling is applied.
fn load_wav_mono_16khz(path: &Path) -> Result<Vec<f32>> {
    let reader = hound::WavReader::open(path)
        .with_context(|| format!("Failed to open WAV file: {:?}", path))?;

    let spec = reader.spec();
    let channels = spec.channels as usize;
    let source_rate = spec.sample_rate;

    // Read samples as f32
    let raw_samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Int => {
            let bits = spec.bits_per_sample;
            let max_val = (1i64 << (bits - 1)) as f32;
            reader
                .into_samples::<i32>()
                .map(|s| s.map(|v| v as f32 / max_val))
                .collect::<std::result::Result<Vec<_>, _>>()
                .context("Failed to read WAV samples")?
        }
        hound::SampleFormat::Float => reader
            .into_samples::<f32>()
            .collect::<std::result::Result<Vec<_>, _>>()
            .context("Failed to read WAV float samples")?,
    };

    // Mix to mono if stereo
    let mono: Vec<f32> = if channels > 1 {
        raw_samples
            .chunks(channels)
            .map(|ch| ch.iter().sum::<f32>() / channels as f32)
            .collect()
    } else {
        raw_samples
    };

    // Resample to 16kHz if needed
    if source_rate == WHISPER_SAMPLE_RATE {
        Ok(mono)
    } else {
        Ok(resample_linear(&mono, source_rate, WHISPER_SAMPLE_RATE))
    }
}

/// Simple linear interpolation resampling.
fn resample_linear(samples: &[f32], from_rate: u32, to_rate: u32) -> Vec<f32> {
    if from_rate == to_rate || samples.is_empty() {
        return samples.to_vec();
    }

    let ratio = from_rate as f64 / to_rate as f64;
    let out_len = (samples.len() as f64 / ratio).ceil() as usize;
    let mut output = Vec::with_capacity(out_len);

    for i in 0..out_len {
        let pos = i as f64 * ratio;
        let idx = pos as usize;
        let frac = pos - idx as f64;

        let sample = if idx + 1 < samples.len() {
            samples[idx] * (1.0 - frac as f32) + samples[idx + 1] * frac as f32
        } else if idx < samples.len() {
            samples[idx]
        } else {
            0.0
        };
        output.push(sample);
    }

    output
}

// ---------------------------------------------------------------------------
// Whisper Configuration
// ---------------------------------------------------------------------------

/// Configuration for the Whisper ONNX pipeline.
#[derive(Debug, Clone)]
pub struct WhisperConfig {
    /// Number of threads for ONNX Runtime intra-op parallelism.
    pub num_threads: usize,
    /// Language code for the language token (default: "en").
    pub language: String,
    /// Whether to translate to English instead of transcribing.
    pub translate: bool,
    /// Model architecture parameters (auto-detected from model if possible).
    pub n_text_layer: usize,
    /// Text context length (number of decoder positions).
    pub n_text_ctx: usize,
    /// Text hidden state dimension.
    pub n_text_state: usize,
    /// Audio encoder layers.
    pub n_audio_layer: usize,
    /// Audio hidden state dimension.
    pub n_audio_state: usize,
}

impl Default for WhisperConfig {
    fn default() -> Self {
        // Defaults for whisper-tiny
        Self {
            num_threads: 4,
            language: "en".to_string(),
            translate: false,
            n_text_layer: 4,
            n_text_ctx: 448,
            n_text_state: 384,
            n_audio_layer: 4,
            n_audio_state: 384,
        }
    }
}

impl WhisperConfig {
    /// Configuration preset for whisper-base model.
    pub fn base() -> Self {
        Self {
            n_text_layer: 6,
            n_text_ctx: 448,
            n_text_state: 512,
            n_audio_layer: 6,
            n_audio_state: 512,
            ..Self::default()
        }
    }

    /// Configuration preset for whisper-small model.
    pub fn small() -> Self {
        Self {
            n_text_layer: 12,
            n_text_ctx: 448,
            n_text_state: 768,
            n_audio_layer: 12,
            n_audio_state: 768,
            ..Self::default()
        }
    }

    /// Get the language token ID for the configured language.
    fn language_token(&self) -> i64 {
        // Map common language codes to Whisper token offsets
        let lang_offset: i64 = match self.language.as_str() {
            "en" => 0,
            "zh" => 1,
            "de" => 2,
            "es" => 3,
            "ru" => 4,
            "ko" => 5,
            "fr" => 6,
            "ja" => 7,
            "pt" => 8,
            "tr" => 9,
            "pl" => 10,
            "ca" => 11,
            "nl" => 12,
            "ar" => 13,
            "sv" => 14,
            "it" => 15,
            _ => 0, // Default to English
        };
        EN_LANG_TOKEN + lang_offset
    }

    /// Get the task token (transcribe or translate).
    fn task_token(&self) -> i64 {
        if self.translate {
            TRANSLATE_TOKEN
        } else {
            TRANSCRIBE_TOKEN
        }
    }
}

// ---------------------------------------------------------------------------
// Audio Metadata
// ---------------------------------------------------------------------------

/// Metadata extracted from an audio file.
#[derive(Debug, Clone)]
pub struct AudioMetadata {
    /// Duration in seconds.
    pub duration_secs: f32,
    /// Original sample rate (before resampling).
    pub sample_rate: u32,
    /// Number of channels in original audio.
    pub channels: u16,
    /// Detected language (if available).
    pub detected_language: Option<String>,
}

// ---------------------------------------------------------------------------
// Transcription Result
// ---------------------------------------------------------------------------

/// Result of audio transcription.
#[derive(Debug, Clone)]
pub struct TranscriptionResult {
    /// Full transcribed text.
    pub text: String,
    /// Per-chunk transcriptions (for audio longer than 30s).
    pub chunks: Vec<ChunkTranscription>,
    /// Audio file metadata.
    pub metadata: AudioMetadata,
}

/// Transcription of a single 30-second audio chunk.
#[derive(Debug, Clone)]
pub struct ChunkTranscription {
    /// Transcribed text for this chunk.
    pub text: String,
    /// Start time in seconds.
    pub start_secs: f32,
    /// End time in seconds.
    pub end_secs: f32,
}

// ---------------------------------------------------------------------------
// Token Map
// ---------------------------------------------------------------------------

/// Load token vocabulary from a tokens.txt file.
///
/// Format: one token per line, token ID is the line number (0-indexed).
/// Special tokens may include `<|token|>` format.
fn load_token_map(path: &Path) -> Result<HashMap<i64, String>> {
    let content = std::fs::read_to_string(path)
        .with_context(|| format!("Failed to read tokens file: {:?}", path))?;

    let mut map = HashMap::new();
    for (idx, line) in content.lines().enumerate() {
        let token = line.trim().to_string();
        if !token.is_empty() {
            map.insert(idx as i64, token);
        }
    }

    Ok(map)
}

/// Decode a sequence of token IDs to text using the token map.
fn decode_tokens(tokens: &[i64], token_map: &HashMap<i64, String>) -> String {
    let mut text = String::new();
    for &token_id in tokens {
        // Skip special tokens (>= 50257)
        if token_id >= SOT_TOKEN - 1 {
            continue;
        }
        if let Some(token_text) = token_map.get(&token_id) {
            // Handle byte-level tokens (Whisper uses byte-level BPE)
            // Tokens starting with 'Ġ' (U+0120) represent a space prefix
            let decoded = token_text
                .replace('\u{0120}', " ") // GPT-2 style space
                .replace("Ġ", " ");
            text.push_str(&decoded);
        }
    }
    text.trim().to_string()
}

// ---------------------------------------------------------------------------
// Whisper ONNX Pipeline
// ---------------------------------------------------------------------------

/// Whisper ONNX-based audio transcription pipeline.
///
/// Uses separate encoder and decoder ONNX models following the sherpa-onnx
/// export format. Thread-safe via Mutex on ONNX sessions.
pub struct WhisperOnnxPipeline {
    encoder: Mutex<Session>,
    decoder: Mutex<Session>,
    token_map: HashMap<i64, String>,
    config: WhisperConfig,
}

impl WhisperOnnxPipeline {
    /// Create a new Whisper ONNX pipeline.
    ///
    /// # Arguments
    /// * `encoder_path` - Path to the encoder ONNX model
    /// * `decoder_path` - Path to the decoder ONNX model
    /// * `tokens_path` - Path to the tokens.txt vocabulary file
    /// * `config` - Pipeline configuration
    pub fn new(
        encoder_path: impl AsRef<Path>,
        decoder_path: impl AsRef<Path>,
        tokens_path: impl AsRef<Path>,
        config: WhisperConfig,
    ) -> Result<Self> {
        let encoder_path = encoder_path.as_ref();
        let decoder_path = decoder_path.as_ref();
        let tokens_path = tokens_path.as_ref();

        let encoder = Session::builder()
            .context("Failed to create encoder session builder")?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .context("Failed to set encoder optimization level")?
            .with_intra_threads(config.num_threads)
            .context("Failed to set encoder threads")?
            .commit_from_file(encoder_path)
            .with_context(|| format!("Failed to load encoder ONNX from {:?}", encoder_path))?;

        let decoder = Session::builder()
            .context("Failed to create decoder session builder")?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .context("Failed to set decoder optimization level")?
            .with_intra_threads(config.num_threads)
            .context("Failed to set decoder threads")?
            .commit_from_file(decoder_path)
            .with_context(|| format!("Failed to load decoder ONNX from {:?}", decoder_path))?;

        let token_map = load_token_map(tokens_path)?;

        Ok(Self {
            encoder: Mutex::new(encoder),
            decoder: Mutex::new(decoder),
            token_map,
            config,
        })
    }

    /// Run the encoder on a mel spectrogram to produce cross-attention KV cache.
    ///
    /// Input: mel spectrogram of shape `[1, 80, 3000]`
    /// Output: (cross_k, cross_v) tensors
    #[allow(clippy::type_complexity)]
    fn run_encoder(
        &self,
        mel_data: Vec<f32>,
    ) -> Result<(Vec<f32>, Vec<f32>, Vec<usize>, Vec<usize>)> {
        let mel_tensor = Tensor::from_array(([1usize, N_MELS, N_FRAMES], mel_data))
            .context("Failed to create mel tensor")?;

        let mut session = self
            .encoder
            .lock()
            .map_err(|e| anyhow::anyhow!("Encoder lock poisoned: {}", e))?;

        let outputs = session
            .run(ort::inputs!["mel" => mel_tensor])
            .context("Encoder inference failed")?;

        // Extract cross-attention KV cache
        let cross_k = outputs["n_layer_cross_k"]
            .try_extract_array::<f32>()
            .context("Failed to extract cross_k")?;
        let cross_v = outputs["n_layer_cross_v"]
            .try_extract_array::<f32>()
            .context("Failed to extract cross_v")?;

        let k_shape = cross_k.shape().to_vec();
        let v_shape = cross_v.shape().to_vec();
        let k_data = cross_k
            .as_slice()
            .context("cross_k not contiguous")?
            .to_vec();
        let v_data = cross_v
            .as_slice()
            .context("cross_v not contiguous")?
            .to_vec();

        Ok((k_data, v_data, k_shape, v_shape))
    }

    /// Run the decoder loop for one chunk, producing transcribed text.
    fn run_decoder(
        &self,
        cross_k: &[f32],
        cross_v: &[f32],
        cross_k_shape: &[usize],
        cross_v_shape: &[usize],
    ) -> Result<Vec<i64>> {
        let n_text_layer = self.config.n_text_layer;
        let n_text_ctx = self.config.n_text_ctx;
        let n_text_state = self.config.n_text_state;

        // Initial SOT sequence: [SOT, language, task, no_timestamps]
        let sot_sequence = vec![
            SOT_TOKEN,
            self.config.language_token(),
            self.config.task_token(),
            NO_TIMESTAMPS_TOKEN,
        ];

        // Initialize self-attention KV cache with zeros
        let cache_size = n_text_layer * n_text_ctx * n_text_state;
        let mut self_k_cache = vec![0.0f32; cache_size];
        let mut self_v_cache = vec![0.0f32; cache_size];

        let mut generated_tokens: Vec<i64> = Vec::new();
        let mut offset: i64 = 0;
        let mut is_first = true;

        let mut session = self
            .decoder
            .lock()
            .map_err(|e| anyhow::anyhow!("Decoder lock poisoned: {}", e))?;

        loop {
            // Tokens to feed: full SOT on first call, single token after
            let current_tokens: Vec<i64> = if is_first {
                sot_sequence.clone()
            } else {
                vec![*generated_tokens.last().unwrap_or(&SOT_TOKEN)]
            };
            let n_tokens = current_tokens.len();

            // Create input tensors
            let tokens_tensor = Tensor::from_array(([1usize, n_tokens], current_tokens))
                .context("Failed to create tokens tensor")?;

            let self_k_tensor = Tensor::from_array((
                [n_text_layer, 1usize, n_text_ctx, n_text_state],
                self_k_cache.clone(),
            ))
            .context("Failed to create self_k_cache tensor")?;

            let self_v_tensor = Tensor::from_array((
                [n_text_layer, 1usize, n_text_ctx, n_text_state],
                self_v_cache.clone(),
            ))
            .context("Failed to create self_v_cache tensor")?;

            let cross_k_tensor = Tensor::from_array((cross_k_shape.to_vec(), cross_k.to_vec()))
                .context("Failed to create cross_k tensor")?;

            let cross_v_tensor = Tensor::from_array((cross_v_shape.to_vec(), cross_v.to_vec()))
                .context("Failed to create cross_v tensor")?;

            let offset_tensor = Tensor::from_array(([1usize], vec![offset]))
                .context("Failed to create offset tensor")?;

            // Run decoder
            let outputs = session
                .run(ort::inputs![
                    "tokens" => tokens_tensor,
                    "in_n_layer_self_k_cache" => self_k_tensor,
                    "in_n_layer_self_v_cache" => self_v_tensor,
                    "n_layer_cross_k" => cross_k_tensor,
                    "n_layer_cross_v" => cross_v_tensor,
                    "offset" => offset_tensor
                ])
                .context("Decoder inference failed")?;

            // Extract logits: [1, n_tokens, vocab_size]
            let logits = outputs["logits"]
                .try_extract_array::<f32>()
                .context("Failed to extract logits")?;

            let logits_shape = logits.shape();
            let vocab_size = logits_shape[2];

            // Greedy decoding: argmax of last token's logits
            let last_token_offset = (logits_shape[1] - 1) * vocab_size;
            let logits_slice = logits.as_slice().context("logits not contiguous")?;
            let last_logits = &logits_slice[last_token_offset..last_token_offset + vocab_size];

            let next_token = last_logits
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(idx, _)| idx as i64)
                .unwrap_or(EOT_TOKEN);

            // Check for EOT
            if next_token == EOT_TOKEN {
                break;
            }

            generated_tokens.push(next_token);

            // Update self-attention KV cache
            let out_k = outputs["out_n_layer_self_k_cache"]
                .try_extract_array::<f32>()
                .context("Failed to extract out self_k_cache")?;
            let out_v = outputs["out_n_layer_self_v_cache"]
                .try_extract_array::<f32>()
                .context("Failed to extract out self_v_cache")?;

            self_k_cache = out_k.as_slice().context("out_k not contiguous")?.to_vec();
            self_v_cache = out_v.as_slice().context("out_v not contiguous")?.to_vec();

            offset += n_tokens as i64;
            is_first = false;

            // Safety limit
            if generated_tokens.len() >= MAX_DECODER_TOKENS {
                break;
            }
        }

        Ok(generated_tokens)
    }

    /// Transcribe a single 30-second chunk of audio samples.
    fn transcribe_chunk(&self, samples: &[f32]) -> Result<String> {
        // Pad or truncate to exactly CHUNK_SAMPLES
        let mut padded = vec![0.0f32; CHUNK_SAMPLES];
        let copy_len = samples.len().min(CHUNK_SAMPLES);
        padded[..copy_len].copy_from_slice(&samples[..copy_len]);

        // Compute log-mel spectrogram
        let mel = log_mel_spectrogram(&padded, N_MELS);

        // Run encoder
        let (cross_k, cross_v, k_shape, v_shape) = self.run_encoder(mel)?;

        // Run decoder
        let tokens = self.run_decoder(&cross_k, &cross_v, &k_shape, &v_shape)?;

        // Decode tokens to text
        Ok(decode_tokens(&tokens, &self.token_map))
    }
}

// Send + Sync: Sessions are behind Mutex
unsafe impl Send for WhisperOnnxPipeline {}
unsafe impl Sync for WhisperOnnxPipeline {}

impl TranscriptionPipeline for WhisperOnnxPipeline {
    fn transcribe_file(&self, path: &Path) -> Result<TranscriptionResult> {
        // Load and preprocess audio
        let reader = hound::WavReader::open(path)
            .with_context(|| format!("Failed to open WAV for metadata: {:?}", path))?;
        let spec = reader.spec();
        drop(reader);

        let samples = load_wav_mono_16khz(path)?;

        let duration_secs = samples.len() as f32 / WHISPER_SAMPLE_RATE as f32;
        let metadata = AudioMetadata {
            duration_secs,
            sample_rate: spec.sample_rate,
            channels: spec.channels,
            detected_language: Some(self.config.language.clone()),
        };

        // Split into 30-second chunks
        let mut chunks = Vec::new();
        let mut full_text = String::new();

        if samples.is_empty() {
            return Ok(TranscriptionResult {
                text: String::new(),
                chunks: Vec::new(),
                metadata,
            });
        }

        let mut offset = 0usize;
        while offset < samples.len() {
            let end = (offset + CHUNK_SAMPLES).min(samples.len());
            let chunk_samples = &samples[offset..end];

            let start_secs = offset as f32 / WHISPER_SAMPLE_RATE as f32;
            let end_secs = end as f32 / WHISPER_SAMPLE_RATE as f32;

            let text = self.transcribe_chunk(chunk_samples)?;

            if !full_text.is_empty() && !text.is_empty() {
                full_text.push(' ');
            }
            full_text.push_str(&text);

            chunks.push(ChunkTranscription {
                text,
                start_secs,
                end_secs,
            });

            offset += CHUNK_SAMPLES;
        }

        Ok(TranscriptionResult {
            text: full_text,
            chunks,
            metadata,
        })
    }

    fn transcribe_samples(&self, samples: &[f32], sample_rate: u32) -> Result<TranscriptionResult> {
        // Resample if needed
        let mono_16k = if sample_rate == WHISPER_SAMPLE_RATE {
            samples.to_vec()
        } else {
            resample_linear(samples, sample_rate, WHISPER_SAMPLE_RATE)
        };

        let duration_secs = mono_16k.len() as f32 / WHISPER_SAMPLE_RATE as f32;
        let metadata = AudioMetadata {
            duration_secs,
            sample_rate,
            channels: 1,
            detected_language: Some(self.config.language.clone()),
        };

        if mono_16k.is_empty() {
            return Ok(TranscriptionResult {
                text: String::new(),
                chunks: Vec::new(),
                metadata,
            });
        }

        let mut chunks = Vec::new();
        let mut full_text = String::new();
        let mut offset = 0usize;

        while offset < mono_16k.len() {
            let end = (offset + CHUNK_SAMPLES).min(mono_16k.len());
            let chunk_samples = &mono_16k[offset..end];

            let start_secs = offset as f32 / WHISPER_SAMPLE_RATE as f32;
            let end_secs = end as f32 / WHISPER_SAMPLE_RATE as f32;

            let text = self.transcribe_chunk(chunk_samples)?;

            if !full_text.is_empty() && !text.is_empty() {
                full_text.push(' ');
            }
            full_text.push_str(&text);

            chunks.push(ChunkTranscription {
                text,
                start_secs,
                end_secs,
            });

            offset += CHUNK_SAMPLES;
        }

        Ok(TranscriptionResult {
            text: full_text,
            chunks,
            metadata,
        })
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    // -----------------------------------------------------------------------
    // Unit tests (no models required)
    // -----------------------------------------------------------------------

    #[test]
    fn test_mel_filterbank_shape() {
        let filters = mel_filterbank(N_MELS, FFT_SIZE, WHISPER_SAMPLE_RATE as f32);
        assert_eq!(filters.len(), N_MELS);
        assert_eq!(filters[0].len(), FFT_SIZE / 2 + 1);
    }

    #[test]
    fn test_mel_filterbank_non_negative() {
        let filters = mel_filterbank(N_MELS, FFT_SIZE, WHISPER_SAMPLE_RATE as f32);
        for row in &filters {
            for &val in row {
                assert!(
                    val >= 0.0,
                    "Mel filter weight should be non-negative, got {}",
                    val
                );
            }
        }
    }

    #[test]
    fn test_mel_filterbank_triangular() {
        let filters = mel_filterbank(N_MELS, FFT_SIZE, WHISPER_SAMPLE_RATE as f32);
        // Each filter should have at most a few non-zero bins (triangular shape)
        for (m, row) in filters.iter().enumerate() {
            let nonzero_count = row.iter().filter(|&&v| v > 0.0).count();
            assert!(
                nonzero_count > 0,
                "Mel filter {} should have at least one non-zero bin",
                m
            );
        }
    }

    #[test]
    fn test_log_mel_spectrogram_silence() {
        // All-zero audio should produce a spectrogram (with clamped log values)
        let silence = vec![0.0f32; CHUNK_SAMPLES];
        let mel = log_mel_spectrogram(&silence, N_MELS);
        assert_eq!(mel.len(), N_MELS * N_FRAMES);
        // Values should be finite
        for &val in &mel {
            assert!(val.is_finite(), "Mel value should be finite, got {}", val);
        }
    }

    #[test]
    fn test_log_mel_spectrogram_shape() {
        // Short audio
        let audio = vec![0.5f32; WHISPER_SAMPLE_RATE as usize]; // 1 second
        let mel = log_mel_spectrogram(&audio, N_MELS);
        assert_eq!(mel.len(), N_MELS * N_FRAMES);
    }

    #[test]
    fn test_log_mel_spectrogram_tone() {
        // Generate a 440 Hz sine tone (1 second)
        let n_samples = WHISPER_SAMPLE_RATE as usize;
        let audio: Vec<f32> = (0..n_samples)
            .map(|i| {
                let t = i as f32 / WHISPER_SAMPLE_RATE as f32;
                (2.0 * std::f32::consts::PI * 440.0 * t).sin()
            })
            .collect();
        let mel = log_mel_spectrogram(&audio, N_MELS);
        assert_eq!(mel.len(), N_MELS * N_FRAMES);
        // Should have higher energy in some mel bins than silence
        let max_val = mel.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        assert!(
            max_val > -10.0,
            "A 440Hz tone should produce non-trivial mel energy"
        );
    }

    #[test]
    fn test_resample_linear_same_rate() {
        let samples = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];
        let result = resample_linear(&samples, 16000, 16000);
        assert_eq!(result, samples);
    }

    #[test]
    fn test_resample_linear_downsample() {
        let samples: Vec<f32> = (0..1000).map(|i| i as f32).collect();
        let result = resample_linear(&samples, 44100, 16000);
        // Output should be shorter
        assert!(result.len() < samples.len());
        // First sample should be preserved
        assert!((result[0] - 0.0).abs() < 0.01);
    }

    #[test]
    fn test_resample_linear_upsample() {
        let samples = vec![0.0f32, 1.0, 0.0];
        let result = resample_linear(&samples, 8000, 16000);
        // Output should be ~2x longer
        assert!(result.len() >= 5);
    }

    #[test]
    fn test_resample_linear_empty() {
        let result = resample_linear(&[], 16000, 44100);
        assert!(result.is_empty());
    }

    #[test]
    fn test_decode_tokens_empty() {
        let map = HashMap::new();
        let result = decode_tokens(&[], &map);
        assert_eq!(result, "");
    }

    #[test]
    fn test_decode_tokens_basic() {
        let mut map = HashMap::new();
        map.insert(0, "hello".to_string());
        map.insert(1, "\u{0120}world".to_string()); // Ġworld = " world"
        let result = decode_tokens(&[0, 1], &map);
        assert_eq!(result, "hello world");
    }

    #[test]
    fn test_decode_tokens_skips_special() {
        let mut map = HashMap::new();
        map.insert(0, "hello".to_string());
        map.insert(SOT_TOKEN, "<|sot|>".to_string());
        // Special tokens (>= 50257) should be skipped
        let result = decode_tokens(&[0, SOT_TOKEN, EOT_TOKEN], &map);
        assert_eq!(result, "hello");
    }

    #[test]
    fn test_load_token_map() {
        // Create a temporary tokens file
        let dir = tempfile::tempdir().unwrap();
        let tokens_path = dir.path().join("tokens.txt");
        std::fs::write(&tokens_path, "hello\nworld\nfoo\n").unwrap();

        let map = load_token_map(&tokens_path).unwrap();
        assert_eq!(map.get(&0), Some(&"hello".to_string()));
        assert_eq!(map.get(&1), Some(&"world".to_string()));
        assert_eq!(map.get(&2), Some(&"foo".to_string()));
    }

    #[test]
    fn test_whisper_config_default() {
        let config = WhisperConfig::default();
        assert_eq!(config.n_text_layer, 4);
        assert_eq!(config.n_text_state, 384);
        assert_eq!(config.language, "en");
        assert!(!config.translate);
    }

    #[test]
    fn test_whisper_config_base() {
        let config = WhisperConfig::base();
        assert_eq!(config.n_text_layer, 6);
        assert_eq!(config.n_text_state, 512);
    }

    #[test]
    fn test_whisper_config_small() {
        let config = WhisperConfig::small();
        assert_eq!(config.n_text_layer, 12);
        assert_eq!(config.n_text_state, 768);
    }

    #[test]
    fn test_whisper_config_language_token() {
        let config = WhisperConfig::default();
        assert_eq!(config.language_token(), EN_LANG_TOKEN);

        let es_config = WhisperConfig {
            language: "es".to_string(),
            ..Default::default()
        };
        assert_eq!(es_config.language_token(), EN_LANG_TOKEN + 3);
    }

    #[test]
    fn test_whisper_config_task_token() {
        let config = WhisperConfig::default();
        assert_eq!(config.task_token(), TRANSCRIBE_TOKEN);

        let translate_config = WhisperConfig {
            translate: true,
            ..Default::default()
        };
        assert_eq!(translate_config.task_token(), TRANSLATE_TOKEN);
    }

    #[test]
    fn test_audio_metadata() {
        let meta = AudioMetadata {
            duration_secs: 5.5,
            sample_rate: 16000,
            channels: 1,
            detected_language: Some("en".to_string()),
        };
        assert_eq!(meta.duration_secs, 5.5);
        assert_eq!(meta.sample_rate, 16000);
    }

    #[test]
    fn test_transcription_result_empty() {
        let result = TranscriptionResult {
            text: String::new(),
            chunks: Vec::new(),
            metadata: AudioMetadata {
                duration_secs: 0.0,
                sample_rate: 16000,
                channels: 1,
                detected_language: None,
            },
        };
        assert!(result.text.is_empty());
        assert!(result.chunks.is_empty());
    }

    // -----------------------------------------------------------------------
    // WAV loading tests (uses tempfile, no models needed)
    // -----------------------------------------------------------------------

    #[test]
    fn test_load_wav_mono_16khz() {
        // Create a test WAV file: 1 second of 440Hz sine at 16kHz mono
        let dir = tempfile::tempdir().unwrap();
        let wav_path = dir.path().join("test.wav");

        let spec = hound::WavSpec {
            channels: 1,
            sample_rate: 16000,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };

        let mut writer = hound::WavWriter::create(&wav_path, spec).unwrap();
        for i in 0..16000 {
            let t = i as f32 / 16000.0;
            let sample = (2.0 * std::f32::consts::PI * 440.0 * t).sin();
            writer.write_sample((sample * 32767.0) as i16).unwrap();
        }
        writer.finalize().unwrap();

        let samples = load_wav_mono_16khz(&wav_path).unwrap();
        assert_eq!(samples.len(), 16000);
        // First sample should be ~0
        assert!(samples[0].abs() < 0.01);
    }

    #[test]
    fn test_load_wav_stereo_to_mono() {
        let dir = tempfile::tempdir().unwrap();
        let wav_path = dir.path().join("stereo.wav");

        let spec = hound::WavSpec {
            channels: 2,
            sample_rate: 16000,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };

        let mut writer = hound::WavWriter::create(&wav_path, spec).unwrap();
        for _ in 0..16000 {
            writer.write_sample(16384i16).unwrap(); // Left
            writer.write_sample(-16384i16).unwrap(); // Right
        }
        writer.finalize().unwrap();

        let samples = load_wav_mono_16khz(&wav_path).unwrap();
        assert_eq!(samples.len(), 16000);
        // Averaged stereo should be ~0
        for s in &samples {
            assert!(s.abs() < 0.01, "Averaged stereo should be ~0, got {}", s);
        }
    }

    #[test]
    fn test_load_wav_resample() {
        let dir = tempfile::tempdir().unwrap();
        let wav_path = dir.path().join("44100.wav");

        let spec = hound::WavSpec {
            channels: 1,
            sample_rate: 44100,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };

        let mut writer = hound::WavWriter::create(&wav_path, spec).unwrap();
        for _ in 0..44100 {
            writer.write_sample(0i16).unwrap();
        }
        writer.finalize().unwrap();

        let samples = load_wav_mono_16khz(&wav_path).unwrap();
        // Should be resampled to ~16000 samples
        assert!(
            (samples.len() as i32 - 16000).abs() < 100,
            "Expected ~16000 samples after resample, got {}",
            samples.len()
        );
    }

    // -----------------------------------------------------------------------
    // Model-dependent tests (skip if models not present)
    // -----------------------------------------------------------------------

    fn whisper_model_path() -> Option<PathBuf> {
        let workspace = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()?
            .to_path_buf();
        let model_dir = workspace.join("models").join("whisper-tiny");
        let encoder = model_dir.join("encoder.onnx");
        let decoder = model_dir.join("decoder.onnx");
        let tokens = model_dir.join("tokens.txt");

        if encoder.exists() && decoder.exists() && tokens.exists() {
            Some(model_dir)
        } else {
            None
        }
    }

    #[test]
    fn test_whisper_pipeline_creation() {
        let model_dir = match whisper_model_path() {
            Some(p) => p,
            None => {
                eprintln!("SKIP: Whisper models not found at models/whisper-tiny/");
                return;
            }
        };

        let pipeline = WhisperOnnxPipeline::new(
            model_dir.join("encoder.onnx"),
            model_dir.join("decoder.onnx"),
            model_dir.join("tokens.txt"),
            WhisperConfig::default(),
        );

        assert!(
            pipeline.is_ok(),
            "Pipeline creation failed: {:?}",
            pipeline.err()
        );
    }

    #[test]
    fn test_whisper_transcribe_silence() {
        let model_dir = match whisper_model_path() {
            Some(p) => p,
            None => {
                eprintln!("SKIP: Whisper models not found at models/whisper-tiny/");
                return;
            }
        };

        let pipeline = WhisperOnnxPipeline::new(
            model_dir.join("encoder.onnx"),
            model_dir.join("decoder.onnx"),
            model_dir.join("tokens.txt"),
            WhisperConfig::default(),
        )
        .unwrap();

        // Transcribe 1 second of silence
        let silence = vec![0.0f32; WHISPER_SAMPLE_RATE as usize];
        let result = pipeline
            .transcribe_samples(&silence, WHISPER_SAMPLE_RATE)
            .unwrap();

        // Silence should produce empty or minimal text
        assert!(result.metadata.duration_secs > 0.0);
        assert_eq!(result.chunks.len(), 1);
    }

    #[test]
    fn test_whisper_transcribe_wav_file() {
        let model_dir = match whisper_model_path() {
            Some(p) => p,
            None => {
                eprintln!("SKIP: Whisper models not found at models/whisper-tiny/");
                return;
            }
        };

        let pipeline = WhisperOnnxPipeline::new(
            model_dir.join("encoder.onnx"),
            model_dir.join("decoder.onnx"),
            model_dir.join("tokens.txt"),
            WhisperConfig::default(),
        )
        .unwrap();

        // Create a test WAV with a sine tone
        let dir = tempfile::tempdir().unwrap();
        let wav_path = dir.path().join("test.wav");
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate: 16000,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        let mut writer = hound::WavWriter::create(&wav_path, spec).unwrap();
        for i in 0..16000 {
            let t = i as f32 / 16000.0;
            let sample = (2.0 * std::f32::consts::PI * 440.0 * t).sin();
            writer.write_sample((sample * 32767.0) as i16).unwrap();
        }
        writer.finalize().unwrap();

        let result = pipeline.transcribe_file(&wav_path).unwrap();
        assert!(result.metadata.duration_secs > 0.0);
        assert_eq!(result.metadata.sample_rate, 16000);
        assert_eq!(result.metadata.channels, 1);
    }
}
