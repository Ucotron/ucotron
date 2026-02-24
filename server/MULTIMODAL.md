# Multimodal Memory Architecture

> Cross-modal memory ingestion, storage, and retrieval in Ucotron.

## Table of Contents

- [Overview](#overview)
- [Dual-Index Architecture](#dual-index-architecture)
- [Media Types](#media-types)
- [Ingestion Pipelines](#ingestion-pipelines)
  - [Text Pipeline](#text-pipeline)
  - [Audio Pipeline (Whisper)](#audio-pipeline-whisper)
  - [Image Pipeline (CLIP)](#image-pipeline-clip)
  - [Video Pipeline (FFmpeg + CLIP + Whisper)](#video-pipeline-ffmpeg--clip--whisper)
- [Cross-Modal Search](#cross-modal-search)
  - [Query Types](#query-types)
  - [Search Flow](#search-flow)
  - [Result Fusion](#result-fusion)
- [Projection Layer](#projection-layer)
  - [Architecture](#projection-architecture)
  - [Training Pipeline](#training-pipeline)
  - [ONNX Export](#onnx-export)
- [API Reference](#api-reference)
  - [POST /api/v1/memories/text](#post-apiv1memoriestext)
  - [POST /api/v1/memories/audio](#post-apiv1memoriesaudio)
  - [POST /api/v1/memories/image](#post-apiv1memoriesimage)
  - [POST /api/v1/memories/video](#post-apiv1memoriesvideo)
  - [POST /api/v1/search/multimodal](#post-apiv1searchmultimodal)
  - [GET /api/v1/media/:id](#get-apiv1mediaid)
- [ONNX Models](#onnx-models)
- [Configuration](#configuration)
- [Performance Targets](#performance-targets)

---

## Overview

Ucotron supports four media types — text, audio, images, and video — through a **dual-index architecture** that maintains two independent HNSW indices:

- **Text Index** (384-dim): all-MiniLM-L6-v2 embeddings for text, audio transcripts, and projected image embeddings.
- **Visual Index** (512-dim): CLIP ViT-B/32 embeddings for images and video keyframes.

A trained **projection layer** (MLP: 512 -> 384) bridges the two spaces, enabling cross-modal queries like "find images similar to this text" or "find text related to this image".

All inference runs locally via ONNX Runtime — no external API calls during ingestion or search.

---

## Dual-Index Architecture

```
                    ┌─────────────────────────────────────────┐
                    │              BackendRegistry             │
                    │                                         │
                    │  ┌─────────────────────────────────┐    │
                    │  │      Text Index (VectorBackend)  │    │
                    │  │  HNSW 384-dim (MiniLM space)     │    │
                    │  │                                   │    │
                    │  │  Contains:                        │    │
                    │  │  - Text node embeddings           │    │
                    │  │  - Audio transcript embeddings    │    │
                    │  │  - Projected image embeddings     │    │
                    │  └─────────────────────────────────┘    │
                    │                                         │
                    │  ┌─────────────────────────────────┐    │
                    │  │  Visual Index (VisualVectorBackend)│   │
                    │  │  HNSW 512-dim (CLIP space)       │    │
                    │  │                                   │    │
                    │  │  Contains:                        │    │
                    │  │  - Image CLIP embeddings          │    │
                    │  │  - Video keyframe CLIP embeddings │    │
                    │  └─────────────────────────────────┘    │
                    │                                         │
                    │  ┌─────────────────────────────────┐    │
                    │  │  Graph (GraphBackend)             │    │
                    │  │  Nodes, edges, communities        │    │
                    │  └─────────────────────────────────┘    │
                    └─────────────────────────────────────────┘
```

The `BackendRegistry` holds all backends as trait objects:

```rust
// core/src/backends.rs
pub trait VectorBackend: Send + Sync {
    fn upsert_embeddings(&self, items: &[(NodeId, Vec<f32>)]) -> Result<()>;
    fn search(&self, query: &[f32], top_k: usize) -> Result<Vec<(NodeId, f32)>>;
    fn delete(&self, ids: &[NodeId]) -> Result<()>;
}

pub trait VisualVectorBackend: Send + Sync {
    fn upsert_visual_embeddings(&self, items: &[(NodeId, Vec<f32>)]) -> Result<()>;
    fn search_visual(&self, query: &[f32], top_k: usize) -> Result<Vec<(NodeId, f32)>>;
    fn delete_visual(&self, ids: &[NodeId]) -> Result<()>;
}
```

Both HNSW indices are backed by LMDB with separate environments to avoid key collisions. The text index uses `HNSW_INDEX_KEY = u64::MAX` and the visual index uses `VISUAL_HNSW_INDEX_KEY = u64::MAX - 1`.

---

## Media Types

Each node carries an optional `media_type` field:

```rust
// core/src/types.rs
pub enum MediaType {
    Text,          // Default — 384-dim embedding only
    Audio,         // Whisper transcript + media_uri
    Image,         // 512-dim CLIP embedding + media_uri
    VideoSegment,  // 512-dim CLIP embedding + parent_video_id + timestamp_range
}
```

Multimodal fields on `Node`:

| Field | Type | Used By |
|-------|------|---------|
| `media_type` | `Option<MediaType>` | All multimodal nodes |
| `media_uri` | `Option<String>` | Audio, Image, Video |
| `embedding_visual` | `Option<Vec<f32>>` | Image (512-dim), VideoSegment (512-dim) |
| `timestamp_range` | `Option<(u64, u64)>` | Audio segments, VideoSegment (start_ms, end_ms) |
| `parent_video_id` | `Option<NodeId>` | VideoSegment (links to parent video node) |

---

## Ingestion Pipelines

### Text Pipeline

```
Text ──→ MiniLM embed ──→ 384-dim ──→ Text Index
  │
  ├──→ NER (GLiNER) ──→ Entity nodes + edges
  └──→ Relation extraction ──→ Predicate edges
```

**Endpoint:** `POST /api/v1/memories/text`

The standard ingestion pipeline: chunk the text, embed each chunk with all-MiniLM-L6-v2 (384-dim), run NER to extract entities, detect relations between entities, and store everything in the graph + text index.

### Audio Pipeline (Whisper)

```
Audio file ──→ Load WAV (mono 16kHz)
    │
    ├──→ Log-mel spectrogram (80 bins, 30s chunks)
    │        │
    │        └──→ Whisper encoder (encoder.onnx)
    │                 │
    │                 └──→ Whisper decoder (greedy, decoder.onnx)
    │                          │
    │                          └──→ Transcript text
    │
    └──→ Transcript ──→ MiniLM embed ──→ 384-dim ──→ Text Index
                  │
                  └──→ NER + Relations ──→ Entity nodes + edges
```

**Endpoint:** `POST /api/v1/memories/audio` (multipart/form-data)

Pipeline steps:

1. **Audio loading**: Accept WAV, convert stereo to mono, resample to 16 kHz
2. **Mel spectrogram**: STFT with 400-sample window, 160-sample hop, 80 mel bins, log transform
3. **Whisper encoder**: Process mel spectrogram `[1, 80, 3000]` through ONNX encoder
4. **Whisper decoder**: Greedy autoregressive decoding with KV cache, stopping at EOT token
5. **Transcript ingestion**: Feed transcript through the full text pipeline (embed, NER, relations)

Whisper configuration:

| Parameter | Value |
|-----------|-------|
| Sample rate | 16,000 Hz |
| Chunk length | 30 seconds |
| FFT size | 400 (25ms window) |
| Hop size | 160 (10ms hop) |
| Mel bins | 80 |
| Frames per chunk | 3,000 |
| Models | whisper-tiny (default), whisper-base |

### Image Pipeline (CLIP)

```
Image bytes ──→ Decode (PNG/JPEG/WebP/etc.)
    │
    ├──→ Center-crop to square
    │        │
    │        └──→ Resize to 224x224
    │                 │
    │                 └──→ Normalize (CLIP mean/std)
    │                          │
    │                          └──→ CLIP visual encoder ──→ 512-dim ──→ Visual Index
    │
    └──→ Optional description ──→ MiniLM embed ──→ 384-dim ──→ Text Index
```

**Endpoint:** `POST /api/v1/memories/image` (multipart/form-data)

CLIP preprocessing constants:

```
Image size:  224 x 224
Mean:        [0.48145466, 0.4578275, 0.40821073]
Std:         [0.26862954, 0.26130258, 0.27577711]
Output dim:  512 (L2-normalized)
```

Processing steps:

1. Decode image bytes (supports PNG, JPEG, WebP, BMP, GIF, TIFF via the `image` crate)
2. Center-crop to square aspect ratio
3. Bilinear resize to 224x224
4. Convert to float32 CHW layout, normalize with CLIP mean/std
5. Run through CLIP visual encoder ONNX model
6. L2-normalize output to 512-dim embedding
7. Store in visual index; optionally ingest description through text pipeline

### Video Pipeline (FFmpeg + CLIP + Whisper)

```
Video file ──→ FFmpeg decode
    │
    ├──→ Frame extraction (configurable FPS)
    │        │
    │        ├──→ Scene change detection (histogram chi-squared)
    │        │
    │        └──→ Temporal segmentation
    │                 │
    │                 └──→ Per-segment: CLIP embed representative frame ──→ Visual Index
    │                          │
    │                          └──→ VideoSegment nodes (linked to parent)
    │
    └──→ Audio track ──→ Whisper transcription ──→ Text Index
```

**Endpoint:** `POST /api/v1/memories/video` (multipart/form-data)

The video pipeline produces a hierarchical node structure:

```
Parent Video Node (NodeType::Event, MediaType::VideoSegment)
    │
    ├──→ [RELATES_TO] ──→ Segment 0 (0ms - 5000ms)
    ├──→ [RELATES_TO] ──→ Segment 1 (5000ms - 10000ms)
    └──→ [RELATES_TO] ──→ Segment 2 (10000ms - 15000ms)
```

**Frame extraction** (`VideoConfig`):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `fps` | 1.0 | Target frames per second to extract |
| `scene_change_threshold` | 0.3 | Histogram diff threshold for scene detection |
| `max_frames` | 0 (unlimited) | Maximum frames to extract |
| `output_width` | 0 (original) | Resize output width |
| `output_height` | 0 (original) | Resize output height |

**Scene change detection**: Normalized chi-squared divergence between consecutive frame grayscale histograms, producing a score in [0, 1] where 1.0 = completely different scene.

**Temporal segmentation** (`SegmentConfig`):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `min_duration_ms` | 5,000 | Minimum segment duration |
| `max_duration_ms` | 30,000 | Maximum segment duration |
| `scene_change_threshold` | 0.3 | Score above which to split |

Algorithm: split when scene change detected AND minimum duration met, or when maximum duration exceeded. Short trailing segments are merged into the previous segment.

Each segment stores:
- `embedding_visual`: CLIP embedding of the first keyframe in the segment
- `timestamp_range`: `(start_ms, end_ms)` time boundaries
- `parent_video_id`: Link to the parent video node

---

## Cross-Modal Search

### Query Types

The `CrossModalSearch` orchestrator supports six query types:

| Query Type | Input | Index | Encoding |
|------------|-------|-------|----------|
| `text` | Text string | Text (384-dim) | MiniLM |
| `text_to_image` | Text string | Visual (512-dim) | CLIP text encoder |
| `image` | Image bytes | Visual (512-dim) | CLIP image encoder |
| `image_to_text` | Image bytes | Text (384-dim) | CLIP image encoder + projection |
| `audio` | Transcript text | Text (384-dim) | MiniLM |
| `video` | Frame embeddings + transcript | Both indices | Fusion |

### Search Flow

```
                          ┌──────────────────────┐
                          │   CrossModalSearch    │
                          │                       │
     text ────────────────┤  MiniLM ──→ Text Index
                          │                       │
     text_to_image ───────┤  CLIP text enc ──→ Visual Index
                          │                       │
     image ───────────────┤  CLIP img enc ──→ Visual Index
                          │                       │
     image_to_text ───────┤  CLIP img enc ──→ Projection ──→ Text Index
                          │                       │
     audio ───────────────┤  MiniLM ──→ Text Index
                          │                       │
     video ───────────────┤  Visual: max-pool per-frame visual results
                          │  Text: MiniLM(transcript) ──→ Text Index
                          │  Fusion: weighted merge
                          └──────────────────────┘
```

### Result Fusion

For video queries and any multi-index search, results are fused using configurable weights:

```
final_score = text_weight * text_score + visual_weight * visual_score
```

Default weights: `text_weight = 0.5`, `visual_weight = 0.5`.

Results below `min_similarity` (default 0.0) are filtered out. The final list is sorted by descending score and truncated to `top_k`.

**Configuration** (`CrossModalSearchConfig`):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `top_k` | 10 | Max results per index search |
| `text_weight` | 0.5 | Weight for text index scores |
| `visual_weight` | 0.5 | Weight for visual index scores |
| `min_similarity` | 0.0 | Minimum score threshold |

**Metrics**: Every search returns `CrossModalSearchMetrics` with timing for query encoding, text search, visual search, fusion, and result counts.

---

## Projection Layer

### Projection Architecture

The projection layer bridges CLIP's 512-dim visual space to MiniLM's 384-dim text space via a 3-layer MLP:

```
Input (512-dim, CLIP space)
    │
    ├──→ Linear(512, 1024) + ReLU + Dropout(0.1)
    │
    ├──→ Linear(1024, 512) + ReLU + Dropout(0.1)
    │
    └──→ Linear(512, 384)
         │
         └──→ L2 Normalize
              │
              Output (384-dim, MiniLM space)
```

This enables **image-to-text search**: encode an image with CLIP (512-dim), project into MiniLM space (384-dim), and search the text index.

```rust
// ucotron_extraction/src/cross_modal.rs
pub trait CrossModalProjection: Send + Sync {
    fn project(&self, embedding: &[f32]) -> Result<Vec<f32>>;
    fn project_batch(&self, embeddings: &[Vec<f32>]) -> Result<Vec<Vec<f32>>>;
    fn input_dim(&self) -> usize;   // 512
    fn output_dim(&self) -> usize;  // 384
}
```

### Training Pipeline

Training uses paired (CLIP embedding, MiniLM embedding) data with cosine similarity loss:

```bash
# 1. Generate 50k paired embeddings via GLM-5 (Fireworks)
python scripts/train_projection_layer.py generate --count 50000

# 2. Validate dataset dimensions and norms
python scripts/train_projection_layer.py validate --input projection_dataset.jsonl

# 3. Train the MLP (cosine similarity loss)
python scripts/train_projection_layer.py train \
    --input projection_dataset.jsonl \
    --output models/projection_layer.pt \
    --epochs 10 \
    --batch-size 256 \
    --learning-rate 1e-4 \
    --hidden-dim 1024 \
    --dropout 0.1

# 4. Export to ONNX with verification
python scripts/train_projection_layer.py export \
    --checkpoint models/projection_layer.pt \
    --output models/projection_layer.onnx
```

Training details:

| Parameter | Value |
|-----------|-------|
| Dataset size | 50,000 text-image description pairs |
| Loss function | Cosine similarity loss |
| Optimizer | Adam, lr=1e-4 |
| Hidden dim | 1024 |
| Dropout | 0.1 |
| Validation split | 10% |
| Early stopping | Patience=3 |
| Target val loss | < 0.15 |
| ONNX opset | 17 |

### ONNX Export

The exported `projection_layer.onnx` model accepts:
- **Input**: `[batch, 512]` (L2-normalized CLIP embeddings, f32)
- **Output**: `[batch, 384]` (L2-normalized MiniLM-compatible embeddings, f32)

Dynamic batch axis allows single and batch inference.

---

## API Reference

All endpoints require authentication via `Authorization: Bearer <api-key>` and support multi-tenancy via the `X-Ucotron-Namespace` header.

### POST /api/v1/memories/text

Ingest text content through the full extraction pipeline.

**Request:**
```json
{
  "text": "Marie Curie won the Nobel Prize in Physics in 1903.",
  "metadata": {
    "source": "wikipedia",
    "language": "en"
  }
}
```

**Response:**
```json
{
  "chunk_node_ids": [1001, 1002],
  "entity_node_ids": [2001, 2002],
  "edges_created": 5,
  "media_type": "text",
  "metrics": {
    "chunks": 2,
    "entities_found": 2,
    "relations_found": 1,
    "duration_us": 12450
  }
}
```

### POST /api/v1/memories/audio

Ingest audio via Whisper transcription. Accepts `multipart/form-data`.

**Request:**
```bash
curl -X POST http://localhost:8420/api/v1/memories/audio \
  -H "Authorization: Bearer $API_KEY" \
  -H "X-Ucotron-Namespace: default" \
  -F "file=@recording.wav" \
  -F "sample_rate=16000"
```

**Response:**
```json
{
  "chunk_node_ids": [3001, 3002, 3003],
  "entity_node_ids": [4001],
  "edges_created": 3,
  "media_type": "audio",
  "transcription": "The meeting discussed quarterly results...",
  "audio": {
    "duration_ms": 45000,
    "sample_rate": 16000,
    "channels": 1
  },
  "metrics": {
    "chunks": 3,
    "entities_found": 1,
    "relations_found": 0,
    "duration_us": 234500
  }
}
```

### POST /api/v1/memories/image

Ingest an image with CLIP visual encoding. Accepts `multipart/form-data`.

**Request:**
```bash
curl -X POST http://localhost:8420/api/v1/memories/image \
  -H "Authorization: Bearer $API_KEY" \
  -F "file=@photo.jpg" \
  -F "description=A sunset over the mountains in Patagonia"
```

**Response:**
```json
{
  "node_id": 5001,
  "width": 1920,
  "height": 1080,
  "format": "jpeg",
  "embedding_dim": 512,
  "media_type": "image",
  "description_ingested": true,
  "metrics": {
    "chunks": 1,
    "entities_found": 1,
    "relations_found": 0,
    "duration_us": 85200
  }
}
```

### POST /api/v1/memories/video

Ingest a video with keyframe extraction, segmentation, and optional transcription. Accepts `multipart/form-data`.

**Request:**
```bash
curl -X POST http://localhost:8420/api/v1/memories/video \
  -H "Authorization: Bearer $API_KEY" \
  -F "file=@lecture.mp4"
```

**Response:**
```json
{
  "video_node_id": 6001,
  "segment_node_ids": [6002, 6003, 6004],
  "edges_created": 3,
  "total_frames": 120,
  "total_segments": 3,
  "duration_ms": 120000,
  "video_width": 1280,
  "video_height": 720,
  "media_type": "video_segment",
  "transcription": "Welcome to today's lecture on...",
  "metrics": {
    "frame_extraction_us": 450000,
    "embedding_us": 120000,
    "transcription_us": 890000,
    "total_us": 1560000
  }
}
```

### POST /api/v1/search/multimodal

Unified cross-modal search endpoint.

**Text search:**
```json
{
  "query_type": "text",
  "query_text": "Nobel Prize winners",
  "limit": 5
}
```

**Text-to-image search** (find images matching a text description):
```json
{
  "query_type": "text_to_image",
  "query_text": "sunset over mountains",
  "limit": 10
}
```

**Image search** (find similar images):
```json
{
  "query_type": "image",
  "query_image": "<base64-encoded image bytes>",
  "limit": 5
}
```

**Image-to-text search** (find text related to an image):
```json
{
  "query_type": "image_to_text",
  "query_image": "<base64-encoded image bytes>",
  "limit": 10,
  "media_filter": ["text", "audio"]
}
```

**Audio transcript search:**
```json
{
  "query_type": "audio",
  "query_text": "quarterly revenue discussion",
  "limit": 5,
  "time_range": [1700000000, 1710000000]
}
```

**Response (all query types):**
```json
{
  "results": [
    {
      "node_id": 5001,
      "content": "A sunset over the mountains in Patagonia",
      "score": 0.892,
      "media_type": "image",
      "source": "visual",
      "metadata": {
        "_image_format": "jpeg",
        "_image_width": "1920",
        "_image_height": "1080"
      }
    }
  ],
  "total": 1,
  "query_type": "text_to_image",
  "metrics": {
    "query_encoding_us": 1200,
    "text_search_us": 0,
    "visual_search_us": 3400,
    "fusion_us": 50,
    "total_us": 4650,
    "text_result_count": 0,
    "visual_result_count": 1,
    "final_result_count": 1
  }
}
```

**Filters:**
- `media_filter`: Restrict results to specific media types (`"text"`, `"audio"`, `"image"`, `"video"`). Accepts a single string or array.
- `time_range`: Only return results with timestamps in `[min_ts, max_ts]`.

### GET /api/v1/media/:id

Serve a persisted media file by node ID with the appropriate Content-Type header.

```bash
curl http://localhost:8420/api/v1/media/5001 \
  -H "Authorization: Bearer $API_KEY" \
  --output photo.jpg
```

---

## ONNX Models

All models are stored in the `models/` directory (gitignored) and downloaded via `scripts/download_models.sh`.

| Model | Directory | Input | Output | Purpose |
|-------|-----------|-------|--------|---------|
| all-MiniLM-L6-v2 | `models/all-MiniLM-L6-v2/` | Token IDs | 384-dim | Text embeddings |
| CLIP ViT-B/32 Visual | `models/clip-vit-base-patch32/visual_model.onnx` | `[1,3,224,224]` | 512-dim | Image embeddings |
| CLIP ViT-B/32 Text | `models/clip-vit-base-patch32/text_model.onnx` | `[1,77]` + mask | 512-dim | Text-to-image queries |
| GLiNER small-v2.1 | `models/gliner_small-v2.1/` | Token IDs | Entity spans | Named entity recognition |
| Whisper tiny | `models/whisper-tiny/` | Mel spectrogram | Token IDs | Audio transcription |
| Projection MLP | `models/projection_layer.onnx` | `[batch,512]` | `[batch,384]` | CLIP -> MiniLM bridge |

All embeddings are L2-normalized before storage. CLIP text encoding uses BPE tokenization with a max sequence length of 77 tokens (SOT + 75 tokens + EOT).

---

## Configuration

Multimodal pipelines are configured via `ucotron.toml` and environment variable overrides:

```toml
[extraction]
model_dir = "models"
embedding_threads = 4

[extraction.clip]
image_size = 224
embed_dim = 512

[extraction.whisper]
language = "en"
model_size = "tiny"        # "tiny" (4 layers) or "base" (6 layers)

[extraction.projection]
num_threads = 4

[extraction.video]
fps = 1.0
scene_change_threshold = 0.3
max_frames = 0             # 0 = unlimited
min_segment_ms = 5000
max_segment_ms = 30000

[server]
media_dir = "media"        # Directory for persisted media files
```

Environment variable overrides use the `UCOTRON_` prefix:

```bash
UCOTRON_EXTRACTION__MODEL_DIR=./models
UCOTRON_EXTRACTION__WHISPER__LANGUAGE=es
UCOTRON_EXTRACTION__VIDEO__FPS=2.0
UCOTRON_SERVER__MEDIA_DIR=/data/media
```

---

## Performance Targets

| Metric | Target |
|--------|--------|
| Cross-modal search Recall@10 | > 0.6 |
| Video frame extraction | > 5 FPS |
| Projection layer cosine loss | < 0.15 |
| Batch NER throughput | > 2,000 texts/s |
| Parallel embedding throughput | > 3,000 texts/s |
| Image embedding latency | < 50ms |
| Audio transcription (30s chunk) | < 5s |
