"""
Ucotron Embedding & Reranker Sidecar

FastAPI service that wraps HuggingFace Transformers models for embedding and
reranking. Designed to run alongside the Rust ucotron_server, providing access
to models that can't run natively in Rust (e.g., Qwen3-VL-Embedding/Reranker).

Usage:
    uvicorn main:app --host 0.0.0.0 --port 8421

Environment variables:
    EMBEDDING_MODEL  - HuggingFace model ID (default: Qwen/Qwen3-VL-Embedding-2B)
    RERANKER_MODEL   - HuggingFace model ID (default: Qwen/Qwen3-VL-Reranker-2B)
    EMBEDDING_DIM    - Output embedding dimension via MRL (default: 384)
    DEVICE           - torch device (default: auto-detect mps/cuda/cpu)
    PORT             - Server port (default: 8421)
"""

import os
import sys
import logging
import time
from contextlib import asynccontextmanager
from typing import Optional

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# ─── Configuration ───────────────────────────────────────────────────────────

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "Qwen/Qwen3-VL-Embedding-2B")
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "Qwen/Qwen3-VL-Reranker-2B")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "384"))
PORT = int(os.getenv("PORT", "8421"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("ucotron-sidecar")


def detect_device() -> str:
    """Auto-detect the best available device."""
    override = os.getenv("DEVICE")
    if override:
        return override
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


# ─── Global model holders ────────────────────────────────────────────────────

embedder = None
reranker = None

# ─── Request/Response schemas ────────────────────────────────────────────────


class EmbedRequest(BaseModel):
    texts: list[str]
    instruction: Optional[str] = None
    dimension: Optional[int] = None  # Override EMBEDDING_DIM per-request


class EmbedResponse(BaseModel):
    embeddings: list[list[float]]
    dimension: int
    model: str


class RerankRequest(BaseModel):
    query: str
    documents: list[str]
    instruction: Optional[str] = None


class RerankResponse(BaseModel):
    scores: list[float]
    model: str


class HealthResponse(BaseModel):
    status: str
    embedding_model: Optional[str] = None
    reranker_model: Optional[str] = None
    embedding_loaded: bool = False
    reranker_loaded: bool = False
    device: str = "unknown"
    embedding_dim: int = 384


# ─── Model loading ───────────────────────────────────────────────────────────


def load_embedding_model(model_name: str, device: str):
    """Load the Qwen3-VL-Embedding model."""
    logger.info(f"Loading embedding model: {model_name} on {device}")
    start = time.time()

    try:
        # Try importing the Qwen3-VL-Embedding wrapper script
        # The model repo includes scripts/qwen3_vl_embedding.py
        from transformers import AutoModel, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16 if device != "cpu" else torch.float32,
        )
        model = model.to(device)
        model.eval()

        elapsed = time.time() - start
        logger.info(f"Embedding model loaded in {elapsed:.1f}s")
        return {"model": model, "tokenizer": tokenizer, "device": device}

    except Exception as e:
        logger.error(f"Failed to load embedding model: {e}")
        return None


def load_reranker_model(model_name: str, device: str):
    """Load the Qwen3-VL-Reranker model (cross-encoder)."""
    logger.info(f"Loading reranker model: {model_name} on {device}")
    start = time.time()

    try:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16 if device != "cpu" else torch.float32,
        )
        model = model.to(device)
        model.eval()

        elapsed = time.time() - start
        logger.info(f"Reranker model loaded in {elapsed:.1f}s")
        return {"model": model, "tokenizer": tokenizer, "device": device}

    except Exception as e:
        logger.error(f"Failed to load reranker model: {e}")
        return None


def embed_texts(texts: list[str], instruction: str | None, dim: int) -> list[list[float]]:
    """Generate embeddings using the loaded model."""
    if embedder is None:
        raise RuntimeError("Embedding model not loaded")

    model = embedder["model"]
    tokenizer = embedder["tokenizer"]
    device = embedder["device"]

    # Prepend instruction if provided (Qwen3-VL-Embedding supports instruction-aware mode)
    if instruction:
        texts = [f"Instruct: {instruction}\nQuery: {t}" for t in texts]

    # Tokenize
    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=8192,
        return_tensors="pt",
    )
    encoded = {k: v.to(device) for k, v in encoded.items()}

    # Inference
    with torch.no_grad():
        outputs = model(**encoded)

    # Mean pooling over last hidden state
    last_hidden = outputs.last_hidden_state  # [batch, seq_len, hidden_dim]
    attention_mask = encoded["attention_mask"].unsqueeze(-1).float()
    summed = (last_hidden * attention_mask).sum(dim=1)
    counts = attention_mask.sum(dim=1).clamp(min=1e-9)
    embeddings = summed / counts  # [batch, hidden_dim]

    # MRL: truncate to requested dimension
    if dim > 0 and dim < embeddings.shape[-1]:
        embeddings = embeddings[:, :dim]

    # L2 normalize
    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=-1)

    return embeddings.cpu().float().tolist()


def rerank_documents(query: str, documents: list[str], instruction: str | None) -> list[float]:
    """Score (query, document) pairs using the cross-encoder reranker."""
    if reranker is None:
        raise RuntimeError("Reranker model not loaded")

    model = reranker["model"]
    tokenizer = reranker["tokenizer"]
    device = reranker["device"]

    # Build (query, document) pairs
    if instruction:
        query_text = f"Instruct: {instruction}\nQuery: {query}"
    else:
        query_text = query

    pairs = [[query_text, doc] for doc in documents]

    # Tokenize pairs
    encoded = tokenizer(
        pairs,
        padding=True,
        truncation=True,
        max_length=8192,
        return_tensors="pt",
    )
    encoded = {k: v.to(device) for k, v in encoded.items()}

    # Inference
    with torch.no_grad():
        outputs = model(**encoded)

    # Logits → scores (sigmoid for relevance probability)
    scores = torch.sigmoid(outputs.logits.squeeze(-1))
    return scores.cpu().float().tolist()


# ─── App lifecycle ───────────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    global embedder, reranker
    device = detect_device()
    logger.info(f"Device: {device}, Embedding dim: {EMBEDDING_DIM}")

    embedder = load_embedding_model(EMBEDDING_MODEL, device)
    reranker = load_reranker_model(RERANKER_MODEL, device)

    if embedder is None and reranker is None:
        logger.warning("No models loaded! Sidecar will return errors for all requests.")

    yield

    # Cleanup
    embedder = None
    reranker = None
    logger.info("Sidecar shutdown complete")


app = FastAPI(
    title="Ucotron Sidecar",
    description="Embedding & Reranker sidecar for HuggingFace Transformers models",
    version="0.1.0",
    lifespan=lifespan,
)


# ─── Endpoints ───────────────────────────────────────────────────────────────


@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="ok",
        embedding_model=EMBEDDING_MODEL if embedder else None,
        reranker_model=RERANKER_MODEL if reranker else None,
        embedding_loaded=embedder is not None,
        reranker_loaded=reranker is not None,
        device=detect_device(),
        embedding_dim=EMBEDDING_DIM,
    )


@app.post("/embed", response_model=EmbedResponse)
async def embed(req: EmbedRequest):
    if embedder is None:
        raise HTTPException(status_code=503, detail="Embedding model not loaded")
    if not req.texts:
        raise HTTPException(status_code=400, detail="texts must not be empty")

    dim = req.dimension or EMBEDDING_DIM
    try:
        vectors = embed_texts(req.texts, req.instruction, dim)
    except Exception as e:
        logger.error(f"Embedding error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    return EmbedResponse(
        embeddings=vectors,
        dimension=dim,
        model=EMBEDDING_MODEL,
    )


@app.post("/rerank", response_model=RerankResponse)
async def rerank(req: RerankRequest):
    if reranker is None:
        raise HTTPException(status_code=503, detail="Reranker model not loaded")
    if not req.documents:
        raise HTTPException(status_code=400, detail="documents must not be empty")

    try:
        scores = rerank_documents(req.query, req.documents, req.instruction)
    except Exception as e:
        logger.error(f"Reranking error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    return RerankResponse(
        scores=scores,
        model=RERANKER_MODEL,
    )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT)
