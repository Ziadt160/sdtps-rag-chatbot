"""Central configuration. All knobs come from the environment (no secrets in code).

Load a local .env if python-dotenv is installed; otherwise rely on real env vars.
"""
from __future__ import annotations

import os
from pathlib import Path

try:  # optional, keeps secrets/config out of source
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # pragma: no cover - dotenv is optional
    pass


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.getenv("RAG_DATA_DIR", BASE_DIR / "data"))
INDEX_DIR = DATA_DIR / "index"
PDF_PATH = Path(os.getenv("RAG_PDF_PATH", DATA_DIR / "SDTPSPublicInformation.pdf"))
SERVICES_PATH = DATA_DIR / "services.jsonl"

# Persisted index artifacts
EMBEDDINGS_PATH = INDEX_DIR / "embeddings.npy"
CHUNKS_PATH = INDEX_DIR / "chunks.jsonl"

# Models (fully local) -------------------------------------------------------
EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-m3")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.2"))


def embed_device() -> str:
    """Prefer CUDA when available, else CPU."""
    if os.getenv("EMBED_DEVICE"):
        return os.environ["EMBED_DEVICE"]
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


# Retrieval params -----------------------------------------------------------
TOP_K = int(os.getenv("RAG_TOP_K", "5"))
RRF_K = int(os.getenv("RAG_RRF_K", "60"))           # Reciprocal Rank Fusion constant
DENSE_WEIGHT = float(os.getenv("RAG_DENSE_WEIGHT", "1.0"))
SPARSE_WEIGHT = float(os.getenv("RAG_SPARSE_WEIGHT", "1.0"))
# rapidfuzz score (0-100) above which a query is locked to one service via metadata filter
SERVICE_MATCH_THRESHOLD = float(os.getenv("RAG_SERVICE_MATCH_THRESHOLD", "82"))


# Semantic answer cache --------------------------------------------------------
CACHE_ENABLED = os.getenv("CACHE_ENABLED", "1").lower() in {"1", "true", "yes"}
CACHE_MAX_SIZE = int(os.getenv("CACHE_MAX_SIZE", "256"))
# cosine similarity required for a cache hit (high = only near-identical queries)
CACHE_SIM_THRESHOLD = float(os.getenv("CACHE_SIM_THRESHOLD", "0.97"))
CACHE_TTL = float(os.getenv("CACHE_TTL", "0"))  # seconds; 0 = no expiry
