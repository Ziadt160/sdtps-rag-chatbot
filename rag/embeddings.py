"""BGE-M3 embeddings via sentence-transformers (GPU when available).

BGE-M3 is strong on Arabic and needs no query/passage prefixes, so the same
encoder serves documents and queries. Embeddings are L2-normalized so cosine
similarity is a plain dot product.
"""
from __future__ import annotations

import numpy as np

from . import config


class Embedder:
    def __init__(self, model_name: str | None = None, device: str | None = None):
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name or config.EMBED_MODEL
        self.device = device or config.embed_device()
        self.model = SentenceTransformer(self.model_name, device=self.device)

    def encode(self, texts: list[str], batch_size: int = 16) -> np.ndarray:
        embs = self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return embs.astype(np.float32)

    def encode_one(self, text: str) -> np.ndarray:
        return self.encode([text])[0]


_EMBEDDER: Embedder | None = None


def get_embedder() -> Embedder:
    """Process-wide singleton so the model loads once."""
    global _EMBEDDER
    if _EMBEDDER is None:
        _EMBEDDER = Embedder()
    return _EMBEDDER
