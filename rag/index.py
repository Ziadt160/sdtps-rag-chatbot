"""Build / load the search index.

With only ~43 services there is no need for a vector database: dense vectors are
a small numpy matrix (brute-force cosine is instant) and BM25 is rebuilt in
memory from the stored corpus. Artifacts persisted to ``data/index/``:
  - embeddings.npy   float32 [N, D], L2-normalized
  - chunks.jsonl     the chunks (text, search_text, metadata)
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

from . import arabic, config
from .chunk import Chunk, load_chunks, records_to_chunks, save_chunks
from .embeddings import Embedder, get_embedder
from .parse import ServiceRecord, load_records


@dataclass
class Index:
    chunks: list[Chunk]
    embeddings: np.ndarray          # [N, D] normalized
    bm25: BM25Okapi
    tokenized: list[list[str]]

    @property
    def size(self) -> int:
        return len(self.chunks)


def _tokenize_corpus(chunks: list[Chunk]) -> list[list[str]]:
    return [arabic.tokenize(c.search_text) for c in chunks]


def build_index(records: list[ServiceRecord], embedder: Embedder | None = None) -> Index:
    embedder = embedder or get_embedder()
    chunks = records_to_chunks(records)
    embeddings = embedder.encode([c.text for c in chunks])
    tokenized = _tokenize_corpus(chunks)
    bm25 = BM25Okapi(tokenized)
    return Index(chunks=chunks, embeddings=embeddings, bm25=bm25, tokenized=tokenized)


def save_index(index: Index) -> None:
    config.INDEX_DIR.mkdir(parents=True, exist_ok=True)
    np.save(config.EMBEDDINGS_PATH, index.embeddings)
    save_chunks(index.chunks, config.CHUNKS_PATH)


def load_index() -> Index:
    if not Path(config.CHUNKS_PATH).exists():
        raise FileNotFoundError(
            f"No index at {config.INDEX_DIR}. Run `python cli.py index` first."
        )
    chunks = load_chunks(config.CHUNKS_PATH)
    embeddings = np.load(config.EMBEDDINGS_PATH).astype(np.float32)
    tokenized = _tokenize_corpus(chunks)
    bm25 = BM25Okapi(tokenized)
    return Index(chunks=chunks, embeddings=embeddings, bm25=bm25, tokenized=tokenized)


def build_and_save() -> Index:
    records = load_records(config.SERVICES_PATH)
    index = build_index(records)
    save_index(index)
    return index
