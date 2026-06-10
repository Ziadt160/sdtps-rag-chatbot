"""Hybrid retriever: metadata (service-name) filter + dense + BM25, fused by RRF.

This is where the "extract the service name and put it in filters" requirement
lives. Three independent rankers are combined with Reciprocal Rank Fusion:

  1. dense   - BGE-M3 cosine similarity
  2. sparse  - BM25 over Arabic-normalized tokens
  3. service - if the query confidently names a known service (rapidfuzz), that
               service is injected as a one-item ranker, boosting it in the fusion
               without nuking recall. A caller may also pass an explicit
               ``service_filter`` (e.g. from conversation memory) to hard-restrict.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from rapidfuzz import fuzz, process

from . import arabic, config
from .chunk import Chunk
from .embeddings import Embedder, get_embedder
from .index import Index


@dataclass
class Retrieved:
    chunk: Chunk
    score: float
    rank: int


def _rrf(rankings: list[list[int]], weights: list[float], k: int) -> dict[int, float]:
    """Reciprocal Rank Fusion. rankings: lists of doc indices best-first."""
    scores: dict[int, float] = {}
    for ranking, w in zip(rankings, weights, strict=False):
        for rank, idx in enumerate(ranking):
            scores[idx] = scores.get(idx, 0.0) + w / (k + rank + 1)
    return scores


class HybridRetriever:
    def __init__(self, index: Index, embedder: Embedder | None = None):
        self.index = index
        self.embedder = embedder or get_embedder()
        # normalized service names aligned with chunk order, for fuzzy detection
        self._norm_names = [
            arabic.normalize(c.metadata.get("service_name", "")) for c in index.chunks
        ]
        self._id_to_idx = {c.metadata["service_id"]: i for i, c in enumerate(index.chunks)}

    # -- service-name metadata detection ------------------------------------
    def detect_service(self, query: str) -> int | None:
        """Return chunk index if the query confidently names one service."""
        norm_q = arabic.normalize(query)
        if not norm_q:
            return None
        match = process.extractOne(
            norm_q, self._norm_names, scorer=fuzz.token_set_ratio
        )
        if match and match[1] >= config.SERVICE_MATCH_THRESHOLD:
            return match[2]  # index
        return None

    # -- main entry ----------------------------------------------------------
    def retrieve(
        self, query: str, k: int | None = None, service_filter: str | None = None
    ) -> list[Retrieved]:
        k = k or config.TOP_K
        n = self.index.size

        # Hard filter (explicit, e.g. from memory): restrict candidate set.
        if service_filter and service_filter in self._id_to_idx:
            allowed = {self._id_to_idx[service_filter]}
        else:
            allowed = set(range(n))

        # dense ranking
        q_emb = self.embedder.encode_one(query)
        dense_sim = self.index.embeddings @ q_emb  # cosine (normalized)
        dense_rank = [i for i in np.argsort(-dense_sim) if i in allowed]

        # sparse (BM25) ranking
        bm25_scores = self.index.bm25.get_scores(arabic.tokenize(query))
        sparse_rank = [i for i in np.argsort(-bm25_scores) if i in allowed]

        rankings = [dense_rank, sparse_rank]
        weights = [config.DENSE_WEIGHT, config.SPARSE_WEIGHT]

        # service-name detection ranker (soft boost), unless already hard-filtered
        if not service_filter:
            detected = self.detect_service(query)
            if detected is not None and detected in allowed:
                rankings.append([detected])
                weights.append(config.DENSE_WEIGHT + config.SPARSE_WEIGHT)

        fused = _rrf(rankings, weights, config.RRF_K)
        ordered = sorted(fused.items(), key=lambda kv: -kv[1])[:k]
        return [
            Retrieved(chunk=self.index.chunks[i], score=s, rank=r)
            for r, (i, s) in enumerate(ordered)
        ]
