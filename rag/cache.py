"""In-process semantic answer cache.

Repeated or near-identical questions are common for a services chatbot. Instead
of re-running retrieval + the LLM every time, we embed the (standalone) query and
return a cached answer when a previous query is sufficiently similar (cosine >=
threshold). It is deliberately backend-agnostic and dependency-light — just numpy
and the embedder already loaded for retrieval — and easy to swap for Redis later.

Keyed on the rewritten standalone query, so it is safe for multi-turn chat too.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np


@dataclass
class _Entry:
    embedding: np.ndarray
    value: dict
    ts: float


class SemanticCache:
    def __init__(
        self,
        embedder,
        max_size: int = 256,
        threshold: float = 0.97,
        ttl: float = 0.0,
        clock=time.time,
    ):
        self.embedder = embedder
        self.max_size = max_size
        self.threshold = threshold
        self.ttl = ttl  # seconds; <= 0 means no expiry
        self._clock = clock
        self._entries: list[_Entry] = []
        self.hits = 0
        self.misses = 0

    def __len__(self) -> int:
        return len(self._entries)

    def _prune(self, now: float) -> None:
        if self.ttl > 0:
            self._entries = [e for e in self._entries if now - e.ts <= self.ttl]

    def get(self, query: str) -> dict | None:
        now = self._clock()
        self._prune(now)
        if not self._entries:
            self.misses += 1
            return None
        q = self.embedder.encode_one(query)
        sims = np.array([float(np.dot(q, e.embedding)) for e in self._entries])
        idx = int(sims.argmax())
        if sims[idx] >= self.threshold:
            self._entries[idx].ts = now  # refresh recency
            self.hits += 1
            return self._entries[idx].value
        self.misses += 1
        return None

    def put(self, query: str, value: dict) -> None:
        now = self._clock()
        emb = self.embedder.encode_one(query)
        self._entries.append(_Entry(embedding=emb, value=value, ts=now))
        if len(self._entries) > self.max_size:
            self._entries.sort(key=lambda e: e.ts)  # oldest first
            self._entries = self._entries[-self.max_size :]

    def stats(self) -> dict:
        total = self.hits + self.misses
        return {
            "size": len(self._entries),
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": round(self.hits / total, 3) if total else 0.0,
        }
