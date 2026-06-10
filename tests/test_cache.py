import numpy as np

from rag.cache import SemanticCache


class DummyEmbedder:
    """Deterministic unit-vector embeddings keyed by exact text."""

    def __init__(self):
        self._vmap: dict[str, np.ndarray] = {}

    def encode_one(self, text: str) -> np.ndarray:
        if text not in self._vmap:
            rng = np.random.default_rng(abs(hash(text)) % (2**32))
            v = rng.standard_normal(16).astype("float32")
            self._vmap[text] = v / np.linalg.norm(v)
        return self._vmap[text]


def test_exact_hit_and_miss_and_stats():
    c = SemanticCache(DummyEmbedder(), threshold=0.99)
    assert c.get("q1") is None
    c.put("q1", {"answer": "A"})
    assert c.get("q1") == {"answer": "A"}              # identical query -> cosine 1.0
    assert c.get("a totally different question") is None
    assert c.hits == 1
    assert c.misses == 2


def test_eviction_respects_max_size():
    c = SemanticCache(DummyEmbedder(), max_size=2, threshold=0.99)
    for i in range(5):
        c.put(f"q{i}", {"answer": str(i)})
    assert len(c) == 2


def test_ttl_expiry():
    clock = {"t": 1000.0}
    c = SemanticCache(DummyEmbedder(), threshold=0.99, ttl=10, clock=lambda: clock["t"])
    c.put("q", {"answer": "A"})
    assert c.get("q") == {"answer": "A"}
    clock["t"] += 20
    assert c.get("q") is None
    assert len(c) == 0
