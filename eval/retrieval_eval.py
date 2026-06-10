"""Offline retrieval metrics: Hit@k, MRR, Recall@k.

No LLM involved — this isolates *retrieval* quality. For each golden question we
run the HybridRetriever and check where an acceptable service_id lands.
"""
from __future__ import annotations

import json
from pathlib import Path

from rag.embeddings import get_embedder
from rag.index import load_index
from rag.retriever import HybridRetriever

GOLDEN_PATH = Path(__file__).resolve().parent / "golden.jsonl"
KS = (1, 3, 5)


def load_golden(n: int | None = None) -> list[dict]:
    rows = []
    with GOLDEN_PATH.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows[:n] if n else rows


def run_retrieval_eval(n: int | None = None, k: int = 10) -> dict:
    golden = load_golden(n)
    index = load_index()
    retriever = HybridRetriever(index, get_embedder())

    hits = {kk: 0 for kk in KS}
    recall = {kk: 0.0 for kk in KS}
    rr_sum = 0.0
    misses: list[tuple[str, list[str]]] = []

    for row in golden:
        gold = set(row["service_ids"])
        results = retriever.retrieve(row["question"], k=k)
        ranked_ids = [r.chunk.metadata["service_id"] for r in results]

        first_rank = next((i for i, sid in enumerate(ranked_ids) if sid in gold), None)
        if first_rank is not None:
            rr_sum += 1.0 / (first_rank + 1)
        for kk in KS:
            topk = ranked_ids[:kk]
            if any(sid in gold for sid in topk):
                hits[kk] += 1
            recall[kk] += len(gold.intersection(topk)) / len(gold)
        if first_rank is None or first_rank >= 3:
            misses.append((row["question"], ranked_ids[:3]))

    total = len(golden)
    print("\n=== Retrieval evaluation ===")
    print(f"questions: {total}")
    for kk in KS:
        print(f"  Hit@{kk}:    {hits[kk] / total:.3f}   "
              f"Recall@{kk}: {recall[kk] / total:.3f}")
    print(f"  MRR:       {rr_sum / total:.3f}")
    if misses:
        print(f"\n  weak/missed ({len(misses)}):")
        for q, top in misses:
            print(f"    - {q[:55]}  ->  {top}")

    return {
        "questions": total,
        "hit": {kk: hits[kk] / total for kk in KS},
        "recall": {kk: recall[kk] / total for kk in KS},
        "mrr": rr_sum / total,
    }


if __name__ == "__main__":
    run_retrieval_eval()
