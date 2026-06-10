"""RAGAS answer-quality metrics with fully-local judges (Ollama LLM + BGE-M3).

Metrics:
  - faithfulness            : is the answer grounded in the retrieved context?
  - response relevancy      : does the answer address the question? (uses embeddings)
  - context precision       : are the retrieved contexts relevant? (no reference)
  - context recall          : did we retrieve what the reference needs? (reference only)

NOTE: RAGAS makes several LLM calls per sample, so with a local 7B model this is
slow. Keep --n small (default 5). If RAGAS won't install on Python 3.14, run this
one command in the Anaconda 3.11 env.
"""
from __future__ import annotations

from eval.retrieval_eval import load_golden


def _build_local_judges():
    """Build the RAGAS judge LLM + embeddings.

    The judge is part of the *eval harness*, not the product, so it can use a
    stronger model than the chatbot. Set RAGAS_JUDGE_MODEL to an Ollama model
    (e.g. a larger/cloud model) for reliable scores; it defaults to the same
    local OLLAMA_MODEL the chatbot uses. The chatbot itself stays fully local.
    """
    import os

    from langchain_core.embeddings import Embeddings
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper

    from rag.embeddings import get_embedder
    from rag.llm import LLM

    class STEmbeddings(Embeddings):
        def __init__(self, embedder):
            self.embedder = embedder

        def embed_documents(self, texts):
            return self.embedder.encode(list(texts)).tolist()

        def embed_query(self, text):
            return self.embedder.encode_one(text).tolist()

    judge_model = os.getenv("RAGAS_JUDGE_MODEL")  # None -> chatbot's OLLAMA_MODEL
    judge_llm = LangchainLLMWrapper(LLM(model=judge_model).client)
    judge_emb = LangchainEmbeddingsWrapper(STEmbeddings(get_embedder()))
    return judge_llm, judge_emb


def _shim_removed_langchain_paths() -> None:
    """RAGAS 0.4.x imports a few langchain_community paths that LangChain v1
    removed (e.g. the Vertex AI chat model). Register lightweight stubs so the
    `import ragas` top-level imports succeed; we never call these classes."""
    import sys
    import types

    for path in ("langchain_community.chat_models.vertexai",):
        if path not in sys.modules:
            mod = types.ModuleType(path)
            mod.ChatVertexAI = type("ChatVertexAI", (), {})
            sys.modules[path] = mod


def run_ragas_eval(n: int | None = 5) -> dict | None:
    try:
        _shim_removed_langchain_paths()
        from ragas import EvaluationDataset, evaluate
        from ragas.dataset_schema import SingleTurnSample
        from ragas.metrics import (
            Faithfulness,
            LLMContextPrecisionWithoutReference,
            LLMContextRecall,
            ResponseRelevancy,
        )
    except Exception as exc:  # pragma: no cover
        print(f"\n[RAGAS] not available ({exc}). `pip install ragas datasets` "
              f"or run this in the Anaconda 3.11 env.")
        return None

    from rag.pipeline import RAGChatbot

    golden = load_golden(n if n else 5)
    bot = RAGChatbot()
    judge_llm, judge_emb = _build_local_judges()

    print(f"\n=== RAGAS evaluation (local judges, n={len(golden)}) ===")
    print("Generating answers (slow with a local LLM)...")
    samples = []
    for row in golden:
        ans = bot.ask(row["question"])
        samples.append(
            SingleTurnSample(
                user_input=row["question"],
                retrieved_contexts=ans.contexts,
                response=ans.answer,
                reference=row.get("reference") or None,
            )
        )

    metrics = [
        Faithfulness(llm=judge_llm),
        ResponseRelevancy(llm=judge_llm, embeddings=judge_emb),
        LLMContextPrecisionWithoutReference(llm=judge_llm),
    ]
    # context recall needs references; include only if any sample has one
    if any(s.reference for s in samples):
        metrics.append(LLMContextRecall(llm=judge_llm))

    # A local 7B judge is slow and Ollama serves requests serially, so run the
    # judge jobs one at a time with a generous timeout (the defaults fire many
    # concurrent jobs that all time out).
    import os

    from ragas.run_config import RunConfig

    # Local Ollama serves serially (keep workers=1); a cloud judge handles
    # concurrency, so bump RAGAS_MAX_WORKERS for speed.
    max_workers = int(os.getenv("RAGAS_MAX_WORKERS", "1"))
    run_config = RunConfig(timeout=600, max_workers=max_workers, max_retries=1)

    dataset = EvaluationDataset(samples=samples)
    result = evaluate(dataset=dataset, metrics=metrics, run_config=run_config)
    print("\nScores:")
    print(result)
    return {"result": str(result)}


if __name__ == "__main__":
    run_ragas_eval()
