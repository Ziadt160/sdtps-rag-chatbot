"""End-to-end RAG chatbot: rewrite -> retrieve -> generate (grounded, in Arabic)."""
from __future__ import annotations

from dataclasses import dataclass, field

from .embeddings import get_embedder
from .index import Index, load_index
from .llm import LLM, get_llm
from .memory import ConversationMemory
from .retriever import HybridRetriever, Retrieved

_SYSTEM = (
    "أنت مساعد ذكي لدائرة التخطيط العمراني والمساحة في الشارقة. مهمتك الإجابة على "
    "أسئلة المستخدمين حول الخدمات بالاعتماد فقط على المعلومات الواردة في السياق. "
    "إذا لم تكن الإجابة موجودة في السياق فاذكر بوضوح أنك لا تملك المعلومة ولا تختلق "
    "إجابة. أجب باللغة العربية بشكل واضح ومنظم."
)


@dataclass
class Answer:
    answer: str
    sources: list[dict] = field(default_factory=list)
    contexts: list[str] = field(default_factory=list)
    query_used: str = ""


def _format_context(results: list[Retrieved]) -> str:
    blocks = []
    for i, r in enumerate(results, 1):
        blocks.append(f"[مصدر {i} | صفحة {r.chunk.metadata.get('page')}]\n{r.chunk.text}")
    return "\n\n".join(blocks)


class RAGChatbot:
    def __init__(
        self,
        index: Index | None = None,
        retriever: HybridRetriever | None = None,
        llm: LLM | None = None,
    ):
        self.index = index or load_index()
        self.retriever = retriever or HybridRetriever(self.index, get_embedder())
        self.llm = llm or get_llm()

    def ask(
        self, question: str, memory: ConversationMemory | None = None, k: int | None = None
    ) -> Answer:
        # 1. history-aware rewrite (only meaningful when memory has prior turns)
        query = memory.rewrite_query(self.llm, question) if memory else question

        # 2. hybrid retrieval (service-name filter + dense + BM25)
        results = self.retriever.retrieve(query, k=k)

        # 3. grounded generation
        context = _format_context(results)
        user = f"السياق:\n{context}\n\nالسؤال: {question}"
        answer_text = self.llm.complete(_SYSTEM, user)

        sources = [
            {
                "service_id": r.chunk.metadata.get("service_id"),
                "service_name": r.chunk.metadata.get("service_name"),
                "page": r.chunk.metadata.get("page"),
                "score": round(r.score, 4),
            }
            for r in results
        ]

        # 4. update memory
        if memory is not None:
            memory.add_user(question)
            memory.add_assistant(answer_text)
            if results:
                top = results[0].chunk.metadata
                memory.last_service_id = top.get("service_id")
                memory.last_service_name = top.get("service_name")

        return Answer(
            answer=answer_text,
            sources=sources,
            contexts=[r.chunk.text for r in results],
            query_used=query,
        )
