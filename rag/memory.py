"""Multi-turn conversation memory with history-aware query rewriting.

Follow-ups like "وكم رسومها؟" ("and what are its fees?") are elliptical. Before
retrieval we rewrite them into a standalone Arabic question using the recent
history (and the last service discussed), so the retriever's own service-name
detection can lock onto the right service.
"""
from __future__ import annotations

from dataclasses import dataclass, field

_REWRITE_SYSTEM = (
    "أنت مساعد يعيد صياغة سؤال المستخدم ليصبح سؤالاً مستقلاً ومفهوماً بذاته "
    "بالاعتماد على سياق المحادثة السابق. أعد كتابة السؤال الأخير فقط بحيث يتضمن "
    "اسم الخدمة أو الموضوع المقصود إن كان ضمنياً. أعد السؤال المعاد صياغته فقط "
    "دون أي شرح أو مقدمات."
)


@dataclass
class ConversationMemory:
    messages: list[tuple[str, str]] = field(default_factory=list)  # (role, content)
    last_service_id: str | None = None
    last_service_name: str | None = None
    max_turns: int = 6

    def add_user(self, text: str) -> None:
        self.messages.append(("user", text))

    def add_assistant(self, text: str) -> None:
        self.messages.append(("assistant", text))

    def recent(self) -> list[tuple[str, str]]:
        return self.messages[-self.max_turns * 2 :]

    def history_text(self) -> str:
        lines = []
        for role, content in self.recent():
            who = "المستخدم" if role == "user" else "المساعد"
            lines.append(f"{who}: {content}")
        return "\n".join(lines)

    def rewrite_query(self, llm, question: str) -> str:
        """Rewrite an elliptical follow-up into a standalone question."""
        if not self.messages:
            return question
        ctx = self.history_text()
        if self.last_service_name:
            ctx += f"\n(الخدمة محل النقاش: {self.last_service_name})"
        user = f"سياق المحادثة:\n{ctx}\n\nالسؤال الأخير: {question}\n\nالسؤال المستقل:"
        try:
            rewritten = llm.complete(_REWRITE_SYSTEM, user).strip()
        except Exception:
            return question
        # guard against the model returning something empty or absurdly long
        if not rewritten or len(rewritten) > 4 * len(question) + 200:
            return question
        return rewritten
