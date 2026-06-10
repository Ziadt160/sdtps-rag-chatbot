"""Local LLM via Ollama (langchain-ollama).

Model is configurable through OLLAMA_MODEL. Reasoning models (e.g. deepseek-r1)
emit <think>...</think> traces, which we strip from the final answer.
"""
from __future__ import annotations

import re

from . import config

_THINK = re.compile(r"<think>.*?</think>", re.DOTALL)


def strip_think(text: str) -> str:
    return _THINK.sub("", text).strip()


class LLM:
    def __init__(self, model: str | None = None, temperature: float | None = None):
        from langchain_ollama import ChatOllama

        self.model = model or config.OLLAMA_MODEL
        self.client = ChatOllama(
            model=self.model,
            base_url=config.OLLAMA_BASE_URL,
            temperature=config.LLM_TEMPERATURE if temperature is None else temperature,
        )

    def chat(self, messages: list[tuple[str, str]]) -> str:
        """messages: list of (role, content) with role in {system,user,assistant}."""
        resp = self.client.invoke(messages)
        return strip_think(resp.content)

    def complete(self, system: str, user: str) -> str:
        return self.chat([("system", system), ("user", user)])


_LLM: LLM | None = None


def get_llm() -> LLM:
    global _LLM
    if _LLM is None:
        _LLM = LLM()
    return _LLM
