"""FastAPI service for the SDTPS Arabic RAG chatbot.

Loads the RAG pipeline once at startup and keeps per-session conversation memory
so the chat simulator gets multi-turn behaviour. Endpoints:

  GET  /            -> the chat simulator (HTML)
  POST /chat        -> {message, session_id?} -> {answer, sources, ...}
  POST /reset       -> {session_id} -> clear that conversation
  GET  /health      -> liveness + loaded model info
"""
from __future__ import annotations

import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from threading import Lock

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from rag import config
from rag.memory import ConversationMemory
from rag.pipeline import RAGChatbot

WEB_DIR = Path(__file__).resolve().parent / "web"

# Process-wide state
_STATE: dict[str, ConversationMemory] = {}
_BOT: RAGChatbot | None = None
# Ollama serves serially; serialize generation so concurrent requests queue cleanly.
_GEN_LOCK = Lock()


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _BOT
    _BOT = RAGChatbot()  # loads index + BGE-M3 + Ollama client
    yield
    _STATE.clear()


app = FastAPI(title="SDTPS RAG Chatbot", version="1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# -- schemas ----------------------------------------------------------------
class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1)
    session_id: str | None = None


class Source(BaseModel):
    service_id: str | None = None
    service_name: str | None = None
    page: int | None = None
    score: float | None = None


class ChatResponse(BaseModel):
    session_id: str
    answer: str
    query_used: str
    sources: list[Source]
    cached: bool = False


class ResetRequest(BaseModel):
    session_id: str


# -- endpoints --------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
def index() -> str:
    html = WEB_DIR / "index.html"
    if not html.exists():
        raise HTTPException(500, "simulator UI not found")
    return html.read_text(encoding="utf-8")


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok" if _BOT is not None else "loading",
        "model": config.OLLAMA_MODEL,
        "embed_model": config.EMBED_MODEL,
        "index_size": _BOT.index.size if _BOT else 0,
        "active_sessions": len(_STATE),
        "cache": _BOT.cache.stats() if (_BOT and _BOT.cache) else None,
    }


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    if _BOT is None:
        raise HTTPException(503, "model still loading")
    message = req.message.strip()
    if not message:
        raise HTTPException(400, "empty message")

    session_id = req.session_id or uuid.uuid4().hex
    memory = _STATE.setdefault(session_id, ConversationMemory())

    with _GEN_LOCK:
        ans = _BOT.ask(message, memory=memory)

    return ChatResponse(
        session_id=session_id,
        answer=ans.answer,
        query_used=ans.query_used,
        sources=[Source(**s) for s in ans.sources],
        cached=ans.cached,
    )


@app.post("/reset")
def reset(req: ResetRequest) -> dict:
    _STATE.pop(req.session_id, None)
    return {"ok": True, "session_id": req.session_id}
