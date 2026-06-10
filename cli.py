"""Command-line interface for the SDTPS Arabic RAG chatbot.

Usage:
    python cli.py index                 # parse PDF -> services.jsonl -> build index
    python cli.py ask "سؤالك هنا"        # one-shot question
    python cli.py chat                   # interactive multi-turn chat
    python cli.py eval [--ragas] [--n N] # evaluation (retrieval metrics [+ RAGAS])
"""
from __future__ import annotations

import argparse
import sys

# Ensure UTF-8 output for Arabic on Windows consoles.
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stdin.reconfigure(encoding="utf-8")
except Exception:
    pass


def cmd_index(args: argparse.Namespace) -> None:
    from rag import config
    from rag.index import build_index, save_index
    from rag.parse import parse_pdf, save_records

    print(f"Extracting + parsing {config.PDF_PATH} ...")
    records = parse_pdf(config.PDF_PATH)
    save_records(records, config.SERVICES_PATH)
    print(f"  parsed {len(records)} services -> {config.SERVICES_PATH}")

    print(f"Embedding with {config.EMBED_MODEL} on {config.embed_device()} ...")
    index = build_index(records)
    save_index(index)
    print(f"  built index: {index.size} chunks, dim={index.embeddings.shape[1]}")
    print(f"  saved -> {config.INDEX_DIR}")


def _print_answer(ans) -> None:
    print("\n" + ans.answer.strip() + "\n")
    if ans.sources:
        top = ans.sources[0]
        print(f"  ↳ المصدر: {top['service_name']} (صفحة {top['page']}, {top['service_id']})")
    if ans.query_used and ans.query_used.strip():
        print(f"  ↳ الاستعلام المستخدم: {ans.query_used}")


def cmd_ask(args: argparse.Namespace) -> None:
    from rag.pipeline import RAGChatbot

    bot = RAGChatbot()
    ans = bot.ask(args.question, k=args.k)
    _print_answer(ans)


def cmd_chat(args: argparse.Namespace) -> None:
    from rag.memory import ConversationMemory
    from rag.pipeline import RAGChatbot

    bot = RAGChatbot()
    memory = ConversationMemory()
    print("محادثة تفاعلية. اكتب 'خروج' أو exit للإنهاء.\n")
    while True:
        try:
            q = input("أنت: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q:
            continue
        if q.lower() in {"exit", "quit", "خروج"}:
            break
        ans = bot.ask(q, memory=memory, k=args.k)
        _print_answer(ans)


def cmd_serve(args: argparse.Namespace) -> None:
    import uvicorn

    print(f"Starting API + chat simulator on http://{args.host}:{args.port}")
    print("Open that URL in your browser. Ctrl+C to stop.")
    uvicorn.run("api.app:app", host=args.host, port=args.port, reload=args.reload)


def cmd_eval(args: argparse.Namespace) -> None:
    from eval.retrieval_eval import run_retrieval_eval

    run_retrieval_eval(n=args.n)
    if args.ragas:
        from eval.ragas_eval import run_ragas_eval

        run_ragas_eval(n=args.n)


def main() -> None:
    parser = argparse.ArgumentParser(description="SDTPS Arabic RAG chatbot")
    sub = parser.add_subparsers(dest="command", required=True)

    p_index = sub.add_parser("index", help="build the search index from the PDF")
    p_index.set_defaults(func=cmd_index)

    p_ask = sub.add_parser("ask", help="ask a single question")
    p_ask.add_argument("question")
    p_ask.add_argument("-k", type=int, default=None, help="top-k passages")
    p_ask.set_defaults(func=cmd_ask)

    p_chat = sub.add_parser("chat", help="interactive multi-turn chat")
    p_chat.add_argument("-k", type=int, default=None, help="top-k passages")
    p_chat.set_defaults(func=cmd_chat)

    p_serve = sub.add_parser("serve", help="run the web API + chat simulator")
    p_serve.add_argument("--host", default="127.0.0.1")
    p_serve.add_argument("--port", type=int, default=8000)
    p_serve.add_argument("--reload", action="store_true", help="auto-reload (dev)")
    p_serve.set_defaults(func=cmd_serve)

    p_eval = sub.add_parser("eval", help="run evaluation")
    p_eval.add_argument("--ragas", action="store_true", help="also run RAGAS metrics")
    p_eval.add_argument("--n", type=int, default=None, help="limit number of questions")
    p_eval.set_defaults(func=cmd_eval)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
