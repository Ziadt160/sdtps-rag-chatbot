"""Container entrypoint: wait for Ollama -> ensure model -> build index -> serve."""
from __future__ import annotations

import json
import os
import subprocess
import time
import urllib.request

BASE = os.environ.get("OLLAMA_BASE_URL", "http://ollama:11434")
MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:7b")
AUTO_PULL = os.environ.get("OLLAMA_AUTO_PULL", "1").lower() in {"1", "true", "yes"}


def _tags() -> dict | None:
    try:
        with urllib.request.urlopen(BASE + "/api/tags", timeout=5) as r:
            return json.load(r)
    except Exception:
        return None


def wait_for_ollama() -> None:
    print(f"[entrypoint] waiting for Ollama at {BASE} ...", flush=True)
    while _tags() is None:
        time.sleep(2)
    print("[entrypoint] Ollama is up.", flush=True)


def ensure_model() -> None:
    if not AUTO_PULL:
        return
    names = [m.get("name", "") for m in (_tags() or {}).get("models", [])]
    if any(n == MODEL or n.startswith(MODEL) for n in names):
        return
    print(f"[entrypoint] pulling model {MODEL} (one-time, may take a while) ...", flush=True)
    req = urllib.request.Request(
        BASE + "/api/pull",
        data=json.dumps({"model": MODEL}).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=3600) as r:
            for _ in r:  # stream the pull progress to completion
                pass
        print("[entrypoint] model ready.", flush=True)
    except Exception as exc:  # don't crash; serving can still start
        print(f"[entrypoint] model pull failed: {exc}", flush=True)


def ensure_index() -> None:
    from rag import config

    if config.EMBEDDINGS_PATH.exists():
        return
    print("[entrypoint] building index (one-time) ...", flush=True)
    subprocess.run(["python", "cli.py", "index"], check=True)


def main() -> None:
    wait_for_ollama()
    ensure_model()
    ensure_index()
    print("[entrypoint] starting API on 0.0.0.0:8000", flush=True)
    os.execvp("python", ["python", "cli.py", "serve", "--host", "0.0.0.0", "--port", "8000"])


if __name__ == "__main__":
    main()
