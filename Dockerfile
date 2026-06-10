# syntax=docker/dockerfile:1

# ---- builder: install CPU-only deps into a venv ----
FROM python:3.12-slim AS builder
ENV PIP_NO_CACHE_DIR=1
WORKDIR /app
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
# Install the CPU build of torch first so sentence-transformers doesn't pull CUDA.
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch
COPY requirements-runtime.txt .
RUN pip install -r requirements-runtime.txt

# ---- runtime ----
FROM python:3.12-slim AS runtime
ENV PYTHONUNBUFFERED=1 \
    PYTHONUTF8=1 \
    PIP_NO_CACHE_DIR=1 \
    PATH="/opt/venv/bin:$PATH" \
    HF_HOME=/home/app/.cache/huggingface \
    OLLAMA_BASE_URL=http://ollama:11434 \
    OLLAMA_MODEL=qwen2.5:7b
WORKDIR /app

# libgomp1 is required by the torch CPU runtime.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/venv /opt/venv

# Non-root user; pre-create the cache + index dirs so mounted named volumes
# inherit the right ownership on first run.
RUN useradd -m -u 1000 app \
    && mkdir -p /home/app/.cache/huggingface /app/data/index \
    && chown -R app:app /home/app /app

COPY --chown=app:app rag/ ./rag/
COPY --chown=app:app api/ ./api/
COPY --chown=app:app cli.py ./
COPY --chown=app:app docker/entrypoint.py ./docker/entrypoint.py
COPY --chown=app:app data/SDTPSPublicInformation.pdf ./data/SDTPSPublicInformation.pdf

USER app
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=180s --retries=5 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health',timeout=3).status==200 else 1)"

ENTRYPOINT ["python", "docker/entrypoint.py"]
