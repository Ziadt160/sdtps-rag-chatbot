"""Local Arabic Q&A RAG pipeline for Sharjah town-planning services (SDTPS).

Fully local: BGE-M3 embeddings (GPU) + Ollama LLM, hybrid retrieval
(metadata filter + dense + BM25), multi-turn memory. No external services.
"""
