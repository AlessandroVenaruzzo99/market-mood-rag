"""Local document ingestion and semantic retrieval through Ollama embeddings."""

from __future__ import annotations

import hashlib
import json
import math
import sqlite3
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


RAG_DIR = Path(__file__).resolve().parent / "rag_files"
RAG_DB = RAG_DIR / "index.sqlite3"
OLLAMA_EMBED_URL = "http://localhost:11434/api/embed"
OLLAMA_LEGACY_EMBED_URL = "http://localhost:11434/api/embeddings"
DEFAULT_EMBEDDING_MODEL = "nomic-embed-text"
SUPPORTED_SUFFIXES = {".txt", ".md", ".csv", ".json"}


def ensure_rag_directory() -> None:
    RAG_DIR.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(RAG_DB) as connection:
        connection.execute(
            """CREATE TABLE IF NOT EXISTS chunks (
                id TEXT PRIMARY KEY,
                source TEXT NOT NULL,
                chunk_index INTEGER NOT NULL,
                content TEXT NOT NULL,
                embedding TEXT NOT NULL
            )"""
        )
        connection.commit()


def _chunks(text: str, size: int = 1200, overlap: int = 180) -> list[str]:
    normalized = " ".join(text.split())
    if not normalized:
        return []
    step = max(1, size - overlap)
    return [normalized[start:start + size] for start in range(0, len(normalized), step)]


def _embedding(model: str, text: str) -> list[float]:
    payload = json.dumps({"model": model, "input": text}).encode("utf-8")
    request = Request(
        OLLAMA_EMBED_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=120) as response:
            result = json.loads(response.read().decode("utf-8"))
        vectors = result.get("embeddings")
        if vectors:
            return vectors[0]
        if result.get("embedding"):
            return result["embedding"]
    except (HTTPError, URLError, TimeoutError, OSError):
        pass

    legacy_request = Request(
        OLLAMA_LEGACY_EMBED_URL,
        data=json.dumps({"model": model, "prompt": text}).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(legacy_request, timeout=120) as response:
        result = json.loads(response.read().decode("utf-8"))
    vector = result.get("embedding")
    if not vector:
        raise ValueError("Ollama non ha restituito un embedding.")
    return vector


def index_file(path: Path, model: str = DEFAULT_EMBEDDING_MODEL) -> int:
    ensure_rag_directory()
    text = path.read_text(encoding="utf-8", errors="replace")
    chunks = _chunks(text)
    with sqlite3.connect(RAG_DB) as connection:
        connection.execute("DELETE FROM chunks WHERE source = ?", (path.name,))
        for index, content in enumerate(chunks):
            chunk_id = hashlib.sha256(
                f"{path.name}:{index}:{content}".encode("utf-8")
            ).hexdigest()
            vector = _embedding(model, content)
            connection.execute(
                "INSERT OR REPLACE INTO chunks VALUES (?, ?, ?, ?, ?)",
                (chunk_id, path.name, index, content, json.dumps(vector)),
            )
        connection.commit()
    return len(chunks)


def index_all(model: str = DEFAULT_EMBEDDING_MODEL) -> tuple[int, int]:
    ensure_rag_directory()
    files = [path for path in RAG_DIR.iterdir() if path.suffix.lower() in SUPPORTED_SUFFIXES]
    chunks = sum(index_file(path, model) for path in files)
    return len(files), chunks


def _cosine(left: list[float], right: list[float]) -> float:
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if not left_norm or not right_norm:
        return 0.0
    return dot / (left_norm * right_norm)


def search(query: str, model: str = DEFAULT_EMBEDDING_MODEL, limit: int = 5) -> list[dict]:
    ensure_rag_directory()
    query_vector = _embedding(model, query)
    with sqlite3.connect(RAG_DB) as connection:
        rows = connection.execute(
            "SELECT source, chunk_index, content, embedding FROM chunks"
        ).fetchall()
    ranked = [
        {
            "source": source,
            "chunk_index": chunk_index,
            "content": content,
            "score": _cosine(query_vector, json.loads(embedding)),
        }
        for source, chunk_index, content, embedding in rows
    ]
    return sorted(ranked, key=lambda item: item["score"], reverse=True)[:limit]


def save_uploaded_file(uploaded_file) -> Path:
    ensure_rag_directory()
    suffix = Path(uploaded_file.name).suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise ValueError("Formato non supportato. Usa TXT, Markdown, CSV o JSON.")
    destination = RAG_DIR / Path(uploaded_file.name).name
    destination.write_bytes(uploaded_file.getvalue())
    return destination
