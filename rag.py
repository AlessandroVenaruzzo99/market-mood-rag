"""Local document ingestion and semantic retrieval through Ollama embeddings."""

from __future__ import annotations

import hashlib
import json
import math
import sqlite3
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from pypdf import PdfReader


RAG_DIR = Path(__file__).resolve().parent / "rag_files"
RAG_DB = RAG_DIR / "index.sqlite3"
OLLAMA_EMBED_URL = "http://localhost:11434/api/embed"
OLLAMA_LEGACY_EMBED_URL = "http://localhost:11434/api/embeddings"
DEFAULT_EMBEDDING_MODEL = "nomic-embed-text"
SUPPORTED_SUFFIXES = {".txt", ".md", ".csv", ".json", ".pdf"}


def ensure_rag_directory() -> None:
    RAG_DIR.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(RAG_DB) as connection:
        connection.execute(
            """CREATE TABLE IF NOT EXISTS chunks (
                id TEXT PRIMARY KEY, source TEXT NOT NULL, chunk_index INTEGER NOT NULL,
                content TEXT NOT NULL, embedding TEXT NOT NULL,
                ticker TEXT NOT NULL DEFAULT '', period TEXT NOT NULL DEFAULT '',
                source_name TEXT NOT NULL DEFAULT '', document_date TEXT NOT NULL DEFAULT ''
            )"""
        )
        columns = {row[1] for row in connection.execute("PRAGMA table_info(chunks)")}
        for name in ("ticker", "period", "source_name", "document_date"):
            if name not in columns:
                connection.execute(f"ALTER TABLE chunks ADD COLUMN {name} TEXT NOT NULL DEFAULT ''")
        connection.commit()


def _chunks(text: str, size: int = 1200, overlap: int = 180) -> list[str]:
    normalized = " ".join(text.split())
    if not normalized:
        return []
    step = max(1, size - overlap)
    return [normalized[start:start + size] for start in range(0, len(normalized), step)]


def extract_text(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        return "\n".join(page.extract_text() or "" for page in PdfReader(str(path)).pages)
    return path.read_text(encoding="utf-8", errors="replace")


def _embedding(model: str, text: str) -> list[float]:
    request = Request(OLLAMA_EMBED_URL, data=json.dumps({"model": model, "input": text}).encode(),
                      headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urlopen(request, timeout=120) as response:
            result = json.loads(response.read().decode("utf-8"))
        if result.get("embeddings"):
            return result["embeddings"][0]
        if result.get("embedding"):
            return result["embedding"]
    except (HTTPError, URLError, TimeoutError, OSError):
        pass
    legacy = Request(OLLAMA_LEGACY_EMBED_URL,
                     data=json.dumps({"model": model, "prompt": text}).encode(),
                     headers={"Content-Type": "application/json"}, method="POST")
    with urlopen(legacy, timeout=120) as response:
        vector = json.loads(response.read().decode("utf-8")).get("embedding")
    if not vector:
        raise ValueError("Ollama non ha restituito un embedding.")
    return vector


def index_file(path: Path, model: str = DEFAULT_EMBEDDING_MODEL, *, ticker: str = "",
               period: str = "", source_name: str = "", document_date: str = "") -> int:
    ensure_rag_directory()
    chunks = _chunks(extract_text(path))
    with sqlite3.connect(RAG_DB) as connection:
        connection.execute("DELETE FROM chunks WHERE source = ?", (path.name,))
        for index, content in enumerate(chunks):
            chunk_id = hashlib.sha256(f"{path.name}:{index}:{content}".encode()).hexdigest()
            connection.execute(
                """INSERT OR REPLACE INTO chunks
                (id, source, chunk_index, content, embedding, ticker, period, source_name, document_date)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (chunk_id, path.name, index, content, json.dumps(_embedding(model, content)),
                 ticker.strip().upper(), period.strip(), source_name.strip(), document_date.strip()),
            )
        connection.commit()
    return len(chunks)


def _cosine(left: list[float], right: list[float]) -> float:
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    return dot / (left_norm * right_norm) if left_norm and right_norm else 0.0


def search(query: str, model: str = DEFAULT_EMBEDDING_MODEL, limit: int = 5, *,
           ticker: str = "", period: str = "", source_name: str = "",
           document_date: str = "") -> list[dict]:
    ensure_rag_directory()
    filters = ["1=1"]
    parameters: list[str] = []
    for column, value in (("ticker", ticker), ("period", period),
                          ("source_name", source_name), ("document_date", document_date)):
        if value.strip():
            filters.append(f"{column} = ?")
            parameters.append(value.strip().upper() if column == "ticker" else value.strip())
    with sqlite3.connect(RAG_DB) as connection:
        rows = connection.execute(
            "SELECT source, chunk_index, content, embedding, ticker, period, source_name, document_date "
            f"FROM chunks WHERE {' AND '.join(filters)}", parameters
        ).fetchall()
    query_vector = _embedding(model, query)
    ranked = []
    for source, chunk_index, content, embedding, row_ticker, row_period, row_source, row_date in rows:
        ranked.append({"source": source, "chunk_index": chunk_index, "content": content,
                       "ticker": row_ticker, "period": row_period, "source_name": row_source,
                       "document_date": row_date, "score": _cosine(query_vector, json.loads(embedding))})
    return sorted(ranked, key=lambda item: item["score"], reverse=True)[:limit]


def save_uploaded_file(uploaded_file) -> Path:
    ensure_rag_directory()
    suffix = Path(uploaded_file.name).suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise ValueError("Formato non supportato. Usa TXT, Markdown, CSV, JSON o PDF.")
    destination = RAG_DIR / Path(uploaded_file.name).name
    destination.write_bytes(uploaded_file.getvalue())
    return destination
