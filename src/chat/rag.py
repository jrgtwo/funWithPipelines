"""Retrieval-Augmented Generation: dataset loading, embedding, indexing, and retrieval."""

import csv
import json
import hashlib
from pathlib import Path
from typing import Optional

import faiss
import numpy as np

from .config import DATASETS_DIR
from .ui import console

_embed_model = None


def _get_embed_model(model_name: str = "all-MiniLM-L6-v2"):
    """Lazy-load and cache the sentence-transformers embedding model."""
    global _embed_model
    if _embed_model is None:
        from sentence_transformers import SentenceTransformer
        console.print(f"[dim]Loading embedding model: {model_name}...[/dim]")
        _embed_model = SentenceTransformer(model_name)
        console.print("[dim]Embedding model ready.[/dim]")
    return _embed_model


def _hash_path(path: str) -> str:
    return hashlib.sha256(path.encode()).hexdigest()[:16]


def load_csv(filepath: str) -> list[str]:
    """Load a CSV file into text chunks (one per row)."""
    chunks = []
    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            text = " | ".join(f"{k}: {v}" for k, v in row.items() if v)
            if text.strip():
                chunks.append(text)
    return chunks


def load_json(filepath: str) -> list[str]:
    """Load a JSON file into text chunks.

    Supports:
      - List of objects: each object becomes a chunk
      - Dict with a list value: items from the largest list become chunks
      - Dict of primitives: entire dict is one chunk
    """
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        items = data
    elif isinstance(data, dict):
        list_vals = {k: v for k, v in data.items() if isinstance(v, list)}
        if list_vals:
            key = max(list_vals, key=lambda k: len(list_vals[k]))
            items = list_vals[key]
        else:
            return [" | ".join(f"{k}: {v}" for k, v in data.items() if v)]
    else:
        return [str(data)]

    chunks = []
    for item in items:
        if isinstance(item, dict):
            text = " | ".join(f"{k}: {v}" for k, v in item.items() if v)
        else:
            text = str(item)
        if text.strip():
            chunks.append(text)
    return chunks


class RAGIndex:
    """Manages a FAISS index over text chunks from a single dataset file."""

    def __init__(self):
        self.index: Optional[faiss.IndexFlatIP] = None
        self.chunks: list[str] = []
        self.source_name: str = ""
        self.dimension: int = 0

    def build(self, filepath: str, embedding_model_name: str = "all-MiniLM-L6-v2") -> int:
        """Load file, embed chunks, build FAISS index. Returns chunk count."""
        path = Path(filepath)
        self.source_name = path.name

        suffix = path.suffix.lower()
        if suffix == ".csv":
            self.chunks = load_csv(filepath)
        elif suffix in (".json", ".jsonl"):
            self.chunks = load_json(filepath)
        else:
            raise ValueError(f"Unsupported file type: {suffix}. Use .csv or .json.")

        if not self.chunks:
            raise ValueError(f"No data found in {filepath}")

        model = _get_embed_model(embedding_model_name)
        console.print(f"[dim]Embedding {len(self.chunks)} chunks...[/dim]")
        embeddings = model.encode(self.chunks, show_progress_bar=True, normalize_embeddings=True)
        embeddings = np.array(embeddings, dtype=np.float32)

        self.dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(self.dimension)
        self.index.add(embeddings)

        self._save_cache(filepath)
        return len(self.chunks)

    def query(self, question: str, top_k: int = 5, embedding_model_name: str = "all-MiniLM-L6-v2") -> list[str]:
        """Retrieve top-k relevant chunks for a question."""
        if self.index is None or not self.chunks:
            return []

        model = _get_embed_model(embedding_model_name)
        q_embedding = model.encode([question], normalize_embeddings=True)
        q_embedding = np.array(q_embedding, dtype=np.float32)

        k = min(top_k, len(self.chunks))
        scores, indices = self.index.search(q_embedding, k)

        results = []
        for idx in indices[0]:
            if idx >= 0:
                results.append(self.chunks[idx])
        return results

    def _cache_dir(self, filepath: str) -> Path:
        h = _hash_path(str(Path(filepath).resolve()))
        return DATASETS_DIR / h

    def _save_cache(self, filepath: str):
        cache = self._cache_dir(filepath)
        cache.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(cache / "index.faiss"))
        with open(cache / "chunks.json", "w", encoding="utf-8") as f:
            json.dump({"source": self.source_name, "chunks": self.chunks}, f)

    def load_cache(self, filepath: str) -> bool:
        """Try to load a cached index. Returns True if successful."""
        cache = self._cache_dir(filepath)
        index_path = cache / "index.faiss"
        chunks_path = cache / "chunks.json"
        if index_path.exists() and chunks_path.exists():
            self.index = faiss.read_index(str(index_path))
            with open(chunks_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.chunks = data["chunks"]
            self.source_name = data["source"]
            self.dimension = self.index.d
            return True
        return False


def pick_dataset() -> Optional[str]:
    """Open a file dialog filtered to CSV/JSON files."""
    import tkinter as tk
    from tkinter import filedialog
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    file_path = filedialog.askopenfilename(
        title="Select a dataset (CSV or JSON)",
        filetypes=[("Data files", "*.csv *.json"), ("CSV", "*.csv"), ("JSON", "*.json")],
    )
    root.destroy()
    return file_path or None


def format_rag_context(chunks: list[str], source_name: str) -> str:
    """Format retrieved chunks into a context block for prompt injection."""
    if not chunks:
        return ""
    header = f"### Relevant data from {source_name}:"
    body = "\n\n".join(f"- {chunk}" for chunk in chunks)
    return f"{header}\n{body}"
