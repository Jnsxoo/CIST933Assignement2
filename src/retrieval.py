"""
Embedding and retrieval utilities.

The default implementation uses sentence-transformers for embeddings
and scikit-learn cosine similarity for retrieval.

Supports caching embeddings to disk to avoid recomputation.
"""

from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from config import PROJECT_ROOT


Chunk = Dict[str, Any]

CACHE_DIR = PROJECT_ROOT / "data" / "cache"


class EmbeddingRetriever:
    """
    Embedding-based retriever with disk caching.
    """

    def __init__(self, embedding_model_name: str):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is required for this starter retriever. "
                "Install with: pip install sentence-transformers"
            ) from exc

        self.model = SentenceTransformer(embedding_model_name)
        self.model_name = embedding_model_name
        self.chunks: List[Chunk] = []
        self.embeddings: np.ndarray | None = None

    def _cache_key(self, chunks: List[Chunk]) -> str:
        """Generate a hash key based on model name and chunk texts."""
        texts = [c["text"] for c in chunks]
        content = json.dumps({"model": self.model_name, "n": len(texts),
                              "hash": hashlib.md5("".join(texts).encode()).hexdigest()})
        return hashlib.md5(content.encode()).hexdigest()

    def build_index(self, chunks: List[Chunk]) -> None:
        """
        Create embeddings for all chunks. Uses disk cache if available.
        """
        if not chunks:
            raise ValueError("No chunks supplied to build_index().")

        self.chunks = chunks
        cache_key = self._cache_key(chunks)
        cache_path = CACHE_DIR / f"embeddings_{cache_key}.pkl"

        if cache_path.exists():
            print(f"Loading cached embeddings from {cache_path.name}")
            with cache_path.open("rb") as f:
                self.embeddings = pickle.load(f)
            return

        texts = [chunk["text"] for chunk in chunks]
        self.embeddings = np.asarray(self.model.encode(texts, show_progress_bar=True))

        # Save to cache
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        with cache_path.open("wb") as f:
            pickle.dump(self.embeddings, f)
        print(f"Embeddings cached to {cache_path.name}")

    def retrieve(self, query: str, top_k: int = 3) -> List[Tuple[Chunk, float]]:
        """
        Retrieve top-k chunks for a query.
        """
        if self.embeddings is None:
            raise RuntimeError("Index has not been built. Call build_index() first.")

        query_embedding = np.asarray(self.model.encode([query]))
        scores = cosine_similarity(query_embedding, self.embeddings)[0]

        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(self.chunks[i], float(scores[i])) for i in top_indices]
