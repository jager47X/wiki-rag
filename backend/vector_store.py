"""FAISS-backed vector store with cosine similarity search."""

import numpy as np
import faiss
from typing import List, Dict, Tuple, Callable


class VectorStore:
    """Vector store backed by a FAISS IndexFlatIP (cosine on normalized vectors)."""

    def __init__(self, embed_fn: Callable[[str], List[float]], index: faiss.Index):
        self.embed_fn = embed_fn
        self.index = index
        self.documents: List[Dict] = []

    def add(self, doc: Dict) -> None:
        """Add a document to the in-memory doc list (vector already in FAISS)."""
        self.documents.append(doc)

    def search(
        self, query_embedding: List[float], top_k: int = 10
    ) -> List[Tuple[Dict, float]]:
        """Return top_k (document, score) pairs by cosine similarity."""
        if not self.documents or self.index.ntotal == 0:
            return []
        q = np.array([query_embedding], dtype=np.float32)
        faiss.normalize_L2(q)
        k = min(top_k, self.index.ntotal)
        scores, indices = self.index.search(q, k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if 0 <= idx < len(self.documents):
                results.append((self.documents[idx], float(score)))
        return results

    @property
    def size(self) -> int:
        return len(self.documents)
