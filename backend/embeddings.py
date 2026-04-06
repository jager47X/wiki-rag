"""Local embeddings via sentence-transformers (no API key needed)."""

from typing import List
from sentence_transformers import SentenceTransformer

from backend.config import EMBED_MODEL

_model = SentenceTransformer(EMBED_MODEL)


def embed(text: str) -> List[float]:
    """Embed a single text string, returns a float vector."""
    return _model.encode(text, normalize_embeddings=True).tolist()


def embed_batch(texts: List[str]) -> List[List[float]]:
    """Embed multiple texts at once. Returns list of vectors."""
    embeddings = _model.encode(texts, normalize_embeddings=True, batch_size=64)
    return embeddings.tolist()
