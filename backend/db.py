"""Persistence layer — FAISS for vectors, JSON for chunk metadata and topics."""

import json
import logging
import os
import numpy as np
import faiss
from typing import List, Dict

from backend.config import EMBED_DIM, DATA_DIR

logger = logging.getLogger("wiki-rag.db")

FAISS_PATH = os.path.join(DATA_DIR, "index.faiss")
META_PATH = os.path.join(DATA_DIR, "meta.json")


def _ensure_data_dir() -> None:
    os.makedirs(DATA_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Metadata (chunks + topics) — JSON file
# ---------------------------------------------------------------------------
def _load_meta() -> dict:
    if os.path.exists(META_PATH):
        with open(META_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"chunks": [], "topics": []}


def _save_meta(meta: dict) -> None:
    _ensure_data_dir()
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False)


def get_all_topics() -> set:
    return set(_load_meta().get("topics", []))


def is_topic_fetched(topic: str) -> bool:
    return topic.strip().lower() in get_all_topics()


def mark_topic_fetched(topic: str) -> None:
    meta = _load_meta()
    key = topic.strip().lower()
    if key not in meta["topics"]:
        meta["topics"].append(key)
        _save_meta(meta)


def save_chunks(docs: List[Dict], index: faiss.Index) -> int:
    """Persist new chunks: metadata to JSON, vectors to FAISS.
    Returns count of newly added chunks."""
    meta = _load_meta()
    existing_ids = {c["id"] for c in meta["chunks"]}

    new_docs = [d for d in docs if d["id"] not in existing_ids]
    if not new_docs:
        return 0

    # Add vectors to FAISS
    vectors = []
    for doc in new_docs:
        vectors.append(doc["embedding"])
        meta["chunks"].append({
            "id": doc["id"],
            "title": doc.get("title", ""),
            "section": doc.get("section", ""),
            "text": doc.get("text", ""),
            "parent_id": doc.get("parent_id", ""),
        })

    mat = np.array(vectors, dtype=np.float32)
    faiss.normalize_L2(mat)
    index.add(mat)

    # Persist both
    _ensure_data_dir()
    faiss.write_index(index, FAISS_PATH)
    _save_meta(meta)
    logger.info(f"Saved {len(new_docs)} new chunks (total: {index.ntotal})")
    return len(new_docs)


def load_all() -> tuple:
    """Load FAISS index + chunk metadata. Returns (index, docs)."""
    _ensure_data_dir()
    meta = _load_meta()
    docs = meta.get("chunks", [])

    if os.path.exists(FAISS_PATH) and docs:
        index = faiss.read_index(FAISS_PATH)
        logger.info(f"Loaded FAISS index ({index.ntotal} vectors) + {len(docs)} chunks")
    else:
        index = faiss.IndexFlatIP(EMBED_DIM)
        logger.info(f"Created new FAISS index (dim={EMBED_DIM})")

    return index, docs
