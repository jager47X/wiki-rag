"""RAG pipeline wiring — connects all components through advanced-rag-framework."""

import re
import logging
from typing import List, Tuple, Dict, Optional, Generator

from arf import (
    Pipeline, DocumentConfig, Triage, FeatureExtractor,
    ingest_documents, follow_rephrase_chain, ChainResult,
)

from backend.config import (
    TRIAGE_MIN_SCORE, TRIAGE_ACCEPT, TRIAGE_VERIFY, TRIAGE_GAP, TRIAGE_TOP_K,
    MAX_REPHRASE, GRAPH_MAX_HOPS,
)
from backend.embeddings import embed, embed_batch
from backend.llm import score_relevance, rephrase_query, extract_search_topics, generate_answer
from backend.ingestion import fetch_topic, chunk_article
from backend.vector_store import VectorStore
from backend import db

logger = logging.getLogger("wiki-rag.pipeline")

DOC_CONFIG = DocumentConfig(
    id_field="id",
    title_field="title",
    text_fields=["text"],
    parent_field="parent_id",
    children_fields=[],
    hierarchy=["article", "chunk"],
    domain_id=1,
)

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------
_store: Optional[VectorStore] = None
_keyword_index: Dict[str, List[Tuple[dict, float]]] = {}
_parent_docs: Dict[str, dict] = {}
_query_cache: Dict[str, dict] = {}
_predict_fn = None
_mlp_trained_at = 0
_initialized = False

STOP_WORDS = frozenset(
    "the a an is are was were in on at to for of and or not with by from as "
    "this that it be has have had do does did will would could can may might "
    "shall should".split()
)


def _ensure_initialized() -> None:
    global _store, _initialized
    if _initialized:
        return
    _initialized = True
    index, docs = db.load_all()
    _store = VectorStore(embed_fn=embed, index=index)
    for doc in docs:
        _store.add(doc)
        _add_keyword_entries(doc)
    logger.info(f"Initialized: {_store.size} chunks from disk")


def _add_keyword_entries(doc: dict) -> None:
    title = doc.get("title", "").lower()
    text = doc.get("text", "").lower()
    words = set(re.findall(r"\b[a-z]{3,}\b", f"{title} {text}")) - STOP_WORDS
    title_words = set(re.findall(r"\b[a-z]{3,}\b", title))
    for word in words:
        weight = 0.7 if word in title_words else 0.3
        _keyword_index.setdefault(word, []).append((doc, weight))


# ---------------------------------------------------------------------------
# Cache / moderation / keyword / resolve callbacks
# ---------------------------------------------------------------------------
def cache_lookup(query: str) -> Optional[dict]:
    return _query_cache.get(query.strip().lower())

def cache_store(query: str, results: list) -> None:
    if results:
        _query_cache[query.strip().lower()] = {"results": results, "next": None}

def link_fn(src: str, tgt: str) -> None:
    s, t = src.strip().lower(), tgt.strip().lower()
    if s in _query_cache:
        _query_cache[s]["next"] = t
    else:
        _query_cache[s] = {"results": None, "next": t}

def store_query_fn(query: str, data: dict) -> None:
    key = query.strip().lower()
    _query_cache.setdefault(key, {"results": None, "next": None}).update(data)

BLOCKED_RE = [re.compile(p) for p in [r"\b(hack|exploit|attack|bomb|weapon)\b"]]
def moderate_fn(query: str) -> bool:
    return not any(p.search(query.lower()) for p in BLOCKED_RE)

def preprocess_fn(query: str) -> str:
    return " ".join(query.split())

def keyword_fn(query: str) -> List[Tuple[dict, float]]:
    words = set(re.findall(r"\b[a-z]{3,}\b", query.lower()))
    scores: Dict[str, Tuple[dict, float]] = {}
    for word in words:
        for doc, weight in _keyword_index.get(word, []):
            did = doc.get("id", id(doc))
            if did in scores:
                _, prev = scores[did]
                scores[did] = (doc, min(prev + weight * 0.1, 0.5))
            else:
                scores[did] = (doc, weight * 0.1)
    return sorted(scores.values(), key=lambda x: x[1], reverse=True)[:20]

def resolve_fn(parent_id: str) -> Optional[dict]:
    return _parent_docs.get(parent_id)

def _llm_fn(query: str, doc_dict: dict) -> str:
    title = doc_dict.get("title", "Unknown")
    text = doc_dict.get("text", doc_dict.get("content", ""))
    return score_relevance(query, title, text)

def _rephrase_fn(query: str, previous: list) -> Optional[str]:
    return rephrase_query(query, previous)


# ---------------------------------------------------------------------------
# MLP reranker
# ---------------------------------------------------------------------------
def _maybe_train_mlp() -> None:
    global _predict_fn, _mlp_trained_at
    if _store.size < 50 or _store.size - _mlp_trained_at < 30:
        return
    try:
        from arf.trainer import train_reranker, load_reranker
        import numpy as np
    except ImportError:
        return

    logger.info(f"Training MLP reranker ({_store.size} docs)")
    extractor = FeatureExtractor(config=DOC_CONFIG)
    X_rows, y_labels = [], []
    seen: set = set()
    sample_queries: List[str] = []

    for doc in _store.documents:
        t = doc.get("title", "")
        if t and t not in seen:
            seen.add(t)
            sample_queries.append(t)
    for doc in _store.documents[::5]:
        first = doc.get("text", "").split(".")[0].strip()
        if first and len(first) > 20 and first not in seen:
            seen.add(first)
            sample_queries.append(first)

    for q in sample_queries:
        q_emb = embed(q)
        results = _store.search(q_emb, top_k=15)
        if not results:
            continue
        batch = extractor.extract_batch(query=q, results=results, query_embedding=q_emb)
        for feat, (_, score) in zip(batch, results):
            X_rows.append(extractor.to_vector(feat))
            y_labels.append(1 if score > 0.45 else 0)

    if len(X_rows) < 20 or len(set(y_labels)) < 2:
        return

    import numpy as np
    X = np.array(X_rows, dtype=np.float32)
    y = np.array(y_labels, dtype=np.int32)
    train_reranker(
        X, y, architecture=(64, 32, 16), max_iter=300, calibrate=True,
        feature_names=extractor.feature_names(), save_path="reranker_model.joblib",
    )
    _predict_fn = load_reranker("reranker_model.joblib", uncertainty_threshold=(0.4, 0.6))
    _mlp_trained_at = _store.size


# ---------------------------------------------------------------------------
# Pipeline factory
# ---------------------------------------------------------------------------
def _build_pipeline() -> Pipeline:
    def search_fn(query_embedding, top_k):
        return _store.search(query_embedding, top_k=top_k)

    triage = Triage(
        min_score=TRIAGE_MIN_SCORE, accept_threshold=TRIAGE_ACCEPT,
        verify_threshold=TRIAGE_VERIFY, gap=TRIAGE_GAP, top_k=TRIAGE_TOP_K,
    )
    return Pipeline(
        doc_config=DOC_CONFIG, triage=triage, search_fn=search_fn, embed_fn=embed,
        predict_fn=_predict_fn, llm_fn=_llm_fn,
        cache_lookup=cache_lookup, cache_store=cache_store,
        link_fn=link_fn, store_query_fn=store_query_fn,
        preprocess_fn=preprocess_fn, moderate_fn=moderate_fn,
        rephrase_fn=_rephrase_fn, keyword_fn=keyword_fn, resolve_fn=resolve_fn,
        graph_max_hops=GRAPH_MAX_HOPS, parser_range=(0.50, 1.50),
        predict_zones=(0.4, 0.6), max_rephrase=MAX_REPHRASE,
    )


def _extract_sources(results: List[Dict]) -> Tuple[List[Dict], List[str]]:
    sources, chunks = [], []
    for r in results:
        doc = r["document"]
        title = doc.title if hasattr(doc, "title") else doc.get("title", "")
        content = doc.content if hasattr(doc, "content") else doc.get("text", "")
        section = doc.get("section", "") if isinstance(doc, dict) else ""
        sources.append({
            "title": title, "section": section,
            "score": round(r["score"], 4), "text": content[:500],
        })
        chunks.append(f"[{title}]\n{content}")
    return sources, chunks


# ---------------------------------------------------------------------------
# Streaming ask — yields status dicts, final dict has "answer"
# ---------------------------------------------------------------------------
def ask_stream(query: str) -> Generator[Dict, None, None]:
    """Yields status updates, then a final result with answer + sources.

    Status:  {"status": "message text"}
    Result:  {"answer": "...", "sources": [...]}
    """
    _ensure_initialized()

    yield {"status": "Thinking..."}

    # Step 1: search existing store
    if _store.size > 0:
        yield {"status": f"Searching {_store.size} stored chunks..."}
        _maybe_train_mlp()
        pipeline = _build_pipeline()
        results = pipeline.run(query, top_k=5)
        if results:
            yield {"status": f"Found {len(results)} relevant results"}
            sources, context_chunks = _extract_sources(results)
            yield {"status": "Generating answer..."}
            answer = generate_answer(query, context_chunks)
            yield {"answer": answer, "sources": sources}
            return

    # Step 2: extract topics
    yield {"status": "Analyzing your question..."}
    topics = extract_search_topics(query)
    yield {"status": f"Search topics: {', '.join(topics)}"}

    # Step 3: filter already-fetched topics
    fetched = db.get_all_topics()
    new_topics = []
    for t in topics:
        key = t.strip().lower()
        if key in fetched or any(e in key or key in e for e in fetched):
            continue
        new_topics.append(t)

    if not new_topics:
        yield {"status": "All topics already in database, no new data found"}
        yield {
            "answer": "I couldn't find relevant information on Wikipedia for that question.",
            "sources": [],
        }
        return

    # Step 4: fetch from Wikipedia
    total_articles = 0
    total_chunks = 0
    for topic in new_topics:
        yield {"status": f"Searching Wikipedia for '{topic}'..."}
        articles = fetch_topic(topic, max_articles=4)
        if not articles:
            yield {"status": f"No articles found for '{topic}'"}
            db.mark_topic_fetched(topic.strip().lower())
            continue

        total_articles += len(articles)
        yield {"status": f"Found {len(articles)} articles on '{topic}'"}

        yield {"status": f"Chunking articles..."}
        all_chunks: List[dict] = []
        for art in articles:
            all_chunks.extend(chunk_article(art))
        total_chunks += len(all_chunks)

        yield {"status": f"Embedding {len(all_chunks)} chunks..."}
        ingested: List[dict] = []
        def store_fn(enriched_doc: dict) -> None:
            ingested.append(enriched_doc)
        ingest_documents(all_chunks, config=DOC_CONFIG, embed_fn=embed, store_fn=store_fn)

        # Persist to FAISS + JSON
        db.save_chunks(ingested, _store.index)
        db.mark_topic_fetched(topic.strip().lower())

        for doc in ingested:
            _store.add(doc)
            _add_keyword_entries(doc)

        for art in articles:
            _parent_docs[art["id"]] = art

    yield {"status": f"Indexed {total_chunks} chunks from {total_articles} articles"}

    # Step 5: run pipeline on new data
    yield {"status": "Running retrieval pipeline..."}
    _maybe_train_mlp()
    pipeline = _build_pipeline()
    results = pipeline.run(query, top_k=5)
    sources, context_chunks = _extract_sources(results)

    if context_chunks:
        yield {"status": "Generating answer..."}
        answer = generate_answer(query, context_chunks)
    else:
        answer = "I couldn't find relevant information on Wikipedia for that question."

    yield {"answer": answer, "sources": sources}
