"""Local LLM via LangChain + Ollama (no API key needed)."""

import re
from typing import List, Optional

from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage

from backend.config import OLLAMA_MODEL, OLLAMA_BASE_URL

_llm = ChatOllama(
    model=OLLAMA_MODEL,
    base_url=OLLAMA_BASE_URL,
    temperature=0.3,
)


def chat(system: str, user: str, max_tokens: int = 512) -> str:
    """Send a system + user message pair and return the assistant's reply."""
    resp = _llm.invoke(
        [SystemMessage(content=system), HumanMessage(content=user)],
        max_tokens=max_tokens,
    )
    return resp.content.strip()


def generate_answer(query: str, context_chunks: List[str]) -> str:
    """Generate a grounded answer from retrieved context chunks."""
    context = "\n\n---\n\n".join(context_chunks)
    return chat(
        system=(
            "You are a helpful assistant. Answer the user's question using the "
            "provided context. Synthesize information from multiple sources when "
            "relevant. If the context is only partially relevant, use what is "
            "available and note any gaps. Be concise and cite which source the "
            "information comes from using [Source Title]."
        ),
        user=f"Context:\n{context}\n\nQuestion: {query}",
    )


def extract_search_topics(query: str) -> List[str]:
    """Use the LLM to pull 2-3 Wikipedia search terms from a question."""
    raw = chat(
        system=(
            "Extract 2-3 Wikipedia search topics from the user's question. "
            "Include both specific and broader related topics that would have "
            "substantial Wikipedia articles. Return ONLY the topics, one per line. "
            "No numbering, no explanation."
        ),
        user=query,
        max_tokens=60,
    )
    topics = [
        re.sub(r"^[\d.\-•*)\s]+", "", t).strip()
        for t in raw.strip().split("\n") if t.strip()
    ]
    return [t for t in topics if len(t) > 2][:3] or [query]


def score_relevance(query: str, title: str, text: str) -> str:
    """Return raw LLM output scoring document relevance (0-9 JSON)."""
    return chat(
        system=(
            "You are a relevance judge. Score how relevant the document is to "
            'the query on a scale of 0-9. Respond with ONLY a JSON object like '
            '{"score": 7}. Nothing else.'
        ),
        user=(
            f"Query: {query}\n\nDocument title: {title}\n"
            f"Document excerpt: {text[:500]}\n\nScore:"
        ),
        max_tokens=30,
    )


def rephrase_query(query: str, previous_attempts: list) -> Optional[str]:
    """Rephrase a search query to improve retrieval results."""
    prev = ", ".join(f'"{p}"' for p in previous_attempts) if previous_attempts else "none"
    rephrased = chat(
        system=(
            "Rephrase the search query to find better results. "
            "Return ONLY the rephrased query, nothing else."
        ),
        user=(
            f"Original query: {query}\n"
            f"Previous attempts: {prev}\nRephrased query:"
        ),
        max_tokens=60,
    ).strip("\"'")
    if rephrased and rephrased.lower() != query.lower():
        return rephrased
    return None
