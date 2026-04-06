"""Wikipedia fetching with section-aware chunking for structured RAG ingestion."""

import re
import requests
from typing import List, Dict

from backend.config import CHUNK_MAX_CHARS, CHUNK_OVERLAP

API_URL = "https://en.wikipedia.org/w/api.php"
HEADERS = {"User-Agent": "WikiRAGBot/1.0 (sample-project; educational)"}


# ---------------------------------------------------------------------------
# Wikipedia API helpers
# ---------------------------------------------------------------------------

def search_titles(topic: str, limit: int = 10) -> List[str]:
    """Search Wikipedia for page titles matching a topic."""
    resp = requests.get(API_URL, headers=HEADERS, params={
        "action": "query", "list": "search",
        "srsearch": topic, "srlimit": limit, "format": "json",
    }, timeout=15)
    resp.raise_for_status()
    return [hit["title"] for hit in resp.json()["query"]["search"]]


def fetch_article(title: str) -> Dict[str, str]:
    """Fetch the plain-text extract of a single Wikipedia article."""
    resp = requests.get(API_URL, headers=HEADERS, params={
        "action": "query", "titles": title,
        "prop": "extracts", "explaintext": True,
        "exlimit": 1, "format": "json",
    }, timeout=15)
    resp.raise_for_status()
    page = next(iter(resp.json()["query"]["pages"].values()))
    return {
        "id": str(page.get("pageid", title)),
        "title": page.get("title", title),
        "text": page.get("extract", ""),
    }


def fetch_topic(topic: str, max_articles: int = 4) -> List[Dict[str, str]]:
    """Search + fetch multiple articles for a topic."""
    titles = search_titles(topic, limit=max_articles)
    articles: List[Dict[str, str]] = []
    for t in titles:
        try:
            art = fetch_article(t)
            if art["text"] and len(art["text"]) > 200:
                articles.append(art)
        except Exception:
            continue
    return articles


# ---------------------------------------------------------------------------
# Section-aware chunking
# ---------------------------------------------------------------------------

_SECTION_RE = re.compile(r"^(={2,})\s*(.+?)\s*\1$", re.MULTILINE)


def _split_sections(text: str) -> List[Dict[str, str]]:
    """Split Wikipedia plain-text into sections with titles.

    Returns a list of {"heading": str, "level": int, "body": str} dicts.
    The first entry (level=0) is the article lead.
    """
    matches = list(_SECTION_RE.finditer(text))
    if not matches:
        return [{"heading": "", "level": 0, "body": text.strip()}]

    sections: List[Dict] = []
    # Lead section (before first heading)
    lead = text[: matches[0].start()].strip()
    if lead:
        sections.append({"heading": "Introduction", "level": 0, "body": lead})

    for i, m in enumerate(matches):
        level = len(m.group(1))
        heading = m.group(2).strip()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        if body:
            sections.append({"heading": heading, "level": level, "body": body})
    return sections


def chunk_article(
    article: Dict[str, str],
    max_chars: int = CHUNK_MAX_CHARS,
    overlap: int = CHUNK_OVERLAP,
) -> List[Dict[str, str]]:
    """Split an article into overlapping chunks that respect section boundaries.

    Each chunk carries metadata: id, title, section, text, parent_id.
    """
    sections = _split_sections(article["text"])
    chunks: List[Dict[str, str]] = []
    idx = 0

    for sec in sections:
        body = sec["body"]
        heading = sec["heading"]

        # Prefix each chunk with the section heading for context
        prefix = f"[{heading}] " if heading else ""

        start = 0
        while start < len(body):
            end = start + max_chars
            segment = body[start:end]

            # Try to break at a paragraph boundary
            if end < len(body):
                last_para = segment.rfind("\n\n")
                if last_para > max_chars // 3:
                    segment = segment[:last_para]
                    end = start + last_para

            chunk_text = (prefix + segment).strip()
            if len(chunk_text) > 30:  # skip tiny fragments
                chunks.append({
                    "id": f"{article['id']}_chunk_{idx}",
                    "title": article["title"],
                    "section": heading,
                    "text": chunk_text,
                    "parent_id": article["id"],
                })
                idx += 1

            start = end - overlap

    return chunks
