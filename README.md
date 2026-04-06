# Wiki RAG

A real-time Retrieval-Augmented Generation system that answers questions using Wikipedia as its knowledge source. Unlike traditional RAG setups with a static corpus, Wiki RAG fetches fresh Wikipedia articles on every query, chunks them with section awareness, and retrieves the most relevant passages through a multi-stage ranking pipeline.

Runs fully local — no API keys required. Built with [advanced-rag-framework](https://pypi.org/project/advanced-rag-framework/), LangChain, and Ollama.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![License](https://img.shields.io/badge/License-MIT-green)

## How It Works

```
User Question
     |
     v
 Topic Extraction (Ollama) ----- "Who built the ISS?" -> ["International Space Station", "NASA"]
     |
     v
 Wikipedia Fetch --------------- MediaWiki API -> plain-text articles
     |
     v
 Section-Aware Chunking -------- Splits on == headings ==, preserves structure
     |
     v
 Embedding (sentence-transformers) --- all-MiniLM-L6-v2, 384-dim vectors
     |
     v
 FAISS Vector Store ------------ Persistent index, search-first retrieval
     |
     v
 Multi-Stage Retrieval --------- Triage -> MLP Rerank -> LLM Verification
     |
     v
 Answer Generation (Ollama) ---- Grounded response with source citations
```

### Pipeline Stages (advanced-rag-framework)

| Stage | What it does |
|---|---|
| **Cache graph walk** | Instant return if the query (or a rephrase) was seen before |
| **Vector search** | Cosine similarity over FAISS index |
| **Triage** | Accept / verify / reject routing based on score thresholds and gaps |
| **MLP reranker** | Lightweight neural reranker trained incrementally on accumulated data |
| **LLM verification** | Ollama scores uncertain candidates on a 0-9 relevance scale |
| **Keyword fallback** | TF-weighted keyword matching as a safety net |
| **Query rephrase** | Up to 2 automatic rephrases when initial retrieval is weak |

## Project Structure

```
wiki-rag/
├── app.py                      # FastAPI entry point
├── backend/
│   ├── config.py               # Model settings and parameters
│   ├── embeddings.py           # sentence-transformers (all-MiniLM-L6-v2)
│   ├── llm.py                  # LangChain + Ollama
│   ├── ingestion.py            # Wikipedia fetch + section-aware chunking
│   ├── vector_store.py         # FAISS-backed vector search
│   ├── db.py                   # FAISS + JSON persistence
│   └── rag_pipeline.py         # advanced-rag-framework pipeline wiring
├── static/
│   ├── index.html              # Chat interface
│   ├── style.css               # Responsive styles
│   └── app.js                  # SSE streaming frontend
├── requirements.txt
└── .env.example
```

## Getting Started

### Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com/) installed and running

### Installation

```bash
git clone https://github.com/jager47X/wiki-rag.git
cd wiki-rag

python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install -r requirements.txt

# Pull an Ollama model
ollama pull llama3.2
```

### Configuration (optional)

```bash
cp .env.example .env
```

```env
OLLAMA_MODEL=llama3.2
OLLAMA_BASE_URL=http://localhost:11434
```

### Run

```bash
uvicorn app:app --reload
```

Open [http://localhost:8000](http://localhost:8000) to use the chat interface.

## Tech Stack

| Component | Technology |
|---|---|
| RAG orchestration | [advanced-rag-framework](https://pypi.org/project/advanced-rag-framework/) 0.2.2 |
| LLM | Ollama (llama3.2) via LangChain |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2, 384-dim) |
| Vector store | FAISS (persistent) |
| Knowledge source | Wikipedia MediaWiki API (real-time) |
| Reranker | scikit-learn MLP (trained incrementally) |
| Backend | FastAPI + Uvicorn |
| Frontend | Vanilla HTML / CSS / JS |

## License

[MIT](LICENSE)
