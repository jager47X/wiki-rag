"""Centralized configuration — model settings and pipeline parameters."""

import os
from dotenv import load_dotenv

load_dotenv()

# --- Local Models ---
EMBED_MODEL = "all-MiniLM-L6-v2"
EMBED_DIM = 384
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2")
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")

# --- Chunking ---
CHUNK_MAX_CHARS = 1200
CHUNK_OVERLAP = 150

# --- Storage ---
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")

# --- RAG pipeline ---
TRIAGE_MIN_SCORE = 0.05
TRIAGE_ACCEPT = 0.35
TRIAGE_VERIFY = 0.15
TRIAGE_GAP = 0.30
TRIAGE_TOP_K = 15
MAX_REPHRASE = 2
GRAPH_MAX_HOPS = 3
