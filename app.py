"""FastAPI server — serves the chat UI with SSE streaming status updates."""

import json
import logging
import traceback
import os

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from starlette.responses import StreamingResponse
from pydantic import BaseModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("wiki-rag")

app = FastAPI(title="Wiki RAG", version="1.0.0")

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")


class ChatRequest(BaseModel):
    message: str


def _sse_generator(query: str):
    """Yield Server-Sent Events from the RAG pipeline."""
    from backend.rag_pipeline import ask_stream
    try:
        for event in ask_stream(query):
            if "status" in event:
                yield f"event: status\ndata: {json.dumps(event)}\n\n"
            else:
                yield f"event: done\ndata: {json.dumps(event)}\n\n"
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        traceback.print_exc()
        error = {"answer": f"Error: {e}", "sources": []}
        yield f"event: done\ndata: {json.dumps(error)}\n\n"


@app.get("/")
async def index():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


@app.post("/api/chat")
async def chat(req: ChatRequest):
    """Stream RAG pipeline progress via Server-Sent Events."""
    logger.info(f"Query: {req.message}")
    return StreamingResponse(
        _sse_generator(req.message),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/api/health")
async def health():
    return {"status": "ok"}


# Mount static AFTER explicit routes so "/" isn't shadowed
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
