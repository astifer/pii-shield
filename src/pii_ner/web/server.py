"""FastAPI backend for the PII Shield browser demo.

Serves the static frontend (``web/``) and a ``/chat`` endpoint that stands in for a real
LLM. By the time a message reaches this server it has ALREADY been obfuscated in the
browser (each PII entity replaced by a ``[TAG]``), so no raw PII ever leaves the client.

Run:
    uv sync --extra web
    pii-serve            # or: uvicorn pii_ner.web.server:app --reload
    open http://127.0.0.1:8000/
"""

from __future__ import annotations

import os
from html import escape
from pathlib import Path

from fastapi import FastAPI, Form
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

# Static frontend lives at the repo root: src/pii_ner/web/server.py -> parents[3].
WEB_DIR = Path(os.environ.get("PII_WEB_DIR", Path(__file__).resolve().parents[3] / "web"))

app = FastAPI(title="PII Shield")

# Serve everything under web/ (app.js, obfuscate.js, styles.css, models/...) at /static.
app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")


@app.get("/", response_class=FileResponse)
def index() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")


@app.post("/chat", response_class=HTMLResponse)
def chat(message: str = Form(...)) -> str:
    """MOCK LLM: echo the (already obfuscated) message back.

    Returns an HTML fragment that htmx appends to the chat log. Replace the body of
    this function with a real LLM call -- the contract (obfuscated text in, HTML
    fragment out) stays the same.
    """
    reply = mock_llm(message)
    return (
        '<div class="msg assistant">'
        '<span class="who">LLM (mock)</span>'
        f'<div class="bubble">{escape(reply)}</div>'
        "</div>"
    )


def mock_llm(obfuscated_message: str) -> str:
    # MOCK LLM -- just echoes the obfuscated text. Swap in a real model here.
    return obfuscated_message
