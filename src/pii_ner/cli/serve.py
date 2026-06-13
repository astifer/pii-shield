"""Run the browser-demo FastAPI server (mock LLM + static frontend)."""

from __future__ import annotations

import argparse


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Serve the PII Shield browser demo")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--reload", action="store_true", help="Auto-reload on code changes")
    return p.parse_args()


def main() -> None:
    import uvicorn

    args = parse_args()
    uvicorn.run("pii_ner.web.server:app", host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
