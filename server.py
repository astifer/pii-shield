# -*- coding: utf-8 -*-
"""
PII Shield web server (stdlib only — no FastAPI/Flask needed).

Serves the single-page UI at "/" and a JSON detection endpoint at
"POST /api/detect". The trained model is loaded once on first request.

Run:
    python server.py            # http://127.0.0.1:8000
    python server.py --port 9000 --host 0.0.0.0
"""
from __future__ import annotations

import argparse
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import infer

HERE = os.path.dirname(os.path.abspath(__file__))
INDEX_PATH = os.path.join(HERE, "static", "index.html")
MAX_BODY = 1_000_000  # 1 MB cap on request bodies


class Handler(BaseHTTPRequestHandler):
    server_version = "PIIShield/1.0"

    # --- helpers ---------------------------------------------------------- #
    def _send(self, code: int, body: bytes, content_type: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _json(self, code: int, obj: dict) -> None:
        self._send(code, json.dumps(obj, ensure_ascii=False).encode("utf-8"),
                   "application/json; charset=utf-8")

    # --- routes ----------------------------------------------------------- #
    def do_GET(self) -> None:
        path = self.path.split("?", 1)[0]
        if path == "/" or path == "/index.html":
            try:
                with open(INDEX_PATH, "rb") as f:
                    self._send(200, f.read(), "text/html; charset=utf-8")
            except FileNotFoundError:
                self._send(500, b"index.html not found", "text/plain; charset=utf-8")
        elif path == "/api/types":
            self._json(200, {"types": infer.entity_types()})
        elif path == "/healthz":
            self._json(200, {"ok": True})
        else:
            self._send(404, b"Not found", "text/plain; charset=utf-8")

    def do_HEAD(self) -> None:
        self.do_GET()

    def do_POST(self) -> None:
        if self.path.split("?", 1)[0] != "/api/detect":
            self._send(404, b"Not found", "text/plain; charset=utf-8")
            return
        try:
            length = int(self.headers.get("Content-Length", 0))
        except ValueError:
            length = 0
        if length <= 0 or length > MAX_BODY:
            self._json(400, {"error": "Invalid or too large body"})
            return
        try:
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            text = str(payload.get("text", ""))
            threshold = float(payload.get("threshold", 0.5))
            threshold = min(max(threshold, 0.0), 0.999)
        except (ValueError, json.JSONDecodeError):
            self._json(400, {"error": "Invalid JSON payload"})
            return
        try:
            result = infer.detect(text, threshold=threshold)
        except Exception as exc:  # surface model errors to the client
            self._json(500, {"error": f"Inference failed: {exc}"})
            return
        self._json(200, result)

    # quieter logging
    def log_message(self, fmt: str, *args) -> None:
        print("[server] " + (fmt % args))


def main() -> None:
    ap = argparse.ArgumentParser(description="PII Shield server")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--preload", action="store_true",
                    help="Load the model at startup instead of on first request")
    args = ap.parse_args()

    if args.preload:
        print("Loading model ...")
        infer.load_model()
        print("Model ready.")

    httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"PII Shield running at http://{args.host}:{args.port}  (Ctrl+C to stop)")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
        httpd.shutdown()


if __name__ == "__main__":
    main()
