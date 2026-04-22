"""
StencilLab FastAPI application.

Serves the SPA frontend and proxies Replicate API requests.
"""
from __future__ import annotations

import base64
import logging
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

app = FastAPI(title="StencilLab", version="1.0.0")

# Permissive CORS so the Chrome extension & any future mobile app can hit it.
# Tighten this in prod to your actual origins.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://stencillab-ns2ljw.fly.dev",
        "https://stencillab.fly.dev",
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
async def health():
    import os
    return {
        "ok": True,
        "providers": {
            "gemini": bool(os.environ.get("GEMINI_API_KEY")),
            "replicate": bool(os.environ.get("REPLICATE_API_TOKEN")),
            "anthropic": bool(os.environ.get("ANTHROPIC_API_KEY")),
        },
    }


# --- Replicate proxy routes (legacy frontend compat) ---------------------
import os as _os_proxy
import httpx as _httpx_proxy
from fastapi import Request as _Request_proxy
from fastapi.responses import Response as _Response_proxy

_REPLICATE_BASE = "https://api.replicate.com/v1"

import time
from collections import defaultdict, deque

_ALLOWED_ORIGINS = {
    "https://stencillab-ns2ljw.fly.dev",
    "https://stencillab.fly.dev",
}
_RATE_WINDOW_SECS = 3600
_RATE_MAX_POST = 60
_RATE_MAX_GET = 1200
_request_log = defaultdict(lambda: deque(maxlen=2000))


def _check_origin_and_rate(request, is_post: bool):
    origin = request.headers.get("origin", "")
    referer = request.headers.get("referer", "")
    origin_ok = (
        origin in _ALLOWED_ORIGINS
        or any(referer.startswith(o) for o in _ALLOWED_ORIGINS)
    )
    if not origin_ok:
        return _Response_proxy(
            content='{"detail":"Origin not allowed"}',
            status_code=403,
            media_type="application/json",
        )
    ip = request.headers.get("fly-client-ip") or (request.client.host if request.client else "unknown")
    now = time.time()
    log = _request_log[ip]
    while log and log[0] < now - _RATE_WINDOW_SECS:
        log.popleft()
    limit = _RATE_MAX_POST if is_post else _RATE_MAX_GET
    if len(log) >= limit:
        return _Response_proxy(
            content='{"detail":"Rate limit exceeded"}',
            status_code=429,
            media_type="application/json",
        )
    log.append(now)
    return None


def _resolve_auth(request):
    # Always prefer server-side Fly secret — ignore stale/invalid client tokens.
    server_token = _os_proxy.environ.get("REPLICATE_API_TOKEN")
    if server_token:
        return f"Token {server_token}"
    incoming = request.headers.get("authorization", "")
    if incoming.lower().startswith("token ") or incoming.lower().startswith("bearer "):
        return incoming
    return None


@app.post("/api/replicate/predictions")
async def replicate_create_prediction(request: _Request_proxy):
    rejected = _check_origin_and_rate(request, is_post=True)
    if rejected:
        return rejected
    auth = _resolve_auth(request)
    if not auth:
        return _Response_proxy(
            content='{"detail":"No Replicate token available"}',
            status_code=401,
            media_type="application/json",
        )
    body = await request.body()
    async with _httpx_proxy.AsyncClient(timeout=60.0) as client:
        resp = await client.post(
            f"{_REPLICATE_BASE}/predictions",
            content=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": auth,
            },
        )
    return _Response_proxy(
        content=resp.content,
        status_code=resp.status_code,
        media_type=resp.headers.get("content-type", "application/json"),
    )


@app.get("/api/replicate/predictions/{prediction_id}")
async def replicate_get_prediction(prediction_id: str, request: _Request_proxy):
    rejected = _check_origin_and_rate(request, is_post=False)
    if rejected:
        return rejected
    auth = _resolve_auth(request)
    if not auth:
        return _Response_proxy(
            content='{"detail":"No Replicate token available"}',
            status_code=401,
            media_type="application/json",
        )
    async with _httpx_proxy.AsyncClient(timeout=30.0) as client:
        resp = await client.get(
            f"{_REPLICATE_BASE}/predictions/{prediction_id}",
            headers={"Authorization": auth},
        )
    return _Response_proxy(
        content=resp.content,
        status_code=resp.status_code,
        media_type=resp.headers.get("content-type", "application/json"),
    )


# Static frontend (mounted last so /api/* takes priority)
STATIC_DIR = Path(__file__).parent.parent / "static"
if STATIC_DIR.exists():
    app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")


