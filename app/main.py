"""
StencilLab FastAPI application.

One endpoint does the real work: POST /api/stencil
Everything else is static file serving for the existing HTML frontend.
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

from .pipeline import run_pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

app = FastAPI(title="StencilLab", version="1.0.0")

# Permissive CORS so the Chrome extension & any future mobile app can hit it.
# Tighten this in prod to your actual origins.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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


@app.post("/api/stencil")
async def generate_stencil(
    image: UploadFile = File(...),
    tier: Literal["simple", "moderate", "expert"] = Form("moderate"),
    line_intensity: float = Form(0.85),
    vectorize: bool = Form(True),
):
    """
    Main pipeline endpoint.

    Returns JSON:
      {
        "png": "<base64>",          # always present
        "svg": "<svg string>" | null,
        "verification": {...},
        "attempts": <int>,
        "provider": "gemini" | "replicate",
        "model": "...",
        "timings": [{"name": ..., "ms": ...}, ...],
        "warnings": [...]
      }
    """
    raw = await image.read()
    if not raw:
        raise HTTPException(400, "empty upload")
    if len(raw) > 20 * 1024 * 1024:
        raise HTTPException(413, "image too large (max 20MB)")
    if not (0.1 <= line_intensity <= 1.5):
        raise HTTPException(400, "line_intensity must be between 0.1 and 1.5")

    try:
        result = await run_pipeline(
            image_bytes=raw,
            tier=tier,
            line_intensity=line_intensity,
            vectorize=vectorize,
        )
    except KeyError as e:
        raise HTTPException(500, f"missing required env var: {e}")
    except Exception as e:
        logging.exception("pipeline error")
        raise HTTPException(500, f"pipeline failed: {e}")

    return JSONResponse({
        "png": base64.b64encode(result.png).decode("ascii"),
        "svg": result.svg,
        "verification": result.verification,
        "attempts": result.attempts,
        "provider": result.provider,
        "model": result.model,
        "timings": [{"name": t.name, "ms": t.ms} for t in result.timings],
        "warnings": result.warnings,
    })


# --- Replicate proxy routes (legacy frontend compat) ---------------------
import os as _os_proxy
import httpx as _httpx_proxy
from fastapi import Request as _Request_proxy
from fastapi.responses import Response as _Response_proxy

_REPLICATE_BASE = "https://api.replicate.com/v1"


def _resolve_auth(request):
    incoming = request.headers.get("authorization", "")
    if incoming.lower().startswith("token ") or incoming.lower().startswith("bearer "):
        return incoming
    server_token = _os_proxy.environ.get("REPLICATE_API_TOKEN")
    if server_token:
        return f"Token {server_token}"
    return None


@app.post("/api/replicate/predictions")
async def replicate_create_prediction(request: _Request_proxy):
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


