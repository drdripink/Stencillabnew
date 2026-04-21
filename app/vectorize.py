"""
Stage 5 — Post-processing and vectorization.

- Threshold to pure black/white (no grey)
- Despeckle (remove orphan pixels under N area)
- Optional: vectorize to SVG via potrace for lossless scaling.
"""
from __future__ import annotations

import io
import subprocess
import tempfile
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image


@dataclass
class PostprocessResult:
    png: bytes
    svg: str | None


def _to_binary(img: Image.Image) -> np.ndarray:
    gray = np.array(img.convert("L"))
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return bw


def _despeckle(bw: np.ndarray, min_area: int = 8) -> np.ndarray:
    inv = 255 - bw
    n, labels, stats, _ = cv2.connectedComponentsWithStats(inv, connectivity=8)
    keep = np.zeros_like(inv)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            keep[labels == i] = 255
    return 255 - keep


def _png_bytes(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def _to_svg_via_potrace(bw: np.ndarray) -> str | None:
    with tempfile.NamedTemporaryFile(suffix=".pbm", delete=False) as pbm_f:
        pbm_path = pbm_f.name
    with tempfile.NamedTemporaryFile(suffix=".svg", delete=False) as svg_f:
        svg_path = svg_f.name

    try:
        Image.fromarray(bw).convert("1").save(pbm_path)
        result = subprocess.run(
            [
                "potrace",
                pbm_path,
                "-s",
                "-o", svg_path,
                "--turdsize", "4",
                "--alphamax", "1.0",
                "--opttolerance", "0.2",
            ],
            capture_output=True,
            timeout=30,
        )
        if result.returncode != 0:
            return None
        with open(svg_path, "r", encoding="utf-8") as f:
            return f.read()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    finally:
        import os
        for p in (pbm_path, svg_path):
            try:
                os.unlink(p)
            except OSError:
                pass


def postprocess(stencil: Image.Image, vectorize: bool = False) -> PostprocessResult:
    bw = _to_binary(stencil)
    bw = _despeckle(bw)
    png = _png_bytes(bw)
    svg = _to_svg_via_potrace(bw) if vectorize else None
    return PostprocessResult(png=png, svg=svg)
