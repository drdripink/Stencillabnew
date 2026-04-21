"""
Stage 2 — Structure extraction.

Produces an edge/structure map that locks the composition before the AI
model gets involved. This is what stops "lion" from becoming "lion in a
Spartan helmet" — the model is forced to respect real geometry.
"""
from __future__ import annotations

import io
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image


@dataclass
class EdgeMaps:
    canny: Image.Image     # grayscale, white edges on black
    lineart: Image.Image   # grayscale, black lines on white (model-friendly)


def _to_gray(img: Image.Image) -> np.ndarray:
    arr = np.array(img.convert("RGB"))
    return cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)


def _canny_map(gray: np.ndarray) -> np.ndarray:
    """Adaptive Canny with thresholds derived from the median."""
    blurred = cv2.GaussianBlur(gray, (5, 5), 1.4)
    median = float(np.median(blurred))
    lower = int(max(0, 0.66 * median))
    upper = int(min(255, 1.33 * median))
    edges = cv2.Canny(blurred, lower, upper, L2gradient=True)
    kernel = np.ones((2, 2), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)
    return edges


def _lineart_map(gray: np.ndarray) -> np.ndarray:
    """
    Difference of Gaussians -> thresholded lineart.
    Approximates HED/lineart ControlNet input. Black lines on white.
    """
    blur1 = cv2.GaussianBlur(gray, (0, 0), sigmaX=1.0)
    blur2 = cv2.GaussianBlur(gray, (0, 0), sigmaX=2.5)
    dog = blur1.astype(np.int16) - blur2.astype(np.int16)

    dog = np.clip(dog, 0, 255).astype(np.uint8)
    _, binary = cv2.threshold(dog, 8, 255, cv2.THRESH_BINARY)

    kernel = np.ones((2, 2), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    inverted = 255 - binary
    return inverted


def extract_edges(img: Image.Image) -> EdgeMaps:
    gray = _to_gray(img)
    canny = _canny_map(gray)
    lineart = _lineart_map(gray)
    return EdgeMaps(
        canny=Image.fromarray(canny),
        lineart=Image.fromarray(lineart),
    )


def pil_to_png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return buf.getvalue()
