"""
Stage 1 — Preprocessing.

Deterministic image cleanup before anything AI touches it. Higher input
quality = dramatically better downstream results. No magic, just solid
image ops.
"""
from __future__ import annotations

import io
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image, ImageOps, ImageEnhance


TARGET_SIZE = 1024  # nano-banana & SDXL sweet spot


@dataclass
class PreprocessResult:
    image: Image.Image          # cleaned PIL image, RGB, max dim = TARGET_SIZE
    original_size: tuple[int, int]
    scale_factor: float         # how much we resized


def _auto_orient(img: Image.Image) -> Image.Image:
    """Respect EXIF rotation so phone photos aren't sideways."""
    return ImageOps.exif_transpose(img)


def _resize_longest(img: Image.Image, target: int) -> tuple[Image.Image, float]:
    w, h = img.size
    longest = max(w, h)
    if longest <= target:
        return img, 1.0
    scale = target / longest
    new_size = (int(w * scale), int(h * scale))
    return img.resize(new_size, Image.LANCZOS), scale


def _denoise(img: Image.Image) -> Image.Image:
    """Light bilateral denoise — preserves edges, kills compression artifacts."""
    arr = np.array(img)
    denoised = cv2.bilateralFilter(arr, d=7, sigmaColor=50, sigmaSpace=50)
    return Image.fromarray(denoised)


def _auto_contrast(img: Image.Image) -> Image.Image:
    """
    Stretch histogram so the subject has full tonal range.
    Critical for low-contrast phone photos.
    """
    arr = np.array(img)
    lab = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    rgb = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    return Image.fromarray(rgb)


def _sharpen(img: Image.Image, amount: float = 1.15) -> Image.Image:
    enhancer = ImageEnhance.Sharpness(img)
    return enhancer.enhance(amount)


def preprocess(image_bytes: bytes) -> PreprocessResult:
    """
    Main entry point. Takes raw upload bytes, returns a clean PIL image
    ready for edge extraction and model conditioning.
    """
    img = Image.open(io.BytesIO(image_bytes))
    img = _auto_orient(img).convert("RGB")
    original_size = img.size

    img, scale = _resize_longest(img, TARGET_SIZE)
    img = _denoise(img)
    img = _auto_contrast(img)
    img = _sharpen(img)

    return PreprocessResult(
        image=img,
        original_size=original_size,
        scale_factor=scale,
    )


def pil_to_png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return buf.getvalue()
