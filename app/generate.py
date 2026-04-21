"""
Stage 3 — Multi-provider generation.

Tries Gemini 2.5 Flash (Image Edit mode) first, falls back to
ControlNet (Replicate SDXL) if Gemini fails or forced via flag.

This is the only spot in the pipeline where things get "creative".
Everything else is deterministic image processing.
"""
from __future__ import annotations

import base64
import json
import logging
import os
from dataclasses import dataclass
from typing import Literal

import google.generativeai as genai
import requests
from PIL import Image

log = logging.getLogger("stencillab.generate")

StyleTier = Literal["simple", "moderate", "expert"]


@dataclass
class GenerateResult:
    image: Image.Image
    provider: str
    model: str


# -----------------------------------------------------------------------------
# GEMINI 2.5 FLASH IMAGE EDIT MODE
# -----------------------------------------------------------------------------

GEMINI_PROMPT_TEMPLATE = """
Transform this image into a clean line drawing/tattoo stencil.

Specific requirements:
1. Preserve exact composition, proportions, pose, and framing
2. Keep key features (face, expression, body parts)
3. Produce ONLY visible elements in the input
4. NO added helmets, accessories, or decorations
5. Black outlines on white background for tattoo ink compatibility

6. Line intensity: %s
7. Style level: %s

%s
Do NOT obscure, hide, or change essential features.
"""

STYLE_DEFINITIONS = {
    "simple": "Minimal lines — clean, stripped, bold outlines, not much detail. Focus on basic shape and contour.",
    "moderate": "Balanced detail — clear lines with some internal structure. Neither too basic nor too complex. Suitable for most tattoos.",
    "expert": "Rich detail — intricate textures, hatching/cross-hatching, fine edges, multi-layered complexity. Hard challenge for expert tattoo artists.",
}

LINE_INTENSITY_DESC = {
    0.1: "extremely light, vague lines",
    0.2: "light lines",
    0.4: "soft, thin edges",
    0.6: "standard line weight",
    0.8: "default engraving depth",
    1.0: "bold lines",
    1.2: "deep cuts",
    1.5: "very dark heavy edges",
}


def _build_gemini_prompt(
    tier: StyleTier,
    line_intensity: float,
    extra_feedback: str | None = None,
) -> str:
    line_desc = LINE_INTENSITY_DESC.get(line_intensity, str(line_intensity))
    style_desc = STYLE_DEFINITIONS[tier]
    feedback = f"\nAdditional instruction: {extra_feedback}" if extra_feedback else ""
    return GEMINI_PROMPT_TEMPLATE % (line_desc, style_desc, feedback)


def _pil_to_b64(img: Image.Image) -> str:
    import io
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


async def _try_gemini_generate(
    img: Image.Image,
    tier: StyleTier,
    line_intensity: float,
    extra_feedback: str | None = None,
) -> Image.Image | None:
    if not os.environ.get("GEMINI_API_KEY"):
        log.warn("Gemini API key not found — skipping")
        return None

    try:
        genai.configure(api_key=os.environ["GEMINI_API_KEY"])

        model = genai.GenerativeModel(model_name="gemini-2.5-flash")
        b64_data = _pil_to_b64(img)
        prompt = _build_gemini_prompt(tier, line_intensity, extra_feedback)

        # Generate image
        response = model.generate_content([
            genai.Part.from_text(prompt),
            genai.Part.from_data(f"data:image/png;base64,{b64_data}"),
        ])

        if response.parts:
            img_data = response.parts[0].inline_data.data
            import io
            return Image.open(io.BytesIO(img_data))
        return None

    except Exception as e:
        log.exception(f"Gemini generation failed: {e}")
        return None


# -----------------------------------------------------------------------------
# REPLICATE CONTROLNET SDXL 1.1
# -----------------------------------------------------------------------------

REPLICATE_MODEL = "sdxl-controlnet-1.1"
REPLICATE_OWNER = "lucataco"

# ControlNet is just diffusion + adaptive — it runs the base model but adds additional input channels
# (edge-maps, canny, depth, etc) that let the model know about structure.
# Here we're using it for lineart.
REPLICATE_CONTROLTYPE = "lineart"

REPLICATE_PROMPT_TEMPLATE = """
Clean line drawing, tattoo stencil, black outlines, white background, %s, %s

Faithful to composition and framing, clear edges, no added elements, no decorations

%s
"""


def _build_replicate_prompt(
    tier: StyleTier,
    line_intensity: float,
    extra_feedback: str | None = None,
) -> str:
    line_desc = LINE_INTENSITY_DESC.get(line_intensity, str(line_intensity))
    style_desc = STYLE_DEFINITIONS[tier]
    feedback = f"\nAdditional guidance: {extra_feedback}" if extra_feedback else ""
    return REPLICATE_PROMPT_TEMPLATE % (line_desc, style_desc, feedback)


async def _try_replicate_generate(
    img: Image.Image,
    lineart_map: Image.Image,
    tier: StyleTier,
    line_intensity: float,
    extra_feedback: str | None = None,
) -> Image.Image | None:
    if not os.environ.get("REPLICATE_API_TOKEN"):
        log.warn("Replicate API token not found — skipping")
        return None

    try:
        token = os.environ["REPLICATE_API_TOKEN"]
        url = f"https://api.replicate.com/v1/models/{REPLICATE_OWNER}/{REPLICATE_MODEL}/predictions"

        # Convert images to base64
        img_b64 = _pil_to_b64(img)
        lineart_b64 = _pil_to_b64(lineart_map)
        prompt = _build_replicate_prompt(tier, line_intensity, extra_feedback)

        payload = {
            "input": {
                "image": f"data:image/png;base64,{img_b64}",
                "control_image": f"data:image/png;base64,{lineart_b64}",
                "control_type": REPLICATE_CONTROLTYPE,
                "prompt": prompt,
                "num_inference_steps": 30,
                "controlnet_conditioning_scale": 0.9,
                "guidance_scale": 7.5,
                "seed": None,
            }
        }

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}",
        }

        response = requests.post(url, json=payload, headers=headers, timeout=120)
        response.raise_for_status()
        data = response.json()

        if data.get("output") and isinstance(data["output"], list) and data["output"]:
            img_url = data["output"][0]
            img_resp = requests.get(img_url, timeout=30)
            img_resp.raise_for_status()
            import io
            return Image.open(io.BytesIO(img_resp.content))

        return None

    except Exception as e:
        log.exception(f"Replicate generation failed: {e}")
        return None


# -----------------------------------------------------------------------------
# MAIN FUNCTION
# -----------------------------------------------------------------------------

async def generate(
    img: Image.Image,
    lineart_map: Image.Image,
    tier: StyleTier = "moderate",
    line_intensity: float = 0.85,
    force_controlnet: bool = False,
    extra_feedback: str | None = None,
) -> GenerateResult:
    """
    Main generation entry point.

    Tries Gemini first (unless force_controlnet), then falls back to Replicate.
    Returns GenerateResult if either provider succeeds, RuntimeError if both fail.
    """
    if not force_controlnet:
        log.info("trying Gemini 2.5 Flash Image Edit...")
        gem_image = await _try_gemini_generate(img, tier, line_intensity, extra_feedback)
        if gem_image:
            log.info("Gemini succeeded")
            return GenerateResult(
                image=gem_image,
                provider="gemini",
                model="gemini-2.5-flash",
            )

        log.info("Gemini failed or unavailable — falling back to ControlNet")

    log.info("Trying Replicate SDXL ControlNet...")
    replica_image = await _try_replicate_generate(
        img, lineart_map, tier, line_intensity, extra_feedback,
    )
    if replica_image:
        log.info("Replicate succeeded")
        return GenerateResult(
            image=replica_image,
            provider="replicate",
            model=f"{REPLICATE_OWNER}/{REPLICATE_MODEL}",
        )

    raise RuntimeError("Both Gemini and Replicate failed for generation")

# -----------------------------------------------------------------------------
# EARLY PROTOTYPE -- for testing either provider only
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    import asyncio
    import sys

    async def main():
        if len(sys.argv) < 2:
            print("Usage: python generate.py <image_path lineart_path>")
            return

        input_path = sys.argv[1]
        lineart_path = sys.argv[2] if len(sys.argv) > 2 else None

        img = Image.open(input_path)
        lineart = Image.open(lineart_path) if lineart_path else img

        try:
            result = await generate(img, lineart)
            print(f"Success! Provider: {result.provider}, Model: {result.model}")
            result.image.show()
        except Exception as e:
            print(f"Error: {e}")

    asyncio.run(main())
