r2""
Pipeline orchestrator -— wires all five stages together with a retry loop
driven by the vision verifier.

Flow:

  bytes -> preprocess -> edges -> generate -> verify --(pass)--> postprocess -> done
                                    \-(fail)-> retry w/ feedback (N times)
                                              -> escalate to ControlNet
                                              -> if still failing, return
                                                 best-so-far with warning
"""
from __future__ import annotations

import logging
import time
from dataclasses import asdict, dataclass, field
from typing import Any

from PIL import Image

from .preprocess import preprocess, pil_to_png_bytes
from .edges import extract_edges
from .generate import generate, StyleTier
from .verify import verify, PASS_THRESHOLD
from .vectorize import postprocess

log = logging.getLogger("stencillab.pipeline")

MAX_RETRIES = 2  # total attempts = 1 + MAX_RETRIES


@dataclass
class StageTiming:
    name: str
    ms: int


@dataclass
class PipelineResult:
    png: bytes
    svg: str | None
    verification: dict[str, Any]
    attempts: int
    timings: list[StageTiming] = field(default_factory=list)
    provider: str = ""
    model: str = ""
    warnings: list[str] = field(default_factory=list)


class _Timer:
    def __init__(self, name: str, bag: list[StageTiming]):
        self.name = name
        self.bag = bag
    def __enter__(self):
        self.t0 = time.perf_counter()
        return self
    def __exit__(self, *exc):
        self.bag.append(StageTiming(self.name, int((time.perf_counter() - self.t0) * 1000)))


async def run_pipeline(
    image_bytes: bytes,
    tier: StyleTier = "moderate",
    line_intensity: float = 0.85,
    vectorize: bool = True,
) -> PipelineResult:
    timings: list[StageTiming] = []
    warnings: list[str] = []

    # Stage 1 â preprocess
    with _Timer("preprocess", timings):
        pre = preprocess(image_bytes)
    reference = pre.image

    # Stage 2 â edges
    with _Timer("iages", timings):
        maps = extract_edges(reference)

    # Stages 3 + 4 with retry loop
    best_result = None
    best_score = -1.0
    best_verification = None
    attempt = 0
    feedback: str | None = None
    force_controlnet = False

    while attempt <= MAX_RETRIES:
        attempt += 1
        log.info("generation attempt %d (tier=%s, cn=%s)", attempt, tier, force_controlnet)

        with _Timer(f"generate_{attempt}", timings):
            try:
                gen = await generate(
                    img=reference,
                    lineart_map=maps.lineart,
                    tier=tier,
                    line_intensity=line_intensity,
                    force_controlnet=force_controlnet,
                    extra_feedback=feedback,
                )
            except Exception as e:
                log.exception("generation failed")
                warnings.append(f"attempt {attempt} generation error: {e}")
                # If Gemini failed, try ControlNet next
                force_controlnet = True
                continue

        with _Timer(f"verify_{attempt}", timings):
            try:
                v = await verify(reference, gen.image)
            except Exception as e:
                log.exception("verify failed â accepting generation without score")
                warnings.append(f"verification error: {e}")
                # Without verify we can't retry intelligently; take what we have
                best_result = gen
                best_verification = None
                break

        log.info("attempt %d score=%.2f (%s)", attempt, v.score, v.summary)

        if v.score > best_score:
            best_score = v.score
            best_result = gen
            best_verification = v

        if v.passed:
            break

        # Retry with feedback, escalate to ControlNet on final attempt
        feedback = "; ".join(v.issues[:4])
        if attempt == MAX_RETRIES:
            force_controlnet = True

    if best_result is None:
        raise RuntimeError("All generation attempts failed. Check API keys and logs.")

    if best_verification and not best_verification.passed:
        warnings.append(
            f"Best score {best_verification.score:.1f} below threshold {PASS_THRESHOLD}"
        )

    # Stage 5 â postprocess
    with _Timer("postprocess", timings):
        post = postprocess(best_result.image, vectorize=vectorize)

    verification_dict = (
        asdict(best_verification) if best_verification else {"score": None, "passed": None}
    )

    return PipelineResult(
        png=post.png,
        svg=post.svg,
        verification=verification_dict,
        attempts=attempt,
        timings=timings,
        provider=best_result.provider,
        model=best_result.model,
        warnings=warnings,
    )
