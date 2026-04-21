"""
[]age 4 â Vision verification.

Sends the original photo AND the generated stencil to Claude Sonnet 4.6
and asks: does this stencil accurately represent the reference?
"""
from __future__ import annotations

import base64
import io
import json
import os
import re
from dataclasses import dataclass

from PIL import Image
from anthropic import AsyncAnthropic


VERIFY_MODEL = os.environ.get("VERIFY_MODEL", "claude-sonnet-4-6")
PASS_THRESHOLD = float(os.environ.get("VERIFY_THRESHOLD", "7.5"))


@dataclass
class VerificationResult:
    score: float
    passed: bool
    composition_score: float
    subject_score: float
    detail_score: float
    issues: list[str]
    summary: str


def _img_to_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


_SYSTEM_PROMPT = """You are a quality auditor for tattoo stencil generation. You will be shown two images:
1. A REFERENCE photograph (the source the user uploaded)
2. A STENCIL output (what the AI pipeline produced)

Your job is to judge whether the stencil faithfully represents the reference.
A good stencil preserves composition, proportions, subject identity, pose,
and key structural features. It should look like a line drawing OF the reference,
not a new creation loosely inspired by it.

Common failures to flag:
- Added elements not in the reference (helmets, accessories, extra objects)
- Removed features (missing eye, wrong number of limbs, etc.)
- Wrong pose, crop, or framing
- Wrong subject identity (different animal, different person's face shape)
- Stylistic drift (too much artistic license)

Score on three axes, 0-10 each:
- composition: framing, crop, pose, proportions
- subject: identity, key features preserved
- detail: level of detail appropriate and accurate

Then give an OVERALL score 0-10.

Respond ONLY with a JSON object, no other text:
{
  "composition": <number>,
  "subject": <number>,
  "detail": <number>,
  "overall": <number>,
  "issues": ["<specific issue 1>", "<specific issue 2>", ...],
  "summary": "<one sentence>"
}
"""


async def verify(
    reference: Image.Image,
    stencil: Image.Image,
) -> VerificationResult:
    client = AsyncAnthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    ref_b64 = _img_to_b64(reference)
    sten_b64 = _img_to_b64(stencil)

    message = await client.messages.create(
        model=VERIFY_MODEL,
        max_tokens=1024,
        system=_SYSTEM_PROMPT,
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": "REFERENCE:"},
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": ref_b64,
                    },
                },
                {"type": "text", "text": "STENCIL:"},
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": sten_b64,
                    },
                },
                {"type": "text", "text": "Score the stencil per the rubric."},
            ],
        }],
    )

    raw = message.content[0].text.strip()
    raw = re.sub(r"^\`\`\`(?:json)?\s*|\s*\`\`\`$", "", raw, flags=re.MULTILINE).strip()
    data = json.loads(raw)

    overall = float(data["overall"])
    return VerificationResult(
        score=overall,
        passed=overall >= PAAS_THRESHOLD,
        composition_score=float(data["composition"]),
        subject_score=float(data["subject"]),
        detail_score=float(data["detail"]),
        issues=[str(x) for x in data.get("issues", [])],
        summary=str(data.get("summary", "")),
    )
