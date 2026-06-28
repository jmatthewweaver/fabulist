"""
Visual continuity: a self-building "style bible" per (game, style).

As each new location is illustrated, we (1) augment its prompt against the running guide so
it matches established style + recurring objects, then (2) analyze the *generated* image and
fold what it actually looks like back into the guide. Over a playthrough the guide accrues a
stable global look plus canonical appearances ("the white house is a white colonial building
with a boarded door on the west side"), keeping later locations consistent with earlier ones.

The guide doc shape: {"style": "<one paragraph>", "entities": {"<name>": "<appearance>"}}.
"""
import base64
import json
import logging

import anthropic

from ..config import settings

log = logging.getLogger(__name__)

_client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)


def _format_guide(guide: dict) -> str:
    lines: list[str] = []
    if guide.get("style"):
        lines.append(f"Global style (apply to every image): {guide['style']}")
    entities = guide.get("entities") or {}
    if entities:
        lines.append("Recurring things must be drawn exactly like this:")
        lines += [f"- {name}: {desc}" for name, desc in entities.items()]
    return "\n".join(lines)


def _parse_json(raw: str) -> dict:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1].rsplit("```", 1)[0]
    start, end = raw.find("{"), raw.rfind("}")
    return json.loads(raw[start:end + 1]) if start != -1 and end != -1 else {}


_AUGMENT_SYSTEM = """You write a single image-generation prompt that keeps a game's locations
visually consistent. You are given a STYLE GUIDE (the established global look, and how specific
recurring things must appear) and a SCENE to depict.

Rewrite the scene into ONE vivid image prompt that:
- applies the guide's global style, palette, lighting, weather and medium,
- draws any entity named in the guide exactly as the guide specifies,
- otherwise faithfully depicts the scene as described.
Phrase everything affirmatively — describe what IS present, never what is absent.
Output only the prompt, no preamble."""


async def augment_prompt(scene_description: str, guide: dict) -> str:
    """Rewrite a scene description into a prompt consistent with the running guide.
    Falls back to the description unchanged when the guide is empty."""
    if not guide or not (guide.get("style") or guide.get("entities")):
        return scene_description
    try:
        response = await _client.messages.create(
            model=settings.model_translation,   # Haiku — cheap, text only
            max_tokens=400,
            system=_AUGMENT_SYSTEM,
            messages=[{"role": "user",
                       "content": f"STYLE GUIDE:\n{_format_guide(guide)}\n\nSCENE:\n{scene_description}"}],
        )
        return response.content[0].text.strip() or scene_description
    except Exception:
        log.warning("prompt augmentation failed; using raw description", exc_info=True)
        return scene_description


_ANALYZE_SYSTEM = """You maintain a visual STYLE GUIDE for a game's illustrations. You are given
the current guide, the scene description, and the IMAGE that was generated. Update the guide
from what the image actually shows.

Return ONLY JSON: {"style": "...", "entities": {"<name>": "<appearance>"}}
- "style": one paragraph capturing the overall medium, palette, lighting, weather and mood.
  This is set from the FIRST image and must stay STABLE — repeat the existing style almost
  verbatim unless it is clearly wrong.
- "entities": canonical, concise appearances of NOTABLE, RECURRING things in the image
  (buildings, landscape features, characters) that may appear in other locations — NOT
  one-off small props. Only include things actually visible. Keep each to one short clause.
Describe appearances affirmatively. Output only JSON."""


async def analyze_image(image_bytes: bytes, scene_description: str, guide: dict) -> dict:
    """Analyze a generated image and return the updated guide (merged with the prior one)."""
    b64 = base64.b64encode(image_bytes).decode()
    content = [
        {"type": "text",
         "text": f"CURRENT GUIDE:\n{json.dumps(guide)}\n\nSCENE:\n{scene_description}\n\n"
                 f"Update the guide from this generated image:"},
        {"type": "image",
         "source": {"type": "base64", "media_type": "image/jpeg", "data": b64}},
    ]
    try:
        response = await _client.messages.create(
            model=settings.model_enrichment,    # Sonnet — vision capable
            max_tokens=700,
            system=_ANALYZE_SYSTEM,
            messages=[{"role": "user", "content": content}],
        )
        new = _parse_json(response.content[0].text)
    except Exception:
        log.warning("image analysis failed; guide unchanged", exc_info=True)
        return guide

    # Merge: keep the first established style stable; accrue/refine entities.
    style = (guide.get("style") or new.get("style") or "").strip()
    entities = {**(guide.get("entities") or {}), **(new.get("entities") or {})}
    return {"style": style, "entities": entities}
