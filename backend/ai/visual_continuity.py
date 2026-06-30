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


def _entity_in_scene(name: str, scene_lower: str) -> bool:
    """True if the named entity is actually referenced in the scene text (match on its
    significant words so 'white house' matches '...white colonial house...')."""
    words = [w for w in name.lower().split() if len(w) >= 4]
    return any(w in scene_lower for w in words) if words else name.lower() in scene_lower


def _format_guide(style: str, entities: dict) -> str:
    lines: list[str] = []
    if style:
        lines.append(f"GLOBAL LOOK (the palette, light and mood every scene shares — weave it "
                     f"into the description as colour/light/mood, never as new scenery): {style}")
    if entities:
        lines.append("If — and only if — the scene includes any of these, draw them like this:")
        lines += [f"- {name}: {desc}" for name, desc in entities.items()]
    return "\n".join(lines)


def _parse_json(raw: str) -> dict:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1].rsplit("```", 1)[0]
    start, end = raw.find("{"), raw.rfind("}")
    return json.loads(raw[start:end + 1]) if start != -1 and end != -1 else {}


_AUGMENT_SYSTEM = """You write a single image-generation prompt for a game location, so the
game's places share a consistent look. You are given a GLOBAL LOOK (the palette, lighting and
mood every scene shares), canonical appearances for specific recurring things, and the SCENE to
depict. The art medium/style is applied SEPARATELY by the system — don't restate it.

Write ONE image prompt that:
- describes the SCENE's physical content (the location and the things in it) as concrete
  subjects, framed wide enough to show the whole space,
- WEAVES the global look into that description as natural language — the same palette, light
  and mood settling over THIS place — rather than tacking it on as a separate clause or prefix,
- when the scene includes one of the listed recurring things, draws it the canonical way,
- applies the global look ONLY as colour/light/mood. NEVER let it (or anything) add buildings,
  objects, places, or scenery the SCENE itself does not mention — no forest, house, or room the
  scene didn't state (the global look is an atmosphere, not a place),
- contains NO medium, art-style, camera, or photography words ("photograph", "illustration",
  "establishing shot", "eye level", "style guide", "render", "wide shot") and NO meta or
  instruction language — only the scene's physical content and its colour/light/mood, or the
  image will draw those words literally.
Phrase everything affirmatively — describe what IS present, never what is absent.
Output only the prompt, no preamble."""


async def augment_prompt(scene_description: str, guide: dict) -> str:
    """Rewrite a scene description into a prompt that carries the running guide.
    The guide's "style" — a transferable ATMOSPHERE (palette, light, mood), NOT a medium or a
    place (see _ANALYZE_SYSTEM) — is woven into the scene as colour/light/mood so the game's
    locations stay consistent scene to scene. The art medium itself is applied separately via
    style_prefix. Only entities actually present in the scene are carried in (so the
    mailbox/house don't follow the player around). Falls back to the description unchanged when
    there's nothing to enforce."""
    guide = guide or {}
    style = guide.get("style") or ""
    scene_lower = scene_description.lower()
    relevant = {n: d for n, d in (guide.get("entities") or {}).items()
                if _entity_in_scene(n, scene_lower)}
    if not style and not relevant:
        return scene_description
    try:
        response = await _client.messages.create(
            model=settings.model_translation,   # Haiku — cheap, text only
            max_tokens=400,
            system=_AUGMENT_SYSTEM,
            messages=[{"role": "user",
                       "content": f"{_format_guide(style, relevant)}\n\nSCENE:\n{scene_description}"}],
        )
        return response.content[0].text.strip() or scene_description
    except Exception:
        log.warning("prompt augmentation failed; using raw description", exc_info=True)
        return scene_description


_ANALYZE_SYSTEM = """You maintain a visual STYLE GUIDE for a game's illustrations. You are given
the current guide, the scene description, and the IMAGE that was generated. Update the guide
from what the image actually shows.

Return ONLY JSON: {"style": "...", "entities": {"<name>": "<appearance>"}}
- "style": one or two natural sentences capturing ONLY the transferable ATMOSPHERE every scene
  shares — colour palette, colour temperature, lighting quality, weather/haze, and mood. Do
  NOT name the art medium or art style (it is fixed elsewhere), do NOT mention camera, framing,
  shot type, or "eye level", and do NOT name any specific PLACE or object ("forest", "house",
  "kitchen") — only qualities that transfer to ANY location. This is set from the FIRST image
  and must stay STABLE — repeat the existing style almost verbatim unless clearly wrong.
  Example: "A muted, cool palette of slate blues and deep greens under soft, low twilight
  light, with a faint ground haze and a quiet, melancholy mood."
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
