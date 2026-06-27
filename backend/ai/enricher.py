"""
Output enrichment: transforms terse game output into immersive narrative prose,
and optionally suggests an image to generate.

Streaming protocol:
  - Narrative text streams as chunks via the Anthropic streaming API.
  - After streaming completes, a second lightweight Haiku call extracts the
    structured image suggestion (if any). This keeps streaming fast — the
    image decision doesn't block text delivery.
"""
import json
from typing import AsyncIterator

import anthropic

from ..config import settings
from .context_manager import ContextBundle

_client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)

# --- Image suggestion instructions per mode ---

_IMAGE_INSTRUCTIONS = {
    "conservative": """
IMAGE GUIDANCE: Only suggest an image for a first visit to a new named location.
Do not suggest images for movement, object examination, or routine actions.""",

    "normal": """
IMAGE GUIDANCE: Suggest an image when the moment is genuinely worth visualizing:
  - First visit to a named location (type: room_wide)
  - Player examines a notable or interesting object (type: object_closeup)
  - A dramatic event: finding something significant, danger, a key discovery (type: scene_moment)
Do NOT suggest images for: routine movement, failed commands, inventory checks, dialogue.""",

    "generous": """
IMAGE GUIDANCE: Suggest an image for any visually interesting moment:
  - First visit to a named location (type: room_wide)
  - Any examine command on a named object (type: object_closeup)
  - Looking through openings, windows, passages (type: view)
  - Dramatic events and discoveries (type: scene_moment)
  - Significant inventory changes (type: inventory_still)
Do NOT suggest images for: failed commands, routine movement with no description.""",
}

_SYSTEM_TEMPLATE = """You are the narrator for an interactive fiction game. Transform the game's
terse output into immersive, atmospheric prose.

Rules:
1. Never contradict the game's explicit statements — they are ground truth.
2. Invent sensory details (smell, sound, texture, temperature) that fit the world.
3. Use the World Bible for tone, style, and period conventions.
4. If an object's description appears in Established Descriptions, use it verbatim.
5. Do not mention items or characters not present in the game's output.
6. Output only narrative prose — no meta-commentary, no score, no inventory lists, no image suggestions.
7. Keep responses concise (2-4 paragraphs maximum)."""

_IMAGE_EXTRACT_SYSTEM = """Given this narrative, determine if an image should be generated.
If yes, return JSON: {{"suggest": true, "type": "<room_wide|object_closeup|scene_moment|view|inventory_still>", "subject": "<what to depict>", "prompt_hint": "<concise visual description for image generation, ~20 words>"}}
If no, return: {{"suggest": false}}
Return only valid JSON."""


def _system_prompt() -> str:
    return _SYSTEM_TEMPLATE


def _scene_knowledge(
    known_objects: dict,
    raw_output: str,
    current_room: str,
    limit: int = 10,
) -> str:
    """
    Build a compact scene-knowledge block for the enricher prompt from the id-keyed
    object tree ({nodes, roots, name_index}).

    Includes:
      - Current room (resolved via name_index) and its description, if known
      - Room's initial contents and their descriptions (one container level deep)
      - Other objects named in raw_output that carry a known description

    Descriptions may be empty (populated in a later pipeline step); contents names
    alone are still useful context.  Returns "" if the structure isn't present.
    """
    nodes = known_objects.get("nodes")
    if not nodes:
        return ""
    name_index = known_objects.get("name_index", {})
    raw_lower = raw_output.lower()

    def labeled(node: dict, tag: str) -> str:
        desc = node.get("description", "")
        return f"{tag} {node['name']}" + (f": {desc}" if desc else "")

    lines: list[str] = []
    seen: set[int] = set()

    room_ids = name_index.get(current_room.strip().lower(), [])
    room_node = nodes.get(str(room_ids[0])) if room_ids else None
    if room_node:
        lines.append(labeled(room_node, "[Room]"))
        seen.add(room_node["id"])
        for cid in room_node.get("children", []):
            child = nodes.get(str(cid))
            if not child:
                continue
            lines.append(labeled(child, "[In room]"))
            seen.add(child["id"])
            for gcid in child.get("children", []):
                gc = nodes.get(str(gcid))
                if not gc or gc["id"] in seen:
                    continue
                lines.append(labeled(gc, f"  [Inside {child['name']}]"))
                seen.add(gc["id"])
            if len(lines) >= limit:
                break

    # Objects mentioned in game output but not already covered (need a description)
    for node in nodes.values():
        if len(lines) >= limit:
            break
        nm = node["name"].strip().lower()
        if node["id"] not in seen and nm and nm in raw_lower and node.get("description"):
            lines.append(labeled(node, "[Mentioned]"))
            seen.add(node["id"])

    return "\n".join(lines)


def _build_user_prompt(raw_output: str, bundle: ContextBundle) -> str:
    world_bible = bundle.world_bible if isinstance(bundle.world_bible, dict) else json.loads(bundle.world_bible)
    # Extract known_objects separately — too large to include in the full bible dump
    world_bible = dict(world_bible)
    known_objects: dict[str, dict] = world_bible.pop("known_objects", {})
    # Back-compat: older ingestions stored flat known_descriptions
    known_descriptions: dict[str, str] = world_bible.pop("known_descriptions", {})

    sections = [
        f"## World Bible\n{json.dumps(world_bible, indent=2)}",
        f"## Current Location\n{bundle.current_room}",
    ]
    if bundle.current_inventory:
        sections.append(f"## Inventory\n{', '.join(bundle.current_inventory)}")

    # Scene knowledge: room description + contents (game ground truth)
    scene = _scene_knowledge(known_objects, raw_output, bundle.current_room)
    if not scene and known_descriptions:
        # Fallback for old world bibles that only have flat descriptions
        raw_lower = raw_output.lower()
        room_lower = bundle.current_room.lower()
        relevant = {k: v for k, v in known_descriptions.items()
                    if k.lower() == room_lower or k.lower() in raw_lower}
        scene = "\n".join(f"- {k}: {v}" for k, v in list(relevant.items())[:8])
    if scene:
        sections.append(
            f"## Scene Knowledge (game ground truth — use verbatim, do not reinvent)\n{scene}"
        )

    if bundle.relevant_inventions:
        inv_text = "\n".join(
            f"- {i['object_key']}: {i['canonical_text']}"
            for i in bundle.relevant_inventions
        )
        sections.append(f"## Established Descriptions (use verbatim, do not reinvent)\n{inv_text}")
    if bundle.episodic_summaries:
        sections.append("## Earlier Events\n" + "\n".join(f"- {s}" for s in bundle.episodic_summaries[-5:]))
    if bundle.recent_turns:
        recent_text = "\n".join(
            f"[Turn {t.turn_num}] {t.user_input}: {t.raw_game_output[:150]}"
            for t in bundle.recent_turns[-5:]
        )
        sections.append(f"## Recent Turns\n{recent_text}")
    sections.append(f"## Game Output to Enrich\n{raw_output}")
    return "\n\n".join(sections)


async def enrich_stream(raw_output: str, bundle: ContextBundle) -> AsyncIterator[str]:
    """Yields narrative text chunks as they stream."""
    prompt = _build_user_prompt(raw_output, bundle)
    async with _client.messages.stream(
        model=settings.model_enrichment,
        max_tokens=600,
        system=_system_prompt(),
        messages=[{"role": "user", "content": prompt}],
    ) as stream:
        async for text in stream.text_stream:
            yield text


async def extract_image_suggestion(
    narrative: str,
    raw_output: str,
    current_room: str,
    is_new_room: bool,
) -> dict | None:
    """
    After streaming completes, decide whether to suggest an image.
    Returns a suggestion dict or None.
    Called as an async task — does not block text delivery.
    """
    if settings.image_mode == "conservative" and not is_new_room:
        return None

    context = f"Room: {current_room}\nGame output: {raw_output[:300]}\nNarrative: {narrative[:500]}"
    response = await _client.messages.create(
        model=settings.model_translation,  # Haiku — cheap
        max_tokens=150,
        system=_IMAGE_EXTRACT_SYSTEM,
        messages=[{"role": "user", "content": context}],
    )
    raw = response.content[0].text.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1].rsplit("```", 1)[0]
    try:
        result = json.loads(raw)
        return result if result.get("suggest") else None
    except json.JSONDecodeError:
        return None
