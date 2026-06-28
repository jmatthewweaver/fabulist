"""
Step 3: turn raw txd-extracted candidate strings into clean, state-neutral visual
descriptions for each object/room (one-time, at game ingestion).

The candidates for one object are a mix of: genuine description prose, mutually
exclusive state messages (lid open vs closed, lamp on vs off), and parser/action
responses ("You can't burn this door."). We ask the model to synthesize a single
description that:
  - combines details that are simultaneously true (augmentation),
  - omits any detail that varies by state (Schrödinger's lid — leave it out),
  - discards parser/action feedback and second-person framing,
  - returns null when nothing stably visual remains.

Runs in batches; failures degrade to "no description" rather than blocking ingestion.
"""
import asyncio
import json
import logging

import anthropic

from ..config import settings

log = logging.getLogger(__name__)

_client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)

_BATCH_SIZE = 20
_MAX_CANDIDATES = 20      # cap per object to bound prompt size (thief has ~20)

_SYSTEM = """You convert raw text fragments from a 1980s text-adventure game into clean,
state-neutral VISUAL descriptions, used for image generation and scene-setting.

For each item you get its name, kind (room/object/...), and a list of candidate strings
pulled straight from the game's code — room text, examine text, and action-response
messages, mixed together in no particular order.

Produce ONE concise description of how the thing physically LOOKS, by these rules:
1. COMBINE details that are true at the same time into a fuller picture (e.g. a waterfall
   AND a rainbow arching over it).
2. OMIT anything that varies by state. If candidates describe conflicting states — a lid
   open in one and closed in another, a lamp on/off, a door open/closed, intact/broken —
   do NOT pick one; leave that detail out entirely. Describe only what holds regardless
   of state.
3. DISCARD parser/action responses and second-person feedback that aren't descriptions
   ("It is far too large to carry.", "You can't burn this door.", "I'm afraid you have
   run out of matches.", "Talking to yourself is...").
4. Strip second-person framing: "You are standing in an open field..." -> "An open
   field...". Write neutral, third-person, present-tense visual prose.
5. Use ONLY what the candidates imply — invent nothing.
6. If no candidate conveys any stable visual description (only actions/parser noise),
   return null for that item.

Keep each description to 1-3 sentences.

Return ONLY a JSON array, no prose: [{"id": <int>, "description": <string or null>}, ...]
Include every id you were given."""


def _parse_json_array(raw: str) -> list[dict]:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1].rsplit("```", 1)[0]
    start, end = raw.find("["), raw.rfind("]")
    if start == -1 or end == -1:
        return []
    return json.loads(raw[start:end + 1])


async def _describe_batch(batch: list[dict]) -> dict[int, str]:
    payload = [
        {
            "id": o["id"],
            "name": o["name"],
            "kind": o.get("kind", "object"),
            "candidates": o["candidates"][:_MAX_CANDIDATES],
        }
        for o in batch
    ]
    response = await _client.messages.create(
        model=settings.model_enrichment,   # Sonnet — the state/augmentation judgment is nuanced
        max_tokens=2000,
        system=_SYSTEM,
        messages=[{"role": "user", "content": json.dumps(payload, ensure_ascii=False)}],
    )
    try:
        items = _parse_json_array(response.content[0].text)
    except (json.JSONDecodeError, IndexError):
        log.warning("description batch returned unparseable JSON", exc_info=True)
        return {}

    out: dict[int, str] = {}
    for item in items:
        desc = item.get("description")
        if isinstance(item.get("id"), int) and isinstance(desc, str) and desc.strip():
            out[item["id"]] = desc.strip()
    return out


async def synthesize_descriptions(objects: list[dict]) -> dict[int, str]:
    """
    objects: [{id, name, kind, candidates}] — only those that HAVE candidates.
    Returns {id: description} for items the model produced a usable description for
    (null/empty results are omitted).
    """
    batches = [objects[i:i + _BATCH_SIZE] for i in range(0, len(objects), _BATCH_SIZE)]
    results = await asyncio.gather(*(_describe_batch(b) for b in batches))
    merged: dict[int, str] = {}
    for r in results:
        merged.update(r)
    return merged


# ---------------------------------------------------------------------------
# Step 4: compose one cohesive description per location
# ---------------------------------------------------------------------------

_LOCATION_BATCH_SIZE = 15

_LOCATION_SYSTEM = """You compose ONE cohesive visual description of a location in a
text-adventure game, used for image generation and scene-setting.

For each location you get: its own base description, and the descriptions of the notable
things present there ("present" = objects in the room and visible scenery like a distant
house or forest).

Weave these into a single natural, present-tense, third-person paragraph describing what
the place LOOKS like as a whole. Rules:
- Integrate smoothly into flowing prose; do not just list the parts.
- Use only what you are given; invent nothing.
- Stay state-neutral (do not assert open/closed, lit/unlit, full/empty).
- Some "present" items have only a name and no description — mention these briefly by
  name (e.g. "a bird's nest sits among the branches"); do not invent their appearance.
  Omit an item only if it is clearly abstract/non-physical.
- Describe only the present items given; do NOT speculate about what might be inside
  containers.
- 2-4 sentences.

Example — base "An open field west of a white house with a boarded front door", present
[white house: "a beautiful colonial house painted white", mailbox: "a small mailbox
securely anchored"] -> "An open field west of a white house with a boarded front door.
The house is a beautiful colonial structure painted white, and a small mailbox stands
securely anchored nearby."

Return ONLY a JSON array, no prose: [{"id": <int>, "location": <string>}, ...]"""


async def _compose_batch(batch: list[dict]) -> dict[int, str]:
    response = await _client.messages.create(
        model=settings.model_enrichment,
        max_tokens=2500,
        system=_LOCATION_SYSTEM,
        messages=[{"role": "user", "content": json.dumps(batch, ensure_ascii=False)}],
    )
    try:
        items = _parse_json_array(response.content[0].text)
    except (json.JSONDecodeError, IndexError):
        log.warning("location batch returned unparseable JSON", exc_info=True)
        return {}

    out: dict[int, str] = {}
    for item in items:
        loc = item.get("location")
        if isinstance(item.get("id"), int) and isinstance(loc, str) and loc.strip():
            out[item["id"]] = loc.strip()
    return out


async def compose_locations(locations: list[dict]) -> dict[int, str]:
    """
    locations: [{id, name, description, present: [{name, description}, ...]}]
    Returns {id: composed_location_description}.
    """
    batches = [locations[i:i + _LOCATION_BATCH_SIZE]
               for i in range(0, len(locations), _LOCATION_BATCH_SIZE)]
    results = await asyncio.gather(*(_compose_batch(b) for b in batches))
    merged: dict[int, str] = {}
    for r in results:
        merged.update(r)
    return merged
