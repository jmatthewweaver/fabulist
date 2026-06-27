"""
One-time generation of the World Bible from infodump output + game opening text.
Runs during game ingestion, result stored in Game.world_bible (JSON).
"""
import json
import anthropic

from ..config import settings
from ..game.adapter import StaticWorldData

_client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)

_SYSTEM = """You are analyzing an interactive fiction game to produce a World Bible.
This document will guide an AI narrator that enriches the game's descriptions with
invented-but-consistent sensory details. It must never contradict the game's own text."""

_PROMPT_TEMPLATE = """Game title: {title}

Infodump summary (rooms, objects, vocabulary extracted from the game file):
{world_summary}

Opening game text (first 1000 chars of gameplay):
{opening_text}

Produce a World Bible as JSON with these fields:
- title: string
- setting: string (2-3 sentence description of the world)
- tone: string (e.g. "darkly comic, puzzle-centric, slightly archaic")
- prose_style: string (narrator voice guidance, e.g. "terse second-person present, dry wit")
- period_conventions: list of strings (what belongs in this world)
- forbidden_inventions: list of strings (what must never be invented, e.g. "electricity", "plastic")
- sensory_palette: object with keys smell, sound, texture, sight — each a list of 4-6 period-appropriate strings
- npc_disposition: string (general characterization of how NPCs behave in this world)

Return only valid JSON, no commentary."""


async def generate_world_bible(
    world_data: StaticWorldData,
    opening_text: str,
    game_title: str,
) -> dict:
    # Summarize the infodump into something manageable
    room_names = [r["name"] for r in world_data.rooms[:30]]
    object_names = [o["name"] for o in world_data.objects[:50]]
    world_summary = (
        f"Rooms ({len(world_data.rooms)} total): {', '.join(room_names)}\n"
        f"Objects ({len(world_data.objects)} total): {', '.join(object_names)}\n"
        f"Verbs: {', '.join(world_data.vocab_verbs[:30])}"
    )

    prompt = _PROMPT_TEMPLATE.format(
        title=game_title,
        world_summary=world_summary,
        opening_text=opening_text[:1000],
    )

    response = await _client.messages.create(
        model=settings.model_enrichment,
        max_tokens=4096,
        system=_SYSTEM,
        messages=[{"role": "user", "content": prompt}],
    )
    text = response.content[0].text.strip()
    # Strip markdown code fences if present
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0]
    return json.loads(text)


def build_vocab_index(world_data: StaticWorldData) -> dict[str, str]:
    """Map common user nouns → exact game object names for command translation."""
    index: dict[str, str] = {}
    for obj in world_data.objects:
        name = obj["name"]
        key = name.lower().strip()
        index[key] = name
        # Also index individual words for multi-word objects ("brass lantern" → "lantern")
        for word in key.split():
            if len(word) > 3 and word not in index:
                index[word] = name
    return index
