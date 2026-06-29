"""
Scene/image text generation.

Two live entry points, both feeding the cached-scene pipeline (see websocket.py):
  - describe_scene: turns the game's own current-surroundings output (a silent LOOK +
    EXAMINEs) into one state-correct visual description, cached by scene-output hash.
  - describe_edit: computes the focused visual delta between a location's reference scene
    and its current state, for pixel-preserving image editing.

The per-turn ACTION narration is sent to the player verbatim (the game's own text), so the
old enrich_stream / image-suggestion path was retired with the runtime-scene refactor.
"""
import json

import anthropic

from ..config import settings

_client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)


_SCENE_SYSTEM = """You are the scene-setter for an interactive fiction game. You are given the
game's own output describing the player's current surroundings — a LOOK plus EXAMINE of the
things present. Write ONE grounded, present-tense visual description of the location AS IT
CURRENTLY IS, for setting the scene and grounding an illustration.

Rules:
1. Describe only what the game's output states or directly implies — never invent new objects,
   exits, or facts.
2. Describe only what IS present. NEVER state the absence of something ("there is no door",
   "nothing here", "no exit") — image generators draw negated objects anyway. Rephrase
   affirmatively: "the wall is solid, lined only with boarded-over windows" instead of "there
   is no door here."
3. Reflect the CURRENT state exactly. If a container is open with something inside, say so; if
   a door is boarded, say so; if it is dark, the scene is dark. Do not describe a state the
   output doesn't support (e.g. don't call an opened thing closed).
4. You may add restrained physical/atmospheric detail (light, material, weather), but NO
   editorializing, backstory, foreshadowing, or authorial commentary ("old money", "frankly",
   "as if it expected you", "indifferent to your presence").
5. Third person, present tense. No second-person "you". Favor concrete visual nouns over mood.
6. The viewpoint IS the player's own eyes — an empty vantage, not a scene with the player in
   it. NEVER introduce a person, figure, observer, or silhouette that the game output does not
   name (e.g. don't add "a figure perches in the tree" for a room the player is simply standing
   in). Still depict characters the output explicitly mentions — a thief, a troll, a gnome.
7. 2-3 plain sentences.
Output only the description prose."""


async def describe_scene(scene_output: str, world_bible: dict | str) -> str:
    """
    Enrich the game's own current-surroundings output (silent LOOK + EXAMINEs) into a
    single state-correct visual description. Deterministic input ⇒ cache by scene hash.
    """
    wb = world_bible if isinstance(world_bible, dict) else json.loads(world_bible or "{}")
    tone = wb.get("tone") or wb.get("setting") or ""
    user = f"## World tone\n{tone}\n\n## Game output (current surroundings)\n{scene_output}"
    response = await _client.messages.create(
        model=settings.model_enrichment,
        max_tokens=400,
        system=_SCENE_SYSTEM,
        messages=[{"role": "user", "content": user}],
    )
    return response.content[0].text.strip()


_EDIT_SYSTEM = """You are given a REFERENCE description and the CURRENT description of the SAME
location in a game. Both are independently-written prose, so their WORDING will differ even
when nothing actually changed. Your job is to find a genuine change to a discrete, nameable
OBJECT — and otherwise say nothing.

Output a single short imperative instruction describing ONLY a real, visible object-state
change, to apply as an edit to the reference image.

What counts as a change (the ONLY things you may report):
- a container opening or closing (mailbox, egg, window, door, trap door)
- an item appearing or disappearing (a leaflet now inside; the egg now gone from the nest)
- a light turning on or off / a room going dark or lit
- a discrete object visibly changing state (a board pried off, a lamp now lit)

What is NEVER a change — ignore completely, even if the two descriptions word it differently:
- background, scenery, landscape, trees, sky, weather, lighting mood, time of day
- sounds, smells, atmosphere, or any non-visual detail (a birdcall is NOT a change)
- any rephrasing, reordering, or added/dropped flavor wording that names no new object state
- framing, camera, composition, colors, or saturation

Rules:
- Report at most ONE object's change. Restate nothing that is unchanged.
- Phrase affirmatively — say what IS there ("the mailbox is empty"), never what is absent.
- When in doubt, or if the only differences are wording/scenery/atmosphere, output exactly:
  no change
Examples:
  ref "a closed mailbox", current "an open mailbox with a leaflet inside" -> "Open the mailbox; a leaflet sits inside it."
  ref "an open mailbox with a leaflet", current "an open, empty mailbox" -> "The open mailbox is now empty."
  ref "a field with a white house, forest to the west", current "a white house in a field, a songbird calling beyond the trees" -> "no change"
Output only the instruction, nothing else."""


async def describe_edit(reference_description: str, current_description: str) -> str:
    """
    Compute the focused visual change from a location's reference scene to its current
    state. FLUX editing drifts far less when given a small change instruction than a full
    re-description. Returns "no change" when nothing visible differs.
    """
    response = await _client.messages.create(
        model=settings.model_translation,   # Haiku — cheap
        max_tokens=120,
        system=_EDIT_SYSTEM,
        messages=[{"role": "user",
                   "content": f"REFERENCE:\n{reference_description}\n\nCURRENT:\n{current_description}"}],
    )
    return response.content[0].text.strip()
