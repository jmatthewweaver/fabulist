"""
Translates natural-language user input into a parser command the IF game accepts.
Uses Haiku with structured JSON output + VocabIndex grounding.
Retries up to MAX_RETRIES times if the game rejects the command.
"""
import json
import re
import anthropic

from ..config import settings

_client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
MAX_RETRIES = 3

_SYSTEM = """You translate natural language into classic interactive fiction parser commands.
Output ONLY a JSON object with keys: verb (string), noun (string or null), prep (string or null), indirect (string or null).
Examples:
  "go north" → {"verb": "go", "noun": "north", "prep": null, "indirect": null}
  "put the key in the box" → {"verb": "put", "noun": "key", "prep": "in", "indirect": "box"}
  "look around" → {"verb": "look", "noun": null, "prep": null, "indirect": null}

Movement: the parser moves by compass DIRECTION (north/south/east/west/ne/nw/se/sw/up/down/
in/out) — never by the name of a path or feature. The SURROUNDINGS text names where each exit
leads. When the user says to walk/go/head toward a described feature (a path, trail, door,
stairs, opening), find that feature in the surroundings and output the DIRECTION it lies in.
  surroundings "To the north a narrow path winds through the trees" + "walk down the path"
    → {"verb": "north", "noun": null, "prep": null, "indirect": null}
  surroundings "a window opens to the east" + "climb through the window"
    → {"verb": "east", "noun": null, "prep": null, "indirect": null}
Prefer the SURROUNDINGS for resolving objects and directions; fall back to the vocabulary
lists for spelling. Use a bare direction as the verb for movement (noun null)."""

_TRANSLATE_PROMPT = """Current room: {room}
Surroundings (the game's own current description — use it to resolve directions and objects):
{surroundings}

Visible objects: {objects}
Known verbs: {verbs}
Known nouns (use exact spelling): {nouns}

User said: "{input}"

Translate to parser command JSON."""

_RETRY_PROMPT = """The game rejected the command "{failed_cmd}" with: "{rejection}"

Visible objects (exact names): {objects}
Known verbs: {verbs}

Try a different command for: "{original_input}"
Output JSON only."""


def _assemble_command(parsed: dict) -> str:
    parts = [parsed["verb"]]
    if parsed.get("noun"):
        parts.append(parsed["noun"])
    # Only include a preposition when it has an object — otherwise it dangles
    # ("go into the path" -> "go path into"). A bare prep is dropped.
    if parsed.get("prep") and parsed.get("indirect"):
        parts.append(parsed["prep"])
        parts.append(parsed["indirect"])
    return " ".join(parts)


def _resolve_nouns(parsed: dict, vocab_index: dict[str, str]) -> dict:
    """Replace user nouns with exact game object names via VocabIndex."""
    for key in ("noun", "indirect"):
        val = parsed.get(key)
        if val:
            resolved = vocab_index.get(val.lower().strip())
            if resolved:
                parsed[key] = resolved
    return parsed


async def translate(
    user_input: str,
    room: str,
    visible_objects: list[str],
    vocab_verbs: list[str],
    vocab_nouns: list[str],
    vocab_index: dict[str, str],
    step_fn,  # async (command: str) -> StepResult
    surroundings: str = "",
) -> tuple[str, str]:
    """
    Returns (final_command, game_output).
    Raises ValueError if all retries are exhausted.

    `surroundings` is the game's own current room text (a live LOOK), which names where the
    exits lead — essential for mapping "walk down the path" to a compass direction.
    """
    objects_str = ", ".join(visible_objects) if visible_objects else "none visible"
    verbs_str = ", ".join(vocab_verbs[:40])
    nouns_str = ", ".join(vocab_nouns[:60])
    surroundings_str = surroundings.strip() or "(not available)"

    prompt = _TRANSLATE_PROMPT.format(
        room=room,
        surroundings=surroundings_str,
        objects=objects_str,
        verbs=verbs_str,
        nouns=nouns_str,
        input=user_input,
    )

    last_command = None
    last_rejection = None
    last_output = None

    for attempt in range(MAX_RETRIES):
        if attempt == 0:
            content = prompt
        else:
            content = _RETRY_PROMPT.format(
                failed_cmd=last_command,
                rejection=last_rejection,
                objects=objects_str,
                verbs=verbs_str,
                original_input=user_input,
            )

        response = _client.messages.create(
            model=settings.model_translation,
            max_tokens=128,
            system=_SYSTEM,
            messages=[{"role": "user", "content": content}],
        )
        raw = response.content[0].text.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1].rsplit("```", 1)[0]

        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            # Fallback: treat the whole text as a raw command
            parsed = {"verb": raw, "noun": None, "prep": None, "indirect": None}

        parsed = _resolve_nouns(parsed, vocab_index)
        command = _assemble_command(parsed)
        result = await step_fn(command)

        if not result.rejected:
            return command, result.raw_text

        last_command = command
        last_rejection = result.raw_text[:200]
        last_output = result.raw_text

    # Exhausted: the parser never understood it. Show the game's OWN last response (e.g.
    # "I don't know the word 'frobozz'.") rather than a generic translation error — it's
    # honest feedback and lets the player rephrase. Falls back to a message if somehow empty.
    return (
        last_command or user_input,
        last_output or "The game didn't understand that. Try rephrasing.",
    )
