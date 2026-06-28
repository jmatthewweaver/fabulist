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
Use only verbs and nouns from the provided vocabulary lists."""

_TRANSLATE_PROMPT = """Current room: {room}
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
) -> tuple[str, str]:
    """
    Returns (final_command, game_output).
    Raises ValueError if all retries are exhausted.
    """
    objects_str = ", ".join(visible_objects) if visible_objects else "none visible"
    verbs_str = ", ".join(vocab_verbs[:40])
    nouns_str = ", ".join(vocab_nouns[:60])

    prompt = _TRANSLATE_PROMPT.format(
        room=room,
        objects=objects_str,
        verbs=verbs_str,
        nouns=nouns_str,
        input=user_input,
    )

    last_command = None
    last_rejection = None

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

    raise ValueError(
        f"Could not translate '{user_input}' into a valid game command after {MAX_RETRIES} attempts."
    )
