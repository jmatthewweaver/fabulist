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
Output ONLY a JSON ARRAY of command objects. Each object has keys: verb (string), noun (string
or null), prep (string or null), indirect (string or null). Most inputs are ONE action → an
array of one. Output several objects ONLY when the user clearly asks for several distinct
actions (a conjunction like "take the sword and the lamp", or "open the door and go through"),
in order. Never invent steps the user didn't ask for.
Examples:
  "go north" → [{"verb": "north", "noun": null, "prep": null, "indirect": null}]
  "put the key in the box" → [{"verb": "put", "noun": "key", "prep": "in", "indirect": "box"}]
  "take the sword and the lamp" → [{"verb": "take", "noun": "sword", "prep": null, "indirect": null}, {"verb": "take", "noun": "lamp", "prep": null, "indirect": null}]

Movement: the parser moves by compass DIRECTION (north/south/east/west/ne/nw/se/sw/up/down/
in/out) — never by the name of a path or feature. The SURROUNDINGS text names where each exit
leads. When the user says to walk/go/head toward a described feature (a path, trail, door,
stairs, opening), find that feature in the surroundings and output the DIRECTION it lies in.
  surroundings "To the north a narrow path winds through the trees" + "walk down the path"
    → [{"verb": "north", "noun": null, "prep": null, "indirect": null}]
  surroundings "a window opens to the east" + "climb through the window"
    → [{"verb": "east", "noun": null, "prep": null, "indirect": null}]
  surroundings names an open window/door/opening but NO direction + "climb in the window"
    → [{"verb": "enter", "noun": "window", "prep": null, "indirect": null}]

Containers: to obtain something that is INSIDE a container or can't be carried by itself (a
liquid like water, an item shown as the contents of a sack/bottle/box), take the CONTAINER
instead — the SURROUNDINGS state what holds what ("the glass bottle contains water"; "a sack
smelling of hot peppers"). But if the thing can be taken on its own (a clove of garlic, a coin,
a sword), take it directly. Only redirect for TAKE intents — never for use ("drink water"
stays drink water).
  surroundings "The glass bottle contains: A quantity of water." + "take the water"
    → [{"verb": "take", "noun": "bottle", "prep": null, "indirect": null}]
  "take the water and the peppers" (water is in the bottle; the sack smells of hot peppers)
    → [{"verb": "take", "noun": "bottle", "prep": null, "indirect": null}, {"verb": "take", "noun": "sack", "prep": null, "indirect": null}]

Prefer the SURROUNDINGS for resolving objects and directions; fall back to the vocabulary
lists for spelling. Use a bare direction as the verb for movement (noun null).

Output the JSON array and NOTHING else — no explanation, no reasoning, no code fences."""

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


def _valid_cmd(d) -> bool:
    return isinstance(d, dict) and isinstance(d.get("verb"), str) and bool(d["verb"].strip())


def _parse_commands(raw: str) -> list[dict]:
    """
    Extract one OR MORE parser-command dicts from the model's reply, defensively.

    The model is asked for a JSON ARRAY of command objects (most inputs → one; a conjunction
    like "take the bottle and the sack" → several). Haiku sometimes ignores that and emits
    reasoning prose around the JSON — occasionally multiple blocks as it "reconsiders". We must
    NEVER feed that prose to the game as a command (it did once, with persist=True, corrupting
    the save). Accepts, in order of preference: a JSON array of command objects, a single JSON
    object, or — only as a last resort — a short bare one-line command. Returns [] if nothing
    usable, so the caller can retry rather than execute garbage.
    """
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip() if "\n" in raw else raw[3:]

    # Prefer a JSON array of command objects. Our objects are flat (braces, no brackets), so
    # the first ']' closes the array; scan from the last array if the model emitted several.
    for chunk in reversed(re.findall(r"\[.*?\]", raw, re.DOTALL)):
        try:
            arr = json.loads(chunk)
        except json.JSONDecodeError:
            continue
        cmds = [d for d in arr if _valid_cmd(d)] if isinstance(arr, list) else []
        if cmds:
            return cmds

    # Else the last well-formed flat object (model emitted a bare object, not an array).
    for chunk in reversed(re.findall(r"\{[^{}]*\}", raw, re.DOTALL)):
        try:
            obj = json.loads(chunk)
        except json.JSONDecodeError:
            continue
        if _valid_cmd(obj):
            return [obj]

    # No JSON: only trust a bare one-liner that actually looks like a command, never a blob.
    first = raw.splitlines()[0].strip() if raw else ""
    if first and "{" not in first and "[" not in first and len(first) <= 40 and len(first.split()) <= 4:
        return [{"verb": first, "noun": None, "prep": None, "indirect": None}]
    return []


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


MAX_COMMANDS = 5    # bound how many game turns one natural-language input may spend


async def _execute_with_retry(
    parsed: dict,
    user_input: str,
    objects_str: str,
    verbs_str: str,
    vocab_index: dict[str, str],
    step_fn,
) -> tuple[str, str]:
    """
    Run one parsed command; if the PARSER rejects it (didn't understand the words), re-ask the
    model up to MAX_RETRIES times with the game's own rejection as a hint. Returns
    (command_run, game_output) — the last attempt's command/output even on failure, so the
    player sees the game's honest response rather than a generic error.

    step_fn restores from the evolving save, and a rejected attempt doesn't advance it, so
    retries restart cleanly from the last good state.
    """
    parsed = _resolve_nouns(parsed, vocab_index)
    command = _assemble_command(parsed)
    result = await step_fn(command)
    if not result.rejected:
        return command, result.raw_text

    for _ in range(MAX_RETRIES):
        retry = _RETRY_PROMPT.format(
            failed_cmd=command,
            rejection=result.raw_text[:200],
            objects=objects_str,
            verbs=verbs_str,
            original_input=user_input,
        )
        response = _client.messages.create(
            model=settings.model_translation, max_tokens=128, system=_SYSTEM,
            messages=[{"role": "user", "content": retry}],
        )
        cmds = _parse_commands(response.content[0].text)
        if not cmds:
            continue
        parsed = _resolve_nouns(cmds[0], vocab_index)
        command = _assemble_command(parsed)
        result = await step_fn(command)
        if not result.rejected:
            return command, result.raw_text
    return command, result.raw_text


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
    Returns (final_command, game_output). One natural-language input may expand to SEVERAL
    parser commands (e.g. "take the bottle and the sack"); they run in order — step_fn chains
    state across them — and their outputs are concatenated. Never raises; on total failure it
    returns the game's own last response so the player can rephrase.

    `surroundings` is the game's own current room text (a live LOOK), which names where the
    exits lead and what each container holds — essential for mapping "walk down the path" to a
    direction and "take the water" to the bottle that holds it.
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

    # First pass: the full (possibly multi-command) plan. One re-ask if the model returns prose.
    commands: list[dict] = []
    for _ in range(2):
        response = _client.messages.create(
            model=settings.model_translation,
            max_tokens=256,
            system=_SYSTEM,
            messages=[{"role": "user", "content": prompt}],
        )
        commands = _parse_commands(response.content[0].text)[:MAX_COMMANDS]
        if commands:
            break

    if not commands:
        return user_input, "The game didn't understand that. Try rephrasing."

    executed: list[str] = []
    outputs: list[str] = []
    for parsed in commands:
        command, output = await _execute_with_retry(
            parsed, user_input, objects_str, verbs_str, vocab_index, step_fn
        )
        executed.append(command)
        outputs.append(output)

    return "; ".join(executed), "\n\n".join(outputs)
