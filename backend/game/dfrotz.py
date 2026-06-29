"""
dfrotz integration via piped subprocess.

All commands (restore, command, save, quit) are written to stdin at once.
dfrotz exits when stdin closes — no interactive I/O synchronization needed.
This matches the technique confirmed working in the diagnostic:
  printf 'look\\nsave\\n/tmp/x.qzl\\n' | dfrotz -p -m game.z3
"""
import asyncio
import logging
import re
import subprocess
import tempfile
from pathlib import Path

from .adapter import WorldExtractor, StaticWorldData, StepResult

log = logging.getLogger(__name__)

# The parser did not UNDERSTAND the input at all — a rephrase might work, so the translator
# should retry. This must NOT match "understood but the action failed" responses ("You can't
# go that way", "You can't see any peppers here") — those are valid game feedback the player
# should see verbatim, not something to retry into a generic translation error.
_NOT_UNDERSTOOD = re.compile(
    r"(i don'?t (understand|know the word)|"
    r"that'?s not a verb|"
    r"i don'?t recognize that|"
    r"that sentence isn'?t one i recognize|"
    r"you used the word .* in a way|"
    r"huh\?|"
    r"what\?)",
    re.IGNORECASE,
)

# An EXAMINE produced no real description of its target (out of scope or concealed). Broader
# than _NOT_UNDERSTOOD — it also catches "you can't see any X here" — so observe_scene drops
# these and they don't pollute the scene text (and thus the scene cache key).
_EXAMINE_MISS = re.compile(
    r"(you can'?t see|"
    r"i don'?t (understand|know the word|recognize)|"
    r"that'?s not (a verb|here)|"
    r"huh\?|"
    r"what\?)",
    re.IGNORECASE,
)


def _run_dfrotz_sync(game_path: str, stdin_text: str, dfrotz_path: str) -> str:
    """Synchronous dfrotz invocation; call via run_in_executor."""
    result = subprocess.run(
        [dfrotz_path, "-p", "-m", game_path],
        input=stdin_text.encode(),
        capture_output=True,
        timeout=30,
    )
    return result.stdout.decode("latin-1", errors="replace")


def _extract_command_output(stdout: str, has_restore: bool) -> str:
    """
    dfrotz output is delimited by '\\n>' prompt markers.
    Layout:  banner \\n> [restore_ok \\n>] command_output \\n> save_ok \\n> quit_prompt
    """
    parts = stdout.split("\n>")
    idx = 2 if has_restore else 1
    return parts[idx].strip() if idx < len(parts) else ""


async def run_one_turn(
    game_path: str,
    command: str,
    save_bytes: bytes | None,
    dfrotz_path: str = "dfrotz",
    persist: bool = True,
) -> tuple[StepResult, bytes | None]:
    """
    Stateless per-turn execution.  persist=False skips the save step (used for
    the opening 'look' where the initial game state is always reproducible).
    """
    log.info("run_one_turn: cmd=%r save=%s bytes persist=%s",
             command, len(save_bytes) if save_bytes else 0, persist)

    restore_path: str | None = None
    save_path: str | None = None
    try:
        lines: list[str] = []

        if save_bytes is not None:
            tf = tempfile.NamedTemporaryFile(suffix=".qzl", delete=False)
            tf.write(save_bytes)
            tf.close()
            restore_path = tf.name
            lines.append(f"restore\n{restore_path}")

        lines.append(command)

        if persist:
            tf = tempfile.NamedTemporaryFile(suffix=".qzl", delete=False)
            tf.close()
            save_path = tf.name
            # dfrotz prompts "Overwrite existing file?" if the path already exists, and our
            # piped stdin never answers it — so the save silently aborts and state never
            # persists. Remove the empty placeholder so dfrotz writes a fresh file, no prompt.
            Path(save_path).unlink(missing_ok=True)
            lines.append(f"save\n{save_path}")

        lines.append("quit\ny")

        stdin_text = "\n".join(lines) + "\n"
        loop = asyncio.get_event_loop()
        try:
            stdout = await loop.run_in_executor(
                None,
                lambda: _run_dfrotz_sync(game_path, stdin_text, dfrotz_path),
            )
        except FileNotFoundError:
            raise RuntimeError(f"dfrotz not found at {dfrotz_path!r} — is it installed?")
        except subprocess.TimeoutExpired:
            raise RuntimeError("dfrotz timed out — check game file path")

        raw_text = _extract_command_output(stdout, has_restore=save_bytes is not None)
        log.info("run_one_turn: got %d chars", len(raw_text))
        rejected = bool(_NOT_UNDERSTOOD.search(raw_text))
        result = StepResult(raw_text=raw_text, rejected=rejected)

        new_save: bytes | None = None
        if persist and save_path:
            p = Path(save_path)
            if p.exists() and p.stat().st_size > 0:
                new_save = p.read_bytes()
                log.info("run_one_turn: saved %d bytes", len(new_save))
            else:
                log.warning("run_one_turn: dfrotz did not create save file")

        return result, new_save

    finally:
        if restore_path:
            Path(restore_path).unlink(missing_ok=True)
        if save_path:
            Path(save_path).unlink(missing_ok=True)


# "Your score is 10 (total of 350 points), in 12 moves." (the SCORE verb's reply)
_SCORE_RE = re.compile(r"score (?:is|would be) (\d+)", re.IGNORECASE)
_MAXSCORE_RE = re.compile(r"total of (\d+)", re.IGNORECASE)
_MOVES_RE = re.compile(r"in (\d+) (?:move|turn)", re.IGNORECASE)


def _parse_status(text: str) -> dict | None:
    m = _SCORE_RE.search(text)
    if not m:
        return None
    mx = _MAXSCORE_RE.search(text)
    mv = _MOVES_RE.search(text)
    return {
        "score": int(m.group(1)),
        "max_score": int(mx.group(1)) if mx else None,
        "moves": int(mv.group(1)) if mv else None,
    }


async def read_game_status(
    game_path: str,
    save_bytes: bytes | None,
    dfrotz_path: str = "dfrotz",
) -> dict | None:
    """
    Non-perturbing read of the game's score/moves via the SCORE verb (restore → score →
    never save). Returns {score, max_score, moves} or None if the game has no score verb.
    """
    restore_path: str | None = None
    try:
        lines: list[str] = []
        if save_bytes is not None:
            tf = tempfile.NamedTemporaryFile(suffix=".qzl", delete=False)
            tf.write(save_bytes)
            tf.close()
            restore_path = tf.name
            lines.append(f"restore\n{restore_path}")
        lines.append("score")
        lines.append("quit\ny")

        stdin_text = "\n".join(lines) + "\n"
        loop = asyncio.get_event_loop()
        try:
            stdout = await loop.run_in_executor(
                None, lambda: _run_dfrotz_sync(game_path, stdin_text, dfrotz_path)
            )
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return None

        outputs = _extract_command_outputs(stdout, has_restore=save_bytes is not None, count=1)
        return _parse_status(outputs[0]) if outputs else None
    finally:
        if restore_path:
            Path(restore_path).unlink(missing_ok=True)


def _extract_command_outputs(stdout: str, has_restore: bool, count: int) -> list[str]:
    """
    Like _extract_command_output but for several commands run back-to-back.
    Layout:  banner \\n> [restore_ok \\n>] out1 \\n> out2 \\n> ... \\n> quit_prompt
    Returns up to `count` command outputs (in order), trimmed.
    """
    parts = stdout.split("\n>")
    base = 2 if has_restore else 1
    return [parts[base + i].strip() for i in range(count) if base + i < len(parts)]


async def observe_scene(
    game_path: str,
    save_bytes: bytes | None,
    dfrotz_path: str = "dfrotz",
    examine_targets: list[str] | None = None,
) -> str:
    """
    Non-perturbing observation of the current scene: restore the save, run `look`
    then `examine <t>` for each target, capture each command's output, and NEVER
    save — the real game state is untouched.

    Returns the combined cleaned text (the state-correct scene). The `look` output
    is always included; an `examine` output is dropped if it's a rejection
    ("you can't see any X here") so out-of-scope/concealed targets add no noise.

    This deterministic text is the scene's state-signature (hash it for caching).
    """
    targets = examine_targets or []
    commands = ["look"] + [f"examine {t}" for t in targets]

    restore_path: str | None = None
    try:
        lines: list[str] = []
        if save_bytes is not None:
            tf = tempfile.NamedTemporaryFile(suffix=".qzl", delete=False)
            tf.write(save_bytes)
            tf.close()
            restore_path = tf.name
            lines.append(f"restore\n{restore_path}")

        lines.extend(commands)
        lines.append("quit\ny")

        stdin_text = "\n".join(lines) + "\n"
        loop = asyncio.get_event_loop()
        try:
            stdout = await loop.run_in_executor(
                None, lambda: _run_dfrotz_sync(game_path, stdin_text, dfrotz_path)
            )
        except FileNotFoundError:
            raise RuntimeError(f"dfrotz not found at {dfrotz_path!r} — is it installed?")
        except subprocess.TimeoutExpired:
            raise RuntimeError("dfrotz timed out — check game file path")

        outputs = _extract_command_outputs(stdout, has_restore=save_bytes is not None, count=len(commands))

        kept: list[str] = []
        for i, text in enumerate(outputs):
            if not text:
                continue
            if i > 0 and _EXAMINE_MISS.search(text):
                continue        # skip examine of out-of-scope/concealed target
            kept.append(text)

        scene = "\n\n".join(kept)
        log.info("observe_scene: %d/%d outputs kept, %d chars", len(kept), len(commands), len(scene))
        return scene

    finally:
        if restore_path:
            Path(restore_path).unlink(missing_ok=True)


class InfodumpExtractor(WorldExtractor):
    def __init__(self, infodump_path: str = "infodump"):
        self._infodump_path = infodump_path

    async def extract(self, game_path: str) -> StaticWorldData:
        try:
            # -f: full dump — includes the '**** Object tree ****' section that
            # _parse_object_tree needs (plain -o omits it), plus grammar/dictionary
            # for vocab. raw_dump isn't sent to the LLM, so the extra size is free.
            result = subprocess.run(
                [self._infodump_path, "-f", game_path],
                capture_output=True, text=True, timeout=30,
            )
            raw = result.stdout
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return StaticWorldData(raw_dump="")

        return StaticWorldData(
            rooms=_parse_rooms(raw),
            objects=_parse_objects(raw),
            vocab_verbs=_parse_verbs(raw),
            vocab_nouns=_parse_nouns(raw),
            object_tree=_parse_object_tree(raw),
            raw_dump=raw,
        )


def _parse_rooms(dump: str) -> list[dict]:
    rooms = []
    for block in re.split(r"\n(?=Object\s+\d+)", dump):
        if "Room" in block or "Outdoors" in block:
            name_match = re.search(r'Short name: "([^"]+)"', block)
            if name_match:
                rooms.append({"name": name_match.group(1), "raw": block})
    return rooms


def _parse_objects(dump: str) -> list[dict]:
    objects = []
    for block in re.split(r"\n(?=Object\s+\d+)", dump):
        name_match = re.search(r'Short name: "([^"]+)"', block)
        if name_match:
            objects.append({"name": name_match.group(1), "raw": block})
    return objects


# Matches one line of infodump's object-tree section, e.g.  ` .  . [230] "small mailbox"`
# The leading indent is a run of spaces and '.' characters; depth = number of dots.
_TREE_LINE = re.compile(r'^(?P<indent>[ .]*)\[\s*(?P<id>\d+)\]\s+"(?P<name>.*)"\s*$')


def _parse_object_tree(dump: str) -> dict:
    """
    Parse infodump's '**** Object tree ****' section into an id-keyed structure:
        {nodes: {str(id): {id, name, parent, children, kind, description, candidates}},
         roots: [id, ...],
         name_index: {lowercased_name: [id, ...]}}

    Blank-named pseudo-containers are relabeled for readability (decision in plan):
      - the blank root with the most children      -> "Rooms"     (kind=container)
      - any other blank container                  -> "Global Objects" (kind=container)
      - a root named like the player avatar         -> "Player"    (kind=player)
    Children of the Rooms container are kind=room; everything else with a real name
    is kind=object.  Returns {} if the section is absent (graceful degrade).
    """
    start = dump.find("**** Object tree ****")
    if start == -1:
        return {}
    rest = dump[start + len("**** Object tree ****"):]
    end = rest.find("****")          # next banner ends the section
    section = rest[:end] if end != -1 else rest

    nodes: dict[str, dict] = {}
    roots: list[int] = []
    stack: list[int] = []            # stack[d] = id of the most recent node at depth d

    for line in section.splitlines():
        m = _TREE_LINE.match(line)
        if not m:
            continue
        depth = m.group("indent").count(".")
        oid = int(m.group("id"))

        del stack[depth:]            # keep ancestors at depths 0..depth-1
        parent = stack[depth - 1] if depth > 0 and len(stack) >= depth else None
        stack.append(oid)

        nodes[str(oid)] = {
            "id": oid,
            "name": m.group("name"),
            "parent": parent,
            "children": [],
            "kind": "object",
            "description": "",
            "candidates": [],
            "scenery": [],      # global/local-global object ids visible from a room
            "location_description": "",   # composed room+contents+scenery (rooms only)
        }
        if parent is None:
            roots.append(oid)
        elif str(parent) in nodes:
            nodes[str(parent)]["children"].append(oid)

    if not nodes:
        return {}

    # --- pseudo-container heuristics ---
    blank_roots = [nodes[str(r)] for r in roots if nodes[str(r)]["name"] == ""]
    rooms_container_id = (
        max(blank_roots, key=lambda n: len(n["children"]))["id"] if blank_roots else None
    )

    # The player avatar is a top-level object; match by common ZIL names (the
    # "you"/"hands" pseudo-objects live under the globals container, not here).
    player_names = {"cretin", "adventurer", "yourself", "you"}
    player_id = next(
        (r for r in roots if nodes[str(r)]["name"].strip().lower() in player_names),
        None,
    )

    for node in nodes.values():
        if node["id"] == rooms_container_id:
            node["name"], node["kind"] = "Rooms", "container"
        elif node["id"] == player_id:
            node["name"], node["kind"] = "Player", "player"
        elif node["name"] == "" and node["children"]:
            node["name"], node["kind"] = "Global Objects", "container"
        elif node["parent"] == rooms_container_id:
            node["kind"] = "room"
        else:
            node["kind"] = "object"

    name_index: dict[str, list[int]] = {}
    for node in nodes.values():
        key = node["name"].strip().lower()
        if key:
            name_index.setdefault(key, []).append(node["id"])

    return {"nodes": nodes, "roots": roots, "name_index": name_index}


def _parse_verbs(dump: str) -> list[str]:
    """
    Verbs from infodump's '**** Parse tables ****' verb grammar. The section opens with
    'Verb entries = N' and lists, one verb per entry:

        247. 11 entries, verb = "go", synonyms = "procee", "run", "step", "walk"

    Only lines carrying `verb =` are real verbs — the same section also contains grammar
    template lines (`[01 00 ..] "go OBJ"`) and a preposition table (`249. "on", ...`), both
    of which we skip by requiring `verb =`. Words are the game's actual 6-char-truncated
    parser tokens (e.g. "procee", "destro"), which is exactly what the parser accepts.
    Returns [] if the section is absent (graceful degrade).
    """
    start = dump.find("Verb entries =")
    if start == -1:
        return []
    end = dump.find("****", start)          # next banner (Dictionary) ends the section
    section = dump[start:end] if end != -1 else dump[start:]

    verbs: list[str] = []
    seen: set[str] = set()
    for line in section.splitlines():
        if "verb =" not in line:
            continue
        for w in re.findall(r'"([^"]+)"', line):   # head verb + same-line synonyms
            w = w.strip().lower()
            if w and not w.startswith("#") and " " not in w and w not in seen:
                seen.add(w)
                verbs.append(w)
    return verbs


# A dictionary entry: `[  16] air` / `[  17] air-p` — bracketed index then a padded token.
_DICT_ENTRY = re.compile(r"\[\s*\d+\]\s+(\S+)")


def _parse_nouns(dump: str) -> list[str]:
    """
    The game's parser vocabulary from infodump's '**** Dictionary ****' section, laid out
    several entries per line:

        [   9] a       [  16] air     [  17] air-p   [  19] altar   [  23] answer

    Tokens are unquoted and 6-char truncated. We keep alphabetic words (allowing an internal
    hyphen) and drop the punctuation/control tokens ($ve, ".", ",", "#comm"). This is the
    full parser vocabulary (nouns, adjectives, verbs, directions); it grounds the translator
    and complements the object-name `vocab_index`. Returns [] if the section is absent.
    """
    start = dump.find("**** Dictionary ****")
    if start == -1:
        return []
    body = dump[start + len("**** Dictionary ****"):]
    end = body.find("****")                 # next banner, if any, ends the section
    section = body[:end] if end != -1 else body

    words: list[str] = []
    seen: set[str] = set()
    for tok in _DICT_ENTRY.findall(section):
        w = tok.strip().lower()
        if re.fullmatch(r"[a-z][a-z\-]*", w) and w not in seen:
            seen.add(w)
            words.append(w)
    return words
