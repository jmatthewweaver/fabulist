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

_REJECTION_PATTERNS = re.compile(
    r"(i don'?t (understand|know the word)|"
    r"you can'?t|"
    r"that'?s not a verb|"
    r"i don'?t recognize that|"
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
        rejected = bool(_REJECTION_PATTERNS.search(raw_text))
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


class InfodumpExtractor(WorldExtractor):
    def __init__(self, infodump_path: str = "infodump"):
        self._infodump_path = infodump_path

    async def extract(self, game_path: str) -> StaticWorldData:
        try:
            result = subprocess.run(
                [self._infodump_path, "-o", game_path],
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
    section = re.search(r"Grammar:(.+?)(?:\n[A-Z]|\Z)", dump, re.DOTALL)
    if not section:
        return []
    return list(set(re.findall(r'"(\w+)"', section.group(1))))


def _parse_nouns(dump: str) -> list[str]:
    section = re.search(r"Dictionary:(.+?)(?:\n[A-Z]|\Z)", dump, re.DOTALL)
    if not section:
        return []
    return [w.strip() for w in re.findall(r'"(\w+)"', section.group(1))]
