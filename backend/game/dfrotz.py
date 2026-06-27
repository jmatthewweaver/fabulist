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
