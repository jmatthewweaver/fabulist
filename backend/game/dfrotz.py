"""
DfrotzAdapter: runs dfrotz as a subprocess, communicates via stdin/stdout.
InfodumpExtractor: runs infodump (ztools) to extract Z-machine world data.

run_one_turn(): stateless helper — start → restore → command → save → stop.
"""
import asyncio
import logging
import re
import subprocess
import tempfile
from pathlib import Path

from .adapter import GameEngineAdapter, WorldExtractor, StaticWorldData, StepResult

log = logging.getLogger(__name__)

# Patterns that indicate the game rejected the command
_REJECTION_PATTERNS = re.compile(
    r"(i don'?t (understand|know the word)|"
    r"you can'?t|"
    r"that'?s not a verb|"
    r"i don'?t recognize that|"
    r"huh\?|"
    r"what\?)",
    re.IGNORECASE,
)

_PROMPT = b">"


class DfrotzAdapter(GameEngineAdapter):
    def __init__(self, dfrotz_path: str = "dfrotz"):
        self._dfrotz_path = dfrotz_path
        self._processes: dict[str, asyncio.subprocess.Process] = {}
        self._locks: dict[str, asyncio.Lock] = {}

    async def start(self, game_path: str, session_id: str) -> None:
        log.debug("Starting dfrotz: %s %s", self._dfrotz_path, game_path)
        proc = await asyncio.create_subprocess_exec(
            self._dfrotz_path, "-p", "-m", game_path,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
        self._processes[session_id] = proc
        self._locks[session_id] = asyncio.Lock()
        # Consume the opening banner
        await self._read_until_prompt(proc)
        log.debug("dfrotz ready (pid=%s)", proc.pid)

    async def step(self, session_id: str, command: str) -> StepResult:
        proc = self._processes[session_id]
        async with self._locks[session_id]:
            proc.stdin.write(f"{command}\n".encode())
            await proc.stdin.drain()
            raw = await self._read_until_prompt(proc)
        text = raw.strip()
        rejected = bool(_REJECTION_PATTERNS.search(text))
        return StepResult(raw_text=text, rejected=rejected)

    async def save(self, session_id: str) -> bytes:
        with tempfile.NamedTemporaryFile(suffix=".qzl", delete=False) as f:
            save_path = f.name
        try:
            await self.step(session_id, f"save\n{save_path}")
            return Path(save_path).read_bytes()
        finally:
            Path(save_path).unlink(missing_ok=True)

    async def restore(self, session_id: str, save_bytes: bytes) -> None:
        with tempfile.NamedTemporaryFile(suffix=".qzl", delete=False) as f:
            f.write(save_bytes)
            save_path = f.name
        try:
            await self.step(session_id, f"restore\n{save_path}")
        finally:
            Path(save_path).unlink(missing_ok=True)

    async def stop(self, session_id: str) -> None:
        proc = self._processes.pop(session_id, None)
        self._locks.pop(session_id, None)
        if proc:
            try:
                proc.stdin.write(b"quit\ny\n")
                await proc.stdin.drain()
                await asyncio.wait_for(proc.wait(), timeout=2.0)
            except Exception:
                proc.kill()

    async def _read_until_prompt(self, proc: asyncio.subprocess.Process) -> str:
        buf = bytearray()
        while True:
            chunk = await asyncio.wait_for(proc.stdout.read(256), timeout=10.0)
            if not chunk:
                break
            buf.extend(chunk)
            if buf.rstrip().endswith(b">"):
                break
        text = buf.decode("latin-1", errors="replace")
        return re.sub(r"\s*>\s*$", "", text)


async def run_one_turn(
    game_path: str,
    command: str,
    save_bytes: bytes | None,
    dfrotz_path: str = "dfrotz",
) -> tuple[StepResult, bytes]:
    """
    Stateless per-turn execution: start → restore (if save_bytes) → command → save → stop.
    Returns (StepResult, new_save_bytes).
    """
    log.info("run_one_turn: cmd=%r save=%s bytes", command, len(save_bytes) if save_bytes else 0)
    adapter = DfrotzAdapter(dfrotz_path=dfrotz_path)
    sid = "_turn"
    try:
        await adapter.start(game_path, sid)
    except FileNotFoundError:
        raise RuntimeError(f"dfrotz not found at {dfrotz_path!r} — is it installed?")
    except asyncio.TimeoutError:
        raise RuntimeError(f"dfrotz timed out starting {game_path!r} — is the game file valid?")
    try:
        if save_bytes is not None:
            await adapter.restore(sid, save_bytes)
        result = await adapter.step(sid, command)
        log.info("run_one_turn: got %d chars, rejected=%s", len(result.raw_text), result.rejected)
        new_save = await adapter.save(sid)
        return result, new_save
    except asyncio.TimeoutError:
        raise RuntimeError("dfrotz timed out waiting for game response")
    finally:
        await adapter.stop(sid)


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


# --- infodump output parsers ---

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
