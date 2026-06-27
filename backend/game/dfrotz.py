"""
DfrotzAdapter: runs dfrotz as a subprocess via a pty so it behaves interactively.
InfodumpExtractor: runs infodump (ztools) to extract Z-machine world data.

run_one_turn(): stateless helper — start → restore → command → save → stop.

Why pty: when dfrotz's stdin/stdout are pipes, isatty() returns False and libc
may buffer stdout. Using a pty makes dfrotz think it's attached to a terminal,
ensuring line-buffered output and correct interactive save/restore prompts.
"""
import asyncio
import fcntl
import logging
import os
import pty
import re
import subprocess
import tempfile
import termios
from pathlib import Path

from .adapter import GameEngineAdapter, WorldExtractor, StaticWorldData, StepResult

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


class DfrotzAdapter(GameEngineAdapter):
    def __init__(self, dfrotz_path: str = "dfrotz"):
        self._dfrotz_path = dfrotz_path
        self._processes: dict[str, asyncio.subprocess.Process] = {}
        self._master_fds: dict[str, int] = {}
        self._locks: dict[str, asyncio.Lock] = {}

    async def start(self, game_path: str, session_id: str) -> None:
        master_fd, slave_fd = pty.openpty()

        # Disable echo on slave so our written commands don't appear in output
        attrs = termios.tcgetattr(slave_fd)
        attrs[3] &= ~termios.ECHO
        termios.tcsetattr(slave_fd, termios.TCSANOW, attrs)

        # Non-blocking master so os.read() never blocks in the event loop callback
        fl = fcntl.fcntl(master_fd, fcntl.F_GETFL)
        fcntl.fcntl(master_fd, fcntl.F_SETFL, fl | os.O_NONBLOCK)

        log.debug("Starting dfrotz via pty: %s %s", self._dfrotz_path, game_path)
        proc = await asyncio.create_subprocess_exec(
            self._dfrotz_path, "-p", "-m", game_path,
            stdin=slave_fd,
            stdout=slave_fd,
            stderr=asyncio.subprocess.DEVNULL,
        )
        os.close(slave_fd)

        self._processes[session_id] = proc
        self._master_fds[session_id] = master_fd
        self._locks[session_id] = asyncio.Lock()

        await self._read_until_prompt(master_fd)
        log.debug("dfrotz ready (pid=%s)", proc.pid)

    async def _read_chunk(self, fd: int, timeout: float = 10.0) -> bytes:
        """Read up to 256 bytes from fd, waiting asynchronously."""
        loop = asyncio.get_event_loop()
        fut: asyncio.Future[bytes] = loop.create_future()

        def on_readable():
            try:
                data = os.read(fd, 256)
            except BlockingIOError:
                return  # spurious wakeup; leave reader registered
            except OSError as exc:
                loop.remove_reader(fd)
                if not fut.done():
                    fut.set_exception(exc)
                return
            loop.remove_reader(fd)
            if not fut.done():
                fut.set_result(data)

        loop.add_reader(fd, on_readable)
        try:
            return await asyncio.wait_for(fut, timeout=timeout)
        except BaseException:
            loop.remove_reader(fd)
            raise

    async def _read_until_prompt(self, fd: int) -> str:
        """Read from the pty master until a bare '>' prompt appears."""
        buf = bytearray()
        while True:
            chunk = await self._read_chunk(fd)
            buf.extend(chunk)
            # Pty ONLCR converts \n→\r\n; normalise before checking
            clean = buf.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
            if clean.rstrip().endswith(b">"):
                break
        text = buf.decode("latin-1", errors="replace")
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        return re.sub(r"\s*>\s*$", "", text)

    async def step(self, session_id: str, command: str) -> StepResult:
        fd = self._master_fds[session_id]
        async with self._locks[session_id]:
            os.write(fd, f"{command}\n".encode())
            raw = await self._read_until_prompt(fd)
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
        fd = self._master_fds.pop(session_id, None)
        self._locks.pop(session_id, None)
        if fd is not None:
            try:
                os.write(fd, b"quit\ny\n")
            except OSError:
                pass
        if proc:
            try:
                await asyncio.wait_for(proc.wait(), timeout=2.0)
            except Exception:
                proc.kill()
        if fd is not None:
            try:
                os.close(fd)
            except OSError:
                pass


async def run_one_turn(
    game_path: str,
    command: str,
    save_bytes: bytes | None,
    dfrotz_path: str = "dfrotz",
    persist: bool = True,
) -> tuple[StepResult, bytes | None]:
    """
    Stateless per-turn execution: start → restore (if save_bytes) → command → save → stop.

    persist=False skips the save step; use for the opening 'look' where the initial
    game state is always reproducible and saving is unnecessary.
    """
    log.info("run_one_turn: cmd=%r save=%s bytes persist=%s",
             command, len(save_bytes) if save_bytes else 0, persist)
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
        new_save = await adapter.save(sid) if persist else None
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
