"""
In-memory store for active game engine processes.
Processes don't survive server restarts — sessions are restored from DB saves on reconnect.
"""
import asyncio
from dataclasses import dataclass, field

from .adapter import GameEngineAdapter


@dataclass
class ActiveSession:
    session_id: str
    game_path: str
    adapter: GameEngineAdapter
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


class SessionStore:
    def __init__(self):
        self._sessions: dict[str, ActiveSession] = {}

    def get(self, session_id: str) -> ActiveSession | None:
        return self._sessions.get(session_id)

    def put(self, session: ActiveSession) -> None:
        self._sessions[session.session_id] = session

    async def remove(self, session_id: str) -> None:
        session = self._sessions.pop(session_id, None)
        if session:
            await session.adapter.stop(session_id)

    def active_ids(self) -> list[str]:
        return list(self._sessions.keys())


# Module-level singleton
session_store = SessionStore()
