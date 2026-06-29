"""
Manages conversation context across a long play session.
Rolling 20-turn window; older turns compressed to episodic summaries by Haiku.
"""
import json
from collections import deque
from dataclasses import dataclass

import anthropic

from ..config import settings

_client = anthropic.Anthropic(api_key=settings.anthropic_api_key)

WINDOW_SIZE = 20
COMPRESS_BATCH = 10  # compress this many oldest turns when window fills

_COMPRESS_SYSTEM = "Summarize these interactive fiction game turns into one concise paragraph (50 tokens max). Focus on what happened and what was discovered."


@dataclass
class Turn:
    turn_num: int
    user_input: str
    raw_game_output: str
    enriched_narrative: str
    room: str


class ContextManager:
    def __init__(self, world_bible: str):
        self.world_bible = world_bible
        self._recent: deque[Turn] = deque(maxlen=WINDOW_SIZE)
        self._summaries: list[str] = []

    def add_turn(self, turn: Turn) -> None:
        if len(self._recent) == WINDOW_SIZE:
            self._compress_oldest()
        self._recent.append(turn)

    def _compress_oldest(self) -> None:
        """Move the COMPRESS_BATCH oldest turns to a summary."""
        to_compress = [self._recent.popleft() for _ in range(min(COMPRESS_BATCH, len(self._recent)))]
        turns_text = "\n".join(
            f"Turn {t.turn_num}: [{t.room}] {t.user_input} → {t.raw_game_output[:200]}"
            for t in to_compress
        )
        response = _client.messages.create(
            model=settings.model_translation,
            max_tokens=100,
            system=_COMPRESS_SYSTEM,
            messages=[{"role": "user", "content": turns_text}],
        )
        summary = response.content[0].text.strip()
        self._summaries.append(summary)
        # Keep summaries bounded too
        if len(self._summaries) > 20:
            self._summaries = self._summaries[-20:]

    def to_json(self) -> str:
        return json.dumps({
            "world_bible": self.world_bible,
            "recent": [
                {
                    "turn_num": t.turn_num,
                    "user_input": t.user_input,
                    "raw_game_output": t.raw_game_output,
                    "enriched_narrative": t.enriched_narrative,
                    "room": t.room,
                }
                for t in self._recent
            ],
            "summaries": self._summaries,
        })

    @classmethod
    def from_json(cls, data: str) -> "ContextManager":
        d = json.loads(data)
        mgr = cls(world_bible=d["world_bible"])
        for t in d["recent"]:
            mgr._recent.append(Turn(**t))
        mgr._summaries = d["summaries"]
        return mgr
