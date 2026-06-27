from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class StaticWorldData:
    """Output of WorldExtractor.extract(). May be partially populated."""
    rooms: list[dict] = field(default_factory=list)        # [{id, name, description, exits}]
    objects: list[dict] = field(default_factory=list)      # [{id, name, parent, description, attrs}]
    vocab_verbs: list[str] = field(default_factory=list)
    vocab_nouns: list[str] = field(default_factory=list)
    object_tree: dict = field(default_factory=dict)        # {nodes:{id:{...}}, roots:[...], name_index:{...}}
    game_title: str = ""
    game_format: str = ""
    raw_dump: str = ""  # full infodump text, for world bible generation


@dataclass
class StepResult:
    raw_text: str
    rejected: bool = False     # True if game said "I don't understand" / "You can't"
    score_delta: int = 0
    done: bool = False


class GameEngineAdapter(ABC):
    """Dumb terminal interface to a running IF game process."""

    @abstractmethod
    async def start(self, game_path: str, session_id: str) -> None:
        """Launch the game process and perform initial handshake."""

    @abstractmethod
    async def step(self, session_id: str, command: str) -> StepResult:
        """Send one command, return the game's response."""

    @abstractmethod
    async def save(self, session_id: str) -> bytes:
        """Serialize current game state to bytes."""

    @abstractmethod
    async def restore(self, session_id: str, save_bytes: bytes) -> None:
        """Restore game state from bytes produced by save()."""

    @abstractmethod
    async def stop(self, session_id: str) -> None:
        """Terminate the game process and release resources."""


class WorldExtractor(ABC):
    """One-time static extraction from a game file before play begins."""

    @abstractmethod
    async def extract(self, game_path: str) -> StaticWorldData:
        """
        Parse the game file and return structured world data.
        Should return a StaticWorldData with whatever fields are available.
        An empty/partial result is valid — the ingestion pipeline degrades gracefully.
        """
