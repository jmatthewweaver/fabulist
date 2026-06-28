from sqlalchemy import (
    Column, String, Integer, DateTime, Text,
    ForeignKey, UniqueConstraint, Index
)
from sqlalchemy.dialects.postgresql import JSONB, BYTEA
from sqlalchemy.orm import DeclarativeBase, relationship
from sqlalchemy.ext.asyncio import AsyncAttrs
from pgvector.sqlalchemy import Vector
from datetime import datetime


class Base(AsyncAttrs, DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"
    id = Column(String, primary_key=True)
    email = Column(String, unique=True, nullable=False)
    display_name = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)

    playthroughs = relationship("Playthrough", back_populates="user")


class Game(Base):
    """One row per ingested game file."""
    __tablename__ = "games"
    id = Column(String, primary_key=True)           # sha256 of game file
    title = Column(String, nullable=False)
    filename = Column(String, nullable=False)
    format = Column(String, nullable=False)         # "zmachine", "glulx", etc.
    description = Column(Text)
    default_style_id = Column(String, ForeignKey("styles.id"))
    world_bible = Column(JSONB)
    vocab_index = Column(JSONB)
    icon_image_url = Column(String)
    ingested_at = Column(DateTime)

    playthroughs = relationship("Playthrough", back_populates="game")
    default_style = relationship("Style", foreign_keys=[default_style_id])


class Style(Base):
    """A named visual/narrative style for image + enrichment generation."""
    __tablename__ = "styles"
    id = Column(String, primary_key=True)
    name = Column(String, nullable=False)
    description = Column(String)
    flux_prompt_prefix = Column(Text)
    flux_negative_prompt = Column(Text)
    tone_instructions = Column(Text)
    seed_image_url = Column(String)


class Playthrough(Base):
    """
    One persistent playthrough per user per game. The URL refers to this.
    engine_save holds the current Z-machine state; updated after every turn.
    """
    __tablename__ = "playthroughs"
    id = Column(String, primary_key=True)
    user_id = Column(String, ForeignKey("users.id"), nullable=False)
    game_id = Column(String, ForeignKey("games.id"), nullable=False)
    style_id = Column(String, ForeignKey("styles.id"), nullable=True)
    engine_save = Column(BYTEA, nullable=True)        # None until first turn completes
    context_json = Column(JSONB)
    current_room = Column(String)
    turn_count = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_active = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="playthroughs")
    game = relationship("Game", back_populates="playthroughs")
    style = relationship("Style")
    inventions = relationship("Invention", back_populates="playthrough")


class Invention(Base):
    """Invented detail for a named object, scoped to a playthrough. Immutable once written."""
    __tablename__ = "inventions"
    id = Column(Integer, primary_key=True, autoincrement=True)
    playthrough_id = Column(String, ForeignKey("playthroughs.id"), nullable=False)
    object_key = Column(String, nullable=False)
    canonical_text = Column(Text, nullable=False)
    # full_text is a generated column (set via DDL in migrations.py) used for BM25 indexing
    full_text = Column(Text)
    embedding = Column(Vector(1024))
    source_turn = Column(Integer)
    created_at = Column(DateTime, default=datetime.utcnow)

    playthrough = relationship("Playthrough", back_populates="inventions")

    __table_args__ = (
        UniqueConstraint("playthrough_id", "object_key", name="uq_invention_playthrough_object"),
        Index("ix_inventions_embedding", "embedding", postgresql_using="hnsw",
              postgresql_with={"m": "16", "ef_construction": "64"},
              postgresql_ops={"embedding": "vector_cosine_ops"}),
    )


class CachedScene(Base):
    """
    Game-global cache of a rendered scene, keyed by the hash of the game's own
    deterministic output (LOOK + EXAMINEs) plus game + style. Identical state across
    playthroughs/visits produces the identical output hash → one render, reused forever.
    Holds both the enriched scene description and the generated image.
    """
    __tablename__ = "cached_scenes"
    cache_key = Column(String, primary_key=True)        # sha256(game_id|style_id|scene_output)
    game_id = Column(String, ForeignKey("games.id"), nullable=False)
    style_id = Column(String, nullable=False)           # may be "default" before Style records exist
    room = Column(String)                               # location name; the earliest row per
                                                        # (game,style,room) is that location's image reference
    scene_description = Column(Text)                     # enriched visual prose
    image_url = Column(String)
    image_url_mobile = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)
