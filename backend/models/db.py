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
    id = Column(String, primary_key=True)           # Google sub claim
    email = Column(String, unique=True, nullable=False)
    display_name = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)

    sessions = relationship("Session", back_populates="user")


class Game(Base):
    """One row per ingested game file."""
    __tablename__ = "games"
    id = Column(String, primary_key=True)           # sha256 of game file
    title = Column(String, nullable=False)
    filename = Column(String, nullable=False)
    format = Column(String, nullable=False)         # "zmachine", "glulx", etc.
    description = Column(Text)
    default_style_id = Column(String, ForeignKey("styles.id"))
    world_bible = Column(JSONB)                     # structured dict from ingestion
    vocab_index = Column(JSONB)                     # noun → object_name map
    icon_image_url = Column(String)
    ingested_at = Column(DateTime)

    sessions = relationship("Session", back_populates="game")
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


class Session(Base):
    """One active playthrough per user per game."""
    __tablename__ = "sessions"
    id = Column(String, primary_key=True)
    user_id = Column(String, ForeignKey("users.id"), nullable=False)
    game_id = Column(String, ForeignKey("games.id"), nullable=False)
    style_id = Column(String, ForeignKey("styles.id"), nullable=True)
    current_room = Column(String)
    turn_count = Column(Integer, default=0)
    context_json = Column(JSONB)                    # serialized ContextManager state
    started_at = Column(DateTime, default=datetime.utcnow)
    last_active = Column(DateTime, default=datetime.utcnow)
    ended_at = Column(DateTime)

    user = relationship("User", back_populates="sessions")
    game = relationship("Game", back_populates="sessions")
    style = relationship("Style")
    saves = relationship("Save", back_populates="session", order_by="Save.created_at.desc()")
    inventions = relationship("Invention", back_populates="session")


class Save(Base):
    """Named save snapshot within a session."""
    __tablename__ = "saves"
    id = Column(String, primary_key=True)
    session_id = Column(String, ForeignKey("sessions.id"), nullable=False)
    name = Column(String, nullable=False)
    engine_save = Column(BYTEA, nullable=False)
    context_json = Column(JSONB, nullable=False)
    inventions_json = Column(JSONB, nullable=False)
    turn_count = Column(Integer)
    room_name = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)

    session = relationship("Session", back_populates="saves")


class Invention(Base):
    """Invented detail for a named object, scoped to a session. Immutable once written."""
    __tablename__ = "inventions"
    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String, ForeignKey("sessions.id"), nullable=False)
    object_key = Column(String, nullable=False)     # normalized lowercase, e.g. "wooden_cup"
    canonical_text = Column(Text, nullable=False)
    # full_text is a generated column (set via DDL in migrations.py) used for BM25 indexing
    full_text = Column(Text)
    embedding = Column(Vector(1024))                # pgvector: semantic similarity search
    source_turn = Column(Integer)
    created_at = Column(DateTime, default=datetime.utcnow)

    session = relationship("Session", back_populates="inventions")

    __table_args__ = (
        UniqueConstraint("session_id", "object_key", name="uq_invention_session_object"),
        # BM25 and HNSW indexes created via raw DDL in migrations.py
        Index("ix_inventions_embedding", "embedding", postgresql_using="hnsw",
              postgresql_with={"m": "16", "ef_construction": "64"},
              postgresql_ops={"embedding": "vector_cosine_ops"}),
    )


class CachedImage(Base):
    """Shared image cache. Key: game + location + visible objects + style."""
    __tablename__ = "cached_images"
    cache_key = Column(String, primary_key=True)
    game_id = Column(String, ForeignKey("games.id"), nullable=False)
    style_id = Column(String, ForeignKey("styles.id"), nullable=False)
    location_id = Column(String, nullable=False)
    image_url = Column(String, nullable=False)
    image_url_mobile = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)
