"""
Session-scoped invention ledger.
Immutable: once an object description is written, it never changes.

Two search strategies:
  - BM25 (pg_search): fast keyword lookup for "do we have an invention for this object?"
  - pgvector cosine similarity: semantic context injection — "what prior inventions are
    relevant to this scene, even if the exact objects aren't mentioned?"
"""
import json
from datetime import datetime

import anthropic
from openai import OpenAI
from sqlalchemy import text, select
from sqlalchemy.ext.asyncio import AsyncSession

from ..config import settings
from ..models.db import Invention

_claude = anthropic.Anthropic(api_key=settings.anthropic_api_key)
_openai = OpenAI(api_key=settings.openai_api_key)

_EXTRACT_SYSTEM = """Extract invented object descriptions from this narrative text.
Return a JSON array of objects with keys: object_key (lowercase, underscored), canonical_text (the description).
Only include concrete named objects that received a specific invented description.
Return [] if nothing was invented. Return only valid JSON."""


def _embed(text_to_embed: str) -> list[float]:
    """Generate a 1024-dim embedding via OpenAI text-embedding-3-small."""
    response = _openai.embeddings.create(
        model="text-embedding-3-small",
        input=text_to_embed,
        dimensions=1024,
    )
    return response.data[0].embedding


# --- Core read/write ---

async def lookup(db: AsyncSession, session_id: str, object_names: list[str]) -> list[dict]:
    """Exact key match for known objects currently in scope."""
    if not object_names:
        return []
    keys = [n.lower().replace(" ", "_") for n in object_names]
    result = await db.execute(
        select(Invention.object_key, Invention.canonical_text).where(
            Invention.session_id == session_id,
            Invention.object_key.in_(keys),
        )
    )
    return [{"object_key": r.object_key, "canonical_text": r.canonical_text} for r in result]


async def write(
    db: AsyncSession, session_id: str, object_key: str, canonical_text: str, turn: int
) -> bool:
    """Write a new invention. No-op if one already exists (immutability). Returns True if written."""
    key = object_key.lower().replace(" ", "_")
    existing = await db.execute(
        select(Invention).where(Invention.session_id == session_id, Invention.object_key == key)
    )
    if existing.scalar_one_or_none():
        return False

    embedding = _embed(f"{key.replace('_', ' ')} — {canonical_text}")
    inv = Invention(
        session_id=session_id,
        object_key=key,
        canonical_text=canonical_text,
        embedding=embedding,
        source_turn=turn,
        created_at=datetime.utcnow(),
    )
    db.add(inv)
    await db.flush()
    return True


# --- BM25 search (pg_textsearch) ---

async def bm25_search(db: AsyncSession, session_id: str, query: str, limit: int = 8) -> list[dict]:
    """
    Keyword relevance search via BM25.
    Uses the pg_textsearch <@> operator against the inventions_bm25_idx index.
    """
    result = await db.execute(
        text(
            "SELECT object_key, canonical_text "
            "FROM inventions "
            "WHERE session_id = :sid "
            "ORDER BY full_text <@> to_bm25query(:query, 'inventions_bm25_idx') "
            "LIMIT :limit"
        ),
        {"sid": session_id, "query": query, "limit": limit},
    )
    return [{"object_key": r.object_key, "canonical_text": r.canonical_text} for r in result]


# --- Vector similarity search (pgvector) ---

async def semantic_context(
    db: AsyncSession, session_id: str, scene_description: str, limit: int = 5
) -> list[dict]:
    """
    Find prior inventions most semantically relevant to the current scene.
    Used to inject helpful context into the enrichment prompt even for objects
    not explicitly in scope — e.g., a fireplace described 10 rooms ago surfacing
    when the player enters a warm, smoky corridor.
    """
    query_embedding = _embed(scene_description)
    emb_str = "[" + ",".join(str(x) for x in query_embedding) + "]"
    result = await db.execute(
        text(
            "SELECT object_key, canonical_text, "
            f"1 - (embedding <=> '{emb_str}'::vector) AS similarity "
            "FROM inventions "
            "WHERE session_id = :sid AND embedding IS NOT NULL "
            f"ORDER BY embedding <=> '{emb_str}'::vector "
            "LIMIT :limit"
        ),
        {"sid": session_id, "limit": limit},
    )
    return [
        {"object_key": r.object_key, "canonical_text": r.canonical_text, "similarity": r.similarity}
        for r in result
        if r.similarity > 0.75  # only inject genuinely relevant results
    ]


# --- Post-enrichment extraction ---

async def extract_and_store(
    db: AsyncSession,
    session_id: str,
    narrative: str,
    turn: int,
) -> list[str]:
    """After enrichment, extract newly invented details and persist them (with embeddings)."""
    response = _claude.messages.create(
        model=settings.model_translation,
        max_tokens=512,
        system=_EXTRACT_SYSTEM,
        messages=[{"role": "user", "content": f"Narrative:\n{narrative}"}],
    )
    raw = response.content[0].text.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1].rsplit("```", 1)[0]

    try:
        items = json.loads(raw)
    except json.JSONDecodeError:
        return []

    written = []
    for item in items:
        key = item.get("object_key", "").strip()
        text_val = item.get("canonical_text", "").strip()
        if key and text_val:
            stored = await write(db, session_id, key, text_val, turn)
            if stored:
                written.append(key)
    if written:
        await db.commit()
    return written


# --- Save/restore snapshots ---

async def export_snapshot(db: AsyncSession, session_id: str) -> list[dict]:
    result = await db.execute(
        select(Invention.object_key, Invention.canonical_text, Invention.source_turn).where(
            Invention.session_id == session_id
        )
    )
    return [dict(r._mapping) for r in result]


async def import_snapshot(db: AsyncSession, session_id: str, snapshot: list[dict]) -> None:
    for item in snapshot:
        await write(db, session_id, item["object_key"], item["canonical_text"], item["source_turn"])
    await db.commit()
