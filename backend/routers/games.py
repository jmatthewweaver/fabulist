"""
Game library endpoints: list games, get game details + user's saves.
Also: game ingestion trigger.
"""
import hashlib
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from ..models.db import Game, Save, Session, Style
from ..game.dfrotz import DfrotzAdapter, InfodumpExtractor
from ..ai.world_bible import generate_world_bible, build_vocab_index
from ..config import settings

router = APIRouter(prefix="/api/games", tags=["games"])


@router.get("")
async def list_games(db: AsyncSession = Depends(lambda: None)):  # dep injected in main
    games = await db.execute(select(Game).order_by(Game.title))
    return [
        {
            "id": g.id,
            "title": g.title,
            "format": g.format,
            "description": g.description,
            "icon_image_url": g.icon_image_url,
            "default_style_id": g.default_style_id,
        }
        for g in games.scalars()
    ]


@router.get("/{game_id}")
async def get_game(game_id: str, user_id: str, db: AsyncSession = Depends(lambda: None)):
    game = await db.get(Game, game_id)
    if not game:
        raise HTTPException(404, "Game not found")

    # User's saves for this game, most recent first
    sessions_q = await db.execute(
        select(Session).where(Session.user_id == user_id, Session.game_id == game_id)
    )
    sessions = sessions_q.scalars().all()
    session_ids = [s.id for s in sessions]

    saves = []
    if session_ids:
        saves_q = await db.execute(
            select(Save)
            .where(Save.session_id.in_(session_ids))
            .order_by(Save.created_at.desc())
            .limit(20)
        )
        saves = [
            {
                "id": s.id,
                "session_id": s.session_id,
                "name": s.name,
                "room_name": s.room_name,
                "turn_count": s.turn_count,
                "created_at": s.created_at.isoformat(),
            }
            for s in saves_q.scalars()
        ]

    styles = await db.execute(select(Style))
    return {
        "id": game.id,
        "title": game.title,
        "description": game.description,
        "icon_image_url": game.icon_image_url,
        "default_style_id": game.default_style_id,
        "saves": saves,
        "available_styles": [
            {"id": s.id, "name": s.name, "description": s.description}
            for s in styles.scalars()
        ],
    }


@router.post("/ingest")
async def ingest_game(filename: str, db: AsyncSession = Depends(lambda: None)):
    """
    Trigger one-time ingestion for a game file in the games/ directory.
    Idempotent: re-running updates metadata but doesn't regenerate the world bible.
    """
    game_path = settings.games_dir / filename
    if not game_path.exists():
        raise HTTPException(404, f"Game file not found: {filename}")

    game_id = hashlib.sha256(game_path.read_bytes()).hexdigest()[:16]
    existing = await db.get(Game, game_id)
    if existing and existing.world_bible:
        return {"id": game_id, "status": "already_ingested"}

    extractor = InfodumpExtractor()
    world_data = await extractor.extract(str(game_path))

    # Run the game briefly to get opening text
    adapter = DfrotzAdapter()
    await adapter.start(str(game_path), f"ingest_{game_id}")
    opening_result = await adapter.step(f"ingest_{game_id}", "look")
    await adapter.stop(f"ingest_{game_id}")
    opening_text = opening_result.raw_text

    world_bible_dict = await generate_world_bible(world_data, opening_text, game_path.stem)
    vocab_index = build_vocab_index(world_data)

    import json
    game = existing or Game(id=game_id)
    game.title = world_bible_dict.get("title", game_path.stem)
    game.filename = filename
    game.format = world_data.game_format or _detect_format(filename)
    game.description = world_bible_dict.get("setting", "")
    game.world_bible = json.dumps(world_bible_dict)
    game.vocab_index = json.dumps(vocab_index)
    game.ingested_at = __import__("datetime").datetime.utcnow()

    if not existing:
        db.add(game)
    await db.commit()
    return {"id": game_id, "title": game.title, "status": "ingested"}


def _detect_format(filename: str) -> str:
    ext = Path(filename).suffix.lower()
    return {"z5": "zmachine", "z8": "zmachine", "ulx": "glulx"}.get(ext.lstrip("."), "unknown")
