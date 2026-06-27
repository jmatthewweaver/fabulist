"""
Game library endpoints: list games, get game details + user's playthroughs.
Also: game ingestion trigger.
"""
import hashlib
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from ..models.db import Game, Playthrough, Style
from ..game.dfrotz import DfrotzAdapter, InfodumpExtractor
from ..ai.world_bible import generate_world_bible, build_vocab_index
from ..config import settings
from ..deps import get_db
from .auth import decode_jwt

router = APIRouter(prefix="/api/games", tags=["games"])


@router.get("")
async def list_games(db: AsyncSession = Depends(get_db)):
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
async def get_game(game_id: str, request: Request, db: AsyncSession = Depends(get_db)):
    token = request.cookies.get("auth_token")
    user_id = decode_jwt(token)["sub"] if token else None
    game = await db.get(Game, game_id)
    if not game:
        raise HTTPException(404, "Game not found")

    playthroughs = []
    if user_id:
        q = await db.execute(
            select(Playthrough)
            .where(Playthrough.user_id == user_id, Playthrough.game_id == game_id)
            .order_by(Playthrough.last_active.desc())
        )
        playthroughs = [
            {
                "id": p.id,
                "current_room": p.current_room,
                "turn_count": p.turn_count,
                "last_active": p.last_active.isoformat() if p.last_active else None,
            }
            for p in q.scalars()
        ]

    styles = await db.execute(select(Style))
    return {
        "id": game.id,
        "title": game.title,
        "description": game.description,
        "icon_image_url": game.icon_image_url,
        "default_style_id": game.default_style_id,
        "playthroughs": playthroughs,
        "available_styles": [
            {"id": s.id, "name": s.name, "description": s.description}
            for s in styles.scalars()
        ],
    }


@router.post("/ingest")
async def ingest_game(filename: str, db: AsyncSession = Depends(get_db)):
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

    extractor = InfodumpExtractor(infodump_path=settings.infodump_path)
    world_data = await extractor.extract(str(game_path))

    adapter = DfrotzAdapter(dfrotz_path=settings.dfrotz_path)
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
    zmachine_exts = {"z1", "z2", "z3", "z4", "z5", "z6", "z7", "z8"}
    return "zmachine" if ext.lstrip(".") in zmachine_exts else {"ulx": "glulx"}.get(ext.lstrip("."), "unknown")
