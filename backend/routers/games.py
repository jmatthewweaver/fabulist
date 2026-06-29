"""
Game library endpoints: list games, get game details + user's playthroughs.
Also: game ingestion trigger.
"""
import hashlib
import json
import logging
from datetime import datetime
from pathlib import Path

log = logging.getLogger(__name__)

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from ..models.db import Game, Playthrough, Style
from ..game.dfrotz import InfodumpExtractor, run_one_turn
from ..game.zmachine import scenery_by_id
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
    try:
        user_id = decode_jwt(token)["sub"] if token else None
    except Exception:
        user_id = None  # expired/invalid cookie → treat as anonymous, don't 500 the page
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

    opening_result, _ = await run_one_turn(
        str(game_path), "look", None, settings.dfrotz_path, persist=False
    )
    opening_text = opening_result.raw_text

    # Object containment tree (id-keyed) parsed from infodump. This is the only
    # ingestion-time description work: it tells the runtime which objects are present
    # in a room and worth EXAMINE-ing. All actual descriptions/images are produced at
    # play time from the live game state and cached by output hash (see websocket.py).
    known_objects = world_data.object_tree or {}
    nodes = known_objects.get("nodes", {})
    log.info("Object tree: %d nodes from %s", len(nodes), filename)

    # Link each room to its scenery globals (white house, forest, ...). Pure structure
    # like the tree — it tells the runtime to EXAMINE these visible globals (which are not
    # direct children) when building a scene, so e.g. the white house's full description
    # makes it in. No pre-baked descriptions here.
    try:
        global_ids = _collect_global_ids(known_objects)
        for room_id, scenery_ids in scenery_by_id(str(game_path), global_ids).items():
            node = nodes.get(str(room_id))
            if node:
                node["scenery"] = scenery_ids
    except Exception:
        log.warning("scenery linkage failed for %s", filename, exc_info=True)

    world_bible_dict = await generate_world_bible(world_data, opening_text, game_path.stem)
    world_bible_dict["known_objects"] = known_objects
    # Carry the parser vocabulary through to the world bible — the command translator reads
    # world_bible["vocab_verbs"/"vocab_nouns"] at play time. generate_world_bible only returns
    # the LLM's prose JSON, so these must be attached here or they arrive empty.
    world_bible_dict["vocab_verbs"] = world_data.vocab_verbs
    world_bible_dict["vocab_nouns"] = world_data.vocab_nouns
    log.info("Vocab: %d verbs, %d dictionary words from %s",
             len(world_data.vocab_verbs), len(world_data.vocab_nouns), filename)
    vocab_index = build_vocab_index(world_data)

    game = existing or Game(id=game_id)
    game.title = world_bible_dict.get("title", game_path.stem)
    game.filename = filename
    game.format = world_data.game_format or _detect_format(filename)
    game.description = world_bible_dict.get("setting", "")
    game.world_bible = json.dumps(world_bible_dict)
    game.vocab_index = json.dumps(vocab_index)
    game.ingested_at = datetime.utcnow()

    if not existing:
        db.add(game)
    await db.commit()
    return {"id": game_id, "title": game.title, "status": "ingested"}


def _collect_global_ids(tree: dict) -> set[int]:
    """All object ids under a 'Global Objects' container — used by scenery_by_id to
    detect each room's scenery property (room exits point to rooms; scenery to globals)."""
    nodes = tree.get("nodes", {})
    global_ids: set[int] = set()
    stack = [n["id"] for n in nodes.values()
             if n["kind"] == "container" and n["name"] == "Global Objects"]
    while stack:
        node = nodes.get(str(stack.pop()))
        if not node:
            continue
        for child_id in node["children"]:
            if child_id not in global_ids:
                global_ids.add(child_id)
                stack.append(child_id)
    return global_ids


def _detect_format(filename: str) -> str:
    ext = Path(filename).suffix.lower()
    zmachine_exts = {"z1", "z2", "z3", "z4", "z5", "z6", "z7", "z8"}
    return "zmachine" if ext.lstrip(".") in zmachine_exts else {"ulx": "glulx"}.get(ext.lstrip("."), "unknown")
