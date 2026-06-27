"""
Session lifecycle: create, get, save, restore, end.
"""
import json
import uuid
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Cookie, Request
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from ..models.db import Session as DbSession, Save, Game, Style
from ..game.dfrotz import DfrotzAdapter
from ..game.session_store import session_store, ActiveSession
from ..ai.context_manager import ContextManager
from ..config import settings
from ..deps import get_db
from .auth import decode_jwt


def _get_user_id(request: Request) -> str:
    token = request.cookies.get("auth_token")
    if not token:
        raise HTTPException(401, "Not authenticated")
    try:
        return decode_jwt(token)["sub"]
    except Exception:
        raise HTTPException(401, "Invalid token")


class StartSessionRequest(BaseModel):
    game_id: str
    style_id: str | None = None

router = APIRouter(prefix="/api/sessions", tags=["sessions"])


@router.post("")
async def create_session(
    body: StartSessionRequest,
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    game_id = body.game_id
    style_id = body.style_id or "default"
    user_id = _get_user_id(request)
    game = await db.get(Game, game_id)
    if not game:
        raise HTTPException(404, "Game not found")

    session_id = str(uuid.uuid4())
    context = ContextManager(world_bible=game.world_bible)

    db_session = DbSession(
        id=session_id,
        user_id=user_id,
        game_id=game_id,
        style_id=style_id,
        context_json=context.to_json(),
    )
    db.add(db_session)
    await db.commit()

    # Start the game engine
    game_path = str(settings.games_dir / game.filename)
    adapter = DfrotzAdapter(dfrotz_path=settings.dfrotz_path)
    await adapter.start(game_path, session_id)
    session_store.put(ActiveSession(session_id=session_id, game_path=game_path, adapter=adapter))

    return {"session_id": session_id}


@router.get("/{session_id}")
async def get_session(session_id: str, db: AsyncSession = Depends(get_db)):
    s = await db.get(DbSession, session_id)
    if not s:
        raise HTTPException(404)
    return {
        "id": s.id,
        "game_id": s.game_id,
        "style_id": s.style_id,
        "current_room": s.current_room,
        "turn_count": s.turn_count,
        "last_active": s.last_active.isoformat(),
    }


@router.post("/{session_id}/save")
async def save_game(session_id: str, name: str | None, db: AsyncSession = Depends(get_db)):
    active = session_store.get(session_id)
    if not active:
        raise HTTPException(400, "Session not running")

    db_session = await db.get(DbSession, session_id)
    engine_bytes = await active.adapter.save(session_id)

    # Snapshot inventions
    from ..ai import invention_ledger
    inv_snapshot = await invention_ledger.export_snapshot(db, session_id)

    save_name = name or f"Turn {db_session.turn_count} — {db_session.current_room or 'Unknown'}"
    save = Save(
        id=str(uuid.uuid4()),
        session_id=session_id,
        name=save_name,
        engine_save=engine_bytes,
        context_json=db_session.context_json,
        inventions_json=inv_snapshot,
        turn_count=db_session.turn_count,
        room_name=db_session.current_room,
    )
    db.add(save)
    await db.commit()
    return {"save_id": save.id, "name": save_name}


@router.post("/{session_id}/restore/{save_id}")
async def restore_save(session_id: str, save_id: str, db: AsyncSession = Depends(get_db)):
    save = await db.get(Save, save_id)
    if not save or save.session_id != session_id:
        raise HTTPException(404)

    active = session_store.get(session_id)
    db_session = await db.get(DbSession, session_id)

    if not active:
        # Engine not running — restart from save
        game = await db.get(Game, db_session.game_id)
        game_path = str(settings.games_dir / game.filename)
        adapter = DfrotzAdapter(dfrotz_path=settings.dfrotz_path)
        await adapter.start(game_path, session_id)
        session_store.put(ActiveSession(session_id=session_id, game_path=game_path, adapter=adapter))
        active = session_store.get(session_id)

    await active.adapter.restore(session_id, save.engine_save)

    # Restore context and inventions
    from ..ai import invention_ledger
    db_session.context_json = save.context_json
    db_session.turn_count = save.turn_count
    db_session.current_room = save.room_name
    await invention_ledger.import_snapshot(db, session_id, save.inventions_json)
    await db.commit()
    return {"ok": True}


@router.delete("/{session_id}")
async def end_session(session_id: str, db: AsyncSession = Depends(get_db)):
    await session_store.remove(session_id)
    db_session = await db.get(DbSession, session_id)
    if db_session:
        db_session.ended_at = datetime.utcnow()
        await db.commit()
    return {"ok": True}
