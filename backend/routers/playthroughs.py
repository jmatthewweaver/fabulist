"""
Playthrough lifecycle: create and delete.
A Playthrough is the durable record the URL refers to; sessions are ephemeral.
"""
import uuid

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel
from sqlalchemy import delete as sa_delete
from sqlalchemy.ext.asyncio import AsyncSession

from ..models.db import Playthrough, Game, Invention
from ..ai.context_manager import ContextManager
from ..deps import get_db
from .auth import decode_jwt

router = APIRouter(prefix="/api/playthroughs", tags=["playthroughs"])


def _get_user_id(request: Request) -> str:
    token = request.cookies.get("auth_token")
    if not token:
        raise HTTPException(401, "Not authenticated")
    try:
        return decode_jwt(token)["sub"]
    except Exception:
        raise HTTPException(401, "Invalid token")


class StartRequest(BaseModel):
    game_id: str
    style_id: str | None = None


@router.post("")
async def create_playthrough(
    body: StartRequest,
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    user_id = _get_user_id(request)
    game = await db.get(Game, body.game_id)
    if not game:
        raise HTTPException(404, "Game not found")

    context = ContextManager(world_bible=game.world_bible)
    playthrough = Playthrough(
        id=str(uuid.uuid4()),
        user_id=user_id,
        game_id=body.game_id,
        style_id=body.style_id or game.default_style_id or "default",
        context_json=context.to_json(),
    )
    db.add(playthrough)
    await db.commit()
    return {"id": playthrough.id}


@router.get("/{playthrough_id}")
async def get_playthrough(playthrough_id: str, db: AsyncSession = Depends(get_db)):
    p = await db.get(Playthrough, playthrough_id)
    if not p:
        raise HTTPException(404)
    return {
        "id": p.id,
        "game_id": p.game_id,
        "current_room": p.current_room,
        "turn_count": p.turn_count,
        "last_active": p.last_active.isoformat() if p.last_active else None,
    }


@router.delete("/{playthrough_id}")
async def delete_playthrough(playthrough_id: str, db: AsyncSession = Depends(get_db)):
    # Remove dependent inventions first (FK is NOT NULL, so they'd block the delete).
    await db.execute(sa_delete(Invention).where(Invention.playthrough_id == playthrough_id))
    p = await db.get(Playthrough, playthrough_id)
    if p:
        await db.delete(p)
    await db.commit()
    return {"ok": True}
