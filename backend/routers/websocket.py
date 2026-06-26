"""
WebSocket game loop. One connection per active session.

Message protocol (client → server):
  {"type": "command", "text": "..."}
  {"type": "request_image"}          # explicit 📷 request, bypasses cooldown

Message protocol (server → client):
  {"type": "narrative_chunk", "text": "..."}   # streaming
  {"type": "narrative_done"}
  {"type": "game_state", "room": "...", "inventory": [...], "turn": N}
  {"type": "image_ready", "url": "...", "subject": "..."}
  {"type": "error", "message": "..."}
"""
import asyncio
import json

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from sqlalchemy.ext.asyncio import AsyncSession

from ..ai import invention_ledger
from ..ai.command_translator import translate
from ..ai.context_manager import ContextManager, Turn
from ..ai.enricher import enrich_stream, extract_image_suggestion
from ..config import settings
from ..game.session_store import session_store
from ..models.db import Session as DbSession, Game

router = APIRouter()


async def _generate_and_push(
    websocket: WebSocket,
    suggestion: dict,
    current_room: str,
    session_id: str,
    style_prefix: str,
) -> None:
    """Fire image generation and push result to client when ready."""
    from ..media.image_generator import build_scene_prompt, generate_scene_image, make_cache_key

    prompt = suggestion.get("prompt_hint") or f"{suggestion.get('subject', current_room)}"
    cache_key = make_cache_key(session_id, current_room, suggestion.get("subject", ""), style_prefix[:16])

    try:
        url = await generate_scene_image(
            scene_prompt=prompt,
            style_prefix=style_prefix,
            style_negative="modern, anachronistic, text, UI elements, watermark",
            reference_image_urls=[],  # TODO: pass style seed + prior room images
            cache_key=cache_key,
        )
        await websocket.send_json({
            "type": "image_ready",
            "url": url,
            "subject": suggestion.get("subject", current_room),
            "image_type": suggestion.get("type", "room_wide"),
        })
    except Exception as e:
        # Image failure is non-fatal — just don't send image_ready
        pass


@router.websocket("/api/sessions/{session_id}/play")
async def play(websocket: WebSocket, session_id: str, db: AsyncSession):
    await websocket.accept()

    active = session_store.get(session_id)
    if not active:
        await websocket.send_json({"type": "error", "message": "Session not running. Try resuming from a save."})
        await websocket.close()
        return

    db_session: DbSession = await db.get(DbSession, session_id)
    game: Game = await db.get(Game, db_session.game_id)
    context = ContextManager.from_json(db_session.context_json)
    vocab_index: dict = game.vocab_index or {}
    world_bible: dict = game.world_bible or {}
    vocab_verbs: list = world_bible.get("vocab_verbs", [])
    vocab_nouns: list = world_bible.get("vocab_nouns", [])
    style_prefix: str = ""  # TODO: load from Style record

    last_image_turn: int = -(settings.image_cooldown_turns)  # allow image on first turn
    last_room: str = db_session.current_room or ""

    try:
        while True:
            data = await websocket.receive_json()
            msg_type = data.get("type")

            # Explicit image request — bypasses cooldown
            if msg_type == "request_image":
                suggestion = {
                    "suggest": True,
                    "type": "room_wide",
                    "subject": last_room,
                    "prompt_hint": f"{last_room} interior",
                }
                asyncio.create_task(_generate_and_push(websocket, suggestion, last_room, session_id, style_prefix))
                continue

            if msg_type != "command":
                continue

            user_input: str = data["text"].strip()
            if not user_input:
                continue

            current_room = db_session.current_room or "Unknown"
            visible_objects: list[str] = []  # TODO: parse from last game output

            # 1. Translate user input → game command
            try:
                command, raw_output = await translate(
                    user_input=user_input,
                    room=current_room,
                    visible_objects=visible_objects,
                    vocab_verbs=vocab_verbs,
                    vocab_nouns=vocab_nouns,
                    vocab_index=vocab_index,
                    step_fn=lambda cmd: active.adapter.step(session_id, cmd),
                )
            except ValueError as e:
                await websocket.send_json({"type": "error", "message": str(e)})
                continue

            # 2. Fetch relevant inventions (exact + semantic)
            exact_inventions = await invention_ledger.lookup(db, session_id, visible_objects)
            scene_desc = f"{current_room}. {raw_output[:300]}"
            semantic_inventions = await invention_ledger.semantic_context(db, session_id, scene_desc)
            seen = {i["object_key"] for i in exact_inventions}
            inventions = exact_inventions + [i for i in semantic_inventions if i["object_key"] not in seen]

            # 3. Build context bundle and stream enriched narrative
            bundle = context.build_bundle(
                current_room=current_room,
                current_inventory=[],
                relevant_inventions=inventions,
            )

            full_narrative: list[str] = []
            async for chunk in enrich_stream(raw_output, bundle):
                await websocket.send_json({"type": "narrative_chunk", "text": chunk})
                full_narrative.append(chunk)
            await websocket.send_json({"type": "narrative_done"})
            narrative_text = "".join(full_narrative)

            # 4. Update turn count and context
            db_session.turn_count = (db_session.turn_count or 0) + 1
            turn_num = db_session.turn_count
            is_new_room = current_room != last_room
            last_room = current_room

            context.add_turn(Turn(
                turn_num=turn_num,
                user_input=user_input,
                raw_game_output=raw_output,
                enriched_narrative=narrative_text,
                room=current_room,
            ))
            db_session.context_json = context.to_json()
            await db.commit()

            # 5. Async tasks: extract inventions + maybe generate image
            asyncio.create_task(
                invention_ledger.extract_and_store(db, session_id, narrative_text, turn_num)
            )

            turns_since_image = turn_num - last_image_turn
            if turns_since_image >= settings.image_cooldown_turns:
                async def _maybe_image():
                    suggestion = await extract_image_suggestion(
                        narrative=narrative_text,
                        raw_output=raw_output,
                        current_room=current_room,
                        is_new_room=is_new_room,
                    )
                    if suggestion:
                        await _generate_and_push(websocket, suggestion, current_room, session_id, style_prefix)
                asyncio.create_task(_maybe_image())
                last_image_turn = turn_num
            # If still in cooldown, no image this turn (explicit request_image always bypasses)

            # 6. Send updated game state
            await websocket.send_json({
                "type": "game_state",
                "room": current_room,
                "inventory": [],
                "turn": turn_num,
            })

    except WebSocketDisconnect:
        pass
