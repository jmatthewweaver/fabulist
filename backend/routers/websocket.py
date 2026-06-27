"""
WebSocket game loop.

Each command spins up a fresh dfrotz process: restore → command → save → kill.
All durable state lives in the Playthrough DB record between turns.

Message protocol (client → server):
  {"type": "command", "text": "..."}
  {"type": "request_image"}

Message protocol (server → client):
  {"type": "narrative_chunk", "text": "..."}
  {"type": "narrative_done"}
  {"type": "game_state", "room": "...", "inventory": [...], "turn": N}
  {"type": "image_ready", "url": "...", "subject": "..."}
  {"type": "error", "message": "..."}
"""
import asyncio
import json
from datetime import datetime

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from ..deps import AsyncSessionLocal
from ..ai import invention_ledger
from ..ai.command_translator import translate
from ..ai.context_manager import ContextManager, Turn
from ..ai.enricher import enrich_stream, extract_image_suggestion
from ..config import settings
from ..models.db import Playthrough, Game
from ..game.dfrotz import run_one_turn

router = APIRouter()


def _extract_new_room(raw_text: str) -> str | None:
    """
    If raw output leads with a room title header, return it.
    Room names are short, start with a capital, and don't end with sentence punctuation.
    """
    first_line = raw_text.strip().split("\n")[0].strip()
    if (first_line
            and len(first_line) < 60
            and first_line[0].isupper()
            and not first_line.endswith((".", "!", "?", ","))):
        return first_line
    return None


async def _generate_and_push(
    websocket: WebSocket,
    suggestion: dict,
    current_room: str,
    playthrough_id: str,
    style_prefix: str,
):
    try:
        from ..media.image_generator import generate_scene_image, make_cache_key
        cache_key = make_cache_key(playthrough_id, current_room, [], "default")
        url = await generate_scene_image(
            scene_prompt=suggestion.get("prompt_hint", current_room),
            style_prefix=style_prefix,
            style_negative="",
            reference_image_urls=[],
            cache_key=cache_key,
        )
        await websocket.send_json({"type": "image_ready", "url": url, "subject": suggestion.get("subject", "")})
    except Exception:
        pass


@router.websocket("/api/playthroughs/{playthrough_id}/play")
async def play(websocket: WebSocket, playthrough_id: str):
    await websocket.accept()

    async with AsyncSessionLocal() as db:
        playthrough: Playthrough = await db.get(Playthrough, playthrough_id)
        if not playthrough:
            await websocket.send_json({"type": "error", "message": "Playthrough not found."})
            await websocket.close()
            return

        game: Game = await db.get(Game, playthrough.game_id)
        game_path = str(settings.games_dir / game.filename)
        context = ContextManager.from_json(playthrough.context_json)

        world_bible: dict = game.world_bible if isinstance(game.world_bible, dict) else json.loads(game.world_bible or "{}")
        vocab_index: dict = game.vocab_index if isinstance(game.vocab_index, dict) else json.loads(game.vocab_index or "{}")
        vocab_verbs: list = world_bible.get("vocab_verbs", [])
        vocab_nouns: list = world_bible.get("vocab_nouns", [])
        style_prefix: str = ""  # TODO: load from Style record

        last_image_turn: int = -(settings.image_cooldown_turns)

        # First connect: run "look" on a fresh process with no prior save
        if (playthrough.turn_count or 0) == 0:
            try:
                opening, initial_save = await run_one_turn(
                    game_path, "look", None, settings.dfrotz_path
                )
            except Exception as e:
                await websocket.send_json({"type": "error", "message": f"Failed to start game: {e}"})
                await websocket.close()
                return

            opening_room = _extract_new_room(opening.raw_text) or ""
            playthrough.engine_save = initial_save
            playthrough.current_room = opening_room
            playthrough.context_json = context.to_json()
            playthrough.last_active = datetime.utcnow()
            await db.commit()

            bundle = context.build_bundle(current_room=opening_room, current_inventory=[], relevant_inventions=[])
            async for chunk in enrich_stream(opening.raw_text, bundle):
                await websocket.send_json({"type": "narrative_chunk", "text": chunk})
            await websocket.send_json({"type": "narrative_done"})
            await websocket.send_json({"type": "game_state", "room": opening_room, "inventory": [], "turn": 0})
            asyncio.create_task(_generate_and_push(
                websocket,
                {"suggest": True, "type": "room_wide", "subject": opening_room,
                 "prompt_hint": f"{opening_room} interior, opening scene"},
                opening_room, playthrough_id, style_prefix,
            ))

        try:
            while True:
                data = await websocket.receive_json()
                msg_type = data.get("type")

                if msg_type == "request_image":
                    room = playthrough.current_room or ""
                    asyncio.create_task(_generate_and_push(
                        websocket,
                        {"suggest": True, "type": "room_wide", "subject": room,
                         "prompt_hint": f"{room} interior"},
                        room, playthrough_id, style_prefix,
                    ))
                    continue

                if msg_type != "command":
                    continue

                user_input: str = data["text"].strip()
                if not user_input:
                    continue

                current_room = playthrough.current_room or "Unknown"
                visible_objects: list[str] = []

                # 1. Translate natural language → game command via dfrotz
                # step_fn runs one turn; captures save bytes on success
                latest_save = playthrough.engine_save

                async def step_fn(cmd: str):
                    nonlocal latest_save
                    result, new_save = await run_one_turn(
                        game_path, cmd, playthrough.engine_save, settings.dfrotz_path
                    )
                    if not result.rejected:
                        latest_save = new_save
                    return result

                try:
                    command, raw_output = await translate(
                        user_input=user_input,
                        room=current_room,
                        visible_objects=visible_objects,
                        vocab_verbs=vocab_verbs,
                        vocab_nouns=vocab_nouns,
                        vocab_index=vocab_index,
                        step_fn=step_fn,
                    )
                except ValueError as e:
                    await websocket.send_json({"type": "error", "message": str(e)})
                    continue

                # 2. Fetch relevant inventions
                exact_inventions = await invention_ledger.lookup(db, playthrough_id, visible_objects)
                scene_desc = f"{current_room}. {raw_output[:300]}"
                semantic_inventions = await invention_ledger.semantic_context(db, playthrough_id, scene_desc)
                seen = {i["object_key"] for i in exact_inventions}
                inventions = exact_inventions + [i for i in semantic_inventions if i["object_key"] not in seen]

                # 3. Stream enriched narrative
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

                # 4. Detect room change and update playthrough
                new_room = _extract_new_room(raw_output)
                is_new_room = new_room is not None and new_room != current_room
                if new_room:
                    current_room = new_room

                turn_num = (playthrough.turn_count or 0) + 1
                playthrough.turn_count = turn_num
                playthrough.engine_save = latest_save
                playthrough.current_room = current_room
                playthrough.last_active = datetime.utcnow()

                context.add_turn(Turn(
                    turn_num=turn_num,
                    user_input=user_input,
                    raw_game_output=raw_output,
                    enriched_narrative=narrative_text,
                    room=current_room,
                ))
                playthrough.context_json = context.to_json()
                await db.commit()

                # 5. Async: extract inventions + maybe generate image
                asyncio.create_task(
                    invention_ledger.extract_and_store(db, playthrough_id, narrative_text, turn_num)
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
                            await _generate_and_push(websocket, suggestion, current_room, playthrough_id, style_prefix)
                    asyncio.create_task(_maybe_image())
                    last_image_turn = turn_num

                # 6. Send updated game state
                await websocket.send_json({
                    "type": "game_state",
                    "room": current_room,
                    "inventory": [],
                    "turn": turn_num,
                })

        except WebSocketDisconnect:
            pass
