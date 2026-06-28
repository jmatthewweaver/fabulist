"""
WebSocket game loop.

Each command spins up a fresh dfrotz process: restore → command → save → kill.
All durable state lives in the Playthrough DB record between turns.

After each turn we take a NON-PERTURBING look at the current surroundings (restore →
look + examine direct children → discard, never save) and render that scene. The scene's
own deterministic game output is hashed into a cache key, so a given state renders once and
is reused forever, shared across playthroughs.

Message protocol (client → server):
  {"type": "command", "text": "..."}
  {"type": "request_image"}

Message protocol (server → client):
  {"type": "narrative_chunk", "text": "..."}
  {"type": "narrative_done"}
  {"type": "game_state", "room": "...", "inventory": [...], "turn": N}
  {"type": "scene_description", "room": "...", "description": "..."}   # location text (fast)
  {"type": "image_ready", "url": "...", "subject": "..."}              # location image (slow)
  {"type": "error", "message": "..."}
"""
import asyncio
import base64
import hashlib
import json
import logging
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from sqlalchemy import select

log = logging.getLogger(__name__)

from ..deps import AsyncSessionLocal
from ..ai.command_translator import translate
from ..ai.context_manager import ContextManager, Turn
from ..ai.enricher import describe_scene, describe_edit
from ..ai.visual_continuity import augment_prompt, analyze_image
from ..config import settings
from ..models.db import Playthrough, Game, CachedScene, VisualGuide
from ..game.dfrotz import run_one_turn, observe_scene
from ..media.image_generator import make_cache_key, generate_scene_image

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


def _room_examine_targets(world_bible: dict, room_name: str) -> list[str]:
    """
    Names to EXAMINE for the room: its direct children (mailbox, door) AND its scenery
    globals (white house, forest) — the latter aren't children but are visible and carry
    their own descriptions (e.g. the white house's "beautiful colonial house" text).
    """
    ko = world_bible.get("known_objects") or {}
    nodes = ko.get("nodes") or {}
    name_index = ko.get("name_index") or {}
    ids = name_index.get(room_name.strip().lower(), [])
    if not ids:
        return []
    node = nodes.get(str(ids[0]))
    if not node:
        return []
    names: list[str] = []
    seen: set[str] = set()
    for cid in list(node.get("children", [])) + list(node.get("scenery", [])):
        n = nodes.get(str(cid))
        if n and n["name"] and n["name"].lower() not in seen:
            names.append(n["name"])
            seen.add(n["name"].lower())
    return names


async def _render_scene(
    websocket: WebSocket,
    game_id: str,
    scene_key: str,
    scene_output: str,
    room: str,
    style_id: str,
    style_prefix: str,
    world_bible: dict,
):
    """
    Render a scene (description + image) for an already-observed scene_output, caching both
    by scene_key (the hash of the game's own output). Cache hit → reuse; miss → generate
    once and store forever. Runs as a background task.
    """
    try:
        async with AsyncSessionLocal() as db:
            cached = None if settings.force_regen else await db.get(CachedScene, scene_key)
            if cached and cached.image_url:
                log.info("scene cache HIT key=%s room=%r", scene_key, room)
                await websocket.send_json({
                    "type": "scene_description", "room": room,
                    "description": cached.scene_description or "",
                })
                await websocket.send_json({"type": "image_ready", "subject": room, "url": cached.image_url})
                return

            log.info("scene cache MISS key=%s room=%r — generating", scene_key, room)
            description = await describe_scene(scene_output, world_bible)
            # Send the location text immediately — the image generation that follows is slow.
            await websocket.send_json({"type": "scene_description", "room": room, "description": description})

            # Location reference image: the EARLIEST rendered scene for this (game, style,
            # room) anchors the look (house/field/weather/perspective) across all of the
            # room's state-variants — stable regardless of action order or cache state.
            ref_b64 = None
            ref = await db.scalar(
                select(CachedScene)
                .where(CachedScene.game_id == game_id, CachedScene.style_id == style_id,
                       CachedScene.room == room, CachedScene.image_url.isnot(None))
                .order_by(CachedScene.created_at.asc())
                .limit(1)
            )
            if ref and ref.cache_key != scene_key and ref.image_url:
                ref_path = Path(settings.images_dir) / f"{ref.cache_key}.jpg"
                try:
                    ref_b64 = base64.b64encode(ref_path.read_bytes()).decode()
                    log.info("scene reference: room=%r anchored to key=%s", room, ref.cache_key)
                except OSError:
                    ref_b64 = None

            # Stable seed per (game, style, location) backs up the reference for consistency.
            seed = int(hashlib.sha256(f"{game_id}|{style_id}|{room}".encode()).hexdigest()[:8], 16)
            guide: dict = {}

            if ref_b64 and ref.scene_description:
                # Edit from the anchor with a FOCUSED change instruction (the delta), not a
                # full re-description — full prompts make flux re-render and drift (e.g.
                # un-board the door). A cheap Haiku call computes what actually changed.
                edit = (await describe_edit(ref.scene_description, description)).strip()
                if edit.lower().startswith("no change"):
                    log.info("scene edit: room=%r no visual change — reusing anchor image", room)
                    url = ref.image_url
                else:
                    log.info("scene edit: room=%r delta=%r", room, edit[:100])
                    url = await generate_scene_image(
                        scene_prompt=edit, style_prefix=style_prefix, style_negative="",
                        cache_key=scene_key, seed=seed, reference_image_b64=ref_b64,
                    )
            else:
                # First image for this location (the anchor): augment the prompt against the
                # running visual guide so a new location matches the established look + objects.
                vg = await db.get(VisualGuide, (game_id, style_id))
                guide = vg.doc if vg else {}
                prompt = await augment_prompt(description, guide)
                url = await generate_scene_image(
                    scene_prompt=prompt, style_prefix=style_prefix, style_negative="",
                    cache_key=scene_key, seed=seed, reference_image_b64=None,
                )

            row = cached or CachedScene(cache_key=scene_key, game_id=game_id, style_id=style_id)
            row.room = room
            row.scene_description = description
            row.image_url = url
            await db.merge(row)
            await db.commit()

            await websocket.send_json({"type": "image_ready", "subject": room, "url": url})

            # Anchor renders: learn from the IMAGE flux actually drew and fold it into the
            # guide so later locations stay consistent. After image_ready, so it's off the
            # user's critical path.
            if not ref_b64:
                try:
                    img_bytes = (Path(settings.images_dir) / f"{scene_key}.jpg").read_bytes()
                    updated = await analyze_image(img_bytes, description, guide)
                    existing = await db.get(VisualGuide, (game_id, style_id))
                    if existing:
                        cur = existing.doc or {}
                        existing.doc = {
                            "style": cur.get("style") or updated.get("style") or "",
                            "entities": {**(cur.get("entities") or {}), **(updated.get("entities") or {})},
                        }
                        existing.updated_at = datetime.utcnow()
                    else:
                        db.add(VisualGuide(game_id=game_id, style_id=style_id, doc=updated))
                    await db.commit()
                    log.info("visual guide updated from room=%r (%d entities)",
                             room, len((updated.get("entities") or {})))
                except Exception:
                    log.warning("visual guide update failed for room=%r", room, exc_info=True)
    except Exception:
        log.exception("Scene render failed: game=%s room=%r", game_id, room)


@router.websocket("/api/playthroughs/{playthrough_id}/play")
async def play(websocket: WebSocket, playthrough_id: str):
    await websocket.accept()
    log.info("WebSocket connected: playthrough=%s", playthrough_id)

    async with AsyncSessionLocal() as db:
        playthrough: Playthrough = await db.get(Playthrough, playthrough_id)
        if not playthrough:
            log.warning("Playthrough not found: %s", playthrough_id)
            await websocket.send_json({"type": "error", "message": "Playthrough not found."})
            await websocket.close()
            return

        game: Game = await db.get(Game, playthrough.game_id)
        game_path = str(settings.games_dir / game.filename)
        log.info("Starting game: %s (turn=%s)", game.title, playthrough.turn_count)
        context = ContextManager.from_json(playthrough.context_json)

        world_bible: dict = game.world_bible if isinstance(game.world_bible, dict) else json.loads(game.world_bible or "{}")
        vocab_index: dict = game.vocab_index if isinstance(game.vocab_index, dict) else json.loads(game.vocab_index or "{}")
        vocab_verbs: list = world_bible.get("vocab_verbs", [])
        vocab_nouns: list = world_bible.get("vocab_nouns", [])
        style_id: str = playthrough.style_id or "default"
        style_prefix: str = ""  # TODO: load from Style record

        last_scene_key: str | None = None

        async def observe_and_render(room: str, save_bytes: bytes | None, *, dedup: bool):
            """
            Non-perturbingly observe the current scene, then render it in the background.
            With dedup=True, skips when the scene hasn't changed since the last render.
            Returns the scene_key (or None if nothing observed).
            """
            nonlocal last_scene_key
            targets = _room_examine_targets(world_bible, room)
            scene_output = await observe_scene(game_path, save_bytes, settings.dfrotz_path, targets)
            if not scene_output:
                return None
            scene_key = make_cache_key(game.id, style_id, scene_output)
            if dedup and scene_key == last_scene_key:
                return scene_key
            last_scene_key = scene_key
            asyncio.create_task(_render_scene(
                websocket, game.id, scene_key, scene_output, room,
                style_id, style_prefix, world_bible,
            ))
            return scene_key

        # On connect, always show the CURRENT scene — works for a fresh playthrough
        # (no save → initial state) and a resumed one (restore the current save). This
        # also unblocks the client's "Entering the world…" state, which clears on the
        # first narrative chunk.
        try:
            # persist=False: never save during a look — non-perturbing, and the save
            # command deadlocks on pipe buffering.
            opening, _ = await run_one_turn(
                game_path, "look", playthrough.engine_save, settings.dfrotz_path, persist=False
            )
        except Exception:
            log.exception("Failed to start game: playthrough=%s game=%s path=%s",
                          playthrough_id, game.title, game_path)
            await websocket.send_json({"type": "error", "message": "Failed to start the game engine. Check server logs."})
            await websocket.close()
            return

        opening_room = _extract_new_room(opening.raw_text) or playthrough.current_room or ""
        playthrough.current_room = opening_room
        playthrough.last_active = datetime.utcnow()
        await db.commit()

        # Command results (here, the opening LOOK) are shown verbatim — the game's own
        # text is ground truth. The AI enriches the SCENE (grey), not the action result.
        await websocket.send_json({"type": "narrative_chunk", "text": opening.raw_text})
        await websocket.send_json({"type": "narrative_done"})
        log.info("Connect scene: room=%r turn=%s", opening_room, playthrough.turn_count)
        await websocket.send_json({
            "type": "game_state", "room": opening_room, "inventory": [], "turn": playthrough.turn_count or 0,
        })

        await observe_and_render(opening_room, playthrough.engine_save, dedup=False)

        try:
            while True:
                data = await websocket.receive_json()
                msg_type = data.get("type")

                if msg_type == "request_image":
                    # Render the current state on demand (reuses cache if present).
                    await observe_and_render(playthrough.current_room or "", playthrough.engine_save, dedup=False)
                    continue

                if msg_type != "command":
                    continue

                user_input: str = data["text"].strip()
                if not user_input:
                    continue

                current_room = playthrough.current_room or "Unknown"
                visible_objects: list[str] = []

                # 1. Translate natural language → game command via dfrotz
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
                    log.info("Turn %s: %r → cmd=%r (%d chars)", playthrough.turn_count + 1,
                             user_input, command, len(raw_output))
                except ValueError as e:
                    log.warning("Translation failed for %r: %s", user_input, e)
                    await websocket.send_json({"type": "error", "message": str(e)})
                    continue
                except Exception:
                    log.exception("Unexpected error running command for playthrough=%s", playthrough_id)
                    await websocket.send_json({"type": "error", "message": "Game engine error. Try again."})
                    continue

                # 2. Show the command result verbatim — the game's own text is ground
                #    truth (no enrichment/hallucination; the scene block carries the AI prose).
                await websocket.send_json({"type": "narrative_chunk", "text": raw_output})
                await websocket.send_json({"type": "narrative_done"})
                narrative_text = raw_output

                # 3. Detect room change and persist turn
                new_room = _extract_new_room(raw_output)
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

                # 4. Send updated game state
                await websocket.send_json({
                    "type": "game_state",
                    "room": current_room,
                    "inventory": [],
                    "turn": turn_num,
                })

                # 5. Observe the current scene; render only if it actually changed.
                #    (cheap dfrotz look; the scene cache makes repeat states free)
                await observe_and_render(current_room, latest_save, dedup=True)

        except WebSocketDisconnect:
            log.info("WebSocket disconnected: playthrough=%s", playthrough_id)
        except Exception:
            log.exception("Unhandled error in WebSocket loop: playthrough=%s", playthrough_id)
