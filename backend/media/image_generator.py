"""
Scene image generation via BFL (Black Forest Labs) API.
Async polling pattern: POST to submit → GET polling_url until Ready.
Images are permanent assets stored locally; shared across sessions when
game + location + visible objects + style all match.
"""
import asyncio
import hashlib
from pathlib import Path

import httpx

from ..config import settings

_BFL_BASE = "https://api.bfl.ai/v1"
_IMAGE_DIR = settings.images_dir
_IMAGE_DIR.mkdir(exist_ok=True)

_POLL_INTERVAL = 2.0   # seconds between polls
_POLL_TIMEOUT = 120.0  # give up after 2 minutes


def make_cache_key(game_id: str, style_id: str, scene_output: str) -> str:
    """
    Scene cache key = hash of the game's own deterministic output for the current
    state, folded with game + style. Identical state (byte-identical LOOK/EXAMINE
    output) → identical key → reuse; any change → new key → re-render.
    """
    content = f"{game_id}|{style_id}|{scene_output.strip()}"
    return hashlib.sha256(content.encode()).hexdigest()[:32]


async def _submit(prompt: str, width: int, height: int, reference_urls: list[str],
                  mobile: bool, seed: int | None = None) -> str:
    """Submit generation request, return polling_url."""
    model = settings.bfl_model_mobile if mobile else settings.bfl_model_desktop
    payload: dict = {
        "prompt": prompt,
        "width": width,
        "height": height,
    }
    # Deterministic seed keeps a location's state-variants visually consistent
    # (same lighting/weather/composition; only the described details change).
    if seed is not None:
        payload["seed"] = seed
    # Reference image support — flux-2-pro only
    if reference_urls and not mobile:
        payload["image_prompt"] = reference_urls[0]  # style seed as primary reference

    async with httpx.AsyncClient() as client:
        resp = await client.post(
            f"{_BFL_BASE}/{model}",
            headers={
                "accept": "application/json",
                "x-key": settings.bfl_api_key,
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=30.0,
        )
        resp.raise_for_status()
        return resp.json()["polling_url"]


async def _poll(polling_url: str) -> str:
    """Poll until Ready, return image URL."""
    deadline = asyncio.get_event_loop().time() + _POLL_TIMEOUT
    async with httpx.AsyncClient() as client:
        while asyncio.get_event_loop().time() < deadline:
            resp = await client.get(
                polling_url,
                headers={"accept": "application/json", "x-key": settings.bfl_api_key},
                timeout=10.0,
            )
            resp.raise_for_status()
            data = resp.json()
            if data.get("status") == "Ready":
                return data["result"]["sample"]
            if data.get("status") in ("Error", "Failed"):
                raise RuntimeError(f"BFL generation failed: {data}")
            await asyncio.sleep(_POLL_INTERVAL)
    raise TimeoutError("BFL image generation timed out")


async def generate_scene_image(
    scene_prompt: str,
    style_prefix: str,
    style_negative: str,         # BFL doesn't support negative prompts yet — kept for API compat
    reference_image_urls: list[str],
    cache_key: str,
    mobile: bool = False,
    seed: int | None = None,
) -> str:
    """Generate image, save locally, return local URL path."""
    width, height = (512, 384) if mobile else (1024, 768)
    full_prompt = f"{style_prefix} {scene_prompt}".strip()

    polling_url = await _submit(full_prompt, width, height, reference_image_urls, mobile, seed)
    bfl_url = await _poll(polling_url)

    # Download and store permanently
    async with httpx.AsyncClient() as client:
        resp = await client.get(bfl_url, timeout=30.0)
        resp.raise_for_status()

    suffix = "_mobile" if mobile else ""
    out_path = _IMAGE_DIR / f"{cache_key}{suffix}.jpg"
    out_path.write_bytes(resp.content)
    return f"/images/{cache_key}{suffix}.jpg"


async def generate_style_seed(
    game_title: str,
    opening_description: str,
    style_prefix: str,
    style_id: str,
) -> str:
    """Generate the establishing shot used as style reference for all session images."""
    prompt = f"{style_prefix} {opening_description[:200]}. Title: {game_title}."
    seed_key = f"seed_{style_id}_{hashlib.sha256(game_title.encode()).hexdigest()[:8]}"
    return await generate_scene_image(
        scene_prompt=prompt,
        style_prefix="",
        style_negative="",
        reference_image_urls=[],
        cache_key=seed_key,
    )


def build_scene_prompt(
    room_name: str,
    room_description: str,
    visible_objects: list[str],
    relevant_inventions: list[dict],
    world_bible: dict,
) -> str:
    """Assemble the image prompt from game state + world knowledge."""
    palette = world_bible.get("sensory_palette", {})
    atmosphere = ", ".join(palette.get("sight", [])[:3])
    objects_desc = ", ".join(visible_objects[:6]) if visible_objects else ""
    inventions_context = "; ".join(i["canonical_text"][:80] for i in relevant_inventions[:3])

    parts = [room_name]
    if room_description:
        parts.append(room_description[:200])
    if objects_desc:
        parts.append(f"containing {objects_desc}")
    if atmosphere:
        parts.append(atmosphere)
    if inventions_context:
        parts.append(inventions_context)
    return ". ".join(parts)
