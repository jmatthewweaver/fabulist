"""
Scene image generation via FLUX 2 Dev on Replicate.
Images are permanent; shared across sessions when game+location+objects+style match.
"""
import hashlib
import json
from pathlib import Path

import replicate

from ..config import settings

_IMAGE_DIR = settings.images_dir
_IMAGE_DIR.mkdir(exist_ok=True)


def make_cache_key(game_id: str, location_id: str, visible_object_ids: list[str], style_id: str) -> str:
    content = f"{game_id}|{location_id}|{sorted(visible_object_ids)}|{style_id}"
    return hashlib.sha256(content.encode()).hexdigest()[:32]


async def generate_scene_image(
    scene_prompt: str,
    style_prefix: str,
    style_negative: str,
    reference_image_urls: list[str],  # style seed + up to 2 prior rooms
    cache_key: str,
    mobile: bool = False,
) -> str:
    """
    Generate a scene image and save it locally. Returns the local file URL.
    Caller is responsible for checking the cache before calling this.
    """
    width, height = (512, 384) if mobile else (1024, 768)
    full_prompt = f"{style_prefix} {scene_prompt}".strip()

    output = replicate.run(
        "black-forest-labs/flux-dev",
        input={
            "prompt": full_prompt,
            "negative_prompt": style_negative or "modern, anachronistic, text, UI elements, watermark",
            "width": width,
            "height": height,
            "num_inference_steps": 20,
            "guidance_scale": 3.5,
            # reference images for style consistency
            "image_urls": reference_image_urls[:3] if reference_image_urls else [],
        },
    )
    # Replicate returns a URL; download and store locally
    import httpx
    async with httpx.AsyncClient() as client:
        response = await client.get(str(output[0]))
        response.raise_for_status()

    suffix = "_mobile" if mobile else ""
    out_path = _IMAGE_DIR / f"{cache_key}{suffix}.jpg"
    out_path.write_bytes(response.content)
    return f"/images/{cache_key}{suffix}.jpg"


async def generate_style_seed(
    game_title: str,
    opening_description: str,
    style_prefix: str,
    style_negative: str,
    style_id: str,
) -> str:
    """Generate the establishing shot used as style reference for all session images."""
    prompt = f"{style_prefix} {opening_description[:200]}. Title: {game_title}."
    return await generate_scene_image(
        scene_prompt=prompt,
        style_prefix="",
        style_negative=style_negative,
        reference_image_urls=[],
        cache_key=f"seed_{style_id}_{hashlib.sha256(game_title.encode()).hexdigest()[:8]}",
    )


def build_scene_prompt(
    room_name: str,
    room_description: str,
    visible_objects: list[str],
    relevant_inventions: list[dict],
    world_bible: dict,
) -> str:
    """Assemble the FLUX image prompt from game state + world knowledge."""
    palette = world_bible.get("sensory_palette", {})
    atmosphere = ", ".join(palette.get("sight", [])[:3])

    objects_desc = ", ".join(visible_objects[:6]) if visible_objects else ""
    inventions_context = "; ".join(
        i["canonical_text"][:80] for i in relevant_inventions[:3]
    )

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
