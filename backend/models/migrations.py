"""
DDL that SQLAlchemy can't express natively — run once after create_all().
Idempotent: uses IF NOT EXISTS / DO blocks throughout.
"""
from sqlalchemy.ext.asyncio import AsyncConnection
from sqlalchemy import text

_DDL = [
    # Extensions (idempotent)
    "CREATE EXTENSION IF NOT EXISTS vector",
    "CREATE EXTENSION IF NOT EXISTS pg_textsearch",

    # Rename inventions.session_id → playthrough_id if still on old schema
    """
    DO $$
    BEGIN
        IF EXISTS (
            SELECT 1 FROM information_schema.columns
            WHERE table_name = 'inventions' AND column_name = 'session_id'
        ) THEN
            ALTER TABLE inventions RENAME COLUMN session_id TO playthrough_id;
            ALTER TABLE inventions DROP CONSTRAINT IF EXISTS uq_invention_session_object;
            ALTER TABLE inventions ADD CONSTRAINT uq_invention_playthrough_object
                UNIQUE (playthrough_id, object_key);
        END IF;
    END
    $$
    """,

    # Make full_text a generated column: object_key (spaces restored) + canonical_text.
    """
    DO $$
    BEGIN
        IF NOT EXISTS (
            SELECT 1 FROM information_schema.columns
            WHERE table_name = 'inventions'
            AND column_name = 'full_text'
            AND is_generated = 'ALWAYS'
        ) THEN
            ALTER TABLE inventions
                DROP COLUMN IF EXISTS full_text,
                ADD COLUMN full_text TEXT
                    GENERATED ALWAYS AS (
                        replace(object_key, '_', ' ') || ' ' || canonical_text
                    ) STORED;
        END IF;
    END
    $$
    """,

    # BM25 index via pg_textsearch
    """
    CREATE INDEX IF NOT EXISTS inventions_bm25_idx
    ON inventions
    USING bm25 (full_text)
    WITH (text_config='english')
    """,

    # cached_scenes gained a `room` column (location reference image lookup)
    "ALTER TABLE IF EXISTS cached_scenes ADD COLUMN IF NOT EXISTS room VARCHAR",

    # Set the default style's look. Guarded on the set of auto-managed prefixes we've shipped
    # (the original oil-painting seed, then flat-vector) so it upgrades an existing default to
    # the current choice and runs at most once per value — never clobbering a hand-edited
    # prefix. When changing the style again, add the prior value to this IN list. Clear
    # cached_scenes + visual_guides afterwards so renders and the guide rebuild against it.
    """
    UPDATE styles
    SET flux_prompt_prefix = 'hand-drawn 2D animation cel, bold ink outlines, painted storybook backgrounds, rich saturated color, soft cinematic lighting, whimsical dark-fantasy mood,',
        flux_negative_prompt = 'photorealistic, 3d render, photograph',
        name = 'Storybook Animation',
        description = 'Hand-drawn 2D animation — inked outlines, painted backgrounds, warm fantasy mood.'
    WHERE id = 'default'
      AND flux_prompt_prefix IN (
        'detailed oil painting, fantasy book illustration, warm lighting,',
        'flat vector illustration, simple bold shapes, limited flat color palette, subtle paper-grain texture, clean modern storybook style,'
      )
    """,
]


async def run(conn: AsyncConnection) -> None:
    for stmt in _DDL:
        await conn.execute(text(stmt))
