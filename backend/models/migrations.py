"""
DDL that SQLAlchemy can't express natively — run once after create_all().
Idempotent: uses IF NOT EXISTS where possible.
"""
from sqlalchemy.ext.asyncio import AsyncConnection
from sqlalchemy import text

_DDL = [
    # Extensions (idempotent)
    "CREATE EXTENSION IF NOT EXISTS vector",
    "CREATE EXTENSION IF NOT EXISTS pg_textsearch",

    # Make full_text a generated column: object_key (spaces restored) + canonical_text.
    # ALTER COLUMN is idempotent-ish — wrapped in a DO block to skip if already generated.
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
]


async def run(conn: AsyncConnection) -> None:
    for stmt in _DDL:
        await conn.execute(text(stmt))
