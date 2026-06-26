"""
DDL that SQLAlchemy can't express natively — run once after create_all().
Idempotent: uses IF NOT EXISTS where possible.
"""
from sqlalchemy.ext.asyncio import AsyncConnection

_DDL = [
    # Extensions (idempotent)
    "CREATE EXTENSION IF NOT EXISTS vector",
    "CREATE EXTENSION IF NOT EXISTS pg_search",

    # BM25 index on inventions via ParadeDB pg_search.
    # Indexes object_key (for exact/keyword lookup) and canonical_text (for content search).
    # key_field must be the integer primary key.
    """
    CREATE INDEX IF NOT EXISTS ix_inventions_bm25
    ON inventions
    USING bm25(id, object_key, canonical_text)
    WITH (key_field='id')
    """,
]


async def run(conn: AsyncConnection) -> None:
    for stmt in _DDL:
        await conn.execute(__import__("sqlalchemy").text(stmt))
