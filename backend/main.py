from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from starlette.middleware.sessions import SessionMiddleware

from .config import settings
from .deps import engine
from .models.db import Base


@asynccontextmanager
async def lifespan(app: FastAPI):
    from .models.migrations import run as run_migrations
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        await run_migrations(conn)
    settings.games_dir.mkdir(exist_ok=True)
    settings.saves_dir.mkdir(exist_ok=True)
    settings.images_dir.mkdir(exist_ok=True)
    await _seed_styles()
    yield


app = FastAPI(title="Fabulist", lifespan=lifespan)

app.add_middleware(SessionMiddleware, secret_key=settings.jwt_secret)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.frontend_url],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/images", StaticFiles(directory=str(settings.images_dir)), name="images")

async def _seed_styles():
    from .models.db import Style
    from .deps import AsyncSessionLocal
    from sqlalchemy import select
    async with AsyncSessionLocal() as db:
        existing = await db.execute(select(Style).limit(1))
        if existing.scalar():
            return
        db.add(Style(
            id="default",
            name="Classic Illustrated",
            description="Painterly illustrations in the style of classic fantasy book covers.",
            flux_prompt_prefix="detailed oil painting, fantasy book illustration, warm lighting,",
            flux_negative_prompt="photorealistic, modern, sci-fi",
            tone_instructions="Enrich descriptions with vivid sensory detail and a sense of wonder.",
        ))
        await db.commit()


from .routers.auth import router as auth_router
from .routers.games import router as games_router
from .routers.sessions import router as sessions_router
from .routers.websocket import router as ws_router

app.include_router(auth_router)
app.include_router(games_router)
app.include_router(sessions_router)
app.include_router(ws_router)


@app.get("/health")
async def health():
    return {"status": "ok"}
