import logging
import logging.config
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from starlette.middleware.sessions import SessionMiddleware

from .config import settings
from .deps import engine
from .models.db import Base

logging.config.dictConfig({
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {"format": "%(asctime)s %(levelname)-8s %(name)s: %(message)s"},
    },
    "handlers": {
        "console": {"class": "logging.StreamHandler", "formatter": "default"},
    },
    "root": {"handlers": ["console"], "level": settings.log_level},
    "loggers": {
        # Silence SQLAlchemy's per-query noise; only show warnings+
        "sqlalchemy.engine": {"level": "WARNING", "propagate": False, "handlers": ["console"]},
        "sqlalchemy.pool": {"level": "WARNING", "propagate": False, "handlers": ["console"]},
        # uvicorn access log is useful at INFO; keep it
        "uvicorn.access": {"level": "INFO", "propagate": False, "handlers": ["console"]},
        "uvicorn.error": {"level": "INFO", "propagate": False, "handlers": ["console"]},
    },
})


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
            name="Storybook Vector",
            description="Flat vector storybook illustrations — bold shapes, limited palette, paper grain.",
            flux_prompt_prefix="flat vector illustration, simple bold shapes, limited flat color palette, subtle paper-grain texture, clean modern storybook style,",
            flux_negative_prompt="photorealistic, 3d render, photograph",
            tone_instructions="Enrich descriptions with vivid sensory detail and a sense of wonder.",
        ))
        await db.commit()


from .routers.auth import router as auth_router
from .routers.games import router as games_router
from .routers.playthroughs import router as playthroughs_router
from .routers.websocket import router as ws_router

app.include_router(auth_router)
app.include_router(games_router)
app.include_router(playthroughs_router)
app.include_router(ws_router)


@app.get("/health")
async def health():
    return {"status": "ok"}
