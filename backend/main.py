from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession

from .config import settings
from .models.db import Base

engine = create_async_engine(settings.database_url, echo=settings.debug)
AsyncSessionLocal = async_sessionmaker(engine, expire_on_commit=False)


@asynccontextmanager
async def lifespan(app: FastAPI):
    from .models.migrations import run as run_migrations
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        await run_migrations(conn)
    settings.games_dir.mkdir(exist_ok=True)
    settings.saves_dir.mkdir(exist_ok=True)
    settings.images_dir.mkdir(exist_ok=True)
    yield


app = FastAPI(title="Fabulist", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.frontend_url],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve generated images as static files
app.mount("/images", StaticFiles(directory=str(settings.images_dir)), name="images")


async def get_db():
    async with AsyncSessionLocal() as session:
        yield session


# Dependency injection helper — patches into routers
def _override_db_dep(app: FastAPI):
    from .routers import games, sessions, websocket
    for router_mod in (games, sessions):
        for route in router_mod.router.routes:
            if hasattr(route, "dependant"):
                for dep in route.dependant.dependencies:
                    if dep.call is lambda: None:
                        dep.call = get_db


# Register routers
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
