#!/usr/bin/env python3
"""
Fabulist admin control panel — mobile-friendly supervisor for the backend + frontend.

It launches and OWNS the backend and frontend as child processes (so it's the single
container process), captures their output to log files, and exposes a small password-
protected web UI to: git pull, restart either service, view logs, re-ingest a game, and
clear the scene/image cache on demand.

Run (from the repo root, inside the venv):
    python admin.py
Then point your tunnel/proxy at ADMIN_PORT (default 8001).

IMPORTANT: stop any backend/frontend you're already running first — this process starts
its own copies, so leftover ones would fight for the ports.

.env settings (single-user login + optional overrides):
    ADMIN_USER=you                 # default "admin"
    ADMIN_PASS=secret              # REQUIRED (login fails until set)
    ADMIN_SECRET=...               # cookie signing; falls back to JWT_SECRET
    ADMIN_API_KEY=...              # enables the token API (X-Admin-Key header or ?key=)
    ADMIN_PORT=8001
    BACKEND_CMD=uvicorn backend.main:app --host 0.0.0.0 --port 8000
    FRONTEND_CMD=npm run dev
    FRONTEND_DIR=<repo>/frontend
    BACKEND_URL=http://localhost:8000     # used to call the ingest endpoint
    GAME_FILENAME=zork1-r119-s880429.z3   # default for the ingest box
    DATABASE_URL=...               # reused from the backend; for clear-cache / re-ingest
    IMAGES_DIR=<repo>/backend/images
"""
import glob
import hashlib
import hmac
import html
import json
import os
import re
import signal
import subprocess
import time
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import (
    FileResponse, HTMLResponse, JSONResponse, PlainTextResponse, RedirectResponse,
)


# --------------------------------------------------------------------------- config
def _load_env(path: Path) -> None:
    """Minimal .env loader (no dependency); does not override existing env."""
    try:
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
    except FileNotFoundError:
        pass


REPO_DIR = Path(os.environ.get("REPO_DIR") or Path(__file__).resolve().parent)
_load_env(REPO_DIR / ".env")

ADMIN_USER = os.environ.get("ADMIN_USER", "admin")
ADMIN_PASS = os.environ.get("ADMIN_PASS", "")
SECRET = os.environ.get("ADMIN_SECRET") or os.environ.get("JWT_SECRET") or "change-me-please"
ADMIN_API_KEY = os.environ.get("ADMIN_API_KEY", "")
ADMIN_PORT = int(os.environ.get("ADMIN_PORT", "8001"))
BACKEND_CMD = os.environ.get("BACKEND_CMD", "uvicorn backend.main:app --host 0.0.0.0 --port 8000")
FRONTEND_CMD = os.environ.get("FRONTEND_CMD", "npm run dev")
FRONTEND_DIR = os.environ.get("FRONTEND_DIR", str(REPO_DIR / "frontend"))
BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8000")
GAME_FILENAME = os.environ.get("GAME_FILENAME", "zork1-r119-s880429.z3")
DATABASE_URL = os.environ.get("DATABASE_URL", "")
IMAGES_DIR = os.environ.get("IMAGES_DIR", str(REPO_DIR / "backend" / "images"))
LOG_DIR = REPO_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)


# ----------------------------------------------------------------- process supervision
class Service:
    def __init__(self, name: str, cmd: str, cwd: str):
        self.name = name
        self.cmd = cmd
        self.cwd = cwd
        self.log = LOG_DIR / f"{name}.log"
        self.proc: subprocess.Popen | None = None

    def running(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def start(self) -> None:
        if self.running():
            return
        f = open(self.log, "ab")
        f.write(f"\n===== start {time.ctime()} :: {self.cmd} =====\n".encode())
        f.flush()
        # shell=True + new session so we can kill the whole process group on restart.
        self.proc = subprocess.Popen(
            self.cmd, cwd=self.cwd, shell=True,
            stdout=f, stderr=subprocess.STDOUT, start_new_session=True,
        )

    def stop(self) -> None:
        if self.running():
            try:
                os.killpg(os.getpgid(self.proc.pid), signal.SIGTERM)
                self.proc.wait(timeout=10)
            except Exception:
                try:
                    os.killpg(os.getpgid(self.proc.pid), signal.SIGKILL)
                except Exception:
                    pass
        self.proc = None

    def restart(self) -> None:
        self.stop()
        time.sleep(1)
        self.start()


backend = Service("backend", BACKEND_CMD, str(REPO_DIR))
frontend = Service("frontend", FRONTEND_CMD, FRONTEND_DIR)


def tail(path: Path, max_bytes: int = 60000) -> str:
    try:
        with open(path, "rb") as f:
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(0, size - max_bytes))
            data = f.read()
        return data.decode("utf-8", "replace") or "(empty)"
    except FileNotFoundError:
        return "(no log yet)"


def run_cmd(args: list[str], cwd: str) -> str:
    try:
        r = subprocess.run(args, cwd=cwd, capture_output=True, text=True, timeout=300)
        return (r.stdout + r.stderr).strip() or "(no output)"
    except Exception as e:
        return f"error: {e}"


def _libpq_url() -> str:
    # psql wants a plain libpq URL, not SQLAlchemy's "+asyncpg" dialect.
    return DATABASE_URL.replace("+asyncpg", "")


def psql(sql: str) -> str:
    """Run a query and return raw text (tuples-only, unaligned). '' if no DB configured."""
    if not DATABASE_URL:
        return ""
    return run_cmd(["psql", _libpq_url(), "-t", "-A", "-c", sql], str(REPO_DIR)).strip()


def psql_json(sql: str):
    """Run a query whose single value is JSON; return the parsed object (or None)."""
    raw = psql(sql)
    if not raw or raw.startswith("error:"):
        return None
    try:
        val = json.loads(raw)
    except Exception:
        return None
    # world_bible / vocab_index are stored as json.dumps(...) INTO a JSONB column, so they read
    # back double-encoded (a JSON string holding more JSON). Peel string layers until we reach
    # the real object; stop on the first layer that isn't itself JSON.
    while isinstance(val, str):
        try:
            val = json.loads(val)
        except Exception:
            break
    return val


def _safe_id(s: str) -> str:
    """Game ids / cache keys are hex-ish; allow only those chars to keep SQL/paths safe."""
    return s if re.fullmatch(r"[A-Za-z0-9_-]+", s or "") else ""


def esc(s) -> str:
    return html.escape(str(s if s is not None else ""))


# ------------------------------------------------------------------------------- auth
def _token() -> str:
    return hmac.new(SECRET.encode(), ADMIN_USER.encode(), hashlib.sha256).hexdigest()


def authed(request: Request) -> bool:
    return hmac.compare_digest(request.cookies.get("admin_session", ""), _token())


def guard(request: Request) -> RedirectResponse | None:
    return None if authed(request) else RedirectResponse("/login", status_code=303)


def has_api_key(request: Request) -> bool:
    if not ADMIN_API_KEY:
        return False
    key = request.headers.get("X-Admin-Key") or request.query_params.get("key", "")
    return hmac.compare_digest(key, ADMIN_API_KEY)


# ---------------------------------------------------------------------- shared actions
def do_ingest(fn: str) -> str:
    steps: list[str] = []
    if DATABASE_URL:
        sql = f"UPDATE games SET world_bible = NULL WHERE filename = '{fn}';"
        steps.append("clear world_bible:\n" + run_cmd(["psql", _libpq_url(), "-c", sql], str(REPO_DIR)))
    else:
        steps.append("(DATABASE_URL not set — skipped world_bible clear; ingest may no-op if already ingested)")
    try:
        r = httpx.post(f"{BACKEND_URL}/api/games/ingest", params={"filename": fn}, timeout=600)
        steps.append(f"ingest HTTP {r.status_code}:\n{r.text}")
    except Exception as e:
        steps.append(f"ingest call error: {e}")
    return "\n\n".join(steps)


def do_clear_cache() -> str:
    steps: list[str] = []
    if DATABASE_URL:
        steps.append("DELETE cached_scenes + visual_guides:\n" + run_cmd(
            ["psql", _libpq_url(), "-c", "DELETE FROM cached_scenes; DELETE FROM visual_guides;"],
            str(REPO_DIR)))
    else:
        steps.append("(DATABASE_URL not set — skipped DB delete)")
    removed = 0
    for p in glob.glob(str(Path(IMAGES_DIR) / "*.jpg")):
        try:
            os.remove(p)
            removed += 1
        except Exception:
            pass
    steps.append(f"removed {removed} image file(s) from {IMAGES_DIR}")
    return "\n\n".join(steps)


# ------------------------------------------------------------------------------- views
_STYLE = """
*{box-sizing:border-box}body{margin:0;font-family:system-ui,-apple-system,sans-serif;
background:#1c1917;color:#e7e5e4;padding:16px;max-width:640px;margin:0 auto}
h1{font-size:19px}h2{font-size:13px;color:#a8a29e;margin:22px 0 4px;text-transform:uppercase;
letter-spacing:.06em}form{margin:0}button,.btn{display:block;width:100%;padding:14px;margin:8px 0;
border:0;border-radius:10px;background:#44403c;color:#e7e5e4;font-size:16px;text-align:center;
text-decoration:none;cursor:pointer}button:active{background:#57534e}.danger{background:#7f1d1d}
.primary{background:#1d4ed8}input{width:100%;padding:12px;margin:6px 0;border-radius:8px;
border:1px solid #44403c;background:#292524;color:#e7e5e4;font-size:16px}
pre{background:#0c0a09;padding:12px;border-radius:8px;overflow:auto;font-size:12px;
white-space:pre-wrap;word-break:break-word;max-height:68vh}.badge{display:inline-block;
padding:3px 10px;border-radius:999px;font-size:12px}.up{background:#14532d}.down{background:#7f1d1d}
.row{display:flex;gap:8px}.row form{flex:1}a{color:#93c5fd}
.dim{color:#a8a29e;font-size:13px}.key{color:#57534e;font-size:11px;word-break:break-all}
details{background:#0c0a09;border-radius:8px;padding:8px 12px;margin:6px 0}
details summary{cursor:pointer;font-weight:600}details>div{margin-top:6px}
.scene{background:#0c0a09;border-radius:10px;padding:10px;margin:10px 0;display:flex;gap:10px}
.thumb{width:120px;height:90px;object-fit:cover;border-radius:6px;flex-shrink:0;background:#1c1917}
.scene .meta{min-width:0}.gamecard{display:block;background:#0c0a09;border-radius:10px;padding:12px;margin:8px 0;text-decoration:none;color:#e7e5e4}
.pill{display:inline-block;background:#292524;border-radius:6px;padding:2px 8px;margin:2px 4px 2px 0;font-size:12px}
"""


def page(title: str, body: str, auto: int = 0) -> HTMLResponse:
    refresh = f'<meta http-equiv="refresh" content="{auto}">' if auto else ""
    return HTMLResponse(
        f"<!doctype html><html><head><meta charset=utf-8>"
        f'<meta name=viewport content="width=device-width,initial-scale=1">{refresh}'
        f"<title>{title}</title><style>{_STYLE}</style></head><body>{body}</body></html>"
    )


def result(title: str, output: str) -> HTMLResponse:
    return page(title, f"<h1>{html.escape(title)}</h1><pre>{html.escape(output)}</pre>"
                       f"<a class=btn href=/>← back</a>")


@asynccontextmanager
async def lifespan(app: FastAPI):
    backend.start()
    frontend.start()
    yield
    backend.stop()
    frontend.stop()


app = FastAPI(lifespan=lifespan)


@app.get("/login")
def login_form():
    return page("Login",
        "<h1>Fabulist Admin</h1>"
        "<form method=post action=/login>"
        "<input name=username placeholder=username autocomplete=username>"
        "<input name=password type=password placeholder=password autocomplete=current-password>"
        "<button class=primary>Sign in</button></form>")


@app.post("/login")
async def login(request: Request):
    form = await request.form()
    if ADMIN_PASS and form.get("username") == ADMIN_USER and form.get("password") == ADMIN_PASS:
        r = RedirectResponse("/", status_code=303)
        r.set_cookie("admin_session", _token(), httponly=True, samesite="lax", max_age=30 * 86400)
        return r
    return page("Login", "<h1>Login failed</h1><a class=btn href=/login>← try again</a>")


@app.post("/logout")
def logout():
    r = RedirectResponse("/login", status_code=303)
    r.delete_cookie("admin_session")
    return r


@app.get("/")
def dashboard(request: Request):
    if (g := guard(request)):
        return g

    def badge(s: Service) -> str:
        cls, label = ("up", "running") if s.running() else ("down", "stopped")
        return f'<span class="badge {cls}">{label}</span>'

    body = f"""
    <h1>Fabulist Admin</h1>
    <p>backend {badge(backend)} &nbsp; frontend {badge(frontend)}</p>

    <h2>Deploy</h2>
    <form method=post action=/action/git-pull><button class=primary>⬇ Git pull latest</button></form>
    <div class=row>
      <form method=post action=/action/restart-backend onsubmit="return confirm('Restart backend?')"><button>↻ Backend</button></form>
      <form method=post action=/action/restart-frontend onsubmit="return confirm('Restart frontend?')"><button>↻ Frontend</button></form>
    </div>

    <h2>Logs</h2>
    <a class=btn href=/logs/backend>📜 Backend logs</a>
    <a class=btn href=/logs/frontend>📜 Frontend logs</a>

    <h2>Inspect</h2>
    <a class=btn href=/inspect>🔎 Ingested data &amp; cached scenes</a>

    <h2>Game</h2>
    <form method=post action=/action/ingest onsubmit="return confirm('Re-ingest (clears the world bible first)?')">
      <input name=filename value="{html.escape(GAME_FILENAME)}">
      <button>⟳ Re-ingest game</button>
    </form>
    <form method=post action=/action/clear-cache onsubmit="return confirm('Delete all cached scenes + images?')">
      <button class=danger>🗑 Clear game cache</button>
    </form>

    <h2></h2>
    <form method=post action=/logout><button>Log out</button></form>
    """
    return page("Admin", body)


@app.post("/action/git-pull")
def a_git_pull(request: Request):
    if (g := guard(request)):
        return g
    return result("Git pull", run_cmd(["git", "pull"], str(REPO_DIR)))


@app.post("/action/restart-backend")
def a_restart_backend(request: Request):
    if (g := guard(request)):
        return g
    backend.restart()
    status = "running" if backend.running() else "FAILED to start"
    return result("Restart backend", f"backend {status}\n\nrecent log:\n{tail(backend.log, 4000)}")


@app.post("/action/restart-frontend")
def a_restart_frontend(request: Request):
    if (g := guard(request)):
        return g
    frontend.restart()
    status = "running" if frontend.running() else "FAILED to start"
    return result("Restart frontend", f"frontend {status}\n\nrecent log:\n{tail(frontend.log, 4000)}")


@app.post("/action/ingest")
async def a_ingest(request: Request):
    if (g := guard(request)):
        return g
    form = await request.form()
    fn = (form.get("filename") or GAME_FILENAME).strip()
    return result(f"Re-ingest {fn}", do_ingest(fn))


@app.post("/action/clear-cache")
def a_clear_cache(request: Request):
    if (g := guard(request)):
        return g
    return result("Clear cache", do_clear_cache())


@app.get("/logs/{service}")
def logs(service: str, request: Request, auto: int = 0):
    if (g := guard(request)):
        return g
    svc = {"backend": backend, "frontend": frontend}.get(service)
    if not svc:
        return result("Logs", "unknown service")
    toggle = f"/logs/{service}?auto={0 if auto else 4}"
    label = "■ stop auto-refresh" if auto else "↻ auto-refresh"
    body = (f"<h1>{service} logs {('• live' if auto else '')}</h1>"
            f'<a class=btn href="{toggle}">{label}</a>'
            f"<pre>{html.escape(tail(svc.log))}</pre>"
            f"<a class=btn href=/>← back</a>")
    return page(f"{service} logs", body, auto=auto)


# ---------------------------------------------------------------------------- inspect
_INSPECT_GAMES_SQL = """
SELECT json_agg(json_build_object(
  'id', id, 'title', title, 'filename', filename,
  'ingested', to_char(ingested_at, 'YYYY-MM-DD HH24:MI'),
  'verbs', jsonb_array_length(COALESCE(world_bible->'vocab_verbs', '[]'::jsonb)),
  'nouns', jsonb_array_length(COALESCE(world_bible->'vocab_nouns', '[]'::jsonb)),
  'scenes', (SELECT count(*) FROM cached_scenes c WHERE c.game_id = games.id)
) ORDER BY title) FROM games;
"""


@app.get("/inspect")
def inspect_index(request: Request):
    if (g := guard(request)):
        return g
    if not DATABASE_URL:
        return result("Inspect", "DATABASE_URL not set — inspector unavailable.")
    games = psql_json(_INSPECT_GAMES_SQL) or []
    cards = []
    for gm in games:
        gid = esc(gm["id"])
        cards.append(
            f"<div class=gamecard><b>{esc(gm['title'])}</b>"
            f"<div class=dim>{esc(gm['filename'])} · ingested {esc(gm['ingested'])}</div>"
            f"<div style='margin:6px 0'>{esc(gm['verbs'])} verbs · {esc(gm['nouns'])} nouns "
            f"· {esc(gm['scenes'])} cached scenes</div>"
            f"<a class=btn href=/inspect/game/{gid}>📖 World bible</a>"
            f"<a class=btn href=/inspect/scenes/{gid}>🖼 Cached scenes</a></div>"
        )
    body = ("<h1>🔎 Inspect</h1>" + ("".join(cards) or "<p class=dim>No games ingested.</p>")
            + "<a class=btn href=/>← back</a>")
    return page("Inspect", body)


def _render_world_bible(wb: dict) -> str:
    verbs = wb.get("vocab_verbs") or []
    nouns = wb.get("vocab_nouns") or []
    ko = (wb.get("known_objects") or {})
    nodes = ko.get("nodes") or {}
    rooms = sorted((n for n in nodes.values() if n.get("kind") == "room"),
                   key=lambda n: (n.get("name") or "").lower())

    out = [f"<h2>Vocab</h2><p>{len(verbs)} verbs · {len(nouns)} dictionary words</p>",
           f"<p class=dim><b>verbs:</b> {esc(', '.join(verbs[:80]))}</p>",
           f"<p class=dim><b>nouns:</b> {esc(', '.join(nouns[:80]))}</p>",
           f"<h2>Object tree — {len(rooms)} rooms</h2>"]
    for r in rooms:
        kids = [nodes[str(c)]["name"] for c in r.get("children", []) if str(c) in nodes]
        scen = [nodes[str(s)]["name"] for s in r.get("scenery", []) if str(s) in nodes]
        block = [f"<details><summary>{esc(r.get('name'))} "
                 f"<span class=key>#{esc(r.get('id'))}</span></summary>"]
        if r.get("description"):
            block.append(f"<div class=dim>{esc(r['description'])}</div>")
        if kids:
            block.append("<div><b>contains</b> "
                         + "".join(f"<span class=pill>{esc(k)}</span>" for k in kids) + "</div>")
        if scen:
            block.append("<div><b>scenery</b> "
                         + "".join(f"<span class=pill>{esc(s)}</span>" for s in scen) + "</div>")
        block.append("</details>")
        out.append("".join(block))
    return "".join(out)


@app.get("/inspect/game/{gid}")
def inspect_game(gid: str, request: Request):
    if (g := guard(request)):
        return g
    gid = _safe_id(gid)
    if not gid:
        return result("Inspect", "bad game id")
    wb = psql_json(f"SELECT world_bible FROM games WHERE id = '{gid}';")
    if wb is None:
        return result("Inspect", "game not found or world_bible empty (mid-reingest?)")

    guide = psql_json(f"SELECT doc FROM visual_guides WHERE game_id = '{gid}' ORDER BY style_id LIMIT 1;")
    guide_html = ""
    if guide:
        ents = (guide.get("entities") or {})
        guide_html = ("<h2>Visual guide</h2>"
                      f"<p class=dim>{esc(guide.get('style') or '(no style yet)')}</p>"
                      + "".join(f"<div><b>{esc(n)}</b> <span class=dim>{esc(d)}</span></div>"
                                for n, d in ents.items()))

    body = (f"<h1>{esc(wb.get('title') or gid)}</h1>"
            + _render_world_bible(wb) + guide_html
            + f"<a class=btn href=/inspect/scenes/{gid}>🖼 Cached scenes</a>"
            + "<a class=btn href=/inspect>← all games</a>")
    return page("World bible", body)


@app.get("/inspect/scenes/{gid}")
def inspect_scenes(gid: str, request: Request):
    if (g := guard(request)):
        return g
    gid = _safe_id(gid)
    if not gid:
        return result("Inspect", "bad game id")
    scenes = psql_json(
        "SELECT json_agg(json_build_object("
        "'key', cache_key, 'room', room, 'desc', scene_description, "
        "'created', to_char(created_at, 'MM-DD HH24:MI')) ORDER BY room, created_at) "
        f"FROM cached_scenes WHERE game_id = '{gid}';"
    ) or []
    cards = []
    for s in scenes:
        cards.append(
            f"<div class=scene><img class=thumb loading=lazy src=/inspect/img/{esc(s['key'])} alt=''>"
            f"<div class=meta><b>{esc(s['room'] or '?')}</b> <span class=dim>{esc(s['created'])}</span>"
            f"<div class=dim>{esc(s['desc'] or '')}</div>"
            f"<div class=key>{esc(s['key'])}</div></div></div>"
        )
    body = (f"<h1>Cached scenes ({len(scenes)})</h1>"
            + ("".join(cards) or "<p class=dim>No cached scenes for this game.</p>")
            + f"<a class=btn href=/inspect/game/{gid}>📖 World bible</a>"
            + "<a class=btn href=/inspect>← all games</a>")
    return page("Cached scenes", body)


@app.get("/inspect/img/{key}")
def inspect_img(key: str, request: Request):
    if (g := guard(request)):
        return g
    key = _safe_id(key)
    if not key:
        return PlainTextResponse("bad key", status_code=400)
    p = Path(IMAGES_DIR) / f"{key}.jpg"
    if not p.exists():
        return PlainTextResponse("not found", status_code=404)
    return FileResponse(str(p), media_type="image/jpeg")


# --------------------------------------------------------------- token API (for tools)
# Auth with the ADMIN_API_KEY via the `X-Admin-Key` header or `?key=` query param. Lets a
# client (or assistant) fetch logs/status and trigger actions without the browser login.
_SERVICES = {"backend": backend, "frontend": frontend}


def _deny():
    return JSONResponse({"error": "unauthorized — set ADMIN_API_KEY and pass X-Admin-Key"}, status_code=401)


@app.get("/api/status")
def api_status(request: Request):
    if not has_api_key(request):
        return _deny()
    return {"backend": backend.running(), "frontend": frontend.running()}


@app.get("/api/logs/{service}", response_class=PlainTextResponse)
def api_logs(service: str, request: Request, bytes: int = 60000):
    if not has_api_key(request):
        return _deny()
    svc = _SERVICES.get(service)
    if not svc:
        return PlainTextResponse("unknown service", status_code=404)
    return PlainTextResponse(tail(svc.log, max(1000, min(bytes, 500000))))


@app.post("/api/git-pull", response_class=PlainTextResponse)
def api_git_pull(request: Request):
    if not has_api_key(request):
        return _deny()
    return PlainTextResponse(run_cmd(["git", "pull"], str(REPO_DIR)))


@app.post("/api/restart/{service}", response_class=PlainTextResponse)
def api_restart(service: str, request: Request):
    if not has_api_key(request):
        return _deny()
    svc = _SERVICES.get(service)
    if not svc:
        return PlainTextResponse("unknown service", status_code=404)
    svc.restart()
    status = "running" if svc.running() else "FAILED to start"
    return PlainTextResponse(f"{service} {status}\n\nrecent log:\n{tail(svc.log, 4000)}")


@app.post("/api/ingest", response_class=PlainTextResponse)
def api_ingest(request: Request, filename: str = ""):
    if not has_api_key(request):
        return _deny()
    return PlainTextResponse(do_ingest(filename.strip() or GAME_FILENAME))


@app.post("/api/clear-cache", response_class=PlainTextResponse)
def api_clear_cache(request: Request):
    if not has_api_key(request):
        return _deny()
    return PlainTextResponse(do_clear_cache())


@app.get("/api/inspect/games")
def api_inspect_games(request: Request):
    if not has_api_key(request):
        return _deny()
    return JSONResponse(psql_json(_INSPECT_GAMES_SQL) or [])


@app.get("/api/inspect/world/{gid}")
def api_inspect_world(gid: str, request: Request):
    if not has_api_key(request):
        return _deny()
    gid = _safe_id(gid)
    return JSONResponse(psql_json(f"SELECT world_bible FROM games WHERE id = '{gid}';") or {})


@app.get("/api/inspect/scenes/{gid}")
def api_inspect_scenes(gid: str, request: Request):
    if not has_api_key(request):
        return _deny()
    gid = _safe_id(gid)
    return JSONResponse(psql_json(
        "SELECT json_agg(json_build_object('key', cache_key, 'room', room, "
        "'desc', scene_description, 'image_url', image_url, "
        "'created', to_char(created_at, 'YYYY-MM-DD HH24:MI')) ORDER BY room, created_at) "
        f"FROM cached_scenes WHERE game_id = '{gid}';"
    ) or [])


if __name__ == "__main__":
    if not ADMIN_PASS:
        print("WARNING: ADMIN_PASS is not set in .env — login will always fail.")
    print(f"Fabulist admin on :{ADMIN_PORT}  (repo={REPO_DIR})")
    uvicorn.run(app, host="0.0.0.0", port=ADMIN_PORT)
