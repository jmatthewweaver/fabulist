from pydantic_settings import BaseSettings
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent


class Settings(BaseSettings):
    # Claude
    anthropic_api_key: str
    model_translation: str = "claude-haiku-4-5"
    model_enrichment: str = "claude-sonnet-4-6"

    # OpenAI (Whisper + TTS)
    openai_api_key: str

    # Image generation (BFL)
    bfl_api_key: str
    bfl_model_desktop: str = "flux-2-pro"       # from-scratch anchors, best quality
    bfl_model_mobile: str = "flux-2-klein-4b"   # fast + cheap for low-res
    # Within-room state edits use the dedicated EDITING model, which preserves the input
    # image's pixels and changes only the requested region. flux-2-pro's input_image is
    # reference-CONDITIONED regeneration (it redraws the whole scene and brightens it ~+50
    # luminance every pass); Kontext keeps the anchor's exposure/grade and edits in place.
    bfl_model_edit: str = "flux-kontext-pro"

    # Auth
    google_client_id: str
    google_client_secret: str
    jwt_secret: str
    jwt_algorithm: str = "HS256"
    jwt_expire_hours: int = 24 * 30  # 30 days

    # Database
    database_url: str = "postgresql+asyncpg://postgres:postgres@localhost/fabulist"

    # Paths
    games_dir: Path = BASE_DIR / "games"
    saves_dir: Path = BASE_DIR / "saves"
    images_dir: Path = BASE_DIR / "images"

    # Game engine binaries
    dfrotz_path: str = "dfrotz"
    infodump_path: str = "infodump"
    txd_path: str = "txd"

    force_regen: bool = False       # dev: bypass + overwrite the scene/image cache

    # App
    frontend_url: str = "http://localhost:3000"
    debug: bool = False
    log_level: str = "INFO"
    # SQL statement logging is its own switch — even under debug the per-query echo floods
    # the logs without being useful. Turn this on only to trace a specific DB problem.
    db_echo: bool = False

    class Config:
        # Absolute so .env loads no matter the working directory (e.g. test scripts
        # run from ~/work, not the project root).
        env_file = str(BASE_DIR / ".env")
        # The .env is shared with admin.py (ADMIN_*, BACKEND_CMD, etc.); ignore keys
        # that aren't fields here instead of erroring.
        extra = "ignore"


settings = Settings()
