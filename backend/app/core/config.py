import re
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_DIR = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=BACKEND_DIR / ".env", extra="ignore")

    environment: str = "development"  # "production" on Railway: HTTPS-only cookies
    secret_key: str = "change-me-in-backend-env"
    # Supabase Postgres, e.g. postgresql+psycopg://postgres:<pw>@db.<ref>.supabase.co:5432/postgres
    database_url: str
    frontend_origins: list[str] = ["http://localhost:5173", "http://127.0.0.1:5173"]

    # Supabase Storage. The project URL is derived from DATABASE_URL when not set.
    supabase_url: str | None = None
    supabase_service_role_key: str | None = None
    media_bucket: str = "booktown"  # covers, category images, profile photos (public-read)
    pdf_bucket: str = "booktown"  # book PDFs (public-read, opened in the browser)

    groq_api_key: str | None = None
    groq_model: str = "openai/gpt-oss-120b"  # recommendation relevance scoring
    groq_fast_model: str = "openai/gpt-oss-20b"  # sentiment scoring

    # Where fastembed keeps the ONNX embedding model (downloaded once, ~90 MB).
    model_cache_dir: Path = BACKEND_DIR / ".cache" / "fastembed"

    razorpay_key_id: str = ""
    razorpay_key_secret: str = ""

    # Session cookie lifetime (seconds): 14 days
    session_max_age: int = 60 * 60 * 24 * 14

    @property
    def is_production(self) -> bool:
        return self.environment == "production"

    @property
    def storage_url(self) -> str:
        if self.supabase_url:
            return self.supabase_url.rstrip("/")
        match = re.search(r"@db\.([a-z0-9]+)\.supabase\.co", self.database_url)
        if not match:
            raise RuntimeError("Set SUPABASE_URL in backend/.env")
        return f"https://{match.group(1)}.supabase.co"


@lru_cache
def get_settings() -> Settings:
    s = Settings()
    if s.is_production and s.secret_key == "change-me-in-backend-env":
        raise RuntimeError("Set SECRET_KEY before running in production.")
    return s


settings = get_settings()
