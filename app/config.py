"""Application settings, read from environment variables (and a local .env file)."""

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy.engine import URL, make_url

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")


@dataclass(frozen=True)
class Settings:
    app_name: str
    environment: str
    database_enabled: bool
    database_url: URL
    model_path: Path
    max_batch_size: int


def _build_database_url() -> URL:
    """``DATABASE_URL`` wins if set (e.g. on Render); otherwise build it from parts.

    Building from parts with ``URL.create`` means passwords containing special
    characters such as ``@`` or ``#`` do not need manual URL-encoding.
    """
    raw = os.getenv("DATABASE_URL")
    if raw:
        url = make_url(raw)
        if url.drivername in ("mysql", "mysql+mysqldb"):
            url = url.set(drivername="mysql+pymysql")
        if not url.drivername.startswith("mysql"):
            raise ValueError("Credence AI requires a MySQL DATABASE_URL (mysql+pymysql://...)")
        return url
    return URL.create(
        drivername="mysql+pymysql",
        username=os.getenv("MYSQL_USER", "root"),
        password=os.getenv("MYSQL_PASSWORD") or None,
        host=os.getenv("MYSQL_HOST", "localhost"),
        port=int(os.getenv("MYSQL_PORT", "3306")),
        database=os.getenv("MYSQL_DATABASE", "credence_ai"),
        query={"charset": "utf8mb4"},
    )


@lru_cache
def get_settings() -> Settings:
    return Settings(
        app_name="Credence AI",
        environment=os.getenv("APP_ENV", "development"),
        database_enabled=os.getenv("DATABASE_ENABLED", "true").strip().lower() in ("1", "true", "yes", "on"),
        database_url=_build_database_url(),
        model_path=Path(os.getenv("MODEL_PATH", PROJECT_ROOT / "models" / "credence_model.joblib")),
        max_batch_size=int(os.getenv("MAX_BATCH_SIZE", "1000")),
    )
