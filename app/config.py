"""Application settings, read from environment variables (and a local .env file).

Credence AI has no database: prediction history is kept in memory only
(see app/store.py).
"""

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")


@dataclass(frozen=True)
class Settings:
    app_name: str
    environment: str
    model_path: Path
    max_batch_size: int
    max_stored_predictions: int


@lru_cache
def get_settings() -> Settings:
    return Settings(
        app_name="Credence AI",
        environment=os.getenv("APP_ENV", "development"),
        model_path=Path(os.getenv("MODEL_PATH", PROJECT_ROOT / "models" / "credence_model.joblib")),
        max_batch_size=int(os.getenv("MAX_BATCH_SIZE", "1000")),
        max_stored_predictions=int(os.getenv("MAX_STORED_PREDICTIONS", "10000")),
    )
