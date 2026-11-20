"""Shared FastAPI dependencies."""

from collections.abc import Generator

from fastapi import HTTPException, Request, status
from sqlalchemy.orm import Session

from app.database import get_db
from app.predictor import CreditDefaultPredictor


def get_predictor(request: Request) -> CreditDefaultPredictor:
    predictor = getattr(request.app.state, "predictor", None)
    if predictor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not loaded. Train it with `python -m ml.train`.",
        )
    return predictor


def database_enabled(request: Request) -> bool:
    return bool(getattr(request.app.state, "database_enabled", True))


def optional_db(request: Request) -> Generator[Session | None, None, None]:
    """A session when the database is enabled, otherwise None (predict-only mode)."""
    if not database_enabled(request):
        yield None
        return
    yield from get_db()


def require_db(request: Request) -> Generator[Session, None, None]:
    """For endpoints that only make sense with stored history."""
    if not database_enabled(request):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="The database is disabled in this deployment (DATABASE_ENABLED=false), so prediction "
                   "history and portfolio analytics are unavailable. POST /predictions still works.",
        )
    yield from get_db()
