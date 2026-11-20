"""Shared FastAPI dependencies."""

from fastapi import HTTPException, Request, status

from app.predictor import CreditDefaultPredictor
from app.store import PredictionStore


def get_predictor(request: Request) -> CreditDefaultPredictor:
    predictor = getattr(request.app.state, "predictor", None)
    if predictor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not loaded. Train it with `python -m ml.train`.",
        )
    return predictor


def get_store(request: Request) -> PredictionStore:
    """The in-memory prediction history created at application startup."""
    return request.app.state.store
