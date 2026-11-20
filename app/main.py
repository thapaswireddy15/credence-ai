"""Credence AI - FastAPI application entry point.

Run locally:  uvicorn app.main:app --reload
Swagger UI:   http://127.0.0.1:8000/docs

No database is used: prediction history is kept in memory for the lifetime of
the process (see app/store.py).
"""

import logging
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Request

from app.config import get_settings
from app.dependencies import get_predictor
from app.predictor import CreditDefaultPredictor
from app.routers import portfolio, predictions
from app.schemas import HealthResponse, ModelInfo
from app.store import PredictionStore

logger = logging.getLogger("credence_ai")
logging.basicConfig(level=logging.INFO, format="%(levelname)s:     %(name)s - %(message)s")

STORAGE_DESCRIPTION = "in-memory (temporary, cleared on restart)"


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    try:
        app.state.predictor = CreditDefaultPredictor(settings.model_path)
        logger.info("Loaded model %s v%s", app.state.predictor.model_name, app.state.predictor.model_version)
    except Exception as exc:  # the API still starts; /health reports the problem
        app.state.predictor = None
        logger.error("Could not load model from %s: %s", settings.model_path, exc)

    app.state.store = PredictionStore(max_predictions=settings.max_stored_predictions)
    logger.info("Prediction history: %s, max %d predictions", STORAGE_DESCRIPTION,
                settings.max_stored_predictions)
    yield
    app.state.store.clear()  # nothing is persisted


app = FastAPI(
    title="Credence AI - Credit Card Default Prediction API",
    version="1.0.0",
    description=(
        "Backend for the **Credit Portfolio Strategist**: predicts whether a credit card customer "
        "is likely to default on next month's payment and provides portfolio-level risk analytics.\n\n"
        "**Missing values:** up to 11 of the 23 features may be omitted or `null`; they are imputed with "
        "training-split medians / modes and listed in `imputed_features`.\n\n"
        "**Storage:** no database. Predictions are kept in a temporary **in-memory history** while the "
        "service runs, so the history and portfolio endpoints work, but everything is cleared when "
        "the service restarts."
    ),
    lifespan=lifespan,
)
app.include_router(predictions.router)
app.include_router(portfolio.router)


@app.get("/", tags=["Service"], summary="Service information")
def root():
    return {
        "service": "Credence AI - Credit Card Default Prediction API",
        "stakeholder": "Credit Portfolio Strategist",
        "docs": "/docs",
        "health": "/health",
        "storage": STORAGE_DESCRIPTION,
    }


@app.get("/health", response_model=HealthResponse, tags=["Service"], summary="Liveness and model availability")
def health(request: Request):
    predictor = getattr(request.app.state, "predictor", None)
    store = getattr(request.app.state, "store", None)
    return HealthResponse(
        status="ok" if predictor is not None else "degraded",
        model_loaded=predictor is not None,
        missing_value_imputation=bool(predictor and predictor.supports_missing_values),
        storage=STORAGE_DESCRIPTION,
        predictions_in_memory=len(store) if store is not None else 0,
    )


@app.get("/model/info", response_model=ModelInfo, tags=["Model"], summary="Selected model, threshold and metrics")
def model_info(predictor: CreditDefaultPredictor = Depends(get_predictor)):
    return ModelInfo(**{k: predictor.metadata[k] for k in ModelInfo.model_fields})
