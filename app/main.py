"""Credence AI - FastAPI application entry point.

Run locally:  uvicorn app.main:app --reload
Swagger UI:   http://127.0.0.1:8000/docs
"""

import logging
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Request, status
from fastapi.responses import JSONResponse
from sqlalchemy.exc import OperationalError, SQLAlchemyError

from app.config import get_settings
from app.database import check_db_connection, init_db
from app.dependencies import get_predictor
from app.predictor import CreditDefaultPredictor
from app.routers import portfolio, predictions
from app.schemas import HealthResponse, ModelInfo

logger = logging.getLogger("credence_ai")
logging.basicConfig(level=logging.INFO, format="%(levelname)s:     %(name)s - %(message)s")


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    try:
        app.state.predictor = CreditDefaultPredictor(settings.model_path)
        logger.info("Loaded model %s v%s", app.state.predictor.model_name, app.state.predictor.model_version)
    except Exception as exc:  # the API still starts; /health reports the problem
        app.state.predictor = None
        logger.error("Could not load model from %s: %s", settings.model_path, exc)

    app.state.database_enabled = settings.database_enabled
    if not settings.database_enabled:
        logger.warning("DATABASE_ENABLED=false: predict-only mode, nothing is stored")
    else:
        try:
            init_db()
            logger.info("MySQL ready at %s:%s/%s", settings.database_url.host,
                        settings.database_url.port, settings.database_url.database)
        except Exception as exc:
            logger.error("MySQL initialisation failed: %s", exc)
    yield


app = FastAPI(
    title="Credence AI - Credit Card Default Prediction API",
    version="1.0.0",
    description=(
        "Backend for the **Credit Portfolio Strategist**: predicts whether a credit card customer "
        "is likely to default on next month's payment, stores every prediction in MySQL and "
        "provides portfolio-level risk analytics.\n\n"
        "**Missing values:** up to 11 of the 23 features may be omitted or `null`; they are imputed with "
        "training-split medians / modes and listed in `imputed_features`.\n\n"
        "**Database mode:** with `DATABASE_ENABLED=false` the API runs in predict-only mode "
        "(`stored: false`); history and portfolio endpoints then return 503."
    ),
    lifespan=lifespan,
)
app.include_router(predictions.router)
app.include_router(portfolio.router)


@app.exception_handler(OperationalError)
async def database_unavailable_handler(request: Request, exc: OperationalError):
    logger.error("Database error: %s", exc.orig)
    return JSONResponse(status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                        content={"detail": "Database is unavailable. Check the MySQL connection settings."})


@app.exception_handler(SQLAlchemyError)
async def database_error_handler(request: Request, exc: SQLAlchemyError):
    logger.exception("Unexpected database error")
    return JSONResponse(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                        content={"detail": "Unexpected database error."})


@app.get("/", tags=["Service"], summary="Service information")
def root():
    return {
        "service": "Credence AI - Credit Card Default Prediction API",
        "stakeholder": "Credit Portfolio Strategist",
        "docs": "/docs",
        "health": "/health",
        "database_mode": "enabled" if getattr(app.state, "database_enabled", True) else "disabled",
    }


@app.get("/health", response_model=HealthResponse, tags=["Service"], summary="Liveness and dependency check")
def health(request: Request):
    predictor = getattr(request.app.state, "predictor", None)
    model_loaded = predictor is not None
    enabled = bool(getattr(request.app.state, "database_enabled", True))
    db_ok = check_db_connection() if enabled else None
    healthy = model_loaded and (db_ok if enabled else True)
    return HealthResponse(
        status="ok" if healthy else "degraded",
        model_loaded=model_loaded,
        missing_value_imputation=bool(predictor and predictor.supports_missing_values),
        database_mode="enabled" if enabled else "disabled",
        database_connected=db_ok,
    )


@app.get("/model/info", response_model=ModelInfo, tags=["Model"], summary="Selected model, threshold and metrics")
def model_info(predictor: CreditDefaultPredictor = Depends(get_predictor)):
    return ModelInfo(**{k: predictor.metadata[k] for k in ModelInfo.model_fields})
