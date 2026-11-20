"""Portfolio-level analytics for the Credit Portfolio Strategist."""

from fastapi import APIRouter, Depends, HTTPException, Query, status

from app.dependencies import get_store
from app.schemas import PortfolioSummary, SegmentAnalysis, SegmentDimension
from app.store import PredictionStore

router = APIRouter(prefix="/portfolio", tags=["Portfolio analytics"])


def _check_batch(store: PredictionStore, batch_id: int | None) -> None:
    if batch_id is not None and store.get_batch(batch_id) is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail=f"Batch {batch_id} not found")


@router.get("/summary", response_model=PortfolioSummary,
            summary="Portfolio default-risk summary (in-memory history, or one batch)")
def summary(batch_id: int | None = Query(None, ge=1), store: PredictionStore = Depends(get_store)):
    _check_batch(store, batch_id)
    return store.portfolio_summary(batch_id)


@router.get("/segments", response_model=SegmentAnalysis,
            summary="Predicted default risk per customer segment, riskiest first")
def segments(
    dimension: SegmentDimension = Query(SegmentDimension.PAY_0),
    batch_id: int | None = Query(None, ge=1),
    store: PredictionStore = Depends(get_store),
):
    _check_batch(store, batch_id)
    return {"dimension": dimension, "batch_id": batch_id,
            "segments": store.segment_analysis(dimension, batch_id)}
