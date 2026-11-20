"""Portfolio-level analytics for the Credit Portfolio Strategist."""

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from app import crud
from app.dependencies import require_db
from app.schemas import PortfolioSummary, SegmentAnalysis, SegmentDimension

router = APIRouter(prefix="/portfolio", tags=["Portfolio analytics"])


def _check_batch(db: Session, batch_id: int | None) -> None:
    if batch_id is not None and crud.get_batch(db, batch_id) is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail=f"Batch {batch_id} not found")


@router.get("/summary", response_model=PortfolioSummary,
            summary="Portfolio default-risk summary (all stored predictions or one batch)")
def summary(batch_id: int | None = Query(None, ge=1), db: Session = Depends(require_db)):
    _check_batch(db, batch_id)
    return crud.portfolio_summary(db, batch_id)


@router.get("/segments", response_model=SegmentAnalysis,
            summary="Predicted default risk per customer segment, riskiest first")
def segments(
    dimension: SegmentDimension = Query(SegmentDimension.PAY_0),
    batch_id: int | None = Query(None, ge=1),
    db: Session = Depends(require_db),
):
    _check_batch(db, batch_id)
    return {"dimension": dimension, "batch_id": batch_id,
            "segments": crud.segment_analysis(db, dimension, batch_id)}
