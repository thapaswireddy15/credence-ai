"""Customer-level prediction endpoints."""

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from app import crud
from app.config import get_settings
from app.dependencies import get_predictor, optional_db, require_db
from app.predictor import CreditDefaultPredictor, ModelOutput
from app.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    CustomerFeatures,
    PredictionList,
    PredictionResult,
    RiskBand,
    StoredPrediction,
)

router = APIRouter(prefix="/predictions", tags=["Predictions"])


def _result(output: ModelOutput, predictor: CreditDefaultPredictor, customer_ref, row=None) -> PredictionResult:
    return PredictionResult(
        prediction_id=row.id if row is not None else None,
        stored=row is not None,
        customer_ref=customer_ref,
        default_probability=output.default_probability,
        predicted_default=output.predicted_default,
        prediction_label="Default" if output.predicted_default else "No default",
        risk_band=output.risk_band,
        decision_threshold=predictor.threshold,
        model_name=predictor.model_name,
        model_version=predictor.model_version,
        imputed_features=output.imputed_features,
        created_at=row.created_at if row is not None else None,
    )


def _predict(predictor: CreditDefaultPredictor, customers: list[CustomerFeatures]) -> list[ModelOutput]:
    try:
        return predictor.predict([c.feature_dict() for c in customers])
    except ValueError as exc:  # e.g. imputation artifact missing
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(exc)) from exc


@router.post("", response_model=PredictionResult, status_code=status.HTTP_201_CREATED,
             summary="Predict default for one customer (missing features are imputed) and store it")
def predict_customer(
    customer: CustomerFeatures,
    db: Session | None = Depends(optional_db),
    predictor: CreditDefaultPredictor = Depends(get_predictor),
):
    output = _predict(predictor, [customer])[0]
    row = crud.create_prediction(db, customer, output, predictor) if db is not None else None
    return _result(output, predictor, customer.customer_ref, row)


@router.post("/batch", response_model=BatchPredictionResponse, status_code=status.HTTP_201_CREATED,
             summary="Score a portfolio of customers in one request")
def predict_batch(
    request: BatchPredictionRequest,
    db: Session | None = Depends(optional_db),
    predictor: CreditDefaultPredictor = Depends(get_predictor),
):
    max_size = get_settings().max_batch_size
    if len(request.customers) > max_size:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT,
                            detail=f"A batch may contain at most {max_size} customers")
    outputs = _predict(predictor, request.customers)

    batch, rows = None, [None] * len(outputs)
    if db is not None:
        batch, rows = crud.create_batch(db, request.batch_name, request.customers, outputs, predictor)

    total = len(outputs)
    defaults = sum(o.predicted_default for o in outputs)
    band_counts: dict[str, int] = {}
    for o in outputs:
        band_counts[o.risk_band] = band_counts.get(o.risk_band, 0) + 1
    return BatchPredictionResponse(
        batch_id=batch.id if batch is not None else None,
        stored=batch is not None,
        batch_name=request.batch_name,
        total_customers=total,
        predicted_defaults=defaults,
        predicted_default_rate=round(defaults / total, 4),
        average_default_probability=round(sum(o.default_probability for o in outputs) / total, 4),
        customers_with_imputed_values=sum(bool(o.imputed_features) for o in outputs),
        risk_band_breakdown=crud.risk_band_breakdown(band_counts, total),
        predictions=[_result(o, predictor, c.customer_ref, r)
                     for o, c, r in zip(outputs, request.customers, rows)],
    )


@router.get("", response_model=PredictionList, summary="List stored predictions (riskiest first)")
def list_predictions(
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    risk_band: RiskBand | None = None,
    predicted_default: bool | None = None,
    batch_id: int | None = Query(None, ge=1),
    customer_ref: str | None = Query(None, max_length=64),
    min_probability: float | None = Query(None, ge=0, le=1),
    db: Session = Depends(require_db),
):
    total, items = crud.list_predictions(
        db, limit, offset,
        risk_band=risk_band.value if risk_band else None,
        predicted_default=predicted_default,
        batch_id=batch_id,
        customer_ref=customer_ref,
        min_probability=min_probability,
    )
    return PredictionList(total=total, limit=limit, offset=offset,
                          items=[StoredPrediction.model_validate(i) for i in items])


@router.get("/{prediction_id}", response_model=StoredPrediction, summary="Get one stored prediction")
def get_prediction(prediction_id: int, db: Session = Depends(require_db)):
    row = crud.get_prediction(db, prediction_id)
    if row is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail=f"Prediction {prediction_id} not found")
    return StoredPrediction.model_validate(row)


@router.delete("/{prediction_id}", status_code=status.HTTP_204_NO_CONTENT, summary="Delete a stored prediction")
def delete_prediction(prediction_id: int, db: Session = Depends(require_db)):
    if not crud.delete_prediction(db, prediction_id):
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail=f"Prediction {prediction_id} not found")
