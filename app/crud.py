"""Database operations for predictions and portfolio analytics."""

from sqlalchemy import case, func, select
from sqlalchemy.orm import Session

from app.models import Prediction, PredictionBatch
from app.predictor import ModelOutput
from app.schemas import CustomerFeatures, SegmentDimension

RISK_BAND_ORDER = ["LOW", "MEDIUM", "HIGH"]


def _build_prediction(customer: CustomerFeatures, output: ModelOutput, predictor, batch_id=None) -> Prediction:
    return Prediction(
        batch_id=batch_id,
        customer_ref=customer.customer_ref,
        **output.model_input,
        imputed_features=",".join(output.imputed_features) or None,
        default_probability=output.default_probability,
        predicted_default=output.predicted_default,
        risk_band=output.risk_band,
        decision_threshold=predictor.threshold,
        model_name=predictor.model_name,
        model_version=predictor.model_version,
    )


def create_prediction(db: Session, customer: CustomerFeatures, output: ModelOutput, predictor) -> Prediction:
    row = _build_prediction(customer, output, predictor)
    db.add(row)
    db.commit()
    db.refresh(row)
    return row


def create_batch(
    db: Session, name: str | None, customers: list[CustomerFeatures], outputs: list[ModelOutput], predictor
) -> tuple[PredictionBatch, list[Prediction]]:
    """Store a batch and all of its predictions in ONE transaction."""
    batch = PredictionBatch(
        name=name,
        total_customers=len(outputs),
        predicted_defaults=sum(o.predicted_default for o in outputs),
        average_default_probability=round(sum(o.default_probability for o in outputs) / len(outputs), 6),
    )
    try:
        db.add(batch)
        db.flush()  # assigns batch.id
        rows = [_build_prediction(c, o, predictor, batch_id=batch.id) for c, o in zip(customers, outputs)]
        db.add_all(rows)
        db.commit()
    except Exception:
        db.rollback()
        raise
    db.refresh(batch)
    for row in rows:
        db.refresh(row)
    return batch, rows


def get_prediction(db: Session, prediction_id: int) -> Prediction | None:
    return db.get(Prediction, prediction_id)


def list_predictions(
    db: Session,
    limit: int,
    offset: int,
    risk_band: str | None = None,
    predicted_default: bool | None = None,
    batch_id: int | None = None,
    customer_ref: str | None = None,
    min_probability: float | None = None,
) -> tuple[int, list[Prediction]]:
    filters = []
    if risk_band is not None:
        filters.append(Prediction.risk_band == risk_band)
    if predicted_default is not None:
        filters.append(Prediction.predicted_default == predicted_default)
    if batch_id is not None:
        filters.append(Prediction.batch_id == batch_id)
    if customer_ref is not None:
        filters.append(Prediction.customer_ref == customer_ref)
    if min_probability is not None:
        filters.append(Prediction.default_probability >= min_probability)

    total = db.scalar(select(func.count(Prediction.id)).where(*filters)) or 0
    items = db.scalars(
        select(Prediction)
        .where(*filters)
        .order_by(Prediction.default_probability.desc(), Prediction.id.desc())
        .limit(limit)
        .offset(offset)
    ).all()
    return total, list(items)


def delete_prediction(db: Session, prediction_id: int) -> bool:
    row = db.get(Prediction, prediction_id)
    if row is None:
        return False
    db.delete(row)
    db.commit()
    return True


def get_batch(db: Session, batch_id: int) -> PredictionBatch | None:
    return db.get(PredictionBatch, batch_id)


def risk_band_breakdown(counts: dict[str, int], total: int) -> list[dict]:
    return [
        {"risk_band": band, "customers": counts.get(band, 0),
         "share": round(counts.get(band, 0) / total, 4) if total else 0.0}
        for band in RISK_BAND_ORDER
    ]


def portfolio_summary(db: Session, batch_id: int | None = None) -> dict:
    where = [Prediction.batch_id == batch_id] if batch_id is not None else []
    defaults_int = case((Prediction.predicted_default.is_(True), 1), else_=0)
    row = db.execute(
        select(
            func.count(Prediction.id),
            func.coalesce(func.sum(defaults_int), 0),
            func.coalesce(func.avg(Prediction.default_probability), 0.0),
            func.coalesce(func.sum(Prediction.LIMIT_BAL), 0),
            func.coalesce(func.sum(case((Prediction.predicted_default.is_(True), Prediction.LIMIT_BAL), else_=0)), 0),
        ).where(*where)
    ).one()
    total, defaults, avg_prob, total_limit, limit_at_risk = row
    band_counts = dict(
        db.execute(
            select(Prediction.risk_band, func.count(Prediction.id)).where(*where).group_by(Prediction.risk_band)
        ).all()
    )
    total, defaults = int(total), int(defaults)
    total_limit, limit_at_risk = int(total_limit), int(limit_at_risk)
    return {
        "batch_id": batch_id,
        "total_customers": total,
        "predicted_defaults": defaults,
        "predicted_default_rate": round(defaults / total, 4) if total else 0.0,
        "average_default_probability": round(float(avg_prob), 4),
        "total_credit_limit": total_limit,
        "credit_limit_of_predicted_defaulters": limit_at_risk,
        "share_of_credit_limit_at_risk": round(limit_at_risk / total_limit, 4) if total_limit else 0.0,
        "risk_band_breakdown": risk_band_breakdown(band_counts, total),
    }


_SEGMENT_LABELS = {
    SegmentDimension.SEX: {1: "Male", 2: "Female"},
    SegmentDimension.EDUCATION: {1: "Graduate school", 2: "University", 3: "High school", 4: "Others",
                                 0: "Undocumented (0)", 5: "Undocumented (5)", 6: "Undocumented (6)"},
    SegmentDimension.MARRIAGE: {1: "Married", 2: "Single", 3: "Others", 0: "Undocumented (0)"},
}


def segment_analysis(db: Session, dimension: SegmentDimension, batch_id: int | None = None) -> list[dict]:
    """Predicted default risk per customer segment -- portfolio risk pockets."""
    if dimension == SegmentDimension.AGE_GROUP:
        key = case(
            (Prediction.AGE < 25, "18-24"),
            (Prediction.AGE < 35, "25-34"),
            (Prediction.AGE < 45, "35-44"),
            (Prediction.AGE < 55, "45-54"),
            else_="55+",
        )
    elif dimension == SegmentDimension.PAY_0:
        key = case(
            (Prediction.PAY_0 <= 0, "Paid duly / no delay"),
            (Prediction.PAY_0 == 1, "1 month late"),
            (Prediction.PAY_0 == 2, "2 months late"),
            else_="3+ months late",
        )
    else:
        key = getattr(Prediction, dimension.value)

    where = [Prediction.batch_id == batch_id] if batch_id is not None else []
    defaults_int = case((Prediction.predicted_default.is_(True), 1), else_=0)
    rows = db.execute(
        select(
            key.label("segment"),
            func.count(Prediction.id),
            func.sum(defaults_int),
            func.avg(Prediction.default_probability),
        )
        .where(*where)
        .group_by("segment")
    ).all()

    labels = _SEGMENT_LABELS.get(dimension, {})
    result = [
        {
            "segment": labels.get(seg, str(seg)),
            "customers": int(n),
            "predicted_defaults": int(d or 0),
            "predicted_default_rate": round(int(d or 0) / int(n), 4) if n else 0.0,
            "average_default_probability": round(float(p or 0), 4),
        }
        for seg, n, d, p in rows
    ]
    return sorted(result, key=lambda r: r["average_default_probability"], reverse=True)
