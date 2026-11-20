"""Temporary in-memory prediction history (no database).

Replaces the former MySQL tables. Predictions and batches live in the API
process's memory only:

* they are available to GET /predictions, GET /predictions/{id},
  DELETE /predictions/{id} and the /portfolio endpoints while the app runs;
* they are lost when the process restarts (e.g. a Render redeploy or the free
  plan going to sleep) -- there is NO permanent storage;
* at most ``MAX_STORED_PREDICTIONS`` predictions are kept; the oldest are
  evicted first so memory stays bounded;
* with several worker processes each worker would hold its own history, so the
  app is meant to run as a single uvicorn process (as configured for Render).

All operations take a lock because FastAPI runs sync endpoints in a thread pool.
"""

import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime

from app.predictor import ModelOutput
from app.schemas import CustomerFeatures, SegmentDimension

RISK_BAND_ORDER = ["LOW", "MEDIUM", "HIGH"]


@dataclass
class PredictionRecord:
    """One customer-level prediction: the 23 values the model used + outputs."""

    id: int
    batch_id: int | None
    customer_ref: str | None
    features: dict
    default_probability: float
    predicted_default: bool
    risk_band: str
    decision_threshold: float
    model_name: str
    model_version: str
    imputed_features: list[str]
    created_at: datetime = field(default_factory=lambda: datetime.now().replace(microsecond=0))

    def __getattr__(self, name):  # expose LIMIT_BAL, PAY_0, ... as attributes
        features = self.__dict__.get("features", {})
        if name in features:
            return features[name]
        raise AttributeError(name)


@dataclass
class BatchRecord:
    id: int
    name: str | None
    total_customers: int
    predicted_defaults: int
    average_default_probability: float
    created_at: datetime = field(default_factory=lambda: datetime.now().replace(microsecond=0))


class PredictionStore:
    def __init__(self, max_predictions: int = 10_000):
        self.max_predictions = max_predictions
        self._predictions: OrderedDict[int, PredictionRecord] = OrderedDict()
        self._batches: dict[int, BatchRecord] = {}
        self._next_prediction_id = 1
        self._next_batch_id = 1
        self._lock = threading.Lock()

    # ---- writes -------------------------------------------------------------
    def _add(self, customer: CustomerFeatures, output: ModelOutput, predictor, batch_id=None) -> PredictionRecord:
        record = PredictionRecord(
            id=self._next_prediction_id,
            batch_id=batch_id,
            customer_ref=customer.customer_ref,
            features=dict(output.model_input),
            default_probability=output.default_probability,
            predicted_default=output.predicted_default,
            risk_band=output.risk_band,
            decision_threshold=predictor.threshold,
            model_name=predictor.model_name,
            model_version=predictor.model_version,
            imputed_features=list(output.imputed_features),
        )
        self._next_prediction_id += 1
        self._predictions[record.id] = record
        while len(self._predictions) > self.max_predictions:
            self._predictions.popitem(last=False)  # evict the oldest
        return record

    def create_prediction(self, customer: CustomerFeatures, output: ModelOutput, predictor) -> PredictionRecord:
        with self._lock:
            return self._add(customer, output, predictor)

    def create_batch(self, name: str | None, customers: list[CustomerFeatures], outputs: list[ModelOutput],
                     predictor) -> tuple[BatchRecord, list[PredictionRecord]]:
        with self._lock:
            batch = BatchRecord(
                id=self._next_batch_id,
                name=name,
                total_customers=len(outputs),
                predicted_defaults=sum(o.predicted_default for o in outputs),
                average_default_probability=round(sum(o.default_probability for o in outputs) / len(outputs), 6),
            )
            self._next_batch_id += 1
            self._batches[batch.id] = batch
            rows = [self._add(c, o, predictor, batch_id=batch.id) for c, o in zip(customers, outputs)]
            return batch, rows

    def delete_prediction(self, prediction_id: int) -> bool:
        with self._lock:
            return self._predictions.pop(prediction_id, None) is not None

    def clear(self) -> None:
        with self._lock:
            self._predictions.clear()
            self._batches.clear()
            self._next_prediction_id = 1
            self._next_batch_id = 1

    # ---- reads --------------------------------------------------------------
    def __len__(self) -> int:
        return len(self._predictions)

    def get_prediction(self, prediction_id: int) -> PredictionRecord | None:
        return self._predictions.get(prediction_id)

    def get_batch(self, batch_id: int) -> BatchRecord | None:
        return self._batches.get(batch_id)

    def _snapshot(self, batch_id: int | None = None) -> list[PredictionRecord]:
        with self._lock:
            rows = list(self._predictions.values())
        return rows if batch_id is None else [r for r in rows if r.batch_id == batch_id]

    def list_predictions(self, limit: int, offset: int, risk_band: str | None = None,
                         predicted_default: bool | None = None, batch_id: int | None = None,
                         customer_ref: str | None = None, min_probability: float | None = None):
        rows = [
            r for r in self._snapshot(batch_id)
            if (risk_band is None or r.risk_band == risk_band)
            and (predicted_default is None or r.predicted_default == predicted_default)
            and (customer_ref is None or r.customer_ref == customer_ref)
            and (min_probability is None or r.default_probability >= min_probability)
        ]
        rows.sort(key=lambda r: (r.default_probability, r.id), reverse=True)  # riskiest first
        return len(rows), rows[offset:offset + limit]

    def portfolio_summary(self, batch_id: int | None = None) -> dict:
        rows = self._snapshot(batch_id)
        total = len(rows)
        defaults = sum(r.predicted_default for r in rows)
        total_limit = sum(r.features["LIMIT_BAL"] for r in rows)
        limit_at_risk = sum(r.features["LIMIT_BAL"] for r in rows if r.predicted_default)
        band_counts: dict[str, int] = {}
        for r in rows:
            band_counts[r.risk_band] = band_counts.get(r.risk_band, 0) + 1
        return {
            "batch_id": batch_id,
            "total_customers": total,
            "predicted_defaults": defaults,
            "predicted_default_rate": round(defaults / total, 4) if total else 0.0,
            "average_default_probability": round(sum(r.default_probability for r in rows) / total, 4) if total else 0.0,
            "total_credit_limit": total_limit,
            "credit_limit_of_predicted_defaulters": limit_at_risk,
            "share_of_credit_limit_at_risk": round(limit_at_risk / total_limit, 4) if total_limit else 0.0,
            "risk_band_breakdown": risk_band_breakdown(band_counts, total),
        }

    def segment_analysis(self, dimension: SegmentDimension, batch_id: int | None = None) -> list[dict]:
        """Predicted default risk per customer segment -- portfolio risk pockets."""
        groups: dict[str, list[PredictionRecord]] = {}
        for r in self._snapshot(batch_id):
            groups.setdefault(_segment_key(dimension, r.features), []).append(r)
        result = [
            {
                "segment": segment,
                "customers": len(rs),
                "predicted_defaults": sum(r.predicted_default for r in rs),
                "predicted_default_rate": round(sum(r.predicted_default for r in rs) / len(rs), 4),
                "average_default_probability": round(sum(r.default_probability for r in rs) / len(rs), 4),
            }
            for segment, rs in groups.items()
        ]
        return sorted(result, key=lambda x: x["average_default_probability"], reverse=True)


def risk_band_breakdown(counts: dict[str, int], total: int) -> list[dict]:
    return [
        {"risk_band": band, "customers": counts.get(band, 0),
         "share": round(counts.get(band, 0) / total, 4) if total else 0.0}
        for band in RISK_BAND_ORDER
    ]


_SEGMENT_LABELS = {
    SegmentDimension.SEX: {1: "Male", 2: "Female"},
    SegmentDimension.EDUCATION: {1: "Graduate school", 2: "University", 3: "High school", 4: "Others",
                                 0: "Undocumented (0)", 5: "Undocumented (5)", 6: "Undocumented (6)"},
    SegmentDimension.MARRIAGE: {1: "Married", 2: "Single", 3: "Others", 0: "Undocumented (0)"},
}


def _segment_key(dimension: SegmentDimension, f: dict) -> str:
    if dimension == SegmentDimension.AGE_GROUP:
        age = f["AGE"]
        return "18-24" if age < 25 else "25-34" if age < 35 else "35-44" if age < 45 else "45-54" if age < 55 else "55+"
    if dimension == SegmentDimension.PAY_0:
        p = f["PAY_0"]
        return "Paid duly / no delay" if p <= 0 else "1 month late" if p == 1 else "2 months late" if p == 2 \
            else "3+ months late"
    value = f[dimension.value]
    return _SEGMENT_LABELS[dimension].get(value, str(value))
