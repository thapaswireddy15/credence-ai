"""Loads the trained pipeline once and serves predictions.

The joblib artifact contains the full scikit-learn Pipeline (preprocessing +
classifier), so inference applies exactly the transformations fitted during
training -- the API never re-implements preprocessing by hand.

Missing features (None) are filled with training-split medians / modes from
``models/imputation_values.json`` (see ``ml/imputation.py``) BEFORE the
unchanged pipeline is called. The trained model itself is not modified.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import pandas as pd

from ml.config import FEATURES, IMPUTATION_PATH, risk_band


@dataclass
class ModelOutput:
    default_probability: float
    predicted_default: bool
    risk_band: str
    imputed_features: list[str] = field(default_factory=list)
    model_input: dict = field(default_factory=dict)  # the 23 values the model actually used


class CreditDefaultPredictor:
    def __init__(self, model_path: Path, imputation_path: Path = IMPUTATION_PATH):
        artifact = joblib.load(model_path)
        self.pipeline = artifact["pipeline"]
        self.metadata: dict = artifact["metadata"]
        self.threshold: float = float(self.metadata["decision_threshold"])
        self.model_name: str = self.metadata["model_name"]
        self.model_version: str = self.metadata["model_version"]

        if list(self.metadata["features"]) != FEATURES:
            raise ValueError("Model artifact features do not match ml.config.FEATURES")

        self.fill_values: dict[str, int] | None = None
        if Path(imputation_path).exists():
            values = json.loads(Path(imputation_path).read_text())["fill_values"]
            if set(values) != set(FEATURES):
                raise ValueError("imputation_values.json does not cover all 23 features")
            self.fill_values = {k: int(v) for k, v in values.items()}

    @property
    def supports_missing_values(self) -> bool:
        return self.fill_values is not None

    def _complete(self, record: dict) -> tuple[dict, list[str]]:
        missing = [f for f in FEATURES if record.get(f) is None]
        if missing and self.fill_values is None:
            raise ValueError("Missing-value imputation is unavailable (models/imputation_values.json not found)")
        filled = {f: (self.fill_values[f] if record.get(f) is None else int(record[f])) for f in FEATURES}
        return filled, missing

    def predict(self, records: list[dict]) -> list[ModelOutput]:
        completed = [self._complete(r) for r in records]
        frame = pd.DataFrame.from_records([c[0] for c in completed], columns=FEATURES)
        probabilities = self.pipeline.predict_proba(frame)[:, 1]
        return [
            ModelOutput(
                default_probability=round(float(p), 6),
                predicted_default=bool(p >= self.threshold),
                risk_band=risk_band(float(p)),
                imputed_features=missing,
                model_input=filled,
            )
            for p, (filled, missing) in zip(probabilities, completed)
        ]
