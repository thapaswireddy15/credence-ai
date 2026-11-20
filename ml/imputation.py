"""Missing-value handling for inference, added WITHOUT retraining the model.

Why a separate step?
--------------------
The training data has no missing values, so the trained pipeline
(``models/credence_model.joblib``) has no imputer. Real portfolio records can
be incomplete, though. Rather than retrain, we compute fill values from the
**training split only** (the same 80:20, random_state=42 split) and apply them
before the unchanged pipeline:

* numeric / ordinal features (LIMIT_BAL, AGE, PAY_*, BILL_AMT*, PAY_AMT*): median
  (robust to the heavy skew and outliers seen in the EDA)
* nominal categorical features (SEX, EDUCATION, MARRIAGE): most frequent value

Because no training row was ever missing, fitting an imputer inside the
pipeline would have been a no-op during training. Applying it in front of the
pipeline therefore keeps training and inference preprocessing consistent.

Run:  python -m ml.imputation
Writes models/imputation_values.json and
reports/metrics/missing_value_robustness.json.
"""

import json
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score

from ml.config import (
    CATEGORICAL_FEATURES,
    FEATURES,
    IMPUTATION_PATH,
    MAX_MISSING_FEATURES,
    METRICS_DIR,
    MODEL_PATH,
    RANDOM_STATE,
)
from ml.data import load_dataset, train_test_split_data


def compute_fill_values(X_train: pd.DataFrame) -> dict[str, int]:
    values = {}
    for col in FEATURES:
        if col in CATEGORICAL_FEATURES:
            values[col] = int(X_train[col].mode().iloc[0])
        else:
            values[col] = int(round(float(X_train[col].median())))
    return values


def impute(frame: pd.DataFrame, fill_values: dict[str, int]) -> pd.DataFrame:
    return frame.fillna(value=fill_values).astype("int64")


def evaluate_robustness(pipeline, threshold, X_test, y_test, fill_values) -> dict:
    """How much do test metrics degrade when values are missing and imputed?"""
    rng = np.random.default_rng(RANDOM_STATE)

    def score(X):
        proba = pipeline.predict_proba(impute(X, fill_values))[:, 1]
        return {"roc_auc": round(roc_auc_score(y_test, proba), 4),
                "f1_default": round(f1_score(y_test, (proba >= threshold).astype(int)), 4)}

    random_missing = {}
    for rate in [0.0, 0.05, 0.10, 0.20, 0.30]:
        X = X_test.astype("float64").copy()
        mask = rng.random(X.shape) < rate
        X = X.mask(mask)
        random_missing[f"{int(rate * 100)}%"] = score(X)

    single_feature = {}
    for col in FEATURES:
        X = X_test.astype("float64").copy()
        X[col] = np.nan
        single_feature[col] = score(X)
    baseline_auc = random_missing["0%"]["roc_auc"]
    worst = sorted(single_feature.items(), key=lambda kv: kv[1]["roc_auc"])[:5]

    return {
        "description": "Held-out test set (6,000 rows). Values are removed, imputed with training "
                       "medians/modes, then scored by the unchanged trained pipeline.",
        "random_missing_cells": random_missing,
        "single_feature_entirely_missing": single_feature,
        "most_damaging_features_when_missing": [
            {"feature": k, "roc_auc": v["roc_auc"], "auc_drop": round(baseline_auc - v["roc_auc"], 4)}
            for k, v in worst
        ],
    }


def main() -> dict:
    df = load_dataset()
    X_train, X_test, _, y_test = train_test_split_data(df)
    fill_values = compute_fill_values(X_train)

    artifact = {
        "description": "Inference-time fill values for missing features, computed from the "
                       "training split only (80:20, stratified, random_state=42).",
        "strategy": {c: ("most_frequent" if c in CATEGORICAL_FEATURES else "median") for c in FEATURES},
        "fill_values": fill_values,
        "max_missing_features": MAX_MISSING_FEATURES,
        "computed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "train_rows": int(len(X_train)),
    }
    IMPUTATION_PATH.write_text(json.dumps(artifact, indent=2))

    model = joblib.load(MODEL_PATH)
    report = evaluate_robustness(model["pipeline"], model["metadata"]["decision_threshold"],
                                 X_test, y_test, fill_values)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    (METRICS_DIR / "missing_value_robustness.json").write_text(json.dumps(report, indent=2))

    print("Fill values (from training split):", fill_values)
    print("\nTest metrics with randomly missing cells (imputed):")
    for rate, m in report["random_missing_cells"].items():
        print(f"  {rate:>4} missing -> ROC-AUC {m['roc_auc']:.4f}  F1(1) {m['f1_default']:.4f}")
    print("\nMost damaging single features when entirely missing:")
    for item in report["most_damaging_features_when_missing"]:
        print(f"  {item['feature']:<10} ROC-AUC {item['roc_auc']:.4f} (drop {item['auc_drop']:.4f})")
    return artifact


if __name__ == "__main__":
    main()
