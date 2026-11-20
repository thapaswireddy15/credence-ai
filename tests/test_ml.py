"""Tests for data loading, preprocessing and the trained model artifact."""

import json

import numpy as np
import pandas as pd
import pytest

from app.predictor import CreditDefaultPredictor
from ml.config import FEATURES, MODEL_METADATA_PATH, MODEL_PATH, RAW_DATA_PATH, TARGET, risk_band
from ml.data import load_dataset, train_test_split_data
from ml.pipeline import build_preprocessor

needs_data = pytest.mark.skipif(not RAW_DATA_PATH.exists(), reason="raw dataset not present")
needs_model = pytest.mark.skipif(not MODEL_PATH.exists(), reason="run `python -m ml.train` first")


@pytest.fixture(scope="module")
def dataset():
    return load_dataset()


@needs_data
def test_dataset_schema(dataset):
    assert dataset.shape == (30000, 24)
    assert list(dataset.columns) == FEATURES + [TARGET]
    assert "ID" not in dataset.columns
    assert set(dataset[TARGET].unique()) == {0, 1}
    assert dataset.isna().sum().sum() == 0


@needs_data
def test_split_is_80_20_and_stratified(dataset):
    X_train, X_test, y_train, y_test = train_test_split_data(dataset)
    assert len(X_train) == 24000 and len(X_test) == 6000
    assert TARGET not in X_train.columns
    assert abs(y_train.mean() - y_test.mean()) < 0.001
    # deterministic with random_state=42
    X_train2, *_ = train_test_split_data(dataset)
    assert X_train.index.equals(X_train2.index)


@needs_data
def test_preprocessor_fits_on_training_data_only(dataset):
    X_train, X_test, *_ = train_test_split_data(dataset)
    pre = build_preprocessor().fit(X_train)
    scaler = pre.named_transformers_["continuous"]
    assert scaler.mean_[0] == pytest.approx(X_train["AGE"].mean())
    assert scaler.mean_[0] != pytest.approx(dataset["AGE"].mean(), abs=1e-9)
    out = pre.transform(X_test)
    assert np.isfinite(out).all()


@needs_data
def test_rare_and_unseen_category_codes_are_grouped(dataset):
    X_train, *_ = train_test_split_data(dataset)
    pre = build_preprocessor().fit(X_train)
    names = list(pre.get_feature_names_out())
    assert "categorical__EDUCATION_infrequent_sklearn" in names
    assert "categorical__MARRIAGE_infrequent_sklearn" in names
    row = X_train.iloc[[0]].copy()
    row["EDUCATION"] = 6  # undocumented code
    encoded = pd.DataFrame(pre.transform(row), columns=names)
    assert encoded["categorical__EDUCATION_infrequent_sklearn"].iloc[0] == 1.0


@pytest.mark.parametrize("p,band", [(0.0, "LOW"), (0.29, "LOW"), (0.30, "MEDIUM"), (0.59, "MEDIUM"),
                                    (0.60, "HIGH"), (1.0, "HIGH")])
def test_risk_band_boundaries(p, band):
    assert risk_band(p) == band


@needs_model
def test_model_artifact_and_metadata():
    predictor = CreditDefaultPredictor(MODEL_PATH)
    meta = json.loads(MODEL_METADATA_PATH.read_text())
    assert predictor.metadata["features"] == FEATURES
    assert meta["model_name"] == predictor.model_name
    assert 0 < predictor.threshold < 1
    assert meta["test_metrics_tuned_threshold"]["roc_auc"] > 0.7


@needs_model
def test_model_predictions_are_valid_probabilities(customer, low_risk_customer):
    predictor = CreditDefaultPredictor(MODEL_PATH)
    records = [{k: c[k] for k in FEATURES} for c in (customer, low_risk_customer)]
    risky, safe = predictor.predict(records)
    for out in (risky, safe):
        assert 0.0 <= out.default_probability <= 1.0
        assert out.risk_band in {"LOW", "MEDIUM", "HIGH"}
    assert risky.default_probability > safe.default_probability
    assert risky.predicted_default is True
    assert safe.predicted_default is False


# ---------------------------------------------------------------------------
# Missing-value imputation artifact (no retraining)
# ---------------------------------------------------------------------------
from ml.config import IMPUTATION_PATH, METRICS_DIR  # noqa: E402
from ml.imputation import compute_fill_values  # noqa: E402


@needs_data
def test_fill_values_come_from_training_split_only(dataset):
    X_train, *_ = train_test_split_data(dataset)
    stored = json.loads(IMPUTATION_PATH.read_text())["fill_values"]
    assert stored == compute_fill_values(X_train)
    assert stored["LIMIT_BAL"] == int(X_train["LIMIT_BAL"].median())
    assert stored["EDUCATION"] == int(X_train["EDUCATION"].mode().iloc[0])
    assert set(stored) == set(FEATURES)


@needs_model
def test_predictor_imputes_only_missing_values(customer):
    predictor = CreditDefaultPredictor(MODEL_PATH)
    complete = {k: customer[k] for k in FEATURES}
    partial = dict(complete, BILL_AMT3=None)
    out_complete, out_partial = predictor.predict([complete, partial])
    assert out_complete.imputed_features == [] and out_complete.model_input == complete
    assert out_partial.imputed_features == ["BILL_AMT3"]
    assert out_partial.model_input["BILL_AMT3"] == predictor.fill_values["BILL_AMT3"]
    assert out_partial.model_input["PAY_0"] == complete["PAY_0"]


def test_robustness_report_exists():
    report = json.loads((METRICS_DIR / "missing_value_robustness.json").read_text())
    rates = report["random_missing_cells"]
    assert rates["0%"]["roc_auc"] >= rates["30%"]["roc_auc"]
