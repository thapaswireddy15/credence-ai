"""Train, compare and persist the Credence AI default-prediction models.

Run:  python -m ml.train

Workflow
--------
1. Load the dataset and make an 80:20 stratified split (random_state=42).
2. For each of the four algorithms, tune a small hyper-parameter grid with
   5-fold stratified cross-validation on the TRAINING set only, scoring on
   F1 for the default class.
3. Evaluate every tuned model once on the held-out test set.
4. Select the best model by cross-validated F1 (class 1), using cross-validated
   ROC-AUC as the tie-breaker -- never by accuracy, and never by peeking at
   the test set.
5. Choose a decision threshold for the selected model from OUT-OF-FOLD
   training predictions (maximising F1), then report test metrics at it.
6. Save the full pipeline (preprocessing + model) with joblib, plus metadata,
   metrics and figures.
"""

import copy
import json
import time
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict

from ml.config import (
    CV_FOLDS,
    FEATURES,
    FIGURES_DIR,
    METRICS_DIR,
    MODEL_METADATA_PATH,
    MODEL_PATH,
    MODEL_VERSION,
    MODELS_DIR,
    RANDOM_STATE,
    RAW_DATA_PATH,
    RISK_BANDS,
    TARGET,
    TEST_SIZE,
)
from ml.data import load_dataset, train_test_split_data
from ml.evaluate import (
    best_f1_threshold,
    compute_metrics,
    plot_confusion_matrices,
    plot_feature_importance,
    plot_metric_comparison,
    plot_roc_curves,
    plot_threshold_tradeoff,
)
from ml.pipeline import build_pipeline, model_candidates
from ml.plotting import apply_style


def tune_and_evaluate(X_train, y_train, X_test, y_test, cv) -> tuple[dict, dict, dict]:
    results, fitted, curves = {}, {}, {}
    for name, spec in model_candidates().items():
        start = time.perf_counter()
        search = GridSearchCV(
            build_pipeline(spec["estimator"]),
            spec["param_grid"],
            scoring={"f1": "f1", "roc_auc": "roc_auc", "recall": "recall", "precision": "precision",
                     "accuracy": "accuracy"},
            refit="f1",
            cv=cv,
            n_jobs=-1,
        )
        search.fit(X_train, y_train)
        best = search.best_index_
        cv_scores = {
            metric: round(float(search.cv_results_[f"mean_test_{metric}"][best]), 4)
            for metric in ["f1", "roc_auc", "recall", "precision", "accuracy"]
        }
        proba = search.best_estimator_.predict_proba(X_test)[:, 1]
        test_metrics = compute_metrics(y_test, proba, threshold=0.5)
        elapsed = time.perf_counter() - start

        results[name] = {
            "best_params": {k.replace("classifier__", ""): v for k, v in search.best_params_.items()},
            "cv_metrics": cv_scores,
            "test_metrics": test_metrics,
            "training_seconds": round(elapsed, 1),
        }
        fitted[name] = search.best_estimator_
        curves[name] = (y_test, proba)
        print(
            f"{name:<22} CV F1={cv_scores['f1']:.4f} CV AUC={cv_scores['roc_auc']:.4f} | "
            f"test acc={test_metrics['accuracy']:.4f} prec={test_metrics['precision_default']:.4f} "
            f"rec={test_metrics['recall_default']:.4f} f1={test_metrics['f1_default']:.4f} "
            f"auc={test_metrics['roc_auc']:.4f} ({elapsed:.1f}s)"
        )
    return results, fitted, curves


def select_best(results: dict) -> str:
    """Highest cross-validated F1 for class 1; ROC-AUC breaks near-ties (< 0.002)."""
    ranked = sorted(results, key=lambda n: results[n]["cv_metrics"]["f1"], reverse=True)
    top = ranked[0]
    for challenger in ranked[1:]:
        f1_gap = results[top]["cv_metrics"]["f1"] - results[challenger]["cv_metrics"]["f1"]
        if f1_gap < 0.002 and results[challenger]["cv_metrics"]["roc_auc"] > results[top]["cv_metrics"]["roc_auc"]:
            top = challenger
    return top


def main() -> dict:
    apply_style()
    for d in (MODELS_DIR, FIGURES_DIR, METRICS_DIR):
        d.mkdir(parents=True, exist_ok=True)

    df = load_dataset(RAW_DATA_PATH)
    X_train, X_test, y_train, y_test = train_test_split_data(df)
    print(f"Train: {X_train.shape}, default rate {y_train.mean():.4f} | "
          f"Test: {X_test.shape}, default rate {y_test.mean():.4f}\n")

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    results, fitted, curves = tune_and_evaluate(X_train, y_train, X_test, y_test, cv)

    best_name = select_best(results)
    best_pipeline = fitted[best_name]
    print(f"\nSelected model: {best_name} (highest cross-validated F1 for the default class)")

    # Decision threshold from out-of-fold predictions on the TRAINING set only.
    oof_proba = cross_val_predict(
        clone(best_pipeline), X_train, y_train, cv=cv, method="predict_proba", n_jobs=-1
    )[:, 1]
    threshold = round(best_f1_threshold(y_train, oof_proba), 4)
    best_test_proba = best_pipeline.predict_proba(X_test)[:, 1]
    tuned_test_metrics = compute_metrics(y_test, best_test_proba, threshold=threshold)
    results[best_name]["tuned_threshold_test_metrics"] = tuned_test_metrics
    print(f"Tuned decision threshold (OOF F1-optimal): {threshold}")
    print(tuned_test_metrics["classification_report_text"])

    figures = [
        plot_confusion_matrices(results),
        plot_roc_curves(curves),
        plot_metric_comparison(results),
        plot_feature_importance(best_pipeline),
        plot_threshold_tradeoff(y_test, best_test_proba, threshold),
    ]

    # ---- Persist the model artifact --------------------------------------
    metadata = {
        "model_name": best_name,
        "model_version": MODEL_VERSION,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sklearn_version": sklearn.__version__,
        "features": FEATURES,
        "target": TARGET,
        "decision_threshold": threshold,
        "risk_bands": [{"upper_bound_exclusive": u, "label": l} for u, l in RISK_BANDS],
        "best_params": results[best_name]["best_params"],
        "cv_metrics": results[best_name]["cv_metrics"],
        "test_metrics_default_threshold": {
            k: v for k, v in results[best_name]["test_metrics"].items() if not k.startswith("classification_report")
        },
        "test_metrics_tuned_threshold": {
            k: v for k, v in tuned_test_metrics.items() if not k.startswith("classification_report")
        },
        "data": {
            "rows": int(len(df)),
            "train_rows": int(len(X_train)),
            "test_rows": int(len(X_test)),
            "test_size": TEST_SIZE,
            "random_state": RANDOM_STATE,
            "stratified": True,
            "train_default_rate": round(float(y_train.mean()), 4),
        },
    }
    joblib.dump({"pipeline": best_pipeline, "metadata": metadata}, MODEL_PATH, compress=3)
    with open(MODEL_METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2)

    # ---- Persist comparison reports --------------------------------------
    serialisable = copy.deepcopy(results)
    for r in serialisable.values():
        r["test_metrics"].pop("classification_report_text", None)
        if "tuned_threshold_test_metrics" in r:
            r["tuned_threshold_test_metrics"].pop("classification_report_text", None)
    with open(METRICS_DIR / "model_comparison.json", "w") as f:
        json.dump({"selected_model": best_name, "selection_criterion": "cross-validated F1 (class 1)",
                   "models": serialisable}, f, indent=2, default=str)

    table = pd.DataFrame(
        {
            name: {
                "cv_f1": r["cv_metrics"]["f1"],
                "cv_roc_auc": r["cv_metrics"]["roc_auc"],
                "test_accuracy": r["test_metrics"]["accuracy"],
                "test_precision_1": r["test_metrics"]["precision_default"],
                "test_recall_1": r["test_metrics"]["recall_default"],
                "test_f1_1": r["test_metrics"]["f1_default"],
                "test_roc_auc": r["test_metrics"]["roc_auc"],
                "false_positives": r["test_metrics"]["confusion_matrix"]["fp"],
                "false_negatives": r["test_metrics"]["confusion_matrix"]["fn"],
            }
            for name, r in results.items()
        }
    ).T
    table.index.name = "model"
    table.to_csv(METRICS_DIR / "model_comparison.csv")

    with open(METRICS_DIR / "classification_reports.txt", "w") as f:
        for name, r in results.items():
            f.write(f"=== {name} (threshold 0.5) ===\n{r['test_metrics']['classification_report_text']}\n")
        f.write(f"=== {best_name} (tuned threshold {threshold}) ===\n"
                f"{tuned_test_metrics['classification_report_text']}\n")

    print("\n" + table.to_string())
    print(f"\nSaved model to {MODEL_PATH.relative_to(MODELS_DIR.parent)} "
          f"({MODEL_PATH.stat().st_size / 1e6:.1f} MB)")
    print(f"Saved {len([f for f in figures if f])} evaluation figures to reports/figures/")
    return metadata


if __name__ == "__main__":
    np.random.seed(RANDOM_STATE)
    main()
