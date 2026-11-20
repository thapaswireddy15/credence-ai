"""Model evaluation: metrics and evaluation figures."""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from ml.config import FIGURES_DIR
from ml.plotting import SEQUENTIAL, SERIES


def compute_metrics(y_true, y_proba, threshold: float = 0.5) -> dict:
    """All required metrics. Precision / recall / F1 are reported for class 1 (default)."""
    y_pred = (np.asarray(y_proba) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "threshold": round(float(threshold), 4),
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "precision_default": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall_default": round(recall_score(y_true, y_pred), 4),
        "f1_default": round(f1_score(y_true, y_pred), 4),
        "roc_auc": round(roc_auc_score(y_true, y_proba), 4),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "classification_report": classification_report(
            y_true, y_pred, target_names=["No default (0)", "Default (1)"], output_dict=True, zero_division=0
        ),
        "classification_report_text": classification_report(
            y_true, y_pred, target_names=["No default (0)", "Default (1)"], digits=4, zero_division=0
        ),
    }


def best_f1_threshold(y_true, y_proba) -> float:
    """Threshold that maximises F1 for class 1 (used on OUT-OF-FOLD training predictions)."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    f1 = 2 * precision * recall / np.clip(precision + recall, 1e-12, None)
    best = int(np.nanargmax(f1[:-1]))  # last P/R point has no threshold
    return float(thresholds[best])


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def plot_confusion_matrices(results: dict, filename="model_confusion_matrices.png") -> str:
    names = list(results)
    fig, axes = plt.subplots(1, len(names), figsize=(4.4 * len(names), 4))
    for ax, name in zip(np.atleast_1d(axes), names):
        cm = results[name]["test_metrics"]["confusion_matrix"]
        matrix = np.array([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]])
        sns.heatmap(matrix, annot=True, fmt=",d", cmap=SEQUENTIAL, cbar=False, ax=ax,
                    xticklabels=["Pred 0", "Pred 1"], yticklabels=["True 0", "True 1"],
                    linewidths=2, linecolor="#fcfcfb", annot_kws={"size": 12})
        ax.set_title(name, fontsize=11)
        ax.grid(False)
    fig.suptitle("Confusion matrices on the test set (threshold 0.5)", fontweight="bold")
    fig.tight_layout()
    path = FIGURES_DIR / filename
    fig.savefig(path)
    plt.close(fig)
    return str(path)


def plot_roc_curves(curves: dict, filename="model_roc_curves.png") -> str:
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    for color, (name, (y_true, y_proba)) in zip(SERIES, curves.items()):
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        ax.plot(fpr, tpr, color=color, linewidth=2,
                label=f"{name} (AUC {roc_auc_score(y_true, y_proba):.3f})")
    ax.plot([0, 1], [0, 1], color="#a3a29b", linestyle="--", linewidth=1, label="Random guess")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate (recall)")
    ax.set_title("ROC curves on the test set")
    ax.legend(loc="lower right", fontsize=9)
    path = FIGURES_DIR / filename
    fig.savefig(path)
    plt.close(fig)
    return str(path)


def plot_metric_comparison(results: dict, filename="model_metric_comparison.png") -> str:
    metrics = ["accuracy", "precision_default", "recall_default", "f1_default", "roc_auc"]
    labels = ["Accuracy", "Precision (1)", "Recall (1)", "F1 (1)", "ROC-AUC"]
    names = list(results)
    x = np.arange(len(metrics))
    width = 0.8 / len(names)
    fig, ax = plt.subplots(figsize=(11, 4.8))
    for i, (color, name) in enumerate(zip(SERIES, names)):
        values = [results[name]["test_metrics"][m] for m in metrics]
        ax.bar(x + i * width - 0.4 + width / 2, values, width * 0.92, color=color, label=name)
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 1)
    ax.set_title("Model comparison on the test set (threshold 0.5)")
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.1))
    ax.grid(axis="x", visible=False)
    path = FIGURES_DIR / filename
    fig.savefig(path)
    plt.close(fig)
    return str(path)


def plot_feature_importance(pipeline, filename="best_model_feature_importance.png", top_n=15) -> str | None:
    clf = pipeline.named_steps["classifier"]
    names = pipeline.named_steps["preprocessor"].get_feature_names_out()
    if hasattr(clf, "feature_importances_"):
        values, label = clf.feature_importances_, "Impurity-based importance"
    elif hasattr(clf, "coef_"):
        values, label = np.abs(clf.coef_[0]), "|coefficient| (standardised features)"
    else:
        return None
    order = np.argsort(values)[-top_n:]
    clean = [n.split("__", 1)[-1] for n in names[order]]
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.barh(clean, values[order], color=SERIES[0], height=0.7)
    ax.set_xlabel(label)
    ax.set_title(f"Top {top_n} features - selected model")
    ax.grid(axis="y", visible=False)
    path = FIGURES_DIR / filename
    fig.savefig(path)
    plt.close(fig)
    return str(path)


def plot_threshold_tradeoff(y_true, y_proba, chosen: float, filename="best_model_threshold_tradeoff.png") -> str:
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    f1 = 2 * precision * recall / np.clip(precision + recall, 1e-12, None)
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(thresholds, precision[:-1], color=SERIES[0], linewidth=2, label="Precision (1)")
    ax.plot(thresholds, recall[:-1], color=SERIES[1], linewidth=2, label="Recall (1)")
    ax.plot(thresholds, f1[:-1], color=SERIES[2], linewidth=2, label="F1 (1)")
    ax.axvline(chosen, color="#52514e", linestyle="--", linewidth=1)
    ax.annotate(f"chosen {chosen:.2f}", (chosen, 0.05), xytext=(5, 0), textcoords="offset points", fontsize=9)
    ax.axvline(0.5, color="#a3a29b", linestyle=":", linewidth=1)
    ax.set_xlabel("Decision threshold on default probability")
    ax.set_title("Precision / recall trade-off (test set) - selected model")
    ax.set_ylim(0, 1.02)
    ax.legend()
    path = FIGURES_DIR / filename
    fig.savefig(path)
    plt.close(fig)
    return str(path)
