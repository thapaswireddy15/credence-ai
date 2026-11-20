"""Exploratory Data Analysis for the Credence AI dataset.

Run:  python -m ml.eda

Outputs
-------
reports/figures/eda_*.png         visualisations
reports/metrics/eda_summary.json  machine-readable summary
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from ml.config import (
    CATEGORICAL_FEATURES,
    FEATURES,
    FIGURES_DIR,
    METRICS_DIR,
    MONETARY_FEATURES,
    RAW_DATA_PATH,
    REPAYMENT_STATUS_FEATURES,
    TARGET,
)
from ml.data import _detect_header_row, load_dataset
from ml.plotting import CLASS_PALETTE, DEFAULT_COLOR, DIVERGING, NO_DEFAULT_COLOR, apply_style

CATEGORY_LABELS = {
    "SEX": {1: "Male", 2: "Female"},
    "EDUCATION": {0: "Undoc. 0", 1: "Graduate school", 2: "University", 3: "High school",
                  4: "Others", 5: "Undoc. 5", 6: "Undoc. 6"},
    "MARRIAGE": {0: "Undoc. 0", 1: "Married", 2: "Single", 3: "Others"},
}


def _save(fig, name: str) -> str:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / name
    fig.savefig(path)
    plt.close(fig)
    return str(path.relative_to(FIGURES_DIR.parent.parent))


def iqr_outlier_counts(df: pd.DataFrame, columns: list[str]) -> dict[str, int]:
    """Number of values outside [Q1 - 1.5*IQR, Q3 + 1.5*IQR] per column."""
    counts = {}
    for col in columns:
        q1, q3 = df[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        mask = (df[col] < q1 - 1.5 * iqr) | (df[col] > q3 + 1.5 * iqr)
        counts[col] = int(mask.sum())
    return counts


def build_summary(df: pd.DataFrame, raw_shape: tuple[int, int]) -> dict:
    target_counts = df[TARGET].value_counts().sort_index()
    return {
        "raw_shape_including_id": list(raw_shape),
        "modelling_shape": list(df.shape),
        "columns": list(df.columns),
        "dtypes": {c: str(t) for c, t in df.dtypes.items()},
        "missing_values_total": int(df.isna().sum().sum()),
        "missing_values_per_column": {c: int(v) for c, v in df.isna().sum().items()},
        "duplicate_rows_excluding_id": int(df.duplicated().sum()),
        "target_distribution": {
            "counts": {str(k): int(v) for k, v in target_counts.items()},
            "proportions": {str(k): round(float(v), 4) for k, v in (target_counts / len(df)).items()},
        },
        "categorical_code_counts": {
            c: {str(k): int(v) for k, v in df[c].value_counts().sort_index().items()}
            for c in CATEGORICAL_FEATURES + REPAYMENT_STATUS_FEATURES
        },
        "negative_bill_amount_values": int((df[[f"BILL_AMT{i}" for i in range(1, 7)]] < 0).sum().sum()),
        "iqr_outlier_counts": iqr_outlier_counts(df, MONETARY_FEATURES + ["AGE"]),
        "descriptive_statistics": df.describe().T.round(2).to_dict(orient="index"),
        "correlation_with_target": df[FEATURES].corrwith(df[TARGET]).round(4).sort_values(
            key=np.abs, ascending=False).to_dict(),
        "default_rate_by_category": {
            c: df.groupby(c)[TARGET].mean().round(4).rename(index=str).to_dict()
            for c in CATEGORICAL_FEATURES + ["PAY_0"]
        },
    }


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def plot_target_distribution(df):
    counts = df[TARGET].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(5.5, 4))
    bars = ax.bar(["No default (0)", "Default (1)"], counts.values,
                  color=[NO_DEFAULT_COLOR, DEFAULT_COLOR], width=0.55)
    for bar, value in zip(bars, counts.values):
        ax.annotate(f"{value:,}\n({value / counts.sum():.1%})", (bar.get_x() + bar.get_width() / 2, value),
                    ha="center", va="bottom", fontsize=9, xytext=(0, 3), textcoords="offset points")
    ax.set_title("Target class distribution")
    ax.set_ylabel("Customers")
    ax.set_ylim(0, counts.max() * 1.18)
    ax.grid(axis="x", visible=False)
    return _save(fig, "eda_01_target_distribution.png")


def plot_feature_distributions(df):
    cols = ["LIMIT_BAL", "AGE", "BILL_AMT1", "PAY_AMT1", "BILL_AMT6", "PAY_AMT6"]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for ax, col in zip(axes.ravel(), cols):
        sns.histplot(data=df, x=col, hue=TARGET, palette=CLASS_PALETTE, bins=50, ax=ax,
                     element="step", stat="density", common_norm=False)
        ax.set_title(col)
        ax.set_xlabel("")
        legend = ax.get_legend()
        if legend:
            legend.set_title("Default")
    fig.suptitle("Feature distributions by class (density, raw scale)", fontweight="bold")
    fig.tight_layout()
    return _save(fig, "eda_02_feature_distributions.png")


def plot_log_monetary_distributions(df):
    cols = ["LIMIT_BAL", "BILL_AMT1", "PAY_AMT1"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for ax, col in zip(axes, cols):
        signed_log = np.sign(df[col]) * np.log1p(np.abs(df[col]))
        ax.hist(signed_log, bins=60, color=NO_DEFAULT_COLOR)
        ax.set_title(f"signed log1p({col})")
    fig.suptitle("Monetary features are highly skewed -> power transform in the pipeline",
                 fontweight="bold")
    fig.tight_layout()
    return _save(fig, "eda_03_monetary_skew_log_scale.png")


def plot_outlier_boxplots(df):
    cols = MONETARY_FEATURES
    fig, ax = plt.subplots(figsize=(13, 5))
    sns.boxplot(data=df[cols], orient="h", ax=ax, color=NO_DEFAULT_COLOR, fliersize=1.5,
                flierprops={"markerfacecolor": DEFAULT_COLOR, "markeredgecolor": DEFAULT_COLOR})
    ax.set_xscale("symlog", linthresh=1000)
    ax.set_title("Outlier analysis: monetary features (symlog scale)")
    ax.set_xlabel("NT dollars")
    return _save(fig, "eda_04_outlier_boxplots.png")


def plot_correlation_heatmap(df):
    corr = df[FEATURES + [TARGET]].corr()
    fig, ax = plt.subplots(figsize=(14, 11.5))
    sns.heatmap(corr, cmap=DIVERGING, vmin=-1, vmax=1, center=0, square=True, linewidths=0.4,
                linecolor="#fcfcfb", cbar_kws={"shrink": 0.7, "label": "Pearson r"}, ax=ax,
                annot=True, fmt=".2f", annot_kws={"size": 6})
    ax.set_title("Correlation matrix (features + target)")
    return _save(fig, "eda_05_correlation_heatmap.png")


def plot_target_correlation(df):
    corr = df[FEATURES].corrwith(df[TARGET]).sort_values()
    fig, ax = plt.subplots(figsize=(7, 7.5))
    colors = [DEFAULT_COLOR if v > 0 else NO_DEFAULT_COLOR for v in corr.values]
    ax.barh(corr.index, corr.values, color=colors, height=0.7)
    ax.axvline(0, color="#52514e", linewidth=0.8)
    ax.set_title("Correlation of each feature with default")
    ax.set_xlabel("Pearson r with default.payment.next.month")
    ax.grid(axis="y", visible=False)
    return _save(fig, "eda_06_feature_target_correlation.png")


def plot_default_rate_by_category(df):
    cols = ["SEX", "EDUCATION", "MARRIAGE", "PAY_0"]
    fig, axes = plt.subplots(1, 4, figsize=(17, 4.2))
    overall = df[TARGET].mean()
    for ax, col in zip(axes, cols):
        grouped = df.groupby(col)[TARGET].agg(["mean", "size"])
        labels = [CATEGORY_LABELS.get(col, {}).get(k, str(k)) for k in grouped.index]
        ax.bar(labels, grouped["mean"], color=NO_DEFAULT_COLOR, width=0.65)
        ax.axhline(overall, color=DEFAULT_COLOR, linestyle="--", linewidth=1.5,
                   label=f"Overall {overall:.1%}")
        ax.set_title(f"Default rate by {col}")
        ax.set_ylim(0, 1)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
        ax.tick_params(axis="x", rotation=45)
        ax.grid(axis="x", visible=False)
        ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    return _save(fig, "eda_07_default_rate_by_category.png")


def plot_numeric_vs_target(df):
    cols = ["LIMIT_BAL", "AGE", "PAY_AMT1", "BILL_AMT1"]
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.2))
    for ax, col in zip(axes, cols):
        sns.boxplot(data=df, x=TARGET, y=col, hue=TARGET, palette=CLASS_PALETTE, legend=False,
                    ax=ax, fliersize=1, width=0.5)
        if col.startswith(("PAY_AMT", "BILL_AMT")):
            ax.set_yscale("symlog", linthresh=1000)
        ax.set_xticks([0, 1], ["No default", "Default"])
        ax.set_xlabel("")
        ax.set_title(col)
    fig.suptitle("Numeric features vs. target", fontweight="bold")
    fig.tight_layout()
    return _save(fig, "eda_08_numeric_vs_target.png")


def plot_repayment_status_trend(df):
    fig, ax = plt.subplots(figsize=(8, 4.2))
    months = ["PAY_6", "PAY_5", "PAY_4", "PAY_3", "PAY_2", "PAY_0"]  # Apr -> Sep 2005
    month_labels = ["Apr", "May", "Jun", "Jul", "Aug", "Sep"]
    for label, color in [(0, NO_DEFAULT_COLOR), (1, DEFAULT_COLOR)]:
        share_late = [(df.loc[df[TARGET] == label, m] >= 1).mean() for m in months]
        ax.plot(month_labels, share_late, marker="o", markersize=7, linewidth=2, color=color,
                label="Default" if label else "No default")
        ax.annotate(f"{share_late[-1]:.0%}", (5, share_late[-1]), xytext=(8, 0),
                    textcoords="offset points", va="center", fontsize=9)
    ax.set_title("Share of customers with a payment delay, by month")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_ylim(0, 1)
    ax.legend()
    return _save(fig, "eda_09_repayment_delay_trend.png")


def run_eda(csv_path=RAW_DATA_PATH) -> dict:
    apply_style()
    raw = pd.read_csv(csv_path, header=_detect_header_row(Path(csv_path)))
    df = load_dataset(csv_path)

    summary = build_summary(df, raw.shape)
    summary["figures"] = [
        plot_target_distribution(df),
        plot_feature_distributions(df),
        plot_log_monetary_distributions(df),
        plot_outlier_boxplots(df),
        plot_correlation_heatmap(df),
        plot_target_correlation(df),
        plot_default_rate_by_category(df),
        plot_numeric_vs_target(df),
        plot_repayment_status_trend(df),
    ]

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    with open(METRICS_DIR / "eda_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    return summary


if __name__ == "__main__":
    s = run_eda()
    print(f"Shape (features + target): {s['modelling_shape']}")
    print(f"Missing values: {s['missing_values_total']}")
    print(f"Duplicate rows (excluding ID): {s['duplicate_rows_excluding_id']}")
    print(f"Target distribution: {s['target_distribution']['proportions']}")
    print("Top correlations with target:")
    for k, v in list(s["correlation_with_target"].items())[:6]:
        print(f"  {k:<10} {v:+.4f}")
    print(f"Saved {len(s['figures'])} figures to reports/figures/")
