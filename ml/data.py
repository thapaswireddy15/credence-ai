"""Dataset loading and train/test splitting."""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from ml.config import FEATURES, RANDOM_STATE, RAW_DATA_PATH, TARGET, TARGET_ALIASES, TEST_SIZE


def _detect_header_row(csv_path: Path) -> int:
    """Return the 0-based line that holds the real column names.

    The original UCI export has an extra first line (",X1,X2,...,Y") above the
    real header ("ID,LIMIT_BAL,..."). Kaggle's copy does not. We only skip the
    first line when it is actually that placeholder line.
    """
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        first_line = f.readline()
    return 0 if "LIMIT_BAL" in first_line else 1


def load_dataset(csv_path: Path | str = RAW_DATA_PATH) -> pd.DataFrame:
    """Load the credit card default dataset with a clean, canonical schema.

    - skips the placeholder header row if present
    - drops the ID column (an identifier, not a predictive feature)
    - renames the target to ``default.payment.next.month``
    - returns the 23 features + target, in canonical order
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found at {csv_path}")

    df = pd.read_csv(csv_path, header=_detect_header_row(csv_path))
    df.columns = [str(c).strip() for c in df.columns]

    # Normalise the target column name.
    for alias in TARGET_ALIASES:
        if alias in df.columns and alias != TARGET:
            df = df.rename(columns={alias: TARGET})
            break

    # Some copies call PAY_0 "PAY_1".
    if "PAY_0" not in df.columns and "PAY_1" in df.columns:
        df = df.rename(columns={"PAY_1": "PAY_0"})

    if "ID" in df.columns:
        df = df.drop(columns="ID")

    missing = [c for c in FEATURES + [TARGET] if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing expected columns: {missing}")

    return df[FEATURES + [TARGET]].copy()


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    return df[FEATURES], df[TARGET].astype(int)


def train_test_split_data(df: pd.DataFrame):
    """80:20 stratified split with a fixed seed (as required by the project spec)."""
    X, y = split_features_target(df)
    return train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)
