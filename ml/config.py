"""Central configuration for the Credence AI machine learning pipeline.

Every path, feature list and constant used by training, evaluation and the API
lives here so that training-time and inference-time code can never drift apart.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "default of credit card clients.csv"
MODELS_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODELS_DIR / "credence_model.joblib"
MODEL_METADATA_PATH = MODELS_DIR / "model_metadata.json"
IMPUTATION_PATH = MODELS_DIR / "imputation_values.json"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
METRICS_DIR = REPORTS_DIR / "metrics"

# ---------------------------------------------------------------------------
# Target and features
# ---------------------------------------------------------------------------
TARGET = "default.payment.next.month"

# The raw UCI file spells the target with spaces; Kaggle's copy uses dots.
TARGET_ALIASES = ("default payment next month", "default.payment.next.month", "Y")

# Nominal categorical codes -> one-hot encoded.
CATEGORICAL_FEATURES = ["SEX", "EDUCATION", "MARRIAGE"]

# Repayment status: -2/-1/0 = paid or no consumption, 1..9 = months of delay.
# The codes are ORDERED (a larger value means a longer delay), so they are
# kept as ordinal numbers and standardised rather than one-hot encoded.
REPAYMENT_STATUS_FEATURES = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]

# Monetary amounts (NT dollars): heavily right-skewed, BILL_AMT can be negative.
MONETARY_FEATURES = (
    ["LIMIT_BAL"]
    + [f"BILL_AMT{i}" for i in range(1, 7)]
    + [f"PAY_AMT{i}" for i in range(1, 7)]
)

# Other continuous features.
CONTINUOUS_FEATURES = ["AGE"]

# The exact 23 model inputs, in the canonical dataset order.
FEATURES = [
    "LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE",
    "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6",
    "BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6",
    "PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6",
]

assert len(FEATURES) == 23
assert set(FEATURES) == set(
    CATEGORICAL_FEATURES + REPAYMENT_STATUS_FEATURES + MONETARY_FEATURES + CONTINUOUS_FEATURES
)

# ---------------------------------------------------------------------------
# Experiment settings
# ---------------------------------------------------------------------------
TEST_SIZE = 0.2
RANDOM_STATE = 42
CV_FOLDS = 5

# Categories seen in fewer than 2% of training rows are merged into one
# "infrequent" bucket by the OneHotEncoder. On this dataset that groups the
# undocumented EDUCATION codes 0/5/6 with 4 ("others") and MARRIAGE 0 with 3
# ("others") -- learned from the training split only, so there is no leakage.
CATEGORY_MIN_FREQUENCY = 0.02

MODEL_VERSION = "1.0.0"

# Missing-value policy at inference time. A record may omit at most this many
# of the 23 features (fewer than half); beyond that the prediction would rest
# mostly on imputed values, so the API rejects it with HTTP 422.
MAX_MISSING_FEATURES = 11

# Risk bands shown to the Credit Portfolio Strategist (on default probability).
RISK_BANDS = [
    # (upper bound exclusive, label)
    (0.30, "LOW"),
    (0.60, "MEDIUM"),
    (1.01, "HIGH"),
]


def risk_band(probability: float) -> str:
    """Map a default probability to a LOW / MEDIUM / HIGH portfolio risk band."""
    for upper, label in RISK_BANDS:
        if probability < upper:
            return label
    return RISK_BANDS[-1][1]
