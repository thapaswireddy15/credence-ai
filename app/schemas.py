"""Pydantic request / response schemas."""

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ml.config import FEATURES, MAX_MISSING_FEATURES

# ---------------------------------------------------------------------------
# Input
# ---------------------------------------------------------------------------
_PAY_STATUS = dict(ge=-2, le=9, description="Repayment status: -2/-1/0 = paid duly or no use, 1..9 = months late")
_BILL = dict(ge=-10_000_000, le=10_000_000, description="Bill statement amount (NT$, negative = credit balance)")
_PAY_AMT = dict(ge=0, le=10_000_000, description="Amount paid (NT$)")

EXAMPLE_CUSTOMER = {
    "customer_ref": "CUST-000123",
    "LIMIT_BAL": 50000, "SEX": 2, "EDUCATION": 2, "MARRIAGE": 1, "AGE": 37,
    "PAY_0": 2, "PAY_2": 2, "PAY_3": 2, "PAY_4": 0, "PAY_5": 0, "PAY_6": 0,
    "BILL_AMT1": 46990, "BILL_AMT2": 48233, "BILL_AMT3": 49291,
    "BILL_AMT4": 28314, "BILL_AMT5": 28959, "BILL_AMT6": 29547,
    "PAY_AMT1": 0, "PAY_AMT2": 0, "PAY_AMT3": 1000, "PAY_AMT4": 1100, "PAY_AMT5": 1069, "PAY_AMT6": 1000,
}


EXAMPLE_CUSTOMER_WITH_MISSING = {
    k: v for k, v in EXAMPLE_CUSTOMER.items() if k not in ("BILL_AMT3", "PAY_AMT2", "EDUCATION")
} | {"customer_ref": "CUST-000124", "AGE": None}


class CustomerFeatures(BaseModel):
    """The 23 inputs of the Default of Credit Card Clients dataset.

    **Missing values:** any feature may be omitted or sent as ``null`` (up to
    ``MAX_MISSING_FEATURES`` = 11 of 23). Missing values are filled with the
    training-split median (numeric/ordinal) or most frequent value
    (SEX / EDUCATION / MARRIAGE), and the response lists which were imputed.
    """

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={"examples": [EXAMPLE_CUSTOMER, EXAMPLE_CUSTOMER_WITH_MISSING]},
    )

    customer_ref: str | None = Field(None, max_length=64, description="Optional institution customer reference")

    LIMIT_BAL: int | None = Field(None, gt=0, le=10_000_000, description="Credit limit (NT$)")
    SEX: int | None = Field(None, ge=1, le=2, description="1 = male, 2 = female")
    EDUCATION: int | None = Field(None, ge=0, le=6, description="1 = graduate school, 2 = university, 3 = high school, 4 = others (0/5/6 = undocumented)")
    MARRIAGE: int | None = Field(None, ge=0, le=3, description="1 = married, 2 = single, 3 = others (0 = undocumented)")
    AGE: int | None = Field(None, ge=18, le=100, description="Age in years")
    PAY_0: int | None = Field(None, **_PAY_STATUS)
    PAY_2: int | None = Field(None, **_PAY_STATUS)
    PAY_3: int | None = Field(None, **_PAY_STATUS)
    PAY_4: int | None = Field(None, **_PAY_STATUS)
    PAY_5: int | None = Field(None, **_PAY_STATUS)
    PAY_6: int | None = Field(None, **_PAY_STATUS)
    BILL_AMT1: int | None = Field(None, **_BILL)
    BILL_AMT2: int | None = Field(None, **_BILL)
    BILL_AMT3: int | None = Field(None, **_BILL)
    BILL_AMT4: int | None = Field(None, **_BILL)
    BILL_AMT5: int | None = Field(None, **_BILL)
    BILL_AMT6: int | None = Field(None, **_BILL)
    PAY_AMT1: int | None = Field(None, **_PAY_AMT)
    PAY_AMT2: int | None = Field(None, **_PAY_AMT)
    PAY_AMT3: int | None = Field(None, **_PAY_AMT)
    PAY_AMT4: int | None = Field(None, **_PAY_AMT)
    PAY_AMT5: int | None = Field(None, **_PAY_AMT)
    PAY_AMT6: int | None = Field(None, **_PAY_AMT)

    @model_validator(mode="after")
    def _limit_missing_features(self):
        missing = [f for f in FEATURES if getattr(self, f) is None]
        if len(missing) > MAX_MISSING_FEATURES:
            raise ValueError(
                f"Too many missing features ({len(missing)} of 23; at most {MAX_MISSING_FEATURES} allowed): "
                + ", ".join(missing)
            )
        return self

    def feature_dict(self) -> dict:
        """The 23 features (None where missing), in canonical order."""
        return {f: getattr(self, f) for f in FEATURES}


class BatchPredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    batch_name: str | None = Field(None, max_length=120, description="Label for this portfolio scoring run")
    customers: list[CustomerFeatures] = Field(..., min_length=1)


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
class RiskBand(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class PredictionResult(BaseModel):
    prediction_id: int | None = Field(None, description="Row id in MySQL (null if not stored)")
    stored: bool = Field(..., description="True if the prediction was saved to MySQL")
    customer_ref: str | None = None
    default_probability: float = Field(..., ge=0, le=1)
    predicted_default: bool
    prediction_label: str = Field(..., description='"Default" or "No default"')
    risk_band: RiskBand
    decision_threshold: float
    model_name: str
    model_version: str
    imputed_features: list[str] = Field(
        default_factory=list, description="Features that were missing and filled with training medians/modes")
    created_at: datetime | None = None


class RiskBandBreakdown(BaseModel):
    risk_band: RiskBand
    customers: int
    share: float


class BatchPredictionResponse(BaseModel):
    batch_id: int | None = Field(None, description="Null when the database is disabled")
    stored: bool
    batch_name: str | None
    total_customers: int
    predicted_defaults: int
    predicted_default_rate: float
    average_default_probability: float
    customers_with_imputed_values: int
    risk_band_breakdown: list[RiskBandBreakdown]
    predictions: list[PredictionResult]


class StoredPrediction(CustomerFeatures):
    """A prediction row read back from MySQL: inputs + outputs."""

    model_config = ConfigDict(from_attributes=True, extra="ignore")

    id: int
    batch_id: int | None
    default_probability: float
    predicted_default: bool
    risk_band: RiskBand
    decision_threshold: float
    model_name: str
    model_version: str
    imputed_features: list[str] = Field(default_factory=list)
    created_at: datetime

    @field_validator("imputed_features", mode="before")
    @classmethod
    def _split_imputed(cls, value):
        if value is None or value == "":
            return []
        return value.split(",") if isinstance(value, str) else value


class PredictionList(BaseModel):
    total: int
    limit: int
    offset: int
    items: list[StoredPrediction]


class PortfolioSummary(BaseModel):
    batch_id: int | None = Field(None, description="Null = whole stored portfolio")
    total_customers: int
    predicted_defaults: int
    predicted_default_rate: float
    average_default_probability: float
    total_credit_limit: int
    credit_limit_of_predicted_defaulters: int
    share_of_credit_limit_at_risk: float
    risk_band_breakdown: list[RiskBandBreakdown]


class SegmentDimension(str, Enum):
    SEX = "SEX"
    EDUCATION = "EDUCATION"
    MARRIAGE = "MARRIAGE"
    AGE_GROUP = "AGE_GROUP"
    PAY_0 = "PAY_0"


class SegmentRisk(BaseModel):
    segment: str
    customers: int
    predicted_defaults: int
    predicted_default_rate: float
    average_default_probability: float


class SegmentAnalysis(BaseModel):
    dimension: SegmentDimension
    batch_id: int | None
    segments: list[SegmentRisk]


class ModelInfo(BaseModel):
    model_name: str
    model_version: str
    trained_at: str
    sklearn_version: str
    features: list[str]
    target: str
    decision_threshold: float
    risk_bands: list[dict]
    cv_metrics: dict
    test_metrics_default_threshold: dict
    test_metrics_tuned_threshold: dict


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    missing_value_imputation: bool
    database_mode: str = Field(..., description='"enabled" or "disabled" (DATABASE_ENABLED)')
    database_connected: bool | None = Field(..., description="Null when the database is disabled")
