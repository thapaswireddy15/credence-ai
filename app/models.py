"""SQLAlchemy ORM models (MySQL tables)."""

from datetime import datetime

from sqlalchemy import (
    BigInteger,
    Boolean,
    DateTime,
    Enum,
    Double,
    ForeignKey,
    Index,
    Integer,
    SmallInteger,
    String,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base

RISK_BAND_ENUM = Enum("LOW", "MEDIUM", "HIGH", name="risk_band")


class PredictionBatch(Base):
    """A portfolio scoring run: many customers scored together."""

    __tablename__ = "prediction_batches"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str | None] = mapped_column(String(120))
    total_customers: Mapped[int] = mapped_column(Integer, nullable=False)
    predicted_defaults: Mapped[int] = mapped_column(Integer, nullable=False)
    average_default_probability: Mapped[float] = mapped_column(Double, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), nullable=False)

    predictions: Mapped[list["Prediction"]] = relationship(back_populates="batch")

    __table_args__ = ({"mysql_engine": "InnoDB", "mysql_charset": "utf8mb4"},)


class Prediction(Base):
    """One customer-level default prediction, with the exact inputs used."""

    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    batch_id: Mapped[int | None] = mapped_column(
        ForeignKey("prediction_batches.id", ondelete="SET NULL"), index=True
    )
    customer_ref: Mapped[str | None] = mapped_column(String(64), index=True)

    # ---- 23 model inputs -------------------------------------------------
    LIMIT_BAL: Mapped[int] = mapped_column(BigInteger, nullable=False)
    SEX: Mapped[int] = mapped_column(SmallInteger, nullable=False)
    EDUCATION: Mapped[int] = mapped_column(SmallInteger, nullable=False)
    MARRIAGE: Mapped[int] = mapped_column(SmallInteger, nullable=False)
    AGE: Mapped[int] = mapped_column(SmallInteger, nullable=False)
    PAY_0: Mapped[int] = mapped_column(SmallInteger, nullable=False)
    PAY_2: Mapped[int] = mapped_column(SmallInteger, nullable=False)
    PAY_3: Mapped[int] = mapped_column(SmallInteger, nullable=False)
    PAY_4: Mapped[int] = mapped_column(SmallInteger, nullable=False)
    PAY_5: Mapped[int] = mapped_column(SmallInteger, nullable=False)
    PAY_6: Mapped[int] = mapped_column(SmallInteger, nullable=False)
    BILL_AMT1: Mapped[int] = mapped_column(BigInteger, nullable=False)
    BILL_AMT2: Mapped[int] = mapped_column(BigInteger, nullable=False)
    BILL_AMT3: Mapped[int] = mapped_column(BigInteger, nullable=False)
    BILL_AMT4: Mapped[int] = mapped_column(BigInteger, nullable=False)
    BILL_AMT5: Mapped[int] = mapped_column(BigInteger, nullable=False)
    BILL_AMT6: Mapped[int] = mapped_column(BigInteger, nullable=False)
    PAY_AMT1: Mapped[int] = mapped_column(BigInteger, nullable=False)
    PAY_AMT2: Mapped[int] = mapped_column(BigInteger, nullable=False)
    PAY_AMT3: Mapped[int] = mapped_column(BigInteger, nullable=False)
    PAY_AMT4: Mapped[int] = mapped_column(BigInteger, nullable=False)
    PAY_AMT5: Mapped[int] = mapped_column(BigInteger, nullable=False)
    PAY_AMT6: Mapped[int] = mapped_column(BigInteger, nullable=False)

    # ---- model output ----------------------------------------------------
    default_probability: Mapped[float] = mapped_column(Double, nullable=False)
    predicted_default: Mapped[bool] = mapped_column(Boolean, nullable=False)
    risk_band: Mapped[str] = mapped_column(RISK_BAND_ENUM, nullable=False)
    decision_threshold: Mapped[float] = mapped_column(Double, nullable=False)
    model_name: Mapped[str] = mapped_column(String(64), nullable=False)
    model_version: Mapped[str] = mapped_column(String(16), nullable=False)
    # Comma-separated names of features that were missing in the request and
    # imputed (the feature columns above hold the values the model actually used).
    imputed_features: Mapped[str | None] = mapped_column(String(300))
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), nullable=False)

    batch: Mapped[PredictionBatch | None] = relationship(back_populates="predictions")

    __table_args__ = (
        Index("ix_predictions_risk_band", "risk_band"),
        Index("ix_predictions_created_at", "created_at"),
        {"mysql_engine": "InnoDB", "mysql_charset": "utf8mb4"},
    )
