"""Shared pytest fixtures.

Database tests run against a SEPARATE MySQL database (default: ``<your db>_test``,
override with ``TEST_DATABASE_URL``) so that your real predictions are never
touched. If MySQL is unreachable, database tests are skipped with a clear
message -- they are never silently redirected to SQLite.
"""

import os

import pytest

# --- Point the application at the test database BEFORE importing it --------
from app import config

_base_url = config._build_database_url()
_test_url = os.getenv("TEST_DATABASE_URL") or _base_url.set(database=f"{_base_url.database}_test")
os.environ["DATABASE_URL"] = (
    _test_url if isinstance(_test_url, str) else _test_url.render_as_string(hide_password=False)
)
config.get_settings.cache_clear()

from fastapi.testclient import TestClient  # noqa: E402
from sqlalchemy import delete  # noqa: E402

from app.database import SessionLocal, engine, init_db  # noqa: E402
from app.main import app  # noqa: E402
from app.models import Prediction, PredictionBatch  # noqa: E402
from app.schemas import EXAMPLE_CUSTOMER  # noqa: E402


def _mysql_available() -> tuple[bool, str]:
    try:
        init_db(engine, create_database=True)
        return True, ""
    except Exception as exc:  # pragma: no cover - depends on environment
        return False, str(exc).splitlines()[0]


_DB_OK, _DB_ERROR = _mysql_available()


@pytest.fixture(scope="session")
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture
def mysql_db():
    """Skip if MySQL is unavailable; otherwise give each test empty tables."""
    if not _DB_OK:
        pytest.skip(f"MySQL test database unavailable ({_DB_ERROR}). Configure .env - see README.")
    with SessionLocal() as session:
        session.execute(delete(Prediction))
        session.execute(delete(PredictionBatch))
        session.commit()
    yield
    with SessionLocal() as session:
        session.execute(delete(Prediction))
        session.execute(delete(PredictionBatch))
        session.commit()


@pytest.fixture
def customer() -> dict:
    return dict(EXAMPLE_CUSTOMER)


@pytest.fixture
def low_risk_customer() -> dict:
    return {
        "customer_ref": "CUST-LOW",
        "LIMIT_BAL": 500000, "SEX": 2, "EDUCATION": 1, "MARRIAGE": 1, "AGE": 40,
        "PAY_0": -1, "PAY_2": -1, "PAY_3": -1, "PAY_4": -1, "PAY_5": -1, "PAY_6": -1,
        "BILL_AMT1": 12000, "BILL_AMT2": 9000, "BILL_AMT3": 15000,
        "BILL_AMT4": 8000, "BILL_AMT5": 11000, "BILL_AMT6": 7000,
        "PAY_AMT1": 9000, "PAY_AMT2": 15000, "PAY_AMT3": 8000,
        "PAY_AMT4": 11000, "PAY_AMT5": 7000, "PAY_AMT6": 10000,
    }
