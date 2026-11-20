"""Shared pytest fixtures.

Credence AI uses no database: the tests need no MySQL server or credentials.
Each test that relies on prediction history starts with an empty in-memory store.
"""

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.schemas import EXAMPLE_CUSTOMER


@pytest.fixture(scope="session")
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture
def store(client):
    """The app's in-memory prediction history, emptied before and after the test."""
    s = client.app.state.store
    s.clear()
    yield s
    s.clear()


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
