"""API tests with FastAPI's TestClient (HTTPX) against the MySQL test database."""

import pytest


# ---------------------------------------------------------------------------
# Service endpoints (no database writes)
# ---------------------------------------------------------------------------
def test_root(client):
    r = client.get("/")
    assert r.status_code == 200
    assert r.json()["stakeholder"] == "Credit Portfolio Strategist"


def test_docs_available(client):
    assert client.get("/docs").status_code == 200
    assert "/predictions" in client.get("/openapi.json").json()["paths"]


def test_model_info(client):
    r = client.get("/model/info")
    assert r.status_code == 200
    body = r.json()
    assert len(body["features"]) == 23
    assert body["target"] == "default.payment.next.month"
    assert 0 < body["decision_threshold"] < 1


def test_health(client, mysql_db):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "model_loaded": True, "missing_value_imputation": True,
                        "database_mode": "enabled", "database_connected": True}


# ---------------------------------------------------------------------------
# Input validation (rejected before touching the database)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "field,value",
    [("SEX", 3), ("AGE", 10), ("LIMIT_BAL", 0), ("PAY_0", -3), ("PAY_AMT1", -5), ("EDUCATION", 7)],
)
def test_invalid_values_rejected(client, customer, field, value):
    customer[field] = value
    assert client.post("/predictions", json=customer).status_code == 422


def test_too_many_missing_features_rejected(client, customer):
    for f in ["LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE", "PAY_0",
              "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6", "BILL_AMT1"]:  # 12 > 11 allowed
        customer.pop(f)
    r = client.post("/predictions", json=customer)
    assert r.status_code == 422
    assert "Too many missing features (12 of 23" in r.text


def test_unknown_field_rejected(client, customer):
    customer["default.payment.next.month"] = 1  # the target must never be an input
    assert client.post("/predictions", json=customer).status_code == 422


def test_empty_batch_rejected(client):
    assert client.post("/predictions/batch", json={"customers": []}).status_code == 422


# ---------------------------------------------------------------------------
# Predictions stored in MySQL
# ---------------------------------------------------------------------------
def test_single_prediction_is_stored(client, mysql_db, customer):
    r = client.post("/predictions", json=customer)
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["prediction_id"] > 0
    assert body["customer_ref"] == "CUST-000123"
    assert 0 <= body["default_probability"] <= 1
    assert body["predicted_default"] is True
    assert body["prediction_label"] == "Default"
    assert body["risk_band"] in {"LOW", "MEDIUM", "HIGH"}

    stored = client.get(f"/predictions/{body['prediction_id']}")
    assert stored.status_code == 200
    s = stored.json()
    assert s["LIMIT_BAL"] == customer["LIMIT_BAL"]
    assert s["PAY_0"] == customer["PAY_0"]
    assert s["default_probability"] == pytest.approx(body["default_probability"])


def test_get_missing_prediction_404(client, mysql_db):
    assert client.get("/predictions/999999").status_code == 404


def test_delete_prediction(client, mysql_db, customer):
    pid = client.post("/predictions", json=customer).json()["prediction_id"]
    assert client.delete(f"/predictions/{pid}").status_code == 204
    assert client.get(f"/predictions/{pid}").status_code == 404
    assert client.delete(f"/predictions/{pid}").status_code == 404


def test_batch_prediction_and_portfolio_analytics(client, mysql_db, customer, low_risk_customer):
    payload = {"batch_name": "Q3 review", "customers": [customer, low_risk_customer, customer]}
    r = client.post("/predictions/batch", json=payload)
    assert r.status_code == 201, r.text
    batch = r.json()
    assert batch["total_customers"] == 3
    assert batch["predicted_defaults"] == 2
    assert batch["predicted_default_rate"] == pytest.approx(2 / 3, abs=1e-4)
    assert sum(b["customers"] for b in batch["risk_band_breakdown"]) == 3
    assert len(batch["predictions"]) == 3
    batch_id = batch["batch_id"]

    # listing + filters
    listed = client.get("/predictions", params={"batch_id": batch_id}).json()
    assert listed["total"] == 3
    probs = [i["default_probability"] for i in listed["items"]]
    assert probs == sorted(probs, reverse=True)  # riskiest first
    assert client.get("/predictions", params={"predicted_default": True}).json()["total"] == 2
    assert client.get("/predictions", params={"customer_ref": "CUST-LOW"}).json()["total"] == 1

    # portfolio summary
    summary = client.get("/portfolio/summary", params={"batch_id": batch_id}).json()
    assert summary["total_customers"] == 3
    assert summary["predicted_defaults"] == 2
    assert summary["total_credit_limit"] == 50000 * 2 + 500000
    assert summary["credit_limit_of_predicted_defaulters"] == 100000
    assert summary["share_of_credit_limit_at_risk"] == pytest.approx(100000 / 600000, abs=1e-4)

    # segment analysis
    seg = client.get("/portfolio/segments", params={"dimension": "PAY_0", "batch_id": batch_id}).json()
    by_name = {s["segment"]: s for s in seg["segments"]}
    assert by_name["2 months late"]["customers"] == 2
    assert by_name["Paid duly / no delay"]["predicted_defaults"] == 0
    for dim in ["SEX", "EDUCATION", "MARRIAGE", "AGE_GROUP"]:
        assert client.get("/portfolio/segments", params={"dimension": dim}).status_code == 200


def test_portfolio_unknown_batch_404(client, mysql_db):
    assert client.get("/portfolio/summary", params={"batch_id": 987654}).status_code == 404


def test_empty_portfolio_summary(client, mysql_db):
    s = client.get("/portfolio/summary").json()
    assert s["total_customers"] == 0 and s["predicted_default_rate"] == 0.0


# ---------------------------------------------------------------------------
# Missing-value handling (imputation with training-split medians / modes)
# ---------------------------------------------------------------------------
FILL = __import__("json").load(open("models/imputation_values.json"))["fill_values"]


def test_complete_record_unchanged_by_imputation(client, mysql_db, customer):
    body = client.post("/predictions", json=customer).json()
    assert body["imputed_features"] == []
    assert body["default_probability"] == pytest.approx(0.749109, abs=1e-6)  # same as before the feature


def test_omitted_feature_is_imputed_and_stored(client, mysql_db, customer):
    customer.pop("BILL_AMT3")
    r = client.post("/predictions", json=customer)
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["imputed_features"] == ["BILL_AMT3"]
    assert body["stored"] is True
    stored = client.get(f"/predictions/{body['prediction_id']}").json()
    assert stored["BILL_AMT3"] == FILL["BILL_AMT3"]  # value the model actually used
    assert stored["imputed_features"] == ["BILL_AMT3"]


def test_explicit_null_is_imputed(client, mysql_db, customer):
    customer.update({"AGE": None, "EDUCATION": None, "PAY_AMT2": None})
    body = client.post("/predictions", json=customer).json()
    assert sorted(body["imputed_features"]) == ["AGE", "EDUCATION", "PAY_AMT2"]
    assert 0 <= body["default_probability"] <= 1


def test_imputation_matches_sending_fill_values(client, mysql_db, customer):
    missing = dict(customer); missing.pop("PAY_0"); missing.pop("LIMIT_BAL")
    filled = dict(customer, PAY_0=FILL["PAY_0"], LIMIT_BAL=FILL["LIMIT_BAL"])
    a = client.post("/predictions", json=missing).json()["default_probability"]
    b = client.post("/predictions", json=filled).json()["default_probability"]
    assert a == pytest.approx(b)


def test_maximum_allowed_missing_features(client, mysql_db, customer):
    for f in ["SEX", "EDUCATION", "MARRIAGE", "AGE", "BILL_AMT1", "BILL_AMT2", "BILL_AMT3",
              "BILL_AMT4", "BILL_AMT5", "BILL_AMT6", "PAY_AMT6"]:  # exactly 11
        customer.pop(f)
    r = client.post("/predictions", json=customer)
    assert r.status_code == 201
    assert len(r.json()["imputed_features"]) == 11


def test_out_of_range_still_rejected_when_others_missing(client, customer):
    customer.pop("AGE")
    customer["SEX"] = 5
    assert client.post("/predictions", json=customer).status_code == 422


def test_batch_with_missing_values(client, mysql_db, customer, low_risk_customer):
    incomplete = dict(low_risk_customer); incomplete.pop("PAY_AMT1"); incomplete["AGE"] = None
    body = client.post("/predictions/batch", json={"customers": [customer, incomplete]}).json()
    assert body["customers_with_imputed_values"] == 1
    assert body["predictions"][0]["imputed_features"] == []
    assert sorted(body["predictions"][1]["imputed_features"]) == ["AGE", "PAY_AMT1"]


# ---------------------------------------------------------------------------
# Database-optional (predict-only) mode: DATABASE_ENABLED=false
# ---------------------------------------------------------------------------
@pytest.fixture
def db_disabled(client):
    client.app.state.database_enabled = False
    yield
    client.app.state.database_enabled = True


def test_db_disabled_predict_still_works(client, db_disabled, customer):
    customer.pop("BILL_AMT3")
    r = client.post("/predictions", json=customer)
    assert r.status_code == 201
    body = r.json()
    assert body["stored"] is False and body["prediction_id"] is None
    assert body["imputed_features"] == ["BILL_AMT3"]


def test_db_disabled_batch_not_stored(client, db_disabled, customer, low_risk_customer):
    body = client.post("/predictions/batch", json={"customers": [customer, low_risk_customer]}).json()
    assert body["stored"] is False and body["batch_id"] is None
    assert body["total_customers"] == 2 and body["predicted_defaults"] == 1


@pytest.mark.parametrize("path", ["/predictions", "/predictions/1", "/portfolio/summary", "/portfolio/segments"])
def test_db_disabled_history_endpoints_503(client, db_disabled, path):
    r = client.get(path)
    assert r.status_code == 503
    assert "DATABASE_ENABLED=false" in r.json()["detail"]


def test_db_disabled_health(client, db_disabled):
    assert client.get("/health").json() == {
        "status": "ok", "model_loaded": True, "missing_value_imputation": True,
        "database_mode": "disabled", "database_connected": None}
    assert client.get("/").json()["database_mode"] == "disabled"


# ---------------------------------------------------------------------------
# Additive schema migration for databases created before imputed_features existed
# ---------------------------------------------------------------------------
def test_migration_adds_imputed_features_column(mysql_db):
    from sqlalchemy import inspect, text

    from app.database import engine, init_db
    with engine.begin() as conn:
        conn.execute(text("ALTER TABLE predictions DROP COLUMN imputed_features"))
    assert "imputed_features" not in {c["name"] for c in inspect(engine).get_columns("predictions")}
    init_db(engine, create_database=False)
    assert "imputed_features" in {c["name"] for c in inspect(engine).get_columns("predictions")}
