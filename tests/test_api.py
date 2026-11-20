"""API tests with FastAPI's TestClient (HTTPX). No database: history is in memory."""

import pytest
from fastapi.testclient import TestClient

from app.main import app


# ---------------------------------------------------------------------------
# Service endpoints
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


def test_health(client, store):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "model_loaded": True, "missing_value_imputation": True,
                        "storage": "in-memory (temporary, cleared on restart)", "predictions_in_memory": 0}


# ---------------------------------------------------------------------------
# Input validation
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
# Predictions kept in the in-memory history
# ---------------------------------------------------------------------------
def test_single_prediction_is_stored(client, store, customer):
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


def test_get_missing_prediction_404(client, store):
    assert client.get("/predictions/999999").status_code == 404


def test_delete_prediction(client, store, customer):
    pid = client.post("/predictions", json=customer).json()["prediction_id"]
    assert client.delete(f"/predictions/{pid}").status_code == 204
    assert client.get(f"/predictions/{pid}").status_code == 404
    assert client.delete(f"/predictions/{pid}").status_code == 404


def test_batch_prediction_and_portfolio_analytics(client, store, customer, low_risk_customer):
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


def test_portfolio_unknown_batch_404(client, store):
    assert client.get("/portfolio/summary", params={"batch_id": 987654}).status_code == 404


def test_empty_portfolio_summary(client, store):
    s = client.get("/portfolio/summary").json()
    assert s["total_customers"] == 0 and s["predicted_default_rate"] == 0.0


# ---------------------------------------------------------------------------
# Missing-value handling (imputation with training-split medians / modes)
# ---------------------------------------------------------------------------
FILL = __import__("json").load(open("models/imputation_values.json"))["fill_values"]


def test_complete_record_unchanged_by_imputation(client, store, customer):
    body = client.post("/predictions", json=customer).json()
    assert body["imputed_features"] == []
    assert body["default_probability"] == pytest.approx(0.749109, abs=1e-6)  # same as before the feature


def test_omitted_feature_is_imputed_and_stored(client, store, customer):
    customer.pop("BILL_AMT3")
    r = client.post("/predictions", json=customer)
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["imputed_features"] == ["BILL_AMT3"]
    assert body["stored"] is True
    stored = client.get(f"/predictions/{body['prediction_id']}").json()
    assert stored["BILL_AMT3"] == FILL["BILL_AMT3"]  # value the model actually used
    assert stored["imputed_features"] == ["BILL_AMT3"]


def test_explicit_null_is_imputed(client, store, customer):
    customer.update({"AGE": None, "EDUCATION": None, "PAY_AMT2": None})
    body = client.post("/predictions", json=customer).json()
    assert sorted(body["imputed_features"]) == ["AGE", "EDUCATION", "PAY_AMT2"]
    assert 0 <= body["default_probability"] <= 1


def test_imputation_matches_sending_fill_values(client, store, customer):
    missing = dict(customer); missing.pop("PAY_0"); missing.pop("LIMIT_BAL")
    filled = dict(customer, PAY_0=FILL["PAY_0"], LIMIT_BAL=FILL["LIMIT_BAL"])
    a = client.post("/predictions", json=missing).json()["default_probability"]
    b = client.post("/predictions", json=filled).json()["default_probability"]
    assert a == pytest.approx(b)


def test_maximum_allowed_missing_features(client, store, customer):
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


def test_batch_with_missing_values(client, store, customer, low_risk_customer):
    incomplete = dict(low_risk_customer); incomplete.pop("PAY_AMT1"); incomplete["AGE"] = None
    body = client.post("/predictions/batch", json={"customers": [customer, incomplete]}).json()
    assert body["customers_with_imputed_values"] == 1
    assert body["predictions"][0]["imputed_features"] == []
    assert sorted(body["predictions"][1]["imputed_features"]) == ["AGE", "PAY_AMT1"]


# ---------------------------------------------------------------------------
# Database-free operation and in-memory history
# ---------------------------------------------------------------------------
def test_app_does_not_import_any_database_library():
    import subprocess
    import sys
    code = ("import sys, app.main; "
            "bad = [m for m in sys.modules if m.split('.')[0] in "
            "('sqlalchemy', 'pymysql', 'MySQLdb', 'mysql', 'sqlite3', 'pymongo', 'motor')]; "
            "print(bad); sys.exit(1 if bad else 0)")
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, f"database modules imported: {result.stdout}"


def test_predict_returns_probability_prediction_and_risk(client, store, customer):
    body = client.post("/predictions", json=customer).json()
    assert body["default_probability"] == pytest.approx(0.749109, abs=1e-6)
    assert body["predicted_default"] is True and body["risk_band"] == "HIGH"
    assert body["stored"] is True and body["prediction_id"] == 1


def test_history_counts_and_ids(client, store, customer, low_risk_customer):
    ids = [client.post("/predictions", json=c).json()["prediction_id"]
           for c in (customer, low_risk_customer, customer)]
    assert ids == [1, 2, 3]
    assert client.get("/health").json()["predictions_in_memory"] == 3
    assert client.get("/predictions").json()["total"] == 3


def test_history_is_not_persistent_across_restarts(customer):
    with TestClient(app) as first:
        pid = first.post("/predictions", json=customer).json()["prediction_id"]
        assert first.get(f"/predictions/{pid}").status_code == 200
    with TestClient(app) as restarted:  # new process lifetime -> empty history
        assert restarted.get(f"/predictions/{pid}").status_code == 404
        assert restarted.get("/health").json()["predictions_in_memory"] == 0


def test_history_capacity_evicts_oldest(client, store, customer):
    old_limit = store.max_predictions
    store.max_predictions = 3
    try:
        ids = [client.post("/predictions", json=customer).json()["prediction_id"] for _ in range(5)]
        assert len(store) == 3
        assert client.get(f"/predictions/{ids[0]}").status_code == 404   # evicted
        assert client.get(f"/predictions/{ids[-1]}").status_code == 200  # newest kept
    finally:
        store.max_predictions = old_limit


def test_deleted_prediction_leaves_portfolio(client, store, customer, low_risk_customer):
    pid = client.post("/predictions", json=customer).json()["prediction_id"]
    client.post("/predictions", json=low_risk_customer)
    assert client.get("/portfolio/summary").json()["total_customers"] == 2
    client.delete(f"/predictions/{pid}")
    summary = client.get("/portfolio/summary").json()
    assert summary["total_customers"] == 1 and summary["predicted_defaults"] == 0


def test_concurrent_predictions_get_unique_ids(client, store, customer):
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=8) as pool:
        ids = list(pool.map(lambda _: client.post("/predictions", json=customer).json()["prediction_id"],
                            range(40)))
    assert len(set(ids)) == 40 and len(store) == 40
