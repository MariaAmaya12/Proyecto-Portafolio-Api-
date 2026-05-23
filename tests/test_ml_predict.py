from fastapi.testclient import TestClient

from backend.main import app

client = TestClient(app)


def test_ml_predict_valid_request():
    response = client.post(
        "/ml/predict",
        json={"close": 110.0, "sma": 100.0, "ema": 105.0, "rsi": 55.0},
    )

    assert response.status_code == 200

    body = response.json()
    assert "prediction" in body
    assert "probability" in body
    assert "model_version" in body
    assert "log_id" in body

    assert body["prediction"] in {"Alcista", "Bajista", "Neutral"}
    assert 0.0 <= body["probability"] <= 1.0
    assert isinstance(body["model_version"], str) and body["model_version"] != ""
    assert body["log_id"] is None or isinstance(body["log_id"], int)


def test_ml_predict_invalid_request():
    response = client.post(
        "/ml/predict",
        json={"close": "no-es-numero"},
    )

    assert response.status_code == 422


def test_predict_alias_matches_ml_predict():
    """El alias /predict debe devolver la misma estructura que /ml/predict."""
    payload = {"close": 110.0, "sma": 100.0, "ema": 105.0, "rsi": 55.0}

    response = client.post("/predict", json=payload)

    assert response.status_code == 200

    body = response.json()
    assert "prediction" in body
    assert "probability" in body
    assert "model_version" in body
    assert body["prediction"] in {"Alcista", "Bajista", "Neutral"}
    assert 0.0 <= body["probability"] <= 1.0
