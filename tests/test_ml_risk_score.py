from fastapi.testclient import TestClient

from backend.main import app

client = TestClient(app)


def test_risk_score_valid_request():
    response = client.post(
        "/ml/risk-score",
        json={
            "ret_1d": -0.005,
            "ret_5d": -0.018,
            "ret_20d": 0.012,
            "vol_5d": 0.014,
            "vol_20d": 0.011,
            "rsi": 42.0,
            "macd_hist": -0.003,
            "bb_position": 0.28,
            "close_over_sma20": 0.975,
            "drawdown_20d": -0.031,
        },
    )

    assert response.status_code == 200

    body = response.json()
    assert "risk_score" in body
    assert "risk_level" in body
    assert "horizon_days" in body
    assert "model_version" in body
    assert "log_id" in body

    assert 0.0 <= body["risk_score"] <= 1.0
    assert body["risk_level"] in {"alto", "moderado", "bajo"}
    assert body["horizon_days"] == 5
    assert isinstance(body["model_version"], str) and body["model_version"] != ""
    assert body["log_id"] is None or isinstance(body["log_id"], int)


def test_risk_score_invalid_request():
    response = client.post(
        "/ml/risk-score",
        json={"rsi": "no-es-numero"},
    )

    assert response.status_code == 422
