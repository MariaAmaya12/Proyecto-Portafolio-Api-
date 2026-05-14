from fastapi.testclient import TestClient

from backend.main import app


client = TestClient(app)


def test_ewma_volatility_valid_request():
    payload = {
        "returns": [0.01, -0.02, 0.015, -0.005, 0.008],
        "lambda_": 0.94,
        "annualize": False,
        "periods_per_year": 252,
    }

    response = client.post("/volatility/ewma", json=payload)

    assert response.status_code == 200
    body = response.json()
    assert "ewma_volatility" in body
    assert "ewma_variance" in body
    assert "lambda_" in body
    assert "annualize" in body
    assert "periods_per_year" in body
    assert "observations" in body
    assert body["ewma_volatility"] >= 0
    assert body["ewma_variance"] >= 0
    assert body["lambda_"] == 0.94
    assert body["annualize"] is False
    assert body["periods_per_year"] == 252
    assert body["observations"] == 5


def test_ewma_volatility_invalid_request():
    payload = {
        "returns": [0.01],
        "lambda_": 1.5,
    }

    response = client.post("/volatility/ewma", json=payload)

    assert response.status_code == 422
