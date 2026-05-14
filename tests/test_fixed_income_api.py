from fastapi.testclient import TestClient

from backend.main import app


client = TestClient(app)


def test_bond_metrics_valid_request():
    payload = {
        "face_value": 1000.0,
        "coupon_rate": 0.05,
        "market_rate": 0.04,
        "maturity_years": 5.0,
        "frequency": 1,
    }

    response = client.post("/fixed-income/bond-metrics", json=payload)

    assert response.status_code == 200
    body = response.json()
    assert "price" in body
    assert "macaulay_duration" in body
    assert "modified_duration" in body
    assert "convexity" in body
    assert body["price"] > 0
    assert body["macaulay_duration"] > 0
    assert body["modified_duration"] > 0
    assert body["convexity"] > 0


def test_bond_metrics_invalid_request():
    payload = {
        "face_value": -1000.0,
        "coupon_rate": 0.05,
        "market_rate": 0.04,
        "maturity_years": 5.0,
        "frequency": 3,
    }

    response = client.post("/fixed-income/bond-metrics", json=payload)

    assert response.status_code == 422


def test_nelson_siegel_valid_request():
    payload = {
        "maturities": [0.5, 1.0, 2.0, 5.0, 10.0],
        "beta0": 0.05,
        "beta1": -0.02,
        "beta2": 0.01,
        "tau": 1.5,
    }

    response = client.post("/fixed-income/nelson-siegel", json=payload)

    assert response.status_code == 200
    body = response.json()
    assert "maturities" in body
    assert "yields" in body
    assert body["maturities"] == payload["maturities"]
    assert len(body["yields"]) == len(payload["maturities"])
    assert all(isinstance(value, float) for value in body["yields"])


def test_nelson_siegel_invalid_request():
    payload = {
        "maturities": [1.0, -2.0, 5.0],
        "beta0": 0.05,
        "beta1": -0.02,
        "beta2": 0.01,
        "tau": 0,
    }

    response = client.post("/fixed-income/nelson-siegel", json=payload)

    assert response.status_code == 422
