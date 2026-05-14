from fastapi.testclient import TestClient

from backend.main import app


client = TestClient(app)


def test_black_scholes_valid_request():
    payload = {
        "spot": 100.0,
        "strike": 100.0,
        "rate": 0.05,
        "volatility": 0.2,
        "time_to_maturity": 1.0,
    }

    response = client.post("/options/black-scholes", json=payload)

    assert response.status_code == 200
    body = response.json()
    assert "call_price" in body
    assert "put_price" in body
    assert body["call_price"] > 0
    assert body["put_price"] > 0


def test_black_scholes_invalid_request():
    payload = {
        "spot": -100.0,
        "strike": 100.0,
        "rate": 0.05,
        "volatility": 0.2,
        "time_to_maturity": 1.0,
    }

    response = client.post("/options/black-scholes", json=payload)

    assert response.status_code == 422


def test_greeks_valid_request():
    payload = {
        "spot": 100.0,
        "strike": 100.0,
        "rate": 0.05,
        "volatility": 0.2,
        "time_to_maturity": 1.0,
    }

    response = client.post("/options/greeks", json=payload)

    assert response.status_code == 200
    body = response.json()
    for key in [
        "delta_call",
        "delta_put",
        "gamma",
        "vega",
        "theta_call",
        "theta_put",
        "rho_call",
        "rho_put",
    ]:
        assert key in body
        assert isinstance(body[key], float)


def test_greeks_invalid_request():
    payload = {
        "spot": 100.0,
        "strike": 0.0,
        "rate": 0.05,
        "volatility": -0.2,
        "time_to_maturity": 1.0,
    }

    response = client.post("/options/greeks", json=payload)

    assert response.status_code == 422
