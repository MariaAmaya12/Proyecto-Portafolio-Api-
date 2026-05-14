from fastapi.testclient import TestClient

from backend.main import app


client = TestClient(app)


def test_stress_portfolio_valid_request():
    payload = {
        "returns": [
            [0.01, -0.005],
            [-0.02, 0.004],
            [0.015, -0.003],
            [-0.01, -0.002],
            [0.008, 0.006],
        ],
        "weights": [0.6, 0.4],
        "shocks": {
            "asset_0": -0.01,
            "asset_1": -0.005,
        },
        "confidence_level": 0.95,
    }

    response = client.post("/stress/portfolio", json=payload)

    assert response.status_code == 200
    body = response.json()
    for key in ["mean_return", "min_return", "max_drawdown", "var", "cvar", "observations"]:
        assert key in body
        assert isinstance(body[key], int | float)
    assert body["observations"] == 5


def test_stress_portfolio_invalid_request():
    payload = {
        "returns": [
            [0.01, -0.005],
            [-0.02, 0.004],
        ],
        "weights": [0.6],
        "shocks": {
            "asset_0": -0.01,
            "asset_1": -0.005,
        },
        "confidence_level": 0.95,
    }

    response = client.post("/stress/portfolio", json=payload)

    assert response.status_code == 422


def test_combined_stress_scenario_valid_request():
    payload = {
        "prices": [100.0, 101.0, 99.0, 98.0],
        "returns": [0.01, -0.02, 0.005, -0.01],
        "weights": 1.0,
        "price_shock": -0.1,
        "volatility_multiplier": 1.5,
        "bond_price": 1000.0,
        "modified_duration": 4.5,
        "convexity": 20.0,
        "rate_shock": 0.01,
    }

    response = client.post("/stress/combined-scenario", json=payload)

    assert response.status_code == 200
    body = response.json()
    assert "scenario_summary" in body
    assert "shocked_bond_price" in body
    assert body["shocked_bond_price"] > 0
    assert body["scenario_summary"] is not None
    assert isinstance(body["scenario_summary"], dict)
    if body.get("shocked_prices") is not None:
        assert isinstance(body["shocked_prices"], list)
    if body.get("shocked_returns") is not None:
        assert isinstance(body["shocked_returns"], list)


def test_combined_stress_scenario_invalid_request():
    payload = {
        "bond_price": 1000.0,
        "modified_duration": 4.5,
    }

    response = client.post("/stress/combined-scenario", json=payload)

    # El contrato invalido puede rechazarse en Pydantic (422) o en la capa de dominio (400).
    assert response.status_code in {400, 422}
