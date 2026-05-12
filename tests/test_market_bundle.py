import pandas as pd
from fastapi.testclient import TestClient

from backend.main import app, get_market_service


class FakeMarketService:
    def load_market_bundle(self, tickers, start, end):
        dates = pd.to_datetime(["2024-01-02", "2024-01-03"])
        ohlcv_aapl = pd.DataFrame(
            {
                "Open": [100.0, 101.0],
                "High": [102.0, 103.0],
                "Low": [99.0, 100.0],
                "Close": [101.0, 102.0],
                "Adj Close": [101.0, 102.0],
                "Volume": [1000, 1100],
            },
            index=dates,
        )
        close = pd.DataFrame({"AAPL": [101.0, 102.0]}, index=dates)
        returns = pd.DataFrame({"AAPL": [0.0099]}, index=pd.to_datetime(["2024-01-03"]))

        return {
            "ohlcv": {
                "AAPL": ohlcv_aapl,
                "3382.T": pd.DataFrame(),
            },
            "close": close,
            "returns": returns,
            "calendar_diagnostics": {
                "calendar_fill_applied": True,
            },
        }


def test_market_bundle_contract_with_ticker_diagnostics():
    app.dependency_overrides[get_market_service] = lambda: FakeMarketService()
    client = TestClient(app)

    try:
        response = client.post(
            "/market/bundle",
            json={
                "tickers": ["AAPL", "3382.T"],
                "start": "2024-01-01",
                "end": "2024-01-10",
            },
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    payload = response.json()

    for field in [
        "ohlcv",
        "close",
        "returns",
        "missing_tickers",
        "last_available_date",
        "calendar_diagnostics",
    ]:
        assert field in payload

    assert "included_tickers" in payload
    assert "metadata" in payload
    assert isinstance(payload["missing_tickers"], list)
    assert isinstance(payload["included_tickers"], list)
    assert isinstance(payload["metadata"], dict)
    assert "AAPL" in payload["included_tickers"]
    assert "3382.T" in payload["missing_tickers"]

    calendar_diagnostics = payload["calendar_diagnostics"]
    assert "by_ticker" in calendar_diagnostics
    assert "AAPL" in calendar_diagnostics["by_ticker"]

    aapl_diagnostics = calendar_diagnostics["by_ticker"]["AAPL"]
    for field in [
        "raw_rows",
        "aligned_rows",
        "close_rows",
        "return_rows",
        "reason",
        "suggestion",
    ]:
        assert field in aapl_diagnostics

    metadata = payload["metadata"]
    for field in [
        "requested_tickers",
        "included_tickers",
        "missing_tickers",
        "source",
        "generated_at",
    ]:
        assert field in metadata
