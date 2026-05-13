import pandas as pd
import pytest

from src.stress_testing import (
    apply_price_shock,
    apply_volatility_shock,
    stress_portfolio_returns,
    summarize_stress_scenario,
)


def test_apply_price_shock():
    prices = pd.Series([100.0, 110.0])
    shocked = apply_price_shock(prices, -0.10)

    assert shocked.tolist() == [90.0, 99.0]


def test_apply_volatility_shock():
    returns = pd.Series([0.01, -0.02])
    shocked = apply_volatility_shock(returns, 2.0)

    assert shocked.tolist() == [0.02, -0.04]


def test_stress_portfolio_returns_dataframe_and_weights():
    returns = pd.DataFrame({"AAPL": [0.01, 0.02], "MSFT": [0.00, 0.01]})
    result = stress_portfolio_returns(returns, weights=[0.6, 0.4], shocks={"AAPL": -0.10, "MSFT": -0.05})

    assert result.tolist() == pytest.approx([-0.074, -0.064])


def test_summarize_stress_scenario_expected_keys():
    summary = summarize_stress_scenario(pd.Series([0.01, -0.05, 0.02, -0.03]))

    assert set(summary) == {"mean_return", "min_return", "max_drawdown", "var", "cvar"}
