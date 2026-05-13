import pandas as pd
import pytest

from src.stress_testing import (
    apply_rate_shock_to_bond,
    apply_price_shock,
    apply_volatility_shock,
    combined_adverse_scenario,
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


def test_apply_rate_shock_to_bond_duration_price_falls_when_rate_rises():
    shocked_price = apply_rate_shock_to_bond(1000.0, modified_duration=4.0, rate_shock=0.01)

    assert shocked_price < 1000.0


def test_apply_rate_shock_to_bond_with_convexity_positive_price():
    shocked_price = apply_rate_shock_to_bond(
        1000.0,
        modified_duration=4.0,
        convexity=30.0,
        rate_shock=0.01,
    )

    assert shocked_price > 0


def test_apply_rate_shock_to_bond_invalid_price():
    with pytest.raises(ValueError):
        apply_rate_shock_to_bond(0.0, modified_duration=4.0, rate_shock=0.01)


def test_combined_adverse_scenario_with_price_shock():
    prices = pd.Series([100.0, 110.0])
    result = combined_adverse_scenario(prices=prices, price_shocks=-0.10)

    assert "shocked_prices" in result
    assert result["shocked_prices"].tolist() == [90.0, 99.0]


def test_combined_adverse_scenario_with_rate_shock():
    result = combined_adverse_scenario(
        bond_price=1000.0,
        modified_duration=4.0,
        rate_shock=0.01,
    )

    assert "shocked_bond_price" in result
    assert result["shocked_bond_price"] < 1000.0


def test_combined_adverse_scenario_with_volatility_shock():
    returns = pd.Series([0.01, -0.02])
    result = combined_adverse_scenario(returns=returns, volatility_multiplier=2.0)

    assert "shocked_returns" in result
    assert result["shocked_returns"].tolist() == [0.02, -0.04]
