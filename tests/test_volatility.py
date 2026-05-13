import numpy as np
import pytest

from src.volatility import ewma_variance, ewma_volatility


def test_ewma_variance_simple_series():
    returns = [0.01, -0.02, 0.03]
    lambda_ = 0.5
    weights = np.array([0.125, 0.25, 0.5])
    expected = np.sum(weights * np.array(returns) ** 2) / np.sum(weights)

    assert ewma_variance(returns, lambda_=lambda_) == pytest.approx(expected)


def test_ewma_invalid_lambda():
    with pytest.raises(ValueError):
        ewma_variance([0.01, 0.02], lambda_=1.0)


def test_ewma_volatility_annualized():
    returns = [0.01, -0.02, 0.03]
    daily = ewma_volatility(returns, lambda_=0.5)

    assert ewma_volatility(returns, lambda_=0.5, annualize=True, periods_per_year=252) == pytest.approx(
        daily * np.sqrt(252)
    )
