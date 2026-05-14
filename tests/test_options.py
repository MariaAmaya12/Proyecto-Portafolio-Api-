import numpy as np
import pytest

from src.options import (
    black_scholes_call,
    black_scholes_put,
    delta_call,
    gamma,
    vega,
)


def test_call_and_put_are_positive():
    call = black_scholes_call(100, 100, 0.05, 0.2, 1)
    put = black_scholes_put(100, 100, 0.05, 0.2, 1)

    assert call > 0
    assert put > 0


def test_put_call_parity():
    spot = 100
    strike = 100
    rate = 0.05
    maturity = 1
    call = black_scholes_call(spot, strike, rate, 0.2, maturity)
    put = black_scholes_put(spot, strike, rate, 0.2, maturity)

    assert call - put == pytest.approx(spot - strike * np.exp(-rate * maturity))


def test_delta_call_between_zero_and_one():
    value = delta_call(100, 100, 0.05, 0.2, 1)

    assert 0 < value < 1


def test_gamma_and_vega_are_positive():
    assert gamma(100, 100, 0.05, 0.2, 1) > 0
    assert vega(100, 100, 0.05, 0.2, 1) > 0


def test_invalid_volatility_raises_error():
    with pytest.raises(ValueError):
        black_scholes_call(100, 100, 0.05, 0.0, 1)
