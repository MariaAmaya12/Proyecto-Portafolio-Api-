from __future__ import annotations

import numpy as np
from scipy.stats import norm


def _validate_inputs(
    spot: float,
    strike: float,
    rate: float,
    volatility: float,
    time_to_maturity: float,
) -> tuple[float, float, float, float, float]:
    if spot <= 0:
        raise ValueError("spot debe ser mayor que 0.")
    if strike <= 0:
        raise ValueError("strike debe ser mayor que 0.")
    if not np.isfinite(rate):
        raise ValueError("rate debe ser numerico y finito.")
    if volatility <= 0:
        raise ValueError("volatility debe ser mayor que 0.")
    if time_to_maturity <= 0:
        raise ValueError("time_to_maturity debe ser mayor que 0.")
    return (
        float(spot),
        float(strike),
        float(rate),
        float(volatility),
        float(time_to_maturity),
    )


def _d1_d2(spot: float, strike: float, rate: float, volatility: float, time_to_maturity: float) -> tuple[float, float]:
    spot, strike, rate, volatility, time_to_maturity = _validate_inputs(
        spot, strike, rate, volatility, time_to_maturity
    )
    sqrt_t = np.sqrt(time_to_maturity)
    d1 = (np.log(spot / strike) + (rate + 0.5 * volatility**2) * time_to_maturity) / (volatility * sqrt_t)
    d2 = d1 - volatility * sqrt_t
    return float(d1), float(d2)


def black_scholes_call(spot: float, strike: float, rate: float, volatility: float, time_to_maturity: float) -> float:
    """Precio de una opcion call europea bajo Black-Scholes."""
    spot, strike, rate, volatility, time_to_maturity = _validate_inputs(
        spot, strike, rate, volatility, time_to_maturity
    )
    d1, d2 = _d1_d2(spot, strike, rate, volatility, time_to_maturity)
    price = spot * norm.cdf(d1) - strike * np.exp(-rate * time_to_maturity) * norm.cdf(d2)
    return float(price)


def black_scholes_put(spot: float, strike: float, rate: float, volatility: float, time_to_maturity: float) -> float:
    """Precio de una opcion put europea bajo Black-Scholes."""
    spot, strike, rate, volatility, time_to_maturity = _validate_inputs(
        spot, strike, rate, volatility, time_to_maturity
    )
    d1, d2 = _d1_d2(spot, strike, rate, volatility, time_to_maturity)
    price = strike * np.exp(-rate * time_to_maturity) * norm.cdf(-d2) - spot * norm.cdf(-d1)
    return float(price)


def delta_call(spot: float, strike: float, rate: float, volatility: float, time_to_maturity: float) -> float:
    """Delta de una call europea bajo Black-Scholes."""
    d1, _ = _d1_d2(spot, strike, rate, volatility, time_to_maturity)
    return float(norm.cdf(d1))


def delta_put(spot: float, strike: float, rate: float, volatility: float, time_to_maturity: float) -> float:
    """Delta de una put europea bajo Black-Scholes."""
    d1, _ = _d1_d2(spot, strike, rate, volatility, time_to_maturity)
    return float(norm.cdf(d1) - 1.0)


def gamma(spot: float, strike: float, rate: float, volatility: float, time_to_maturity: float) -> float:
    """Gamma de una opcion europea bajo Black-Scholes."""
    spot, strike, rate, volatility, time_to_maturity = _validate_inputs(
        spot, strike, rate, volatility, time_to_maturity
    )
    d1, _ = _d1_d2(spot, strike, rate, volatility, time_to_maturity)
    return float(norm.pdf(d1) / (spot * volatility * np.sqrt(time_to_maturity)))


def vega(spot: float, strike: float, rate: float, volatility: float, time_to_maturity: float) -> float:
    """Vega de una opcion europea bajo Black-Scholes."""
    spot, strike, rate, volatility, time_to_maturity = _validate_inputs(
        spot, strike, rate, volatility, time_to_maturity
    )
    d1, _ = _d1_d2(spot, strike, rate, volatility, time_to_maturity)
    return float(spot * norm.pdf(d1) * np.sqrt(time_to_maturity))


def theta_call(spot: float, strike: float, rate: float, volatility: float, time_to_maturity: float) -> float:
    """Theta de una call europea bajo Black-Scholes."""
    spot, strike, rate, volatility, time_to_maturity = _validate_inputs(
        spot, strike, rate, volatility, time_to_maturity
    )
    d1, d2 = _d1_d2(spot, strike, rate, volatility, time_to_maturity)
    theta = -(spot * norm.pdf(d1) * volatility) / (2.0 * np.sqrt(time_to_maturity))
    theta -= rate * strike * np.exp(-rate * time_to_maturity) * norm.cdf(d2)
    return float(theta)


def theta_put(spot: float, strike: float, rate: float, volatility: float, time_to_maturity: float) -> float:
    """Theta de una put europea bajo Black-Scholes."""
    spot, strike, rate, volatility, time_to_maturity = _validate_inputs(
        spot, strike, rate, volatility, time_to_maturity
    )
    d1, d2 = _d1_d2(spot, strike, rate, volatility, time_to_maturity)
    theta = -(spot * norm.pdf(d1) * volatility) / (2.0 * np.sqrt(time_to_maturity))
    theta += rate * strike * np.exp(-rate * time_to_maturity) * norm.cdf(-d2)
    return float(theta)


def rho_call(spot: float, strike: float, rate: float, volatility: float, time_to_maturity: float) -> float:
    """Rho de una call europea bajo Black-Scholes."""
    spot, strike, rate, volatility, time_to_maturity = _validate_inputs(
        spot, strike, rate, volatility, time_to_maturity
    )
    _, d2 = _d1_d2(spot, strike, rate, volatility, time_to_maturity)
    return float(strike * time_to_maturity * np.exp(-rate * time_to_maturity) * norm.cdf(d2))


def rho_put(spot: float, strike: float, rate: float, volatility: float, time_to_maturity: float) -> float:
    """Rho de una put europea bajo Black-Scholes."""
    spot, strike, rate, volatility, time_to_maturity = _validate_inputs(
        spot, strike, rate, volatility, time_to_maturity
    )
    _, d2 = _d1_d2(spot, strike, rate, volatility, time_to_maturity)
    return float(-strike * time_to_maturity * np.exp(-rate * time_to_maturity) * norm.cdf(-d2))
