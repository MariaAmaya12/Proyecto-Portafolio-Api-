from __future__ import annotations

import numpy as np


def _validate_bond_inputs(
    face_value: float,
    coupon_rate: float,
    market_rate: float,
    maturity_years: float,
    frequency: int,
) -> tuple[float, float, float, float, int]:
    if face_value <= 0:
        raise ValueError("face_value debe ser mayor que 0.")
    if coupon_rate < 0 or market_rate < 0:
        raise ValueError("Las tasas no pueden ser negativas.")
    if maturity_years <= 0:
        raise ValueError("maturity_years debe ser mayor que 0.")
    if frequency <= 0:
        raise ValueError("frequency debe ser mayor que 0.")

    periods = maturity_years * frequency
    if not np.isclose(periods, round(periods)):
        raise ValueError("maturity_years * frequency debe producir un numero entero de periodos.")

    return (
        float(face_value),
        float(coupon_rate),
        float(market_rate),
        float(maturity_years),
        int(frequency),
    )


def _cash_flows(face_value: float, coupon_rate: float, maturity_years: float, frequency: int) -> np.ndarray:
    periods = int(round(maturity_years * frequency))
    coupon = face_value * coupon_rate / frequency
    flows = np.full(periods, coupon, dtype=float)
    flows[-1] += face_value
    return flows


def bond_price(
    face_value: float,
    coupon_rate: float,
    market_rate: float,
    maturity_years: float,
    frequency: int = 1,
) -> float:
    """Calcula el precio presente de un bono con cupones periodicos."""
    face_value, coupon_rate, market_rate, maturity_years, frequency = _validate_bond_inputs(
        face_value, coupon_rate, market_rate, maturity_years, frequency
    )
    flows = _cash_flows(face_value, coupon_rate, maturity_years, frequency)
    period_rate = market_rate / frequency
    periods = np.arange(1, len(flows) + 1)
    price = np.sum(flows / (1.0 + period_rate) ** periods)
    return float(price)


def macaulay_duration(
    face_value: float,
    coupon_rate: float,
    market_rate: float,
    maturity_years: float,
    frequency: int = 1,
) -> float:
    """Calcula la duracion de Macaulay de un bono en anos."""
    face_value, coupon_rate, market_rate, maturity_years, frequency = _validate_bond_inputs(
        face_value, coupon_rate, market_rate, maturity_years, frequency
    )
    flows = _cash_flows(face_value, coupon_rate, maturity_years, frequency)
    period_rate = market_rate / frequency
    periods = np.arange(1, len(flows) + 1)
    discounted = flows / (1.0 + period_rate) ** periods
    price = np.sum(discounted)
    duration = np.sum((periods / frequency) * discounted) / price
    return float(duration)


def modified_duration(
    face_value: float,
    coupon_rate: float,
    market_rate: float,
    maturity_years: float,
    frequency: int = 1,
) -> float:
    """Calcula la duracion modificada de un bono."""
    duration = macaulay_duration(face_value, coupon_rate, market_rate, maturity_years, frequency)
    return float(duration / (1.0 + market_rate / frequency))


def convexity(
    face_value: float,
    coupon_rate: float,
    market_rate: float,
    maturity_years: float,
    frequency: int = 1,
) -> float:
    """Calcula la convexidad discreta de un bono."""
    face_value, coupon_rate, market_rate, maturity_years, frequency = _validate_bond_inputs(
        face_value, coupon_rate, market_rate, maturity_years, frequency
    )
    flows = _cash_flows(face_value, coupon_rate, maturity_years, frequency)
    period_rate = market_rate / frequency
    periods = np.arange(1, len(flows) + 1)
    price = np.sum(flows / (1.0 + period_rate) ** periods)
    convex = np.sum(flows * periods * (periods + 1) / (1.0 + period_rate) ** (periods + 2))
    convex /= price * frequency**2
    return float(convex)


def nelson_siegel_yield(maturity, beta0: float, beta1: float, beta2: float, tau: float):
    """Calcula la tasa Nelson-Siegel para uno o varios vencimientos."""
    if tau <= 0:
        raise ValueError("tau debe ser mayor que 0.")

    maturities = np.asarray(maturity, dtype=float)
    if np.any(maturities <= 0) or np.any(~np.isfinite(maturities)):
        raise ValueError("maturity debe contener valores positivos y finitos.")

    scaled = maturities / tau
    factor1 = (1.0 - np.exp(-scaled)) / scaled
    yields = beta0 + beta1 * factor1 + beta2 * (factor1 - np.exp(-scaled))
    if np.isscalar(maturity):
        return float(yields)
    return yields.astype(float)
