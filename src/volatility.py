from __future__ import annotations

import numpy as np
import pandas as pd


def _clean_returns(returns) -> np.ndarray:
    values = pd.Series(returns, dtype="float64").replace([np.inf, -np.inf], np.nan).dropna()
    if len(values) < 2:
        raise ValueError("Se requieren al menos 2 observaciones validas.")
    return values.to_numpy(dtype=float)


def ewma_variance(returns, lambda_: float = 0.94) -> float:
    """Calcula la varianza EWMA de una serie de rendimientos."""
    if not np.isscalar(lambda_) or not 0 < float(lambda_) < 1:
        raise ValueError("lambda_ debe estar entre 0 y 1.")

    clean = _clean_returns(returns)
    weights = (1.0 - lambda_) * lambda_ ** np.arange(len(clean) - 1, -1, -1)
    variance = np.sum(weights * clean**2) / np.sum(weights)
    return float(variance)


def ewma_volatility(
    returns,
    lambda_: float = 0.94,
    annualize: bool = False,
    periods_per_year: int = 252,
) -> float:
    """Calcula la volatilidad EWMA, opcionalmente anualizada."""
    if periods_per_year <= 0:
        raise ValueError("periods_per_year debe ser mayor que 0.")

    volatility = float(np.sqrt(ewma_variance(returns, lambda_=lambda_)))
    if annualize:
        volatility *= float(np.sqrt(periods_per_year))
    return float(volatility)
