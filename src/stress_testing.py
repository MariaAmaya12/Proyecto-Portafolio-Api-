from __future__ import annotations

import numpy as np
import pandas as pd


def _to_series_or_frame(data, name: str):
    if isinstance(data, (pd.Series, pd.DataFrame)):
        clean = data.copy()
    else:
        clean = pd.Series(data, dtype="float64")

    clean = clean.apply(pd.to_numeric, errors="coerce") if isinstance(clean, pd.DataFrame) else pd.to_numeric(clean, errors="coerce")
    clean = clean.replace([np.inf, -np.inf], np.nan).dropna()
    if clean.empty:
        raise ValueError(f"{name} no contiene datos validos.")
    return clean


def apply_price_shock(prices, shock_pct: float):
    """Aplica un shock porcentual a precios en Series o DataFrame."""
    if not np.isscalar(shock_pct) or not np.isfinite(shock_pct):
        raise ValueError("shock_pct debe ser numerico y finito.")
    clean = _to_series_or_frame(prices, "prices")
    return clean * (1.0 + float(shock_pct))


def apply_volatility_shock(returns, multiplier: float):
    """Escala rendimientos por un multiplicador de volatilidad."""
    if not np.isscalar(multiplier) or not np.isfinite(multiplier) or multiplier <= 0:
        raise ValueError("multiplier debe ser mayor que 0.")
    clean = _to_series_or_frame(returns, "returns")
    return clean * float(multiplier)


def _validate_weights(weights, n_assets: int) -> np.ndarray:
    values = np.asarray(weights, dtype=float)
    if values.ndim != 1 or len(values) != n_assets:
        raise ValueError("weights debe ser un vector con un peso por activo.")
    if np.any(~np.isfinite(values)):
        raise ValueError("weights contiene valores no validos.")
    if not np.isclose(values.sum(), 1.0, atol=1e-6):
        raise ValueError("weights debe sumar aproximadamente 1.")
    return values


def _shock_frame(returns: pd.DataFrame, shocks: dict) -> pd.DataFrame:
    if not isinstance(shocks, dict):
        raise ValueError("shocks debe ser un diccionario para DataFrame.")
    unknown = set(shocks) - set(returns.columns)
    if unknown:
        raise ValueError("shocks contiene columnas que no existen en returns.")
    shocked = returns.copy()
    for column, shock in shocks.items():
        if not np.isscalar(shock) or not np.isfinite(shock):
            raise ValueError("Cada shock debe ser numerico y finito.")
        shocked[column] = shocked[column] + float(shock)
    return shocked


def stress_portfolio_returns(returns, weights, shocks):
    """Calcula rendimientos de portafolio despues de aplicar shocks por activo."""
    clean = _to_series_or_frame(returns, "returns")

    if isinstance(clean, pd.DataFrame):
        w = _validate_weights(weights, clean.shape[1])
        shocked = _shock_frame(clean, shocks)
        portfolio = shocked.to_numpy(dtype=float) @ w
        return pd.Series(portfolio, index=shocked.index, name="portfolio_return")

    if not np.isscalar(weights) or not np.isfinite(weights):
        raise ValueError("weights debe ser numerico para una Series.")
    if not np.isclose(float(weights), 1.0, atol=1e-6):
        raise ValueError("weights debe ser 1 para una Series.")
    if not np.isscalar(shocks) or not np.isfinite(shocks):
        raise ValueError("shocks debe ser numerico para una Series.")
    return clean + float(shocks)


def summarize_stress_scenario(portfolio_returns, confidence_level: float = 0.95) -> dict:
    """Resume un escenario estresado con retorno medio, VaR, CVaR y drawdown."""
    if not 0 < float(confidence_level) < 1:
        raise ValueError("confidence_level debe estar entre 0 y 1.")

    returns = _to_series_or_frame(portfolio_returns, "portfolio_returns")
    if isinstance(returns, pd.DataFrame):
        if returns.shape[1] != 1:
            raise ValueError("portfolio_returns debe ser Series o DataFrame de una columna.")
        returns = returns.iloc[:, 0]

    cumulative = (1.0 + returns).cumprod()
    drawdown = cumulative / cumulative.cummax() - 1.0
    q = 1.0 - float(confidence_level)
    return_cutoff = float(np.quantile(returns, q))
    tail_returns = returns[returns <= return_cutoff]
    var = max(0.0, -return_cutoff)
    cvar = max(var, max(0.0, float(-tail_returns.mean()))) if not tail_returns.empty else var

    return {
        "mean_return": float(returns.mean()),
        "min_return": float(returns.min()),
        "max_drawdown": float(drawdown.min()),
        "var": float(var),
        "cvar": float(cvar),
    }
