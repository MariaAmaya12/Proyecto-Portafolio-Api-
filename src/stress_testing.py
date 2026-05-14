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


def apply_rate_shock_to_bond(
    price: float,
    modified_duration: float,
    convexity: float | None = None,
    rate_shock: float = 0.01,
) -> float:
    """Estima el precio de un bono ante un shock de tasa usando duracion y convexidad."""
    if not np.isscalar(price) or not np.isfinite(price) or price <= 0:
        raise ValueError("price debe ser mayor que 0.")
    if (
        not np.isscalar(modified_duration)
        or not np.isfinite(modified_duration)
        or modified_duration < 0
    ):
        raise ValueError("modified_duration debe ser mayor o igual que 0.")
    if not np.isscalar(rate_shock) or not np.isfinite(rate_shock):
        raise ValueError("rate_shock debe ser numerico y finito.")
    if convexity is not None and (
        not np.isscalar(convexity) or not np.isfinite(convexity) or convexity < 0
    ):
        raise ValueError("convexity debe ser mayor o igual que 0.")

    price_change_pct = -float(modified_duration) * float(rate_shock)
    if convexity is not None:
        price_change_pct += 0.5 * float(convexity) * float(rate_shock) ** 2
    return float(price * (1.0 + price_change_pct))


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


def _apply_column_price_shocks(prices, price_shocks: dict):
    clean = _to_series_or_frame(prices, "prices")
    if not isinstance(clean, pd.DataFrame):
        raise ValueError("price_shocks como diccionario requiere prices como DataFrame.")

    unknown = set(price_shocks) - set(clean.columns)
    if unknown:
        raise ValueError("price_shocks contiene columnas que no existen en prices.")

    shocked = clean.copy()
    for column, shock in price_shocks.items():
        shocked[column] = apply_price_shock(clean[column], shock)
    return shocked


def _portfolio_from_returns(returns, weights):
    clean = _to_series_or_frame(returns, "returns")
    if isinstance(clean, pd.DataFrame):
        w = _validate_weights(weights, clean.shape[1])
        return pd.Series(clean.to_numpy(dtype=float) @ w, index=clean.index, name="portfolio_return")

    if weights is None:
        return clean
    if not np.isscalar(weights) or not np.isfinite(weights) or not np.isclose(float(weights), 1.0):
        raise ValueError("weights debe ser 1 para una Series.")
    return clean * float(weights)


def combined_adverse_scenario(
    prices=None,
    returns=None,
    weights=None,
    price_shocks=None,
    volatility_multiplier=None,
    bond_price=None,
    modified_duration=None,
    convexity=None,
    rate_shock=None,
) -> dict:
    """Aplica un escenario adverso combinado de precios, volatilidad y tasas."""
    scenario = {}

    if prices is not None and price_shocks is not None:
        if isinstance(price_shocks, dict):
            scenario["shocked_prices"] = _apply_column_price_shocks(prices, price_shocks)
        else:
            scenario["shocked_prices"] = apply_price_shock(prices, price_shocks)

    shocked_returns = None
    if returns is not None and volatility_multiplier is not None:
        shocked_returns = apply_volatility_shock(returns, volatility_multiplier)
        scenario["shocked_returns"] = shocked_returns

    if bond_price is not None or modified_duration is not None or rate_shock is not None:
        if bond_price is None or modified_duration is None or rate_shock is None:
            raise ValueError("bond_price, modified_duration y rate_shock deben enviarse juntos.")
        scenario["shocked_bond_price"] = apply_rate_shock_to_bond(
            bond_price,
            modified_duration,
            convexity=convexity,
            rate_shock=rate_shock,
        )

    returns_for_portfolio = shocked_returns if shocked_returns is not None else returns
    if returns_for_portfolio is not None and weights is not None:
        if isinstance(returns_for_portfolio, pd.DataFrame) and isinstance(price_shocks, dict):
            portfolio = stress_portfolio_returns(returns_for_portfolio, weights, price_shocks)
        else:
            portfolio = _portfolio_from_returns(returns_for_portfolio, weights)
        scenario["stressed_portfolio_returns"] = portfolio
        scenario["scenario_summary"] = summarize_stress_scenario(portfolio)
    elif shocked_returns is not None and isinstance(shocked_returns, pd.Series):
        scenario["scenario_summary"] = summarize_stress_scenario(shocked_returns)

    if not scenario:
        raise ValueError("Debe proporcionarse al menos un componente del escenario.")

    return scenario
