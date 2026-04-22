from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import TRADING_DAYS


def clean_price_frame(df: pd.DataFrame) -> pd.DataFrame:
    """
    Limpia precios:
    - elimina duplicados de índice,
    - ordena por fecha,
    - elimina filas completamente vacías,
    - convierte valores a numéricos,
    - reemplaza infinitos por NaN.
    """
    if df.empty:
        return df

    out = df.copy()
    out.index = pd.to_datetime(out.index)
    out = out[~out.index.duplicated(keep="last")]
    out = out.sort_index()
    out = out.apply(pd.to_numeric, errors="coerce")
    out = out.replace([np.inf, -np.inf], np.nan)
    out = out.dropna(how="all")
    return out


def clean_price_series(price_series: pd.Series) -> pd.Series:
    """
    Limpia una serie de precios:
    - convierte a numérico,
    - elimina NaN e infinitos,
    - conserva solo valores positivos.
    """
    if price_series is None or price_series.empty:
        return pd.Series(dtype=float)

    out = pd.to_numeric(price_series, errors="coerce")
    out = out.replace([np.inf, -np.inf], np.nan).dropna()
    out = out[out > 0]
    return out


def align_close_prices(close: pd.DataFrame, dropna: bool = True) -> pd.DataFrame:
    """
    Alinea la matriz de cierres.
    """
    if close.empty:
        return close

    out = clean_price_frame(close)
    if dropna:
        out = out.dropna()
    return out


def simple_returns(price_series: pd.Series) -> pd.Series:
    """
    Rendimiento simple diario.
    """
    clean = clean_price_series(price_series)
    out = clean.pct_change().dropna()
    out = out.replace([np.inf, -np.inf], np.nan).dropna()
    return out


def log_returns(price_series: pd.Series) -> pd.Series:
    """
    Rendimiento logarítmico diario.
    """
    clean = clean_price_series(price_series)
    out = np.log(clean / clean.shift(1)).dropna()
    out = out.replace([np.inf, -np.inf], np.nan).dropna()
    return out


def validate_min_sample(series: pd.Series, min_obs: int = 30) -> tuple[bool, str]:
    """
    Valida si una serie tiene observaciones suficientes para análisis estadístico.
    """
    if series is None or len(series) == 0:
        return False, "La serie no contiene observaciones válidas."
    if len(series.dropna()) < min_obs:
        return (
            False,
            f"La serie tiene {len(series.dropna())} observaciones; se requieren al menos {min_obs}.",
        )
    return True, ""


def equal_weight_vector(n_assets: int) -> np.ndarray:
    """
    Genera pesos iguales para n activos.
    """
    if n_assets <= 0:
        return np.array([])
    return np.repeat(1 / n_assets, n_assets)


def equal_weight_portfolio(returns: pd.DataFrame) -> pd.Series:
    """
    Retorno de portafolio equiponderado.
    """
    if returns is None or returns.empty:
        return pd.Series(dtype=float)

    numeric_returns = returns.apply(pd.to_numeric, errors="coerce")
    clean = numeric_returns.replace([np.inf, -np.inf], np.nan).dropna(how="all", axis=1)
    if clean.empty or clean.shape[1] == 0:
        return pd.Series(dtype=float)

    weights = pd.Series(1.0 / clean.shape[1], index=clean.columns)
    port = clean.mul(weights, axis=1).sum(axis=1, min_count=1)
    port.name = "portfolio_equal_weight"
    return port.dropna()


def annualize_return(daily_returns: pd.Series) -> float:
    """
    Anualiza el retorno medio diario.
    """
    if daily_returns.empty:
        return float("nan")
    return float((1 + daily_returns.mean()) ** TRADING_DAYS - 1)


def annualize_volatility(daily_returns: pd.Series) -> float:
    """
    Anualiza la volatilidad diaria.
    """
    if daily_returns.empty:
        return float("nan")
    return float(daily_returns.std(ddof=1) * np.sqrt(TRADING_DAYS))


def base_100(series: pd.Series) -> pd.Series:
    """
    Convierte una serie a base 100.
    """
    if series.empty:
        return series

    clean = series.dropna()
    if clean.empty:
        return series

    return 100 * clean / clean.iloc[0]
