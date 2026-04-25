from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd
from scipy.stats import chi2, gaussian_kde, norm

from src.config import TRADING_DAYS


def annual_to_daily_rf(rf_annual: float, periods: int = TRADING_DAYS) -> float:
    """
    Convierte tasa libre de riesgo anual en decimal a tasa diaria equivalente.
    """
    if rf_annual is None or pd.isna(rf_annual):
        raise ValueError("La tasa libre de riesgo anual no es válida.")
    if rf_annual <= -1:
        raise ValueError("La tasa libre de riesgo anual debe ser mayor que -100%.")
    return (1 + rf_annual) ** (1 / periods) - 1


def validate_confidence_level(alpha: float) -> None:
    """
    Valida el nivel de confianza para VaR/CVaR.
    """
    if alpha is None or pd.isna(alpha):
        raise ValueError("El nivel de confianza no es válido.")
    if not (0 < alpha < 1):
        raise ValueError("El nivel de confianza debe estar entre 0 y 1.")


def validate_returns_series(returns: pd.Series, min_obs: int = 30) -> pd.Series:
    """
    Limpia y valida una serie de rendimientos.
    """
    if returns is None:
        raise ValueError("La serie de rendimientos es None.")

    clean = pd.to_numeric(returns, errors="coerce")
    clean = clean.replace([np.inf, -np.inf], np.nan).dropna()

    if len(clean) < min_obs:
        raise ValueError(
            f"La serie de rendimientos tiene {len(clean)} observaciones; "
            f"se requieren al menos {min_obs}."
        )

    if np.isclose(clean.std(ddof=1), 0):
        raise ValueError("La serie de rendimientos tiene varianza nula o casi nula.")

    return clean


def validar_serie_para_garch(
    returns: pd.Series,
    min_obs: int = 120,
    max_null_ratio: float = 0.05,
) -> Dict[str, Any]:
    """
    Valida si una serie de rendimientos es apta para ajustar un modelo GARCH.

    Retorna un diccionario con:
    - ok: bool
    - serie_limpia: pd.Series
    - errores: list[str]
    - advertencias: list[str]
    - resumen: dict
    """
    errores: List[str] = []
    advertencias: List[str] = []

    if returns is None:
        return {
            "ok": False,
            "serie_limpia": pd.Series(dtype=float),
            "errores": ["La serie de rendimientos no existe."],
            "advertencias": [],
            "resumen": {},
        }

    if not isinstance(returns, pd.Series):
        try:
            returns = pd.Series(returns)
        except Exception:
            return {
                "ok": False,
                "serie_limpia": pd.Series(dtype=float),
                "errores": ["No fue posible convertir la entrada a una serie de pandas."],
                "advertencias": [],
                "resumen": {},
            }

    n_original = len(returns)

    if n_original == 0:
        return {
            "ok": False,
            "serie_limpia": pd.Series(dtype=float),
            "errores": ["La serie de rendimientos está vacía."],
            "advertencias": [],
            "resumen": {},
        }

    if not isinstance(returns.index, pd.DatetimeIndex):
        advertencias.append(
            "La serie no tiene índice temporal tipo DatetimeIndex. "
            "El ajuste puede ejecutarse, pero pierdes trazabilidad cronológica."
        )

    serie = pd.to_numeric(returns, errors="coerce").replace([np.inf, -np.inf], np.nan)

    null_ratio = float(serie.isna().mean())
    if null_ratio > max_null_ratio:
        advertencias.append(
            f"La serie tiene {null_ratio:.1%} de valores faltantes o no válidos antes de limpiar."
        )

    serie = serie.dropna()
    n_limpio = len(serie)

    if n_limpio < min_obs:
        errores.append(
            f"La serie limpia tiene solo {n_limpio} observaciones; "
            f"se requieren al menos {min_obs} para ajustar GARCH con mayor estabilidad."
        )

    if n_limpio < 30:
        errores.append(
            "La muestra es demasiado corta para modelar volatilidad condicional de forma defendible."
        )

    if serie.empty:
        errores.append("Después de limpiar nulos e infinitos, la serie quedó vacía.")

    std = float(serie.std(ddof=1)) if n_limpio > 1 else 0.0
    if np.isclose(std, 0.0):
        errores.append(
            "La desviación estándar de la serie es cero o casi cero; no hay variabilidad suficiente para GARCH."
        )

    zero_ratio = float((np.isclose(serie, 0.0)).mean()) if n_limpio > 0 else 1.0
    if zero_ratio > 0.80:
        advertencias.append(
            f"El {zero_ratio:.1%} de la serie son ceros o casi ceros. "
            "Esto puede afectar la estabilidad del ajuste."
        )

    if isinstance(serie.index, pd.DatetimeIndex):
        if not serie.index.is_monotonic_increasing:
            advertencias.append(
                "La serie temporal no está ordenada cronológicamente. "
                "Se ordenará antes del ajuste."
            )
            serie = serie.sort_index()

        if serie.index.has_duplicates:
            advertencias.append(
                "La serie contiene fechas duplicadas. "
                "Revisa si hubo problemas al descargar o combinar datos."
            )

    resumen = {
        "n_original": n_original,
        "n_limpio": n_limpio,
        "null_ratio": null_ratio,
        "zero_ratio": zero_ratio,
        "std": std,
        "mean": float(serie.mean()) if n_limpio > 0 else np.nan,
    }

    return {
        "ok": len(errores) == 0,
        "serie_limpia": serie,
        "errores": errores,
        "advertencias": advertencias,
        "resumen": resumen,
    }


def validate_weights(weights: np.ndarray, n_assets: int | None = None) -> np.ndarray:
    """
    Valida pesos de portafolio.
    """
    w = np.asarray(weights, dtype=float)

    if w.ndim != 1:
        raise ValueError("Los pesos deben ser un vector unidimensional.")

    if len(w) == 0:
        raise ValueError("El vector de pesos está vacío.")

    if np.any(np.isnan(w)) or np.any(np.isinf(w)):
        raise ValueError("Los pesos contienen NaN o infinitos.")

    if n_assets is not None and len(w) != n_assets:
        raise ValueError(
            f"El número de pesos ({len(w)}) no coincide con el número de activos ({n_assets})."
        )

    if not np.isclose(w.sum(), 1.0, atol=1e-6):
        raise ValueError("Los pesos del portafolio deben sumar 1.")

    return w


def parametric_var_cvar(returns: pd.Series, alpha: float = 0.95) -> dict:
    """
    VaR y CVaR paramétricos bajo supuesto de normalidad.

    Convención:
    - Se define la pérdida como L = -R
    - VaR y CVaR se reportan como pérdidas positivas
    """
    validate_confidence_level(alpha)

    try:
        r = validate_returns_series(returns)
    except ValueError:
        return {}

    mu = r.mean()
    sigma = r.std(ddof=1)
    q = 1 - alpha

    if sigma == 0 or np.isnan(sigma):
        loss_level = max(0.0, -mu)
        return {
            "VaR_diario": float(loss_level),
            "CVaR_diario": float(loss_level),
            "VaR_anualizado": float(loss_level * np.sqrt(TRADING_DAYS)),
            "CVaR_anualizado": float(loss_level * np.sqrt(TRADING_DAYS)),
        }

    z_q = norm.ppf(q)

    # Cuantil del rendimiento y conversión a pérdida usando L = -R
    return_quantile = mu + sigma * z_q
    var_daily = max(0.0, -return_quantile)

    # CVaR paramétrico para pérdidas bajo normalidad
    cvar_daily = max(0.0, -mu + sigma * (norm.pdf(z_q) / q))

    # Coherencia mínima: CVaR no debe ser menor que VaR
    cvar_daily = max(var_daily, cvar_daily)

    return {
        "VaR_diario": float(var_daily),
        "CVaR_diario": float(cvar_daily),
        "VaR_anualizado": float(var_daily * np.sqrt(TRADING_DAYS)),
        "CVaR_anualizado": float(cvar_daily * np.sqrt(TRADING_DAYS)),
    }


def historical_var_cvar(returns: pd.Series, alpha: float = 0.95) -> dict:
    """
    VaR y CVaR históricos.

    Convención:
    - Se define la pérdida como L = -R
    - VaR y CVaR se reportan como pérdidas positivas
    """
    validate_confidence_level(alpha)

    try:
        r = validate_returns_series(returns)
    except ValueError:
        return {}

    q = 1 - alpha

    # Cuantil izquierdo de los rendimientos
    return_cutoff = np.quantile(r, q)

    # Cola de pérdidas: rendimientos peores o iguales al cuantil
    tail_returns = r[r <= return_cutoff]

    # Conversión a pérdidas usando L = -R
    var_daily = max(0.0, -return_cutoff)

    if len(tail_returns) > 0:
        cvar_daily = max(0.0, -tail_returns.mean())
    else:
        cvar_daily = var_daily

    # Coherencia mínima: CVaR no debe ser menor que VaR
    cvar_daily = max(var_daily, cvar_daily)

    return {
        "VaR_diario": float(var_daily),
        "CVaR_diario": float(cvar_daily),
        "VaR_anualizado": float(var_daily * np.sqrt(TRADING_DAYS)),
        "CVaR_anualizado": float(cvar_daily * np.sqrt(TRADING_DAYS)),
    }


def monte_carlo_var_cvar(
    returns_df: pd.DataFrame,
    weights: np.ndarray,
    alpha: float = 0.95,
    n_sim: int = 10000,
    seed: int = 42,
) -> dict:
    """
    VaR y CVaR de Monte Carlo usando normal multivariada.

    Convención:
    - Se define la pérdida como L = -R_p
    - VaR y CVaR se reportan como pérdidas positivas
    """
    validate_confidence_level(alpha)

    if n_sim < 1000:
        raise ValueError("El número de simulaciones debe ser al menos 1000.")

    clean = returns_df.copy()
    clean = clean.apply(pd.to_numeric, errors="coerce")
    clean = clean.replace([np.inf, -np.inf], np.nan).dropna()

    if clean.empty or len(clean) < 30:
        return {}

    try:
        w = validate_weights(weights, n_assets=clean.shape[1])
    except ValueError:
        return {}

    mu = clean.mean().values
    cov = clean.cov().values

    if np.any(np.isnan(cov)) or np.any(np.isinf(cov)):
        return {}

    rng = np.random.default_rng(seed)
    sims = rng.multivariate_normal(mu, cov, size=n_sim)

    # Rendimientos simulados del portafolio
    port_sim = sims @ w

    q = 1 - alpha

    # Cuantil de los retornos simulados
    return_cutoff = np.quantile(port_sim, q)

    # Cola de pérdidas (peores escenarios)
    tail_returns = port_sim[port_sim <= return_cutoff]

    # Conversión a pérdidas: L = -R
    var_daily = max(0.0, -return_cutoff)

    if len(tail_returns) > 0:
        cvar_daily = max(0.0, -tail_returns.mean())
    else:
        cvar_daily = var_daily

    # Coherencia mínima
    cvar_daily = max(var_daily, cvar_daily)

    return {
        "VaR_diario": float(var_daily),
        "CVaR_diario": float(cvar_daily),
        "VaR_anualizado": float(var_daily * np.sqrt(TRADING_DAYS)),
        "CVaR_anualizado": float(cvar_daily * np.sqrt(TRADING_DAYS)),
        "simulated_returns": port_sim,
    }


def monte_carlo_kde_var_cvar(
    returns_df: pd.DataFrame,
    weights: np.ndarray,
    alpha: float = 0.95,
    n_sim: int = 10000,
    seed: int = 42,
    min_obs: int = 30,
) -> dict:
    """
    VaR y CVaR de Monte Carlo con KDE univariada sobre el portafolio.

    Convencion:
    - Se construye R_p = returns @ weights.
    - La KDE gaussiana se ajusta sobre los rendimientos historicos de R_p.
    - VaR y CVaR se calculan sobre perdidas positivas L = -R_p simulado.
    """
    validate_confidence_level(alpha)

    if n_sim < 1000:
        raise ValueError("El numero de simulaciones debe ser al menos 1000.")

    if returns_df is None or returns_df.empty:
        return {}

    clean = returns_df.copy()
    clean = clean.apply(pd.to_numeric, errors="coerce")
    clean = clean.replace([np.inf, -np.inf], np.nan).dropna()

    if clean.empty or len(clean) < min_obs:
        return {}

    try:
        w = validate_weights(weights, n_assets=clean.shape[1])
    except ValueError:
        return {}

    portfolio_returns = clean.to_numpy(dtype=float) @ w
    portfolio_returns = pd.Series(portfolio_returns, index=clean.index)

    try:
        r = validate_returns_series(portfolio_returns, min_obs=min_obs)
    except ValueError:
        return {}

    try:
        kde = gaussian_kde(r.to_numpy(dtype=float), bw_method="scott")
    except (ValueError, np.linalg.LinAlgError):
        return {}

    rng = np.random.default_rng(seed)
    simulated_returns = kde.resample(size=n_sim, seed=rng).reshape(-1)
    simulated_returns = simulated_returns[np.isfinite(simulated_returns)]

    if len(simulated_returns) < min_obs:
        return {}

    losses = -simulated_returns
    var_daily = max(0.0, float(np.quantile(losses, alpha)))
    tail_losses = losses[losses >= var_daily]

    if len(tail_losses) > 0:
        cvar_daily = max(0.0, float(tail_losses.mean()))
    else:
        cvar_daily = var_daily

    cvar_daily = max(var_daily, cvar_daily)

    return {
        "VaR_diario": float(var_daily),
        "CVaR_diario": float(cvar_daily),
        "VaR_anualizado": float(var_daily * np.sqrt(TRADING_DAYS)),
        "CVaR_anualizado": float(cvar_daily * np.sqrt(TRADING_DAYS)),
    }


def risk_comparison_table(
    portfolio_returns: pd.Series,
    asset_returns_df: pd.DataFrame,
    weights: np.ndarray,
    alpha: float = 0.95,
    n_sim: int = 10000,
) -> pd.DataFrame:
    """
    Tabla comparativa de VaR y CVaR.
    """
    p = parametric_var_cvar(portfolio_returns, alpha=alpha)
    h = historical_var_cvar(portfolio_returns, alpha=alpha)
    m = monte_carlo_var_cvar(
        asset_returns_df,
        weights=weights,
        alpha=alpha,
        n_sim=n_sim,
    )
    mk = monte_carlo_kde_var_cvar(
        asset_returns_df,
        weights=weights,
        alpha=alpha,
        n_sim=n_sim,
    )

    rows = []
    for name, res in [
        ("Paramétrico", p),
        ("Histórico", h),
        ("Monte Carlo", m),
        ("Monte Carlo KDE", mk),
    ]:
        if res:
            rows.append(
                {
                    "método": name,
                    "VaR_diario": res["VaR_diario"],
                    "CVaR_diario": res["CVaR_diario"],
                    "VaR_anualizado": res["VaR_anualizado"],
                    "CVaR_anualizado": res["CVaR_anualizado"],
                }
            )

    return pd.DataFrame(rows)


def kupiec_test(returns: pd.Series, var: float, alpha: float = 0.95) -> dict:
    """
    Test de Kupiec (Proportion of Failures - POF).

    Evalúa si el número de violaciones del VaR es consistente
    con el nivel de confianza esperado.

    - returns: serie de rendimientos
    - var: VaR como pérdida positiva (ej: 0.03)
    - alpha: nivel de confianza (ej: 0.95)
    """
    validate_confidence_level(alpha)

    try:
        r = validate_returns_series(returns)
    except ValueError:
        return {}

    violations = r < -var

    n = len(r)
    x = int(violations.sum())

    if n == 0:
        return {}

    p = 1 - alpha
    p_hat = x / n

    if p_hat == 0 or p_hat == 1:
        return {
            "violations": x,
            "n_obs": n,
            "expected_fail_rate": float(p),
            "observed_fail_rate": float(p_hat),
            "lr_stat": np.nan,
            "p_value": np.nan,
            "conclusion": "Insuficiente variabilidad para test",
        }

    lr = -2 * (
        (n - x) * np.log((1 - p) / (1 - p_hat)) +
        x * np.log(p / p_hat)
    )

    p_value = 1 - chi2.cdf(lr, df=1)

    conclusion = (
        "No se rechaza el modelo (VaR consistente)"
        if p_value > 0.05
        else "Se rechaza el modelo (VaR inconsistente)"
    )

    return {
        "violations": x,
        "n_obs": n,
        "expected_fail_rate": float(p),
        "observed_fail_rate": float(p_hat),
        "lr_stat": float(lr),
        "p_value": float(p_value),
        "conclusion": conclusion,
    }
