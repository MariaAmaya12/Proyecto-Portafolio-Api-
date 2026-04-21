from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def compute_return_series(price_series: pd.Series) -> pd.DataFrame:
    """
    Devuelve DataFrame con rendimiento simple y logarítmico.
    """
    df = pd.DataFrame(index=price_series.index)
    df["simple_return"] = price_series.pct_change()
    df["log_return"] = np.log(price_series / price_series.shift(1))
    return df.dropna()


def descriptive_stats(returns: pd.Series) -> pd.DataFrame:
    """
    Estadísticos descriptivos principales.
    """
    r = returns.dropna()
    data = {
        "media": r.mean(),
        "mediana": r.median(),
        "std": r.std(ddof=1),
        "min": r.min(),
        "max": r.max(),
        "asimetria": stats.skew(r, bias=False),
        "curtosis": stats.kurtosis(r, fisher=True, bias=False),
    }
    df = pd.DataFrame.from_dict(data, orient="index", columns=["valor"])
    df["valor"] = pd.to_numeric(df["valor"], errors="coerce")
    return df


def normality_tests(returns: pd.Series) -> pd.DataFrame:
    """
    Jarque-Bera y Shapiro-Wilk.
    """
    r = returns.dropna()
    if len(r) < 8:
        return pd.DataFrame(
            {
                "test": ["Jarque-Bera", "Shapiro-Wilk"],
                "estadistico": [np.nan, np.nan],
                "p_value": [np.nan, np.nan],
            }
        )

    jb = stats.jarque_bera(r)
    sample = r if len(r) <= 5000 else r.sample(5000, random_state=42)
    sw = stats.shapiro(sample)

    return pd.DataFrame(
        {
            "test": ["Jarque-Bera", "Shapiro-Wilk"],
            "estadistico": [jb.statistic, sw.statistic],
            "p_value": [jb.pvalue, sw.pvalue],
        }
    )


def qq_plot_data(returns: pd.Series) -> pd.DataFrame:
    """
    Datos para Q-Q plot contra normal.

    La muestra se estandariza para que los cuantiles muestrales y teóricos
    queden en una escala comparable frente a la línea de 45 grados.
    """
    r = returns.dropna()
    sigma = r.std(ddof=1)
    if sigma and not np.isnan(sigma):
        r = (r - r.mean()) / sigma
    osm, osr = stats.probplot(r, dist="norm", fit=False)
    return pd.DataFrame({"theoretical_quantiles": osm, "sample_quantiles": osr})


def stylized_facts_comment(returns: pd.Series) -> str:
    """
    Texto breve para interpretar propiedades empíricas.
    """
    r = returns.dropna()
    if r.empty:
        return "No hay suficientes datos para comentar propiedades empíricas."

    skew = stats.skew(r, bias=False)
    kurt = stats.kurtosis(r, fisher=True, bias=False)

    comments = []
    if abs(skew) > 0.5:
        comments.append("La serie presenta asimetría apreciable.")
    else:
        comments.append("La asimetría es moderada o baja.")

    if kurt > 3:
        comments.append("Hay evidencia de colas pesadas respecto a una normal.")
    else:
        comments.append("La curtosis no sugiere colas extremadamente pesadas.")

    comments.append(
        "En activos financieros es común observar clustering de volatilidad; "
        "esto se evalúa mejor con modelos ARCH/GARCH."
    )
    return " ".join(comments)
