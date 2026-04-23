from __future__ import annotations

import numpy as np
import pandas as pd

from src.preprocess import equal_weight_portfolio, equal_weight_vector
from src.risk_metrics import risk_comparison_table


class RiskAnalyzer:
    """Utilidades de limpieza y calculo para modulos de riesgo."""

    @staticmethod
    def clean_returns(returns: pd.DataFrame, drop_how: str = "any") -> pd.DataFrame:
        """Limpia infinitos, convierte a numerico y descarta filas incompletas."""
        if returns is None or returns.empty:
            return pd.DataFrame()
        cleaned = returns.replace([np.inf, -np.inf], np.nan).apply(pd.to_numeric, errors="coerce")
        return cleaned.dropna(how=drop_how)

    @staticmethod
    def validate_sample(returns: pd.DataFrame, min_rows: int = 30, min_cols: int = 1) -> bool:
        """Valida tamano minimo de muestra."""
        return returns is not None and not returns.empty and returns.shape[0] >= min_rows and returns.shape[1] >= min_cols

    @staticmethod
    def portfolio_returns(returns: pd.DataFrame, weights: np.ndarray | None = None) -> tuple[pd.Series, np.ndarray]:
        """Construye retornos de portafolio equiponderado o ponderado."""
        if weights is None:
            weights = equal_weight_vector(returns.shape[1])
            return equal_weight_portfolio(returns), weights
        return returns.mul(weights, axis=1).sum(axis=1), weights

    @staticmethod
    def compute_var_tables(
        portfolio_returns: pd.Series,
        asset_returns_df: pd.DataFrame,
        weights: np.ndarray,
        confidence_levels: list[float],
        n_sim: int,
    ) -> dict[float, pd.DataFrame]:
        """Calcula tablas VaR/CVaR para varios niveles de confianza."""
        tables: dict[float, pd.DataFrame] = {}
        for level in confidence_levels:
            table = risk_comparison_table(
                portfolio_returns=portfolio_returns,
                asset_returns_df=asset_returns_df,
                weights=weights,
                alpha=level,
                n_sim=n_sim,
            )
            if not table.empty:
                table = table.copy()
                table.insert(0, "confianza", level)
            tables[level] = table
        return tables

