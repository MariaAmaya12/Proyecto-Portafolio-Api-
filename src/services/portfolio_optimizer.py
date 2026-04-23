from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.config import TRADING_DAYS
from src.markowitz import (
    efficient_frontier,
    maximum_sharpe_portfolio,
    minimum_variance_portfolio,
    simulate_portfolios,
    weights_table,
)


class PortfolioOptimizer:
    """Fachada de negocio para preparacion y optimizacion Markowitz."""

    @staticmethod
    def prepare_returns(returns: pd.DataFrame) -> pd.DataFrame:
        """Limpia retornos para optimizacion media-varianza."""
        if returns is None or returns.empty:
            return pd.DataFrame()
        return returns.replace([np.inf, -np.inf], np.nan).apply(pd.to_numeric, errors="coerce").dropna(how="any")

    @staticmethod
    def simulate(returns: pd.DataFrame, rf_annual: float, n_portfolios: int) -> pd.DataFrame:
        """Simula portafolios aleatorios."""
        return simulate_portfolios(returns, rf_annual=rf_annual, n_portfolios=n_portfolios)

    @staticmethod
    def efficient_frontier(sim_df: pd.DataFrame) -> pd.DataFrame:
        """Calcula frontera eficiente desde simulaciones."""
        return efficient_frontier(sim_df)

    @staticmethod
    def optimal_portfolios(sim_df: pd.DataFrame) -> tuple[Any, Any]:
        """Devuelve portafolios de minima varianza y maximo Sharpe."""
        return minimum_variance_portfolio(sim_df), maximum_sharpe_portfolio(sim_df)

    @staticmethod
    def manual_portfolio(returns: pd.DataFrame, weights: np.ndarray, rf_annual: float) -> dict[str, Any]:
        """Evalua un portafolio manual con pesos validados por la pagina."""
        mean_returns = returns.mean().values * TRADING_DAYS
        cov_matrix = returns.cov().values * TRADING_DAYS
        port_return = float(weights @ mean_returns)
        port_vol = float(np.sqrt(weights.T @ cov_matrix @ weights))
        sharpe = np.nan if port_vol <= 0 else float((port_return - rf_annual) / port_vol)
        return {
            "return": port_return,
            "volatility": port_vol,
            "sharpe": sharpe,
            "weights": weights,
        }

    @staticmethod
    def weights_frame(portfolio: Any) -> pd.DataFrame:
        """Normaliza tabla de pesos para render en Streamlit."""
        out = weights_table(portfolio).copy()
        cols = {c.lower(): c for c in out.columns}
        activo_col = cols.get("activo", list(out.columns)[0])
        peso_col = cols.get("peso", list(out.columns)[1])
        out = out[[activo_col, peso_col]].copy()
        out.columns = ["Activo", "Peso"]
        out["Peso"] = pd.to_numeric(out["Peso"], errors="coerce")
        out["Participacion"] = out["Peso"].map(lambda x: f"{x:.2%}" if pd.notna(x) else "N/D")
        out["Peso"] = out["Peso"].round(4)
        return out.sort_values("Peso", ascending=False).reset_index(drop=True)

