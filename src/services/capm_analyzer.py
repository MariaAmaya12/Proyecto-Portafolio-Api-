from __future__ import annotations

from typing import Any

import pandas as pd

from src.capm import compute_beta_and_capm
from src.returns_analysis import compute_return_series


class CAPMAnalyzer:
    """Encapsula preparacion de retornos y calculos CAPM."""

    def __init__(self, close: pd.DataFrame, returns: pd.DataFrame | None = None, rf_annual: float = 0.03) -> None:
        self.close = close if close is not None else pd.DataFrame()
        self.returns = returns if returns is not None else pd.DataFrame()
        self.rf_annual = rf_annual

    def get_close_series(self, ticker: str) -> pd.Series:
        """Obtiene una serie de precios numerica para un ticker."""
        if self.close.empty or ticker not in self.close.columns:
            return pd.Series(dtype=float)
        return pd.to_numeric(self.close[ticker], errors="coerce").dropna()

    def get_asset_returns(self, ticker: str) -> pd.Series:
        """Obtiene retornos desde la matriz de returns o los calcula desde close."""
        if not self.returns.empty and ticker in self.returns.columns:
            return pd.to_numeric(self.returns[ticker], errors="coerce").dropna()
        close_series = self.get_close_series(ticker)
        if close_series.empty:
            return pd.Series(dtype=float)
        return compute_return_series(close_series)["simple_return"]

    def get_benchmark_returns(self, benchmark_ticker: str) -> pd.Series:
        """Alias explicito para retornos del benchmark."""
        return self.get_asset_returns(benchmark_ticker)

    def build_portfolio_returns(
        self,
        name_to_ticker: dict[str, str],
        weights: dict[str, float] | None = None,
    ) -> tuple[pd.Series, list[str], list[str], pd.Series]:
        """Construye retornos de portafolio y reporta componentes incluidos/faltantes."""
        series_by_name = []
        included: list[str] = []
        missing: list[str] = []

        for name, ticker in name_to_ticker.items():
            returns = self.get_asset_returns(ticker)
            if returns.empty:
                missing.append(f"{name} ({ticker})")
                continue
            series_by_name.append(returns.rename(name))
            included.append(name)

        if not series_by_name:
            return pd.Series(dtype=float), included, missing, pd.Series(dtype=float)

        returns_df = pd.concat(series_by_name, axis=1).dropna(how="any")
        if returns_df.empty:
            return pd.Series(dtype=float), included, missing, pd.Series(dtype=float)

        if weights:
            weights_series = pd.Series(
                {name: max(float(weights.get(name, 0.0)), 0.0) for name in returns_df.columns},
                index=returns_df.columns,
                dtype=float,
            )
        else:
            weights_series = pd.Series(1 / len(returns_df.columns), index=returns_df.columns, dtype=float)

        return returns_df.mul(weights_series, axis=1).sum(axis=1), included, missing, weights_series

    def compute_for_asset(self, ticker: str, benchmark_ticker: str) -> dict[str, Any]:
        """Calcula CAPM de un activo contra un benchmark."""
        return compute_beta_and_capm(
            self.get_asset_returns(ticker),
            self.get_benchmark_returns(benchmark_ticker),
            rf_annual=self.rf_annual,
        )

    def compute_for_portfolio(self, portfolio_returns: pd.Series, benchmark_ticker: str) -> dict[str, Any]:
        """Calcula CAPM de una serie de portafolio contra un benchmark."""
        return compute_beta_and_capm(
            portfolio_returns,
            self.get_benchmark_returns(benchmark_ticker),
            rf_annual=self.rf_annual,
        )

