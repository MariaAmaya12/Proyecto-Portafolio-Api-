from __future__ import annotations

from typing import Any

import pandas as pd

from src.api.backend_client import fetch_market_bundle_from_backend


class MarketDataClient:
    """Cliente reutilizable para obtener datos de mercado desde el backend FastAPI."""

    def fetch_bundle(self, tickers: list[str] | tuple[str, ...], start: str, end: str) -> dict[str, Any]:
        """Obtiene el bundle completo y normaliza listas de tickers duplicados."""
        unique_tickers = list(dict.fromkeys(str(ticker).strip() for ticker in tickers if str(ticker).strip()))
        return fetch_market_bundle_from_backend(tickers=unique_tickers, start=start, end=end)

    def get_close(self, tickers: list[str] | tuple[str, ...], start: str, end: str) -> pd.DataFrame:
        """Devuelve la matriz de precios de cierre del backend."""
        return self.fetch_bundle(tickers, start, end).get("close", pd.DataFrame())

    def get_returns(self, tickers: list[str] | tuple[str, ...], start: str, end: str) -> pd.DataFrame:
        """Devuelve la matriz de retornos del backend."""
        return self.fetch_bundle(tickers, start, end).get("returns", pd.DataFrame())

    def get_ohlcv(self, tickers: list[str] | tuple[str, ...], start: str, end: str) -> dict[str, pd.DataFrame]:
        """Devuelve series OHLCV por ticker desde el backend."""
        return self.fetch_bundle(tickers, start, end).get("ohlcv", {})

    @staticmethod
    def missing_tickers(bundle: dict[str, Any]) -> list[str]:
        """Extrae tickers faltantes con compatibilidad para metadata anidada."""
        missing = bundle.get("missing_tickers")
        if missing is None:
            missing = bundle.get("metadata", {}).get("missing_tickers", [])
        return list(missing or [])

