from __future__ import annotations

import logging
from typing import Dict, List

import pandas as pd
import yfinance as yf

from src.config import DATA_DIR

logger = logging.getLogger(__name__)

_YFINANCE_CACHE_CONFIGURED = False


def _configure_yfinance_cache() -> None:
    global _YFINANCE_CACHE_CONFIGURED
    if _YFINANCE_CACHE_CONFIGURED:
        return

    cache_dir = DATA_DIR / "yfinance_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    try:
        if hasattr(yf, "cache") and hasattr(yf.cache, "set_cache_location"):
            yf.cache.set_cache_location(str(cache_dir.resolve()))
        elif hasattr(yf, "set_tz_cache_location"):
            yf.set_tz_cache_location(str(cache_dir.resolve()))
    except Exception as exc:
        logger.warning("[YFINANCE CACHE] No se pudo configurar caché en %s: %s", cache_dir, exc)

    _YFINANCE_CACHE_CONFIGURED = True


class MarketService:
    """
    Servicio responsable de descarga, normalizacion y armado de datos de mercado.
    """

    def standardize_ohlcv(self, df: pd.DataFrame) -> pd.DataFrame:
        return _standardize_ohlcv(df)

    def validate_ohlcv(self, df: pd.DataFrame, ticker: str) -> pd.DataFrame:
        return _validate_ohlcv(df, ticker)

    def get_prices(self, ticker: str, start: str, end: str) -> pd.DataFrame:
        return _get_prices(ticker=ticker, start=start, end=end)

    def get_multiple_prices(self, tickers: List[str], start: str, end: str) -> Dict[str, pd.DataFrame]:
        return _get_multiple_prices(tickers=tickers, start=start, end=end)

    def build_close_matrix(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        return _build_close_matrix(data)

    def build_returns_matrix(self, close: pd.DataFrame) -> pd.DataFrame:
        return _build_returns_matrix(close)

    def load_market_bundle(self, tickers: List[str], start: str, end: str) -> Dict[str, object]:
        return _load_market_bundle_service(tickers=tickers, start=start, end=end)


def _standardize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    out = df.copy()

    if isinstance(out.columns, pd.MultiIndex):
        level0 = out.columns.get_level_values(0)
        level1 = out.columns.get_level_values(1)
        expected = {"Open", "High", "Low", "Close", "Adj Close", "Volume"}

        if set(level0).intersection(expected):
            out.columns = level0
        elif set(level1).intersection(expected):
            out.columns = level1
        else:
            out.columns = ["_".join(map(str, c)).strip() for c in out.columns.to_flat_index()]

    cols = {str(c).lower(): c for c in out.columns}
    rename_map = {}

    for desired in ["open", "high", "low", "close", "adj close", "volume"]:
        if desired in cols:
            rename_map[cols[desired]] = desired.title() if desired != "adj close" else "Adj Close"

    out = out.rename(columns=rename_map).copy()

    keep_cols = [c for c in ["Open", "High", "Low", "Close", "Adj Close", "Volume"] if c in out.columns]
    if keep_cols:
        out = out[keep_cols]

    out.index = pd.to_datetime(out.index)
    out = out.sort_index()
    out = out[~out.index.duplicated(keep="last")]
    return out


def _validate_ohlcv(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    if df is None or df.empty:
        raise ValueError(f"[API ERROR] {ticker}: datos vacíos")

    valid_price_cols = {"Close", "Adj Close"}
    if not valid_price_cols.intersection(set(df.columns)):
        raise ValueError(
            f"[API ERROR] {ticker}: no tiene columnas de precio válidas. "
            f"Columnas encontradas: {list(df.columns)}"
        )

    return df


def _get_prices(ticker: str, start: str, end: str) -> pd.DataFrame:
    try:
        _configure_yfinance_cache()

        df = yf.download(
            ticker,
            start=start,
            end=end,
            auto_adjust=False,
            progress=False,
            actions=False,
            threads=False,
        )

        df = _standardize_ohlcv(df)
        return _validate_ohlcv(df, ticker)

    except Exception as e:
        logger.error(f"[YFINANCE ERROR] {ticker}: {e}")
        raise RuntimeError(f"No se pudo descargar datos para {ticker}")


def _get_multiple_prices(tickers: List[str], start: str, end: str) -> Dict[str, pd.DataFrame]:
    data = {}
    for ticker in tickers:
        try:
            data[ticker] = _get_prices(ticker, start, end)
        except Exception as e:
            logger.warning(f"[WARNING] {ticker} falló: {e}")
            data[ticker] = pd.DataFrame()
    return data


def _build_close_matrix(data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    series = {}

    for ticker, df in data.items():
        if df.empty:
            continue

        if "Adj Close" in df.columns:
            adj_close = df["Adj Close"]
        elif "Close" in df.columns:
            adj_close = df["Close"]
        else:
            continue

        if isinstance(adj_close, pd.DataFrame):
            adj_close = adj_close.iloc[:, 0]

        series[ticker] = adj_close.rename(ticker)

    if not series:
        return pd.DataFrame()

    close = pd.concat(series.values(), axis=1).sort_index()
    close = close.dropna(how="all")
    return close


def _build_returns_matrix(close: pd.DataFrame) -> pd.DataFrame:
    if close.empty:
        return pd.DataFrame()

    returns = close.pct_change(fill_method=None).dropna(how="all")
    return returns


def _load_market_bundle_service(tickers: List[str], start: str, end: str) -> Dict[str, object]:
    data = _get_multiple_prices(tickers=tickers, start=start, end=end)
    close = _build_close_matrix(data)
    returns = _build_returns_matrix(close)

    return {
        "ohlcv": data,
        "close": close,
        "returns": returns,
    }


DEFAULT_MARKET_SERVICE = MarketService()


def standardize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    return DEFAULT_MARKET_SERVICE.standardize_ohlcv(df)


def validate_ohlcv(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    return DEFAULT_MARKET_SERVICE.validate_ohlcv(df, ticker)


def get_prices(ticker: str, start: str, end: str) -> pd.DataFrame:
    return DEFAULT_MARKET_SERVICE.get_prices(ticker=ticker, start=start, end=end)


def get_multiple_prices(tickers: List[str], start: str, end: str) -> Dict[str, pd.DataFrame]:
    return DEFAULT_MARKET_SERVICE.get_multiple_prices(tickers=tickers, start=start, end=end)


def build_close_matrix(data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    return DEFAULT_MARKET_SERVICE.build_close_matrix(data)


def build_returns_matrix(close: pd.DataFrame) -> pd.DataFrame:
    return DEFAULT_MARKET_SERVICE.build_returns_matrix(close)


def load_market_bundle_service(tickers: List[str], start: str, end: str) -> Dict[str, object]:
    return DEFAULT_MARKET_SERVICE.load_market_bundle(tickers=tickers, start=start, end=end)
