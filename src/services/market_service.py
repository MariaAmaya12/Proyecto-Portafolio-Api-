from __future__ import annotations

import logging
from typing import Dict, List

import pandas as pd
import yfinance as yf

from src.config import DATA_DIR
from src.date_utils import yfinance_exclusive_end

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
        # yfinance interpreta `end` como exclusivo; el dashboard lo trata como inclusivo.
        yf_end = yfinance_exclusive_end(end)

        df = yf.download(
            ticker,
            start=start,
            end=yf_end,
            auto_adjust=False,
            progress=False,
            actions=False,
            threads=False,
        )

        df = _standardize_ohlcv(df)
        return _validate_ohlcv(df, ticker)

    except Exception as exc:
        logger.error(f"[YFINANCE ERROR] {ticker}: {exc}")
        raise RuntimeError(f"No se pudo descargar datos para {ticker}")


def _get_multiple_prices(tickers: List[str], start: str, end: str) -> Dict[str, pd.DataFrame]:
    data = {}
    for ticker in tickers:
        try:
            data[ticker] = _get_prices(ticker, start, end)
        except Exception as exc:
            logger.warning(f"[WARNING] {ticker} falló: {exc}")
            data[ticker] = pd.DataFrame()
    return data


def _missing_tickers_from_data(data: Dict[str, pd.DataFrame], tickers: List[str]) -> List[str]:
    return [
        ticker
        for ticker in tickers
        if data.get(ticker) is None or data.get(ticker).empty
    ]


def _retry_missing_tickers(
    data: Dict[str, pd.DataFrame],
    tickers: List[str],
    start: str,
    end: str,
) -> dict[str, object]:
    initial_missing = _missing_tickers_from_data(data, tickers)
    retried_tickers: List[str] = []
    recovered_tickers: List[str] = []

    for ticker in initial_missing:
        retried_tickers.append(ticker)
        try:
            recovered_df = _get_prices(ticker=ticker, start=start, end=end)
            if recovered_df is not None and not recovered_df.empty:
                data[ticker] = recovered_df
                recovered_tickers.append(ticker)
            else:
                data[ticker] = pd.DataFrame()
        except Exception as exc:
            logger.warning(f"[RETRY WARNING] {ticker} falló en reintento individual: {exc}")
            data[ticker] = pd.DataFrame()

    still_missing_tickers = _missing_tickers_from_data(data, tickers)
    return {
        "retry_missing_tickers_applied": bool(initial_missing),
        "retried_tickers": retried_tickers,
        "recovered_tickers": recovered_tickers,
        "still_missing_tickers": still_missing_tickers,
    }


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
    close.index = pd.to_datetime(close.index, errors="coerce")
    close = close[~close.index.isna()]
    close = close.apply(pd.to_numeric, errors="coerce")
    close = close.dropna(how="all")
    close = close.dropna(axis=1, how="all")
    return _calendar_fill_close(close)


def _calendar_fill_close(close: pd.DataFrame) -> pd.DataFrame:
    if close.empty:
        return close

    if not isinstance(close.index, pd.DatetimeIndex):
        return pd.DataFrame()

    raw_close = close.apply(pd.to_numeric, errors="coerce").dropna(axis=1, how="all").sort_index()
    first_valid_dates = [
        raw_close[ticker].first_valid_index()
        for ticker in raw_close.columns
        if raw_close[ticker].first_valid_index() is not None
    ]
    if not first_valid_dates:
        return pd.DataFrame()

    start_effective = max(first_valid_dates)
    calendar = pd.bdate_range(start=raw_close.index.min(), end=raw_close.index.max())
    raw_returns = raw_close.pct_change(fill_method=None).dropna(how="all")

    aligned_index = raw_close.index.union(calendar)
    filled = (
        raw_close.reindex(aligned_index)
        .sort_index()
        .ffill()
        .reindex(calendar)
        .loc[start_effective:]
        .dropna(how="all")
    )

    filled.attrs["calendar_diagnostics"] = {
        "calendar_fill_applied": True,
        "fill_method": "ffill",
        "fill_limit": None,
        "start_effective": pd.Timestamp(start_effective).date().isoformat(),
        "close_shape_before_fill": tuple(raw_close.shape),
        "close_shape_after_fill": tuple(filled.shape),
        "na_por_activo_close_antes_fill": raw_close.isna().sum().to_dict(),
        "na_por_activo_close_despues_fill": filled.isna().sum().to_dict(),
        "na_por_activo_retornos_antes_fill": raw_returns.isna().sum().to_dict(),
        "returns_shape_before_fill": tuple(raw_returns.shape),
    }
    return filled


def _build_returns_matrix(close: pd.DataFrame) -> pd.DataFrame:
    if close.empty:
        return pd.DataFrame()

    if not isinstance(close.index, pd.DatetimeIndex):
        return pd.DataFrame()

    aligned = close.apply(pd.to_numeric, errors="coerce").dropna(axis=1, how="all").sort_index()
    returns = aligned.pct_change(fill_method=None).dropna(how="all")
    returns = returns.apply(pd.to_numeric, errors="coerce").dropna(axis=1, how="all")
    diagnostics = dict(close.attrs.get("calendar_diagnostics", {}))
    diagnostics.update(
        {
            "returns_shape_after_fill": tuple(returns.shape),
            "na_por_activo_retornos_despues_fill": returns.isna().sum().to_dict(),
        }
    )
    returns.attrs["calendar_diagnostics"] = diagnostics
    return returns


def _load_market_bundle_service(tickers: List[str], start: str, end: str) -> Dict[str, object]:
    data = _get_multiple_prices(tickers=tickers, start=start, end=end)
    retry_diagnostics = _retry_missing_tickers(data=data, tickers=tickers, start=start, end=end)
    close = _build_close_matrix(data)
    returns = _build_returns_matrix(close)

    calendar_diagnostics = dict(returns.attrs.get("calendar_diagnostics", {}))
    calendar_diagnostics.update(retry_diagnostics)
    close.attrs["calendar_diagnostics"] = calendar_diagnostics
    returns.attrs["calendar_diagnostics"] = calendar_diagnostics

    return {
        "ohlcv": data,
        "close": close,
        "returns": returns,
        "calendar_diagnostics": calendar_diagnostics,
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
