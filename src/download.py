from __future__ import annotations

from typing import Dict, List
import pandas as pd
import streamlit as st

from src.config import RAW_DIR, PROCESSED_DIR, ensure_project_dirs
from src.api.market import get_market_bundle, get_multiple_prices, get_prices
from src.api.backend_client import friendly_error_message

_LAST_DATA_ERROR_MESSAGE: str | None = None
_LAST_DATA_ERROR_DEBUG: str | None = None


def _remember_data_error(exc: Exception) -> None:
    global _LAST_DATA_ERROR_MESSAGE, _LAST_DATA_ERROR_DEBUG
    _LAST_DATA_ERROR_MESSAGE = friendly_error_message(exc)
    technical_detail = getattr(exc, "technical_detail", None)
    status_code = getattr(exc, "status_code", None)
    parts = []
    if status_code is not None:
        parts.append(f"status_code={status_code}")
    parts.append(f"exception={technical_detail or repr(exc)}")
    _LAST_DATA_ERROR_DEBUG = " | ".join(parts)


def _remember_empty_response(ticker: str, df: pd.DataFrame | None) -> None:
    global _LAST_DATA_ERROR_MESSAGE, _LAST_DATA_ERROR_DEBUG
    _LAST_DATA_ERROR_MESSAGE = None
    columns = list(df.columns) if df is not None else []
    _LAST_DATA_ERROR_DEBUG = (
        f"ticker={ticker} | respuesta vacia desde API/proveedor | "
        f"columnas={columns} | longitud_final=0"
    )


def clear_data_error_message() -> None:
    global _LAST_DATA_ERROR_MESSAGE, _LAST_DATA_ERROR_DEBUG
    _LAST_DATA_ERROR_MESSAGE = None
    _LAST_DATA_ERROR_DEBUG = None


def data_error_message(default: str) -> str:
    return _LAST_DATA_ERROR_MESSAGE or default


def data_error_debug(default: str = "") -> str:
    return _LAST_DATA_ERROR_DEBUG or default


def _standardize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """
    Estandariza columnas OHLCV descargadas por yfinance.
    Soporta columnas simples y columnas MultiIndex.
    """
    if df is None or df.empty:
        return pd.DataFrame()

    out = df.copy()

    # yfinance puede devolver MultiIndex incluso para un solo ticker
    if isinstance(out.columns, pd.MultiIndex):
        level0 = out.columns.get_level_values(0)
        level1 = out.columns.get_level_values(1)

        expected = {"Open", "High", "Low", "Close", "Adj Close", "Volume"}

        # Caso más común: nivel 0 = tipo de precio, nivel 1 = ticker
        if set(level0).intersection(expected):
            out.columns = level0

        # Respaldo: por si el orden estuviera invertido
        elif set(level1).intersection(expected):
            out.columns = level1

        else:
            # Último recurso: aplanar columnas
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


@st.cache_data(show_spinner=False, ttl=3600)
def download_single_ticker(ticker: str, start: str, end: str) -> pd.DataFrame:
    """
    Obtiene OHLCV de un ticker desde el backend FastAPI propio.
    """
    try:
        raw_df = get_prices(ticker=ticker, start=start, end=end)
        df = _standardize_ohlcv(raw_df)

        if df.empty:
            _remember_empty_response(ticker, raw_df)
        else:
            clear_data_error_message()

        if not df.empty:
            try:
                df.to_csv(RAW_DIR / f"{ticker.replace('^', '')}_raw.csv")
            except Exception:
                pass

        return df

    except Exception as e:
        _remember_data_error(e)
        return pd.DataFrame()


@st.cache_data(show_spinner=False, ttl=3600)
def download_multiple_tickers(tickers: List[str], start: str, end: str) -> Dict[str, pd.DataFrame]:
    """
    Descarga múltiples tickers usando la nueva capa API.
    """
    ensure_project_dirs()

    try:
        data = get_multiple_prices(tickers=tickers, start=start, end=end)
        clear_data_error_message()
    except Exception as e:
        _remember_data_error(e)
        data = {ticker: pd.DataFrame() for ticker in tickers}

    return data


def build_close_matrix(data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Construye matriz de cierres ajustados si existen; si no, usa Close.
    Maneja correctamente casos donde yfinance devuelve DataFrame en lugar de Series.
    """
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

        # 🔥 FIX CLAVE
        if isinstance(adj_close, pd.DataFrame):
            adj_close = adj_close.iloc[:, 0]

        series[ticker] = adj_close.rename(ticker)

    if not series:
        return pd.DataFrame()

    close = pd.concat(series.values(), axis=1).sort_index()
    close = close.dropna(how="all")
    return close

def build_returns_matrix(close: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula rendimientos simples diarios.
    """
    if close.empty:
        return pd.DataFrame()

    aligned = close.sort_index().ffill(limit=3)
    returns = aligned.pct_change(fill_method=None).dropna(how="all")
    return returns


def market_bundle_metadata(data: Dict[str, pd.DataFrame], close: pd.DataFrame, returns: pd.DataFrame) -> Dict[str, object]:
    valid_frames = {
        ticker: df
        for ticker, df in data.items()
        if df is not None and not df.empty
    }
    last_available = None
    if valid_frames:
        last_available = max(df.index.max() for df in valid_frames.values())

    return {
        "missing_tickers": [
            ticker
            for ticker, df in data.items()
            if df is None or df.empty
        ],
        "ohlcv_shapes": {
            ticker: tuple(df.shape) if df is not None else (0, 0)
            for ticker, df in data.items()
        },
        "close_shape": tuple(close.shape),
        "returns_shape": tuple(returns.shape),
        "last_available_date": (
            pd.Timestamp(last_available).date().isoformat()
            if last_available is not None and pd.notna(last_available)
            else None
        ),
    }


@st.cache_data(show_spinner=False, ttl=3600)
def load_market_bundle(tickers: List[str], start: str, end: str) -> Dict[str, object]:
    """
    Bundle central de mercado usado en todo el dashboard.
    """
    ensure_project_dirs()
    try:
        bundle = get_market_bundle(tickers=tickers, start=start, end=end)
        data = bundle["ohlcv"]
        close = bundle["close"]
        returns = bundle["returns"]
        computed_metadata = market_bundle_metadata(data, close, returns)
        metadata = {
            **computed_metadata,
            "missing_tickers": bundle.get("missing_tickers", []),
            "last_available_date": bundle.get("last_available_date")
            or computed_metadata.get("last_available_date"),
        }
        clear_data_error_message()
    except Exception as e:
        _remember_data_error(e)
        data = download_multiple_tickers(tickers=tickers, start=start, end=end)
        close = build_close_matrix(data)
        returns = build_returns_matrix(close)
        metadata = market_bundle_metadata(data, close, returns)

    try:
        close.to_csv(PROCESSED_DIR / "close_prices.csv")
        returns.to_csv(PROCESSED_DIR / "returns.csv")
    except Exception:
        pass

    return {
        "ohlcv": data,
        "close": close,
        "returns": returns,
        "metadata": metadata,
    }
