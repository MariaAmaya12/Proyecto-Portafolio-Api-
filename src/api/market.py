from __future__ import annotations

from typing import Dict, List

import pandas as pd
import streamlit as st

from src.api.backend_client import backend_post, records_to_dataframe


def _numeric_market_frame(records: list[dict[str, object]] | dict[str, object]) -> pd.DataFrame:
    df = records_to_dataframe(records)
    if df.empty:
        return df

    df = df.drop(columns=[c for c in df.columns if str(c).lower() in {"index", "date"}], errors="ignore")
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.dropna(axis=1, how="all")
    return df


@st.cache_data(show_spinner=False, ttl=3600)
def get_market_bundle(tickers: List[str], start: str, end: str) -> Dict[str, object]:
    payload = {
        "tickers": tickers,
        "start": start,
        "end": end,
    }
    data = backend_post("/market/bundle", payload)
    return {
        "ohlcv": {
            ticker: records_to_dataframe(records)
            for ticker, records in data.get("ohlcv", {}).items()
        },
        "close": _numeric_market_frame(data.get("close", [])),
        "returns": _numeric_market_frame(data.get("returns", [])),
        "missing_tickers": data.get("missing_tickers", []),
        "last_available_date": data.get("last_available_date"),
        "calendar_diagnostics": data.get("calendar_diagnostics", {}),
    }


@st.cache_data(show_spinner=False, ttl=3600)
def get_prices(ticker: str, start: str, end: str) -> pd.DataFrame:
    bundle = get_market_bundle(tickers=[ticker], start=start, end=end)
    return bundle["ohlcv"].get(ticker, pd.DataFrame())


def get_multiple_prices(tickers: List[str], start: str, end: str) -> Dict[str, pd.DataFrame]:
    return get_market_bundle(tickers=tickers, start=start, end=end)["ohlcv"]
