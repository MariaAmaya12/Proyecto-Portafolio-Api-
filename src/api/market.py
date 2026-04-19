from __future__ import annotations

from typing import Dict, List

import logging

import pandas as pd
import streamlit as st

from src.api.backend_client import backend_post, records_to_dataframe

logger = logging.getLogger(__name__)


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
        "close": records_to_dataframe(data.get("close", [])),
        "returns": records_to_dataframe(data.get("returns", [])),
    }


@st.cache_data(show_spinner=False, ttl=3600)
def get_prices(ticker: str, start: str, end: str) -> pd.DataFrame:
    bundle = get_market_bundle(tickers=[ticker], start=start, end=end)
    return bundle["ohlcv"].get(ticker, pd.DataFrame())


def get_multiple_prices(tickers: List[str], start: str, end: str) -> Dict[str, pd.DataFrame]:
    try:
        return get_market_bundle(tickers=tickers, start=start, end=end)["ohlcv"]
    except Exception as e:
        logger.warning(f"[WARNING] bundle de mercado falló: {e}")
        return {ticker: pd.DataFrame() for ticker in tickers}
