from __future__ import annotations

import os
from typing import Any

import pandas as pd
import requests

DEFAULT_BACKEND_BASE_URL = "http://127.0.0.1:8000"
BACKEND_TIMEOUT_SECONDS = 30


def backend_base_url() -> str:
    return os.getenv("BACKEND_API_BASE_URL", DEFAULT_BACKEND_BASE_URL).rstrip("/")


def backend_get(path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
    url = f"{backend_base_url()}{path}"
    try:
        response = requests.get(url, params=params, timeout=BACKEND_TIMEOUT_SECONDS)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as exc:
        raise RuntimeError(f"No se pudo consultar el backend en {url}: {exc}") from exc


def backend_post(path: str, payload: dict[str, Any]) -> dict[str, Any]:
    url = f"{backend_base_url()}{path}"
    try:
        response = requests.post(url, json=payload, timeout=BACKEND_TIMEOUT_SECONDS)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as exc:
        raise RuntimeError(f"No se pudo consultar el backend en {url}: {exc}") from exc


def records_to_dataframe(records: list[dict[str, Any]]) -> pd.DataFrame:
    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.dropna(subset=["Date"]).set_index("Date").sort_index()

    return df
