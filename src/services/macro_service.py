from __future__ import annotations

import io
import logging
import os
from typing import Dict

import pandas as pd
import requests
import wbgapi as wb
from dotenv import load_dotenv
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from src.config import FRED_SERIES
from src.services.market_service import get_prices

load_dotenv()
logger = logging.getLogger(__name__)

FRED_TIMEOUT = 5
MACRO_CACHE_URL = "https://raw.githubusercontent.com/MariaAmaya12/Portafolio_Api/feat/macro-cache/data/macro_cache.json"


class MacroService:
    """
    Servicio responsable de obtener y consolidar variables macroeconomicas.
    """

    def build_session(self) -> requests.Session:
        return _build_session()

    def get_fred_series(self, series_id: str) -> pd.DataFrame:
        return _get_fred_series(series_id)

    def get_worldbank_inflation(self) -> pd.DataFrame:
        return _get_worldbank_inflation()

    def get_worldbank_fx(self) -> pd.DataFrame:
        return _get_worldbank_fx()

    def get_yfinance_usdcop(self) -> float:
        return _get_yfinance_usdcop()

    def get_github_macro_cache(self) -> dict:
        return _get_github_macro_cache()

    def get_macro_snapshot(self) -> Dict[str, float]:
        return _get_macro_snapshot()


def empty_fred_df() -> pd.DataFrame:
    return pd.DataFrame(columns=["date", "value"])


def _build_session() -> requests.Session:
    retry = Retry(
        total=3,
        backoff_factor=0.7,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    adapter = HTTPAdapter(max_retries=retry)
    session = requests.Session()
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update(
        {
            "User-Agent": "Mozilla/5.0",
            "Accept": "text/csv,application/json,text/plain,*/*",
        }
    )
    return session


def fred_csv_url(series_id: str) -> str:
    return f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"


def clean_fred_df(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    if df is None or df.empty or value_col not in df.columns or "date" not in df.columns:
        return empty_fred_df()

    out = df.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out["value"] = pd.to_numeric(out[value_col], errors="coerce")
    out = out[["date", "value"]].dropna()
    out = out.sort_values("date").drop_duplicates(subset="date", keep="last")

    if out.empty:
        return empty_fred_df()

    return out


def _get_fred_series_json(series_id: str, api_key: str, session: requests.Session) -> pd.DataFrame:
    try:
        url = "https://api.stlouisfed.org/fred/series/observations"
        params = {
            "series_id": series_id,
            "api_key": api_key,
            "file_type": "json",
        }
        response = session.get(url, params=params, timeout=FRED_TIMEOUT)
        response.raise_for_status()

        payload = response.json()
        obs = payload.get("observations", [])
        if not obs:
            return empty_fred_df()

        df = pd.DataFrame(obs)
        return clean_fred_df(df, value_col="value")

    except Exception as e:
        logger.warning(f"[FRED JSON] Error en {series_id}: {e}")
        return empty_fred_df()


def _get_fred_series_csv(series_id: str, session: requests.Session) -> pd.DataFrame:
    try:
        response = session.get(fred_csv_url(series_id), timeout=FRED_TIMEOUT)
        response.raise_for_status()

        df = pd.read_csv(io.StringIO(response.text))
        df.columns = [c.lower() for c in df.columns]

        value_cols = [c for c in df.columns if c != "date"]
        if not value_cols:
            return empty_fred_df()

        return clean_fred_df(df, value_col=value_cols[0])

    except Exception as e:
        logger.warning(f"[FRED CSV] Error en {series_id}: {e}")
        return empty_fred_df()


def _get_fred_series(series_id: str) -> pd.DataFrame:
    session = _build_session()
    api_key = os.getenv("FRED_API_KEY", "").strip()

    if api_key:
        df_json = _get_fred_series_json(series_id, api_key, session)
        if not df_json.empty:
            return df_json

    return _get_fred_series_csv(series_id, session)


def _get_worldbank_inflation() -> pd.DataFrame:
    try:
        rows = list(
            wb.data.fetch(
                "FP.CPI.TOTL.ZG",
                economy="USA",
                time=range(2000, 2031),
            )
        )

        if not rows:
            return empty_fred_df()

        df = pd.DataFrame(rows)
        if "time" not in df.columns or "value" not in df.columns:
            return empty_fred_df()

        df["date"] = df["time"].astype(str).str.replace("YR", "", regex=False)
        df["date"] = pd.to_datetime(df["date"], format="%Y", errors="coerce")
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df = df[["date", "value"]].dropna().sort_values("date")

        if df.empty:
            return empty_fred_df()

        return df

    except Exception as e:
        logger.warning(f"[WBGAPI] Error inflación: {e}")
        return empty_fred_df()


def _get_worldbank_fx() -> pd.DataFrame:
    try:
        rows = list(
            wb.data.fetch(
                "PA.NUS.FCRF",
                economy="COL",
                time=range(2000, 2031),
            )
        )

        if not rows:
            return empty_fred_df()

        df = pd.DataFrame(rows)
        if "time" not in df.columns or "value" not in df.columns:
            return empty_fred_df()

        df["date"] = df["time"].astype(str).str.replace("YR", "", regex=False)
        df["date"] = pd.to_datetime(df["date"], format="%Y", errors="coerce")
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df = df[["date", "value"]].dropna().sort_values("date")

        if df.empty:
            return empty_fred_df()

        return df

    except Exception as e:
        logger.warning(f"[WBGAPI] Error FX: {e}")
        return empty_fred_df()


def _get_yfinance_usdcop() -> float:
    try:
        end = pd.Timestamp.today().normalize().date()
        start = (pd.Timestamp(end) - pd.DateOffset(days=7)).date()
        df = get_prices("USDCOP=X", start=start.isoformat(), end=end.isoformat())

        if df is None or df.empty:
            return float("nan")

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        close_col = "Adj Close" if "Adj Close" in df.columns else "Close"
        if close_col not in df.columns:
            return float("nan")

        series = df[close_col]
        if isinstance(series, pd.DataFrame):
            series = series.iloc[:, 0]

        series = pd.to_numeric(series, errors="coerce").dropna()
        if series.empty:
            return float("nan")

        return float(series.iloc[-1])

    except Exception as e:
        logger.warning(f"[YFINANCE FX] Error USD/COP: {e}")
        return float("nan")


def _get_github_macro_cache() -> dict:
    try:
        session = _build_session()
        response = session.get(MACRO_CACHE_URL, timeout=10)
        response.raise_for_status()
        data = response.json()

        return {
            "risk_free_rate_pct": data.get("risk_free_rate_pct"),
            "inflation_yoy": data.get("inflation_yoy"),
            "cop_per_usd": data.get("cop_per_usd"),
            "usdcop_market": data.get("usdcop_market"),
            "source": data.get("source", "github_actions_cache"),
            "last_updated": data.get("last_updated"),
        }

    except Exception as e:
        logger.warning(f"[GITHUB CACHE] Error leyendo macro_cache.json: {e}")
        return {}


def latest_value(df: pd.DataFrame) -> float:
    if df.empty:
        return float("nan")
    return float(df.iloc[-1]["value"])


def yoy_inflation(cpi_df: pd.DataFrame) -> float:
    if cpi_df.empty:
        return float("nan")

    c = cpi_df.set_index("date")["value"].sort_index()
    if len(c) >= 13:
        yoy = c.pct_change(12).dropna()
        if not yoy.empty:
            return float(yoy.iloc[-1])

    return float(c.iloc[-1]) / 100


def _get_macro_snapshot() -> Dict[str, float]:
    cache = _get_github_macro_cache()

    if cache:
        rf = cache.get("risk_free_rate_pct")
        inf = cache.get("inflation_yoy")
        cop = cache.get("cop_per_usd")
        usdcop_market = cache.get("usdcop_market")

        if any(x == x for x in [rf, inf, cop, usdcop_market] if x is not None):
            return {
                "risk_free_rate_pct": rf if rf is not None else float("nan"),
                "inflation_yoy": inf if inf is not None else float("nan"),
                "cop_per_usd": cop if cop is not None else float("nan"),
                "usdcop_market": usdcop_market if usdcop_market is not None else float("nan"),
                "source": cache.get("source", "github_actions_cache"),
                "last_updated": cache.get("last_updated"),
            }

    rf_df = _get_fred_series(FRED_SERIES["risk_free_rate"])
    cpi_df = _get_fred_series(FRED_SERIES["inflation"])
    cop_df = _get_fred_series(FRED_SERIES["cop_usd"])

    if cpi_df.empty:
        cpi_df = _get_worldbank_inflation()

    if cop_df.empty:
        cop_df = _get_worldbank_fx()

    return {
        "risk_free_rate_pct": latest_value(rf_df) if not rf_df.empty else 3.0,
        "inflation_yoy": yoy_inflation(cpi_df) if not cpi_df.empty else float("nan"),
        "cop_per_usd": latest_value(cop_df) if not cop_df.empty else float("nan"),
        "usdcop_market": _get_yfinance_usdcop(),
        "source": "live_api_fallback",
        "last_updated": None,
    }


DEFAULT_MACRO_SERVICE = MacroService()


def build_session() -> requests.Session:
    return DEFAULT_MACRO_SERVICE.build_session()


def get_fred_series_json(series_id: str, api_key: str, session: requests.Session) -> pd.DataFrame:
    return _get_fred_series_json(series_id=series_id, api_key=api_key, session=session)


def get_fred_series_csv(series_id: str, session: requests.Session) -> pd.DataFrame:
    return _get_fred_series_csv(series_id=series_id, session=session)


def get_fred_series(series_id: str) -> pd.DataFrame:
    return DEFAULT_MACRO_SERVICE.get_fred_series(series_id)


def get_worldbank_inflation() -> pd.DataFrame:
    return DEFAULT_MACRO_SERVICE.get_worldbank_inflation()


def get_worldbank_fx() -> pd.DataFrame:
    return DEFAULT_MACRO_SERVICE.get_worldbank_fx()


def get_yfinance_usdcop() -> float:
    return DEFAULT_MACRO_SERVICE.get_yfinance_usdcop()


def get_github_macro_cache() -> dict:
    return DEFAULT_MACRO_SERVICE.get_github_macro_cache()


def get_macro_snapshot() -> Dict[str, float]:
    return DEFAULT_MACRO_SERVICE.get_macro_snapshot()
