from __future__ import annotations

import os
from pathlib import Path
from datetime import date
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
REPORT_DIR = BASE_DIR / "report"

APP_TITLE = "RiskLab USTA - Dashboard de Riesgo Financiero"
APP_SUBTITLE = (
    "Proyecto integrador con APIs, análisis técnico, VaR, GARCH, CAPM, "
    "Markowitz, señales y benchmark."
)

ASSETS = {
    "Seven & i Holdings": {
        "ticker": "3382.T",
        "country": "Japón",
        "benchmark_local": "^N225",
    },
    "Alimentation Couche-Tard": {
        "ticker": "ATD.TO",
        "country": "Canadá",
        "benchmark_local": "^GSPTSE",
    },
    "FEMSA": {
        "ticker": "FEMSAUBD.MX",
        "country": "México",
        "benchmark_local": "^MXX",
    },
    "BP": {
        "ticker": "BP.L",
        "country": "Reino Unido",
        "benchmark_local": "^FTSE",
    },
    "Carrefour": {
        "ticker": "CA.PA",
        "country": "Francia",
        "benchmark_local": "^FCHI",
    },
}

ASSET_TICKERS = {name: meta["ticker"] for name, meta in ASSETS.items()}
TICKER_TO_NAME = {meta["ticker"]: name for name, meta in ASSETS.items()}

GLOBAL_BENCHMARK = "ACWI"

FRED_SERIES = {
    "risk_free_rate": "DGS3MO",
    "inflation": "CPIAUCSL",
    "cop_usd": "COLCCUSMA02STM",
}

DEFAULT_START_DATE = os.getenv("DEFAULT_START_DATE", "2021-01-01")
DEFAULT_END_DATE = os.getenv("DEFAULT_END_DATE") or str(date.today())

TRADING_DAYS = 252

def ensure_project_dirs() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

def get_asset_names() -> list[str]:
    return list(ASSETS.keys())

def get_asset_tickers() -> list[str]:
    return list(ASSET_TICKERS.values())

def get_local_benchmark(asset_name: str) -> str:
    return ASSETS[asset_name]["benchmark_local"]

def get_ticker(asset_name: str) -> str:
    return ASSETS[asset_name]["ticker"]
