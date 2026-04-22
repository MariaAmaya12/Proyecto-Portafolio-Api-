from __future__ import annotations

from datetime import timedelta

import pandas as pd


def yfinance_exclusive_end(end: str) -> str:
    return (pd.to_datetime(end).date() + timedelta(days=1)).isoformat()
