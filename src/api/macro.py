from __future__ import annotations

from typing import Any, Dict

import streamlit as st

from src.api.backend_client import backend_get, friendly_error_message


@st.cache_data(show_spinner=False, ttl=3600)
def _macro_snapshot_cached() -> Dict[str, Any]:
    return backend_get("/macro/snapshot")


def macro_snapshot() -> Dict[str, Any]:
    try:
        return _macro_snapshot_cached()
    except Exception as exc:
        st.warning(friendly_error_message(exc))
        return {
            "risk_free_rate_pct": 3.0,
            "inflation_yoy": float("nan"),
            "cop_per_usd": float("nan"),
            "usdcop_market": float("nan"),
            "source": "backend_unavailable",
            "last_updated": None,
        }
