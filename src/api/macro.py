from __future__ import annotations

from typing import Any, Dict

import streamlit as st

from src.api.backend_client import backend_get


@st.cache_data(show_spinner=False, ttl=3600)
def macro_snapshot() -> Dict[str, Any]:
    return backend_get("/macro/snapshot")
