from __future__ import annotations

import streamlit as st

PORTFOLIO_CONFIG_SESSION_KEY = "portfolio_config"


def get_default_portfolio_config() -> dict:
    return {
        "portfolio_name": "Portafolio base del proyecto",
        "selected_tickers": ["3382.T", "ATD.TO", "FEMSAUBD.MX", "BP.L", "CA.PA"],
        "selected_asset_names": [
            "Seven & i Holdings",
            "Alimentation Couche-Tard",
            "FEMSA",
            "BP",
            "Carrefour",
        ],
        "selected_weights": {
            "3382.T": 0.2,
            "ATD.TO": 0.2,
            "FEMSAUBD.MX": 0.2,
            "BP.L": 0.2,
            "CA.PA": 0.2,
        },
        "selected_horizon": "1 año",
        "selected_modules": [
            "M1 Análisis técnico",
            "M2 Rendimientos",
            "M3 GARCH",
            "M4 CAPM y Beta",
            "M5 VaR/CVaR",
            "M6 Markowitz",
            "M7 Señales",
            "M8 Macro y Benchmark",
            "M9 Panel de decisión",
            "M10 Modelos financieros",
        ],
        "portfolio_config_ready": False,
    }


def save_portfolio_config(config: dict) -> None:
    stored_config = {**config, "portfolio_config_ready": True}
    st.session_state[PORTFOLIO_CONFIG_SESSION_KEY] = stored_config

    for key, value in stored_config.items():
        st.session_state[key] = value


def get_portfolio_config() -> dict:
    return st.session_state.get(
        PORTFOLIO_CONFIG_SESSION_KEY,
        get_default_portfolio_config(),
    )


def is_portfolio_config_ready() -> bool:
    return bool(get_portfolio_config().get("portfolio_config_ready", False))


def reset_portfolio_config() -> None:
    st.session_state.pop(PORTFOLIO_CONFIG_SESSION_KEY, None)
    default_config = get_default_portfolio_config()

    for key, value in default_config.items():
        st.session_state[key] = value
