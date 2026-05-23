from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import streamlit as st

from src.auth import AUTH_USER_SESSION_KEY

PORTFOLIO_CONFIG_SESSION_KEY = "portfolio_config"
USER_PORTFOLIOS_PATH = Path("data") / "user_portfolios.json"
ACTIVE_USER_SESSION_KEY = "active_portfolio_user"

PORTFOLIO_CONFIG_KEYS = (
    "portfolio_name",
    "selected_tickers",
    "selected_asset_names",
    "selected_weights",
    "selected_horizon",
    "selected_modules",
    "portfolio_config_ready",
)


def get_default_portfolio_config() -> dict:
    return {
        "portfolio_name": "Portafolio RiskLab USTA",
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
            "M9 Panel de decision",
            "M10 Modelos financieros",
        ],
        "portfolio_config_ready": False,
    }


def save_portfolio_config(config: dict) -> None:
    stored_config = {**config, "portfolio_config_ready": True}
    st.session_state[PORTFOLIO_CONFIG_SESSION_KEY] = stored_config

    for key, value in stored_config.items():
        st.session_state[key] = value

    save_user_portfolio(stored_config)


def load_portfolio_config(config: dict) -> None:
    loaded_config = {**get_default_portfolio_config(), **config, "portfolio_config_ready": True}
    st.session_state[PORTFOLIO_CONFIG_SESSION_KEY] = loaded_config

    for key, value in loaded_config.items():
        st.session_state[key] = value


def get_portfolio_config() -> dict:
    return st.session_state.get(
        PORTFOLIO_CONFIG_SESSION_KEY,
        get_default_portfolio_config(),
    )


def is_portfolio_config_ready() -> bool:
    return bool(get_portfolio_config().get("portfolio_config_ready", False))


def reset_portfolio_config() -> None:
    clear_active_portfolio_config()


def current_username() -> str:
    username = st.session_state.get(AUTH_USER_SESSION_KEY)
    return str(username).strip() if username else "anonymous"


def clear_active_portfolio_config() -> None:
    st.session_state.pop(PORTFOLIO_CONFIG_SESSION_KEY, None)
    default_config = {**get_default_portfolio_config(), "portfolio_config_ready": False}
    for key in PORTFOLIO_CONFIG_KEYS:
        st.session_state[key] = default_config.get(key)


def _empty_store() -> dict:
    return {"users": {}}


def _read_portfolio_store() -> dict:
    if not USER_PORTFOLIOS_PATH.exists():
        return _empty_store()
    try:
        with USER_PORTFOLIOS_PATH.open("r", encoding="utf-8") as file:
            data = json.load(file)
    except (json.JSONDecodeError, OSError):
        return _empty_store()
    if not isinstance(data, dict):
        return _empty_store()
    data.setdefault("users", {})
    return data


def _write_portfolio_store(store: dict) -> None:
    USER_PORTFOLIOS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with USER_PORTFOLIOS_PATH.open("w", encoding="utf-8") as file:
        json.dump(store, file, ensure_ascii=True, indent=2)


def _persist_portfolio_sqlite(payload: dict) -> None:
    """Complementary SQLite persistence — does not replace the JSON flow."""
    try:
        from backend.database import SessionLocal
        from backend.models import Portfolio as _PortfolioORM
        db = SessionLocal()
        try:
            name = payload.get("portfolio_name") or "Portafolio RiskLab USTA"
            tickers_json = json.dumps(payload.get("selected_tickers", []))
            weights_json = json.dumps(payload.get("selected_weights", {}))
            horizon = str(payload.get("selected_horizon", ""))
            existing = db.query(_PortfolioORM).filter_by(name=name).first()
            if existing:
                existing.tickers = tickers_json
                existing.weights = weights_json
                existing.horizon = horizon
            else:
                db.add(_PortfolioORM(
                    name=name,
                    tickers=tickers_json,
                    weights=weights_json,
                    horizon=horizon,
                ))
            db.commit()
        finally:
            db.close()
    except Exception:
        pass


def _portfolio_payload(config: dict) -> dict:
    return {
        "portfolio_name": config.get("portfolio_name") or "Portafolio RiskLab USTA",
        "selected_tickers": list(config.get("selected_tickers") or []),
        "selected_asset_names": list(config.get("selected_asset_names") or []),
        "selected_weights": dict(config.get("selected_weights") or {}),
        "selected_horizon": config.get("selected_horizon") or "1 año",
        "selected_modules": list(config.get("selected_modules") or []),
    }


def save_user_portfolio(config: dict, username: str | None = None) -> dict:
    user = username or current_username()
    now = datetime.now(timezone.utc).isoformat()
    payload = _portfolio_payload(config)
    store = _read_portfolio_store()
    portfolios = store.setdefault("users", {}).setdefault(user, [])

    existing = next(
        (
            portfolio
            for portfolio in portfolios
            if portfolio.get("portfolio_name") == payload["portfolio_name"]
        ),
        None,
    )
    if existing is None:
        saved = {
            "portfolio_id": str(uuid4()),
            "created_at": now,
            "updated_at": now,
            **payload,
        }
        portfolios.append(saved)
    else:
        existing.update(payload)
        existing["updated_at"] = now
        existing.setdefault("created_at", now)
        existing.setdefault("portfolio_id", str(uuid4()))
        saved = existing

    _write_portfolio_store(store)
    _persist_portfolio_sqlite(payload)
    return dict(saved)


def list_user_portfolios(username: str | None = None) -> list[dict]:
    user = username or current_username()
    store = _read_portfolio_store()
    portfolios = store.get("users", {}).get(user, [])
    if not isinstance(portfolios, list):
        return []
    normalized = []
    for index, portfolio in enumerate(portfolios):
        item = dict(portfolio)
        item.setdefault("portfolio_id", f"legacy-{index}")
        normalized.append(item)
    return sorted(
        normalized,
        key=lambda item: str(item.get("updated_at", "")),
        reverse=True,
    )


def has_saved_portfolios(username: str | None = None) -> bool:
    return bool(list_user_portfolios(username))


def load_user_portfolio(portfolio_id: str, username: str | None = None) -> dict | None:
    for portfolio in list_user_portfolios(username):
        if portfolio.get("portfolio_id") == portfolio_id:
            return {
                **_portfolio_payload(portfolio),
                "portfolio_config_ready": True,
            }
    return None


def delete_user_portfolio(portfolio_id: str, username: str | None = None) -> bool:
    user = username or current_username()
    store = _read_portfolio_store()
    portfolios = store.get("users", {}).get(user, [])
    kept = [
        portfolio
        for portfolio in portfolios
        if portfolio.get("portfolio_id") != portfolio_id
    ]
    if len(kept) == len(portfolios):
        return False
    store.setdefault("users", {})[user] = kept
    _write_portfolio_store(store)
    return True


def mark_active_user_initialized() -> None:
    st.session_state[ACTIVE_USER_SESSION_KEY] = current_username()


def active_user_initialized() -> bool:
    return st.session_state.get(ACTIVE_USER_SESSION_KEY) == current_username()

