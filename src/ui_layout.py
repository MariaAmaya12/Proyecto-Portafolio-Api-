from __future__ import annotations

from collections.abc import Mapping
import pandas as pd
import streamlit as st
from streamlit.delta_generator import context_dg_stack

from src.app_state import (
    clear_active_portfolio_config,
    get_portfolio_config,
    is_portfolio_config_ready,
    reset_portfolio_config,
    save_portfolio_config,
)
from src.auth import AUTH_SESSION_KEY, AUTH_USER_SESSION_KEY
from src.config import DEFAULT_END_DATE, GLOBAL_BENCHMARK


SHOW_PORTFOLIO_CONFIG_SESSION_KEY = "show_portfolio_configurator"
PORTFOLIO_SAVE_MESSAGE_SESSION_KEY = "portfolio_save_message"
SELECTED_ASSET_PANEL_MESSAGE_KEY = "selected_asset_panel_message"
_MODULE_PARAMS_CONTAINER = None

MODULE_REGISTRY = [
    {"id": "M1", "label": "M1 Análisis técnico", "path": "pages/01_tecnico.py"},
    {"id": "M2", "label": "M2 Rendimientos", "path": "pages/02_rendimientos.py"},
    {"id": "M3", "label": "M3 GARCH", "path": "pages/03_garch.py"},
    {"id": "M4", "label": "M4 CAPM y Beta", "path": "pages/04_capm.py"},
    {"id": "M5", "label": "M5 VaR/CVaR", "path": "pages/05_var_cvar.py"},
    {"id": "M6", "label": "M6 Markowitz", "path": "pages/06_markowitz.py"},
    {"id": "M7", "label": "M7 Señales", "path": "pages/07_senales.py"},
    {"id": "M8", "label": "M8 Macro y Benchmark", "path": "pages/08_macro_benchmark.py"},
    {"id": "M9", "label": "M9 Panel de decisión", "path": "pages/09_panel_decision.py"},
    {"id": "M10", "label": "M10 Modelos financieros", "path": "pages/10_modelos_financieros.py"},
]
MODULE_PAGE_LINKS = {module["label"]: module["path"] for module in MODULE_REGISTRY}
MODULES_BY_ID = {module["id"]: module for module in MODULE_REGISTRY}


def sanitize_text(text: object) -> str:
    if text is None:
        return ""
    return str(text).replace("<", "").replace(">", "")


def display_horizon(value: object) -> str:
    return str(value or "1 año").replace("anos", "años").replace("ano", "año")


def selected_asset_names() -> list[str]:
    config = get_portfolio_config()
    names = config.get("selected_asset_names") or []
    return [str(name) for name in names]


def selected_tickers() -> list[str]:
    config = get_portfolio_config()
    tickers = config.get("selected_tickers") or []
    return [str(ticker) for ticker in tickers]


def configured_assets(assets: Mapping[str, dict]) -> dict:
    names = selected_asset_names()
    filtered = {name: assets[name] for name in names if name in assets}
    return filtered or dict(assets)


def configured_period(default_start=None, default_end=None) -> tuple[str, object, object]:
    config = get_portfolio_config()
    horizon = config.get("selected_horizon") or "1 año"
    end_ref = pd.to_datetime(default_end or DEFAULT_END_DATE).date()
    fallback_start = pd.to_datetime(default_start).date() if default_start is not None else (
        pd.Timestamp(end_ref) - pd.DateOffset(years=1)
    ).date()

    if horizon == "1 mes":
        start = (pd.Timestamp(end_ref) - pd.DateOffset(months=1)).date()
    elif horizon == "Trimestre":
        start = (pd.Timestamp(end_ref) - pd.DateOffset(months=3)).date()
    elif horizon in ("Semestre", "6 meses"):
        start = (pd.Timestamp(end_ref) - pd.DateOffset(months=6)).date()
    elif horizon in ("1 año", "1 ano"):
        start = (pd.Timestamp(end_ref) - pd.DateOffset(years=1)).date()
    elif horizon in ("2 años", "2 anos"):
        start = (pd.Timestamp(end_ref) - pd.DateOffset(years=2)).date()
    elif horizon in ("3 años", "3 anos"):
        start = (pd.Timestamp(end_ref) - pd.DateOffset(years=3)).date()
    elif horizon in ("5 años", "5 anos"):
        start = (pd.Timestamp(end_ref) - pd.DateOffset(years=5)).date()
    else:
        start = fallback_start

    return display_horizon(horizon), start, end_ref


def render_selected_asset_card(assets: Mapping[str, dict], *, key: str, default_index: int = 0) -> tuple[str, str]:
    asset_options = list(assets.keys())
    if not asset_options:
        st.error("No hay activos configurados para analizar.")
        st.stop()

    selected_index = min(default_index, len(asset_options) - 1)
    st.markdown("### Activo seleccionado")
    with st.container(border=True):
        selector_col, ticker_col, period_col = st.columns([1.5, 0.75, 1.1])
        with selector_col:
            asset_name = st.selectbox("Activo", asset_options, index=selected_index, key=key)

        ticker = assets[asset_name].get("ticker", asset_name)

        with ticker_col:
            st.markdown("**Ticker**")
            st.caption(str(ticker))

        horizon, start_date, end_date = configured_period()
        with period_col:
            st.markdown("**Periodo analizado**")
            st.caption(f"{horizon}: {start_date} a {end_date}")

    return asset_name, str(ticker)


def render_portfolio_summary_card(assets: Mapping[str, dict]) -> None:
    config = get_portfolio_config()
    horizon, start_date, end_date = configured_period()
    tickers = config.get("selected_tickers") or [meta.get("ticker", name) for name, meta in assets.items()]
    asset_names = config.get("selected_asset_names") or list(assets.keys())

    st.markdown("### Portafolio analizado")
    with st.container(border=True):
        name_col, ticker_col, period_col = st.columns([1.25, 1.2, 1])
        with name_col:
            st.markdown("**Activos**")
            st.caption(", ".join(str(name) for name in asset_names))
        with ticker_col:
            st.markdown("**Tickers**")
            st.caption(", ".join(str(ticker) for ticker in tickers))
        with period_col:
            st.markdown("**Periodo analizado**")
            st.caption(f"{horizon}: {start_date} a {end_date}")


def _module_short_label(module: str) -> str:
    code, name = module.split(" ", maxsplit=1)
    return f"{code} {name}"


def _selected_modules_from_state() -> list[str]:
    modules = st.session_state.get("selected_modules", [])
    return [str(module) for module in modules]


def _module_entry(module: str) -> dict | None:
    module_id = module.split(" ", maxsplit=1)[0]
    if module_id in MODULES_BY_ID:
        return MODULES_BY_ID[module_id]
    return next((entry for entry in MODULE_REGISTRY if entry["label"] == module), None)


def inject_app_shell_css() -> None:
    st.markdown(
        """
        <style>
        section[data-testid="stSidebar"] {
            display: none;
        }
        div[data-testid="stSidebarCollapsedControl"] {
            display: none;
        }
        .main .block-container {
            max-width: 1240px;
            padding-top: 1.15rem;
        }
        .active-app-shell {
            background: #ffffff;
            border: 1px solid rgba(239, 111, 97, 0.22);
            border-radius: 20px;
            box-shadow: 0 12px 34px rgba(15, 23, 42, 0.08);
            margin: 0.25rem 0 1rem;
            padding: 1rem;
        }
        .active-app-title {
            color: #0f172a;
            font-size: 1.05rem;
            font-weight: 900;
            line-height: 1.2;
        }
        .active-app-meta {
            color: #64748b;
            font-size: 0.84rem;
            font-weight: 650;
            line-height: 1.35;
            margin-top: 0.2rem;
        }
        .module-page-heading {
            margin: 0 0 0.9rem;
        }
        .module-page-heading h1 {
            color: #0f172a;
            font-size: 1.65rem !important;
            font-weight: 900 !important;
            letter-spacing: 0 !important;
            line-height: 1.15 !important;
            margin: 0 0 0.25rem !important;
        }
        .module-page-heading p {
            color: #64748b;
            font-size: 0.92rem;
            line-height: 1.45;
            margin: 0;
        }
        .portfolio-options-heading {
            color: #0f172a;
            font-size: 0.98rem;
            font-weight: 850;
            margin-bottom: 0.15rem;
        }
        .portfolio-options-subtitle {
            color: #64748b;
            font-size: 0.78rem;
            line-height: 1.35;
            margin-bottom: 0.8rem;
        }
        .portfolio-options-divider {
            border-top: 1px solid #e2e8f0;
            margin: 0.85rem 0;
        }
        .module-nav-title {
            color: #475569;
            font-size: 0.76rem;
            font-weight: 900;
            letter-spacing: 0.08em;
            margin-bottom: 0.55rem;
            text-transform: uppercase;
        }
        div[data-testid="stVerticalBlockBorderWrapper"] {
            border-radius: 16px;
        }
        div[data-testid="stPageLink"] a {
            border: 1px solid #e2e8f0;
            border-radius: 999px;
            color: #334155;
            font-size: 0.82rem;
            font-weight: 800;
            min-height: 2.2rem;
            padding: 0.35rem 0.65rem;
        }
        div[data-testid="stPageLink"] a:hover {
            border-color: #ef6f61;
            color: #be3f34;
        }
        div[data-testid="stExpander"] {
            border-radius: 16px;
            box-shadow: 0 4px 14px rgba(15, 23, 42, 0.04);
        }
        @media (max-width: 900px) {
            .module-page-heading h1 {
                font-size: 1.35rem !important;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _show_portfolio_config() -> None:
    st.session_state[SHOW_PORTFOLIO_CONFIG_SESSION_KEY] = True
    st.session_state["onboarding_step"] = "assets"
    st.switch_page("app.py")


def _reset_portfolio_config() -> None:
    reset_portfolio_config()
    st.session_state[SHOW_PORTFOLIO_CONFIG_SESSION_KEY] = True
    st.session_state["onboarding_step"] = "welcome"
    st.session_state.pop(PORTFOLIO_SAVE_MESSAGE_SESSION_KEY, None)
    st.switch_page("app.py")


def _logout_current_user() -> None:
    clear_active_portfolio_config()
    st.session_state.pop("active_portfolio_user", None)
    st.session_state.pop(AUTH_SESSION_KEY, None)
    st.session_state.pop(AUTH_USER_SESSION_KEY, None)
    st.session_state[SHOW_PORTFOLIO_CONFIG_SESSION_KEY] = True
    st.session_state["onboarding_step"] = "welcome"
    st.switch_page("app.py")


def _render_active_portfolio_header(config: dict) -> None:
    selected = config.get("selected_tickers", [])
    modules = config.get("selected_modules", [])
    st.markdown(
        f"""
        <div class="active-app-shell">
            <div class="active-app-title">{sanitize_text(config.get("portfolio_name", "Portafolio activo"))}</div>
            <div class="active-app-meta">
                {len(selected)} activos - Horizonte {sanitize_text(display_horizon(config.get("selected_horizon")))} - {len(modules)} módulos - Benchmark {sanitize_text(GLOBAL_BENCHMARK)}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_module_links(config: dict) -> None:
    modules = [entry for module in _selected_modules_from_state() if (entry := _module_entry(module))]
    if not modules:
        return

    st.markdown(
        """
        <div class="portfolio-options-divider"></div>
        <div class="module-nav-title">Módulos seleccionados</div>
        """,
        unsafe_allow_html=True,
    )
    for module in modules:
        st.page_link(
            module["path"],
            label=_module_short_label(module["label"]),
            use_container_width=True,
        )


def _remove_selected_asset(ticker_to_remove: str) -> None:
    config = get_portfolio_config()
    tickers = [str(ticker) for ticker in config.get("selected_tickers", [])]
    names = [str(name) for name in config.get("selected_asset_names", [])]

    if len(tickers) <= 1:
        st.session_state[SELECTED_ASSET_PANEL_MESSAGE_KEY] = (
            "El portafolio debe conservar al menos un activo. Agrega otro activo antes de eliminar este."
        )
        st.rerun()

    pairs = []
    for index, ticker in enumerate(tickers):
        if ticker == ticker_to_remove:
            continue
        asset_name = names[index] if index < len(names) else ticker
        pairs.append((asset_name, ticker))

    remaining_names = [name for name, _ in pairs]
    remaining_tickers = [ticker for _, ticker in pairs]
    if remaining_tickers:
        current_weights = config.get("selected_weights") or {}
        remaining_weights = {
            ticker: float(current_weights.get(ticker, 0.0) or 0.0)
            for ticker in remaining_tickers
        }
        weight_sum = sum(remaining_weights.values())
        if weight_sum > 0:
            selected_weights = {
                ticker: weight / weight_sum
                for ticker, weight in remaining_weights.items()
            }
        else:
            equal_weight = 1 / len(remaining_tickers)
            selected_weights = {ticker: equal_weight for ticker in remaining_tickers}
        save_portfolio_config(
            {
                **config,
                "selected_tickers": remaining_tickers,
                "selected_asset_names": remaining_names,
                "selected_weights": selected_weights,
            }
        )
        st.rerun()

    empty_config = {
        **config,
        "selected_tickers": [],
        "selected_asset_names": [],
        "selected_weights": {},
        "portfolio_config_ready": False,
    }
    st.session_state["portfolio_config"] = empty_config
    for key, value in empty_config.items():
        st.session_state[key] = value
    st.session_state[SHOW_PORTFOLIO_CONFIG_SESSION_KEY] = True
    st.session_state["onboarding_step"] = "assets"
    st.session_state[PORTFOLIO_SAVE_MESSAGE_SESSION_KEY] = (
        "El portafolio quedó sin activos. Selecciona al menos 1 activo para continuar."
    )
    st.switch_page("app.py")


def render_selected_assets_panel() -> None:
    config = get_portfolio_config()
    tickers = [str(ticker) for ticker in config.get("selected_tickers", [])]
    names = [str(name) for name in config.get("selected_asset_names", [])]

    st.markdown(
        """
        <div class="portfolio-options-divider"></div>
        <div class="module-nav-title">Activos seleccionados</div>
        """,
        unsafe_allow_html=True,
    )
    panel_message = st.session_state.pop(SELECTED_ASSET_PANEL_MESSAGE_KEY, None)
    if panel_message:
        st.warning(panel_message)

    if not tickers:
        st.warning("No hay activos seleccionados. Vuelve a configuración para continuar.")
        if st.button("Volver a configuración", key="selected_assets_back_to_config", use_container_width=True):
            _show_portfolio_config()
        return

    for index, ticker in enumerate(tickers):
        name = names[index] if index < len(names) else ticker
        with st.container(border=True):
            st.markdown(f"**{sanitize_text(name)}**")
            st.caption(sanitize_text(ticker))
            st.button(
                "Eliminar",
                key=f"remove_selected_asset_{ticker}",
                use_container_width=True,
                on_click=_remove_selected_asset,
                args=(ticker,),
            )


def _render_options_panel(config: dict) -> None:
    global _MODULE_PARAMS_CONTAINER

    with st.container(border=True):
        st.markdown(
            """
            <div class="portfolio-options-heading">Opciones del portafolio</div>
            <div class="portfolio-options-subtitle">Ajustes rápidos para esta sesión.</div>
            """,
            unsafe_allow_html=True,
        )
        st.button(
            "Editar configuración",
            type="primary",
            use_container_width=True,
            on_click=_show_portfolio_config,
        )
        if st.button("Restablecer", key="shell_reset", use_container_width=True):
            _reset_portfolio_config()
        if st.button("Cerrar sesión", key="shell_logout", use_container_width=True):
            _logout_current_user()

        _render_module_links(config)
        render_selected_assets_panel()
        st.markdown(
            """
            <div class="portfolio-options-divider"></div>
            """,
            unsafe_allow_html=True,
        )
        _MODULE_PARAMS_CONTAINER = st.expander("Parámetros del módulo", expanded=True)


def module_params():
    if _MODULE_PARAMS_CONTAINER is None:
        return st.container()
    return _MODULE_PARAMS_CONTAINER


def _activate_main_column(main_col) -> None:
    context_stack = context_dg_stack.get()
    if context_stack and context_stack[-1] is main_col:
        return
    context_dg_stack.set((*context_stack, main_col))


def render_app_shell(page_title: str, page_subtitle: str | None = None):
    inject_app_shell_css()

    if not is_portfolio_config_ready():
        st.warning("Primero configura y guarda un portafolio desde Inicio para usar este módulo.")
        st.page_link("app.py", label="Ir a Inicio", use_container_width=False)
        st.stop()

    config = get_portfolio_config()
    _render_active_portfolio_header(config)

    main_col, options_col = st.columns([3.2, 1.05], gap="large")
    _activate_main_column(main_col)

    with options_col:
        _render_options_panel(config)

    st.markdown(
        f"""
        <div class="module-page-heading">
            <h1>{sanitize_text(page_title)}</h1>
            {f'<p>{sanitize_text(page_subtitle)}</p>' if page_subtitle else ''}
        </div>
        """,
        unsafe_allow_html=True,
    )
    return main_col, module_params()


