import streamlit as st
import pandas as pd

from src.app_state import (
    active_user_initialized,
    clear_active_portfolio_config,
    delete_user_portfolio,
    get_default_portfolio_config,
    get_portfolio_config,
    has_saved_portfolios,
    is_portfolio_config_ready,
    list_user_portfolios,
    load_user_portfolio,
    load_portfolio_config,
    mark_active_user_initialized,
    reset_portfolio_config,
    save_portfolio_config,
)
from src.auth import AUTH_SESSION_KEY, AUTH_USER_SESSION_KEY, require_login
from src.config import (
    APP_TITLE,
    DEFAULT_END_DATE,
    GLOBAL_BENCHMARK,
    ensure_project_dirs,
)
from src.api.backend_client import (
    backend_base_url,
    fetch_market_bundle_from_backend,
    last_backend_call,
)
from src.date_utils import yfinance_exclusive_end
from src.download import data_error_message
from src.preprocess import (
    equal_weight_portfolio,
    annualize_return,
    annualize_volatility,
)
from src.plots import plot_normalized_prices
from src.ui_components import (
    kpi_card,
    render_explanation_expander,
    render_section,
    render_table,
)
from src.ui_style import apply_global_typography

RESERVED_PRICE_COLUMNS = {"index", "date"}
AVAILABLE_ASSETS = {
    "Seven & i Holdings": "3382.T",
    "Alimentation Couche-Tard": "ATD.TO",
    "FEMSA": "FEMSAUBD.MX",
    "BP": "BP.L",
    "Carrefour": "CA.PA",
}
ASSET_DESCRIPTIONS = {
    "Seven & i Holdings": "Retail y tiendas de conveniencia con exposición a Japón.",
    "Alimentation Couche-Tard": "Operador global de conveniencia y combustibles.",
    "FEMSA": "Consumo, comercio y bebidas con presencia latinoamericana.",
    "BP": "Energía integrada con exposición a petróleo, gas y transición energética.",
    "Carrefour": "Retail alimentario europeo con operación multiformato.",
}
ASSET_DETAILS = {
    "Seven & i Holdings": {
        "market": "Japón",
        "sector": "Retail y tiendas de conveniencia",
        "role": "Exposición defensiva a consumo asiático.",
        "exposure": "Consumo básico y conveniencia en Asia.",
        "relevance": "Aporta diversificación geográfica y sectorial frente a activos europeos, latinoamericanos y energéticos.",
        "description": "Compañía japonesa de retail y tiendas de conveniencia. Aporta exposición al mercado asiático y al sector consumo defensivo.",
    },
    "Alimentation Couche-Tard": {
        "market": "Canadá / global",
        "sector": "Conveniencia y combustibles",
        "role": "Consumo defensivo con ingresos diversificados.",
        "exposure": "Operación global de tiendas de conveniencia y estaciones de servicio.",
        "relevance": "Complementa el portafolio con ingresos recurrentes y presencia en Norteamérica y otros mercados desarrollados.",
        "description": "Operador de tiendas de conveniencia y combustibles con presencia internacional. Aporta estabilidad operativa y exposición a consumo recurrente.",
    },
    "FEMSA": {
        "market": "México / Latinoamérica",
        "sector": "Comercio, bebidas y servicios",
        "role": "Exposición regional a comercio y bebidas.",
        "exposure": "Consumo latinoamericano y redes comerciales de gran escala.",
        "relevance": "Introduce diversificación regional y sensibilidad a dinámicas de consumo en mercados emergentes.",
        "description": "Grupo mexicano vinculado a comercio, bebidas y servicios. Aporta diversificación latinoamericana y consumo de gran escala.",
    },
    "BP": {
        "market": "Reino Unido / global",
        "sector": "Energía integrada",
        "role": "Exposición al ciclo energético.",
        "exposure": "Petróleo, gas, refinación y transición energética.",
        "relevance": "Añade exposición a materias primas y un perfil de riesgo distinto al consumo defensivo.",
        "description": "Empresa integrada de energía con operaciones globales. Aporta sensibilidad a materias primas, petróleo, gas y transición energética.",
    },
    "Carrefour": {
        "market": "Francia / Europa",
        "sector": "Retail alimentario",
        "role": "Consumo básico europeo.",
        "exposure": "Supermercados e hipermercados en Europa y mercados relacionados.",
        "relevance": "Refuerza la diversificación defensiva del portafolio con exposición al consumo básico europeo.",
        "description": "Retail alimentario europeo con operación multiformato. Aporta exposición al consumo defensivo y al mercado europeo.",
    },
}
HORIZON_OPTIONS = ["6 meses", "1 año", "2 años", "5 años"]
MODULE_OPTIONS = [
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
]
MODULE_DESCRIPTIONS = {
    "M1": "Estudia tendencia, medias móviles, RSI, Bollinger, MACD y señales del precio.",
    "M2": "Analiza rendimientos, distribución, estadísticos y normalidad.",
    "M3": "Modela volatilidad condicional y pronóstico de riesgo.",
    "M4": "Estima sensibilidad frente al benchmark y riesgo sistemático.",
    "M5": "Estima pérdidas potenciales bajo distintos enfoques de riesgo.",
    "M6": "Optimiza portafolios según riesgo-retorno.",
    "M7": "Resume alertas técnicas y criterios de decisión.",
    "M8": "Compara el portafolio con contexto macro y referencia global.",
    "M9": "Consolida métricas para una lectura ejecutiva.",
    "M10": "Reúne modelos financieros avanzados disponibles.",
}
MODULE_PAGE_LINKS = {
    "M1 Análisis técnico": "pages/01_tecnico.py",
    "M2 Rendimientos": "pages/02_rendimientos.py",
    "M3 GARCH": "pages/03_garch.py",
    "M4 CAPM y Beta": "pages/04_capm.py",
    "M5 VaR/CVaR": "pages/05_var_cvar.py",
    "M6 Markowitz": "pages/06_markowitz.py",
    "M7 Señales": "pages/07_senales.py",
    "M8 Macro y Benchmark": "pages/08_macro_benchmark.py",
    "M9 Panel de decisión": "pages/09_panel_decision.py",
    "M10 Modelos financieros": "pages/10_modelos_financieros.py",
}
SHOW_PORTFOLIO_CONFIG_SESSION_KEY = "show_portfolio_configurator"
PORTFOLIO_SAVE_MESSAGE_SESSION_KEY = "portfolio_save_message"
ONBOARDING_STEP_SESSION_KEY = "onboarding_step"


def normalize_market_frame(frame: pd.DataFrame, label: str, stop_on_invalid: bool = False) -> tuple[pd.DataFrame, str | None]:
    if frame is None or frame.empty:
        return pd.DataFrame(), None

    original_index_type = type(frame.index).__name__
    normalized = frame.copy()

    if isinstance(normalized.index, pd.DatetimeIndex):
        pass
    else:
        date_col = next((col for col in ("Date", "date") if col in normalized.columns), None)
        if date_col is not None:
            normalized[date_col] = pd.to_datetime(normalized[date_col], errors="coerce")
            normalized = normalized.dropna(subset=[date_col]).set_index(date_col)
        elif isinstance(normalized.index, pd.RangeIndex):
            message = (
                f"{label}: indice RangeIndex sin columna Date/date. "
                "No se pueden recuperar fechas reales; se detienen los calculos para evitar fechas 1970."
            )
            if stop_on_invalid:
                st.error(message)
                st.stop()
            return normalized, message
        else:
            parsed_index = pd.to_datetime(normalized.index, errors="coerce")
            if pd.isna(parsed_index).any():
                message = f"{label}: indice {original_index_type} no contiene fechas validas."
                if stop_on_invalid:
                    st.error(message)
                    st.stop()
                return normalized, message
            normalized.index = parsed_index

    normalized = normalized[~normalized.index.isna()].sort_index()
    normalized = normalized.drop(columns=[c for c in normalized.columns if str(c).lower() in RESERVED_PRICE_COLUMNS], errors="ignore")
    normalized = normalized.apply(pd.to_numeric, errors="coerce")
    normalized = normalized.dropna(axis=1, how="all")

    if normalized.empty or normalized.dropna(how="all").empty:
        message = f"{label}: no quedaron columnas numericas validas despues de normalizar."
        if stop_on_invalid:
            st.error(message)
            st.stop()
        return normalized, message

    return normalized, None


# ---------------------------------------------------------
# Configuración inicial
# ---------------------------------------------------------
ensure_project_dirs()

st.set_page_config(
    page_title=APP_TITLE,
    page_icon=":bar_chart:",
    layout="wide",
    initial_sidebar_state="collapsed",
)
apply_global_typography()
require_login()

if not active_user_initialized():
    clear_active_portfolio_config()
    st.session_state[SHOW_PORTFOLIO_CONFIG_SESSION_KEY] = True
    st.session_state[ONBOARDING_STEP_SESSION_KEY] = (
        "portfolio_choice" if has_saved_portfolios() else "welcome"
    )
    st.session_state.pop(PORTFOLIO_SAVE_MESSAGE_SESSION_KEY, None)
    mark_active_user_initialized()

# ---------------------------------------------------------
# UI helpers
# ---------------------------------------------------------
def inject_ui_css():
    st.markdown(
        """
        <style>
        .section-box {
            background: #ffffff;
            border: 1px solid rgba(15, 23, 42, 0.08);
            border-radius: 18px;
            padding: 16px 18px;
            box-shadow: 0 4px 14px rgba(15, 23, 42, 0.06);
            margin-bottom: 0.8rem;
        }
        .section-title {
            font-size: 1rem;
            font-weight: 700;
            color: #0f172a;
            margin-bottom: 0.2rem;
        }
        .section-subtitle {
            font-size: 0.86rem;
            color: #64748b;
            line-height: 1.45;
        }
        .hero-subtitle {
            color: #475569;
            font-size: 1.02rem;
            line-height: 1.5;
            margin-top: -0.25rem;
            margin-bottom: 1.1rem;
        }
        .summary-chip {
            background: linear-gradient(180deg, #ffffff 0%, #f2f7ff 100%);
            border: 1px solid #c9ddfc;
            border-radius: 16px;
            padding: 14px 16px;
            min-height: 86px;
            box-shadow: 0 6px 18px rgba(37, 99, 235, 0.07);
            display: flex;
            flex-direction: column;
            justify-content: center;
        }
        .summary-chip-label {
            color: #5271a3;
            font-size: 0.72rem;
            font-weight: 800;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            margin-bottom: 0.35rem;
        }
        .summary-chip-value {
            color: #0f3d75;
            font-size: 0.98rem;
            font-weight: 750;
            line-height: 1.35;
            word-break: break-word;
        }
        .insight-box {
            background: linear-gradient(180deg, #f8fbff 0%, #eef6ff 100%);
            border: 1px solid #d5e4fb;
            border-radius: 16px;
            padding: 16px 18px;
            color: #334155;
            margin: 0.5rem 0 1rem 0;
            box-shadow: 0 4px 14px rgba(15, 23, 42, 0.04);
        }
        .insight-box.positive {
            background: linear-gradient(180deg, #f7fef9 0%, #edf9f1 100%);
            border-color: #c8ead4;
        }
        .insight-title {
            color: #274c77;
            font-weight: 800;
            margin-bottom: 0.45rem;
        }
        .insight-box.positive .insight-title {
            color: #166534;
        }
        .insight-box ul {
            margin: 0.2rem 0 0 1.1rem;
            padding: 0;
        }
        .insight-box li {
            margin-bottom: 0.35rem;
            line-height: 1.45;
        }
        section[data-testid="stSidebar"] div.stButton > button {
            border-radius: 999px;
            border: 1px solid #c9ddfc;
            background: #ffffff;
            color: #274c77;
            font-weight: 700;
            min-height: 2.25rem;
            height: auto;
            white-space: normal;
            line-height: 1.2;
            padding: 0.5rem 0.65rem;
            box-shadow: none;
        }
        section[data-testid="stSidebar"] div.stButton > button[kind="primary"] {
            background: #eaf3ff;
            border-color: #93c5fd;
            color: #0f3d75;
        }
        section[data-testid="stSidebar"] div.stButton > button:hover {
            border-color: #60a5fa;
            color: #0f3d75;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _inject_home_styles() -> None:
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
            max-width: 1180px;
            padding-top: 1.25rem;
        }
        .home-topbar {
            align-items: center;
            background: rgba(255, 255, 255, 0.92);
            border: 1px solid rgba(15, 23, 42, 0.08);
            border-radius: 18px;
            box-shadow: 0 10px 28px rgba(15, 23, 42, 0.07);
            display: flex;
            justify-content: space-between;
            gap: 1rem;
            margin-bottom: 1.2rem;
            padding: 0.85rem 1rem;
        }
        .home-topbar-title {
            color: #0f172a;
            font-size: 0.95rem;
            font-weight: 850;
            line-height: 1.2;
        }
        .home-topbar-meta {
            color: #64748b;
            font-size: 0.8rem;
            font-weight: 650;
            margin-top: 0.18rem;
        }
        .portfolio-builder-heading {
            margin-bottom: 0.35rem;
            text-align: center;
        }
        .portfolio-builder-eyebrow {
            color: #ef6f61;
            font-size: 0.76rem;
            font-weight: 850;
            letter-spacing: 0.08em;
            text-transform: uppercase;
        }
        .portfolio-builder-title {
            color: #0f172a;
            font-size: 1.55rem;
            font-weight: 900;
            letter-spacing: 0;
            line-height: 1.15;
            margin: 0.1rem 0 0.25rem;
        }
        .portfolio-builder-copy {
            color: #64748b;
            font-size: 0.9rem;
            line-height: 1.45;
            margin: 0 auto;
            max-width: 720px;
        }
        .portfolio-mode-card {
            margin-bottom: 0.35rem;
        }
        .portfolio-mode-card .stButton > button {
            align-items: flex-start;
            background: linear-gradient(180deg, #ffffff 0%, #fbfdff 100%);
            border: 1px solid #e2e8f0;
            border-radius: 18px;
            box-shadow: 0 8px 22px rgba(15, 23, 42, 0.05);
            color: #0f172a;
            font-size: 0.86rem;
            font-weight: 700;
            justify-content: flex-start;
            line-height: 1.38;
            min-height: 5.8rem;
            padding: 0.9rem 1rem;
            text-align: left;
            white-space: pre-line;
            width: 100%;
        }
        .portfolio-mode-card.selected .stButton > button {
            background: linear-gradient(180deg, #fff7f5 0%, #fff1ee 100%);
            border-color: #ef6f61;
            box-shadow: 0 14px 32px rgba(239, 111, 97, 0.18);
            color: #9f3128;
            font-weight: 850;
        }
        .portfolio-mode-card .stButton > button:hover {
            border-color: #ef6f61;
            color: #9f3128;
            transform: translateY(-1px);
        }
        .saved-portfolio-card {
            background: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 16px;
            box-shadow: 0 8px 22px rgba(15, 23, 42, 0.05);
            cursor: pointer;
            margin-bottom: 0;
            padding: 1rem;
            width: 100%;
        }
        .saved-portfolio-card.selected {
            background: linear-gradient(180deg, #fff7f5 0%, #ffffff 100%);
            border-color: #ef6f61;
            box-shadow: 0 14px 32px rgba(239, 111, 97, 0.16);
        }
        .saved-portfolio-link {
            color: inherit !important;
            display: block;
            margin-bottom: 0.9rem;
            text-decoration: none !important;
        }
        .saved-portfolio-link:hover {
            text-decoration: none !important;
        }
        .saved-portfolio-link:hover .saved-portfolio-card {
            border-color: #ef6f61;
            box-shadow: 0 14px 32px rgba(239, 111, 97, 0.12);
            transform: translateY(-1px);
        }
        div[class*="st-key-saved_portfolio_click_"] {
            position: relative;
        }
        div[class*="st-key-saved_portfolio_click_"] .stButton {
            bottom: 0;
            left: 0;
            position: absolute;
            right: 0;
            top: 0;
            z-index: 2;
        }
        div[class*="st-key-saved_portfolio_click_"] .stButton > button {
            height: 100%;
            min-height: 100%;
            opacity: 0;
            width: 100%;
        }
        div[class*="st-key-saved_portfolio_click_"]:has(.stButton > button:hover) .saved-portfolio-card {
            border-color: #ef6f61;
            box-shadow: 0 14px 32px rgba(239, 111, 97, 0.12);
            transform: translateY(-1px);
        }
        div[class*="st-key-saved_portfolio_click_"] div[class*="st-key-select_saved_portfolio_"] {
            bottom: 0;
            left: 0;
            position: absolute;
            right: 0;
            top: 0;
            z-index: 2;
        }
        div[class*="st-key-saved_portfolio_click_"] div[class*="st-key-select_saved_portfolio_"] .stButton {
            height: 100%;
            position: static;
        }
        div[class*="st-key-saved_portfolio_click_"] div[class*="st-key-select_saved_portfolio_"] .stButton > button {
            height: 100%;
            min-height: 100%;
            opacity: 0;
            width: 100%;
        }
        .saved-portfolio-name {
            color: #0f172a;
            font-size: 1rem;
            font-weight: 900;
            line-height: 1.25;
            margin-bottom: 0.35rem;
        }
        .saved-portfolio-meta {
            color: #64748b;
            font-size: 0.8rem;
            line-height: 1.4;
            margin: 0.18rem 0;
        }
        .saved-portfolio-meta strong {
            color: #334155;
        }
        .saved-portfolio-state {
            border-radius: 999px;
            display: inline-block;
            font-size: 0.72rem;
            font-weight: 850;
            margin: 0.45rem 0 0.55rem;
            padding: 0.22rem 0.55rem;
        }
        .saved-portfolio-state.selected {
            background: rgba(239, 111, 97, 0.14);
            color: #be3f34;
        }
        .saved-portfolio-state.idle {
            background: #f1f5f9;
            color: #64748b;
        }
        div[data-testid="stVerticalBlockBorderWrapper"]:has(.asset-card-click-target) {
            background: linear-gradient(180deg, #ffffff 0%, #fbfdff 100%);
            border: 1px solid #e2e8f0;
            border-radius: 18px;
            box-shadow: 0 10px 26px rgba(15, 23, 42, 0.06);
            margin-bottom: 1rem;
            overflow: hidden;
            position: relative;
            transition: border-color 0.18s ease, box-shadow 0.18s ease, transform 0.18s ease;
        }
        div[data-testid="stVerticalBlockBorderWrapper"]:has(.asset-card-click-target):hover {
            border-color: rgba(239, 111, 97, 0.58);
            box-shadow: 0 16px 34px rgba(15, 23, 42, 0.1);
            transform: translateY(-1px);
        }
        div[data-testid="stVerticalBlockBorderWrapper"]:has(.asset-card-click-target.selected) {
            background: linear-gradient(180deg, #fff8f6 0%, #ffffff 100%);
            border-color: rgba(239, 111, 97, 0.62);
            box-shadow: 0 16px 34px rgba(239, 111, 97, 0.14);
        }
        div[data-testid="stVerticalBlockBorderWrapper"]:has(.asset-card-click-target) > div {
            padding: 1.05rem 1.1rem 0.85rem;
        }
        .asset-card-click-target {
            display: block;
            left: 0;
            margin: 0;
            pointer-events: none;
            position: absolute;
            right: 0;
            text-decoration: none !important;
            top: 0;
            bottom: 3.25rem;
            z-index: 3;
        }
        div[data-testid="stVerticalBlockBorderWrapper"]:has(.asset-card-click-target) div[data-testid="stElementContainer"]:has(div[data-testid="stButton"]) {
            height: auto;
            left: 0;
            margin: 0;
            min-height: 0;
            opacity: 0;
            overflow: hidden;
            padding: 0;
            position: absolute;
            right: 0;
            top: 0;
            bottom: 3.25rem;
            z-index: 3;
        }
        div[data-testid="stVerticalBlockBorderWrapper"]:has(.asset-card-click-target) div[data-testid="stElementContainer"]:has(div[class*="st-key-asset_card_click_"]) {
            bottom: 3.25rem;
            height: auto !important;
            left: 0;
            margin: 0 !important;
            min-height: 0 !important;
            opacity: 0;
            overflow: hidden;
            padding: 0 !important;
            position: absolute;
            right: 0;
            top: 0;
            z-index: 3;
        }
        div[data-testid="stVerticalBlockBorderWrapper"]:has(.asset-card-click-target) div[data-testid="stButton"] {
            left: 0;
            margin: 0;
            min-height: 0;
            opacity: 0;
            overflow: hidden;
            padding: 0;
            position: absolute;
            right: 0;
            top: 0;
            bottom: 3.25rem;
            z-index: 3;
        }
        div[data-testid="stVerticalBlockBorderWrapper"]:has(.asset-card-click-target) div[data-testid="stButton"] > button {
            background: transparent;
            border: 0;
            box-shadow: none;
            cursor: pointer;
            margin: 0;
            min-height: 0;
            padding: 0;
            width: 100%;
            position: absolute;
            top: 0;
            right: 0;
            bottom: 3.25rem;
            left: 0;
        }
        div[data-testid="stVerticalBlockBorderWrapper"]:has(.asset-card-click-target) div[class*="st-key-asset_card_click_"] {
            bottom: 3.25rem;
            height: auto !important;
            left: 0;
            margin: 0 !important;
            min-height: 0 !important;
            opacity: 0;
            overflow: hidden;
            padding: 0 !important;
            position: absolute;
            right: 0;
            top: 0;
            z-index: 3;
        }
        div[data-testid="stVerticalBlockBorderWrapper"]:has(.asset-card-click-target) div[class*="st-key-asset_card_click_"] .stButton {
            bottom: 0 !important;
            left: 0 !important;
            margin: 0 !important;
            min-height: 0 !important;
            opacity: 0;
            padding: 0 !important;
            position: absolute !important;
            right: 0 !important;
            top: 0 !important;
        }
        div[data-testid="stVerticalBlockBorderWrapper"]:has(.asset-card-click-target) div[class*="st-key-asset_card_click_"] .stButton > button {
            background: transparent !important;
            border: 0 !important;
            bottom: 0 !important;
            box-shadow: none !important;
            color: transparent !important;
            cursor: pointer !important;
            left: 0 !important;
            margin: 0 !important;
            min-height: 0 !important;
            outline: 0 !important;
            padding: 0 !important;
            pointer-events: auto !important;
            position: absolute !important;
            right: 0 !important;
            top: 0 !important;
        }
        div[class*="st-key-asset_card_shell_"] {
            position: relative;
        }
        div[class*="st-key-asset_card_shell_"] div[data-testid="stElementContainer"]:has(div[class*="st-key-asset_card_click_"]) {
            bottom: 3.25rem !important;
            height: auto !important;
            left: 0 !important;
            margin: 0 !important;
            min-height: 0 !important;
            opacity: 0 !important;
            overflow: hidden !important;
            padding: 0 !important;
            position: absolute !important;
            right: 0 !important;
            top: 0 !important;
            z-index: 3 !important;
        }
        div[class*="st-key-asset_card_shell_"] div[class*="st-key-asset_card_click_"] {
            bottom: 3.25rem !important;
            left: 0 !important;
            margin: 0 !important;
            min-height: 0 !important;
            opacity: 0 !important;
            overflow: visible !important;
            padding: 0 !important;
            position: absolute !important;
            right: 0 !important;
            top: 0 !important;
            z-index: 4 !important;
        }
        div[class*="st-key-asset_card_shell_"] div[class*="st-key-asset_card_click_"] button {
            background: transparent !important;
            border: 0 !important;
            bottom: 0 !important;
            box-shadow: none !important;
            color: transparent !important;
            cursor: pointer !important;
            left: 0 !important;
            margin: 0 !important;
            min-height: 0 !important;
            outline: 0 !important;
            padding: 0 !important;
            pointer-events: auto !important;
            position: absolute !important;
            right: 0 !important;
            top: 0 !important;
        }
        div[data-testid="stVerticalBlockBorderWrapper"]:has(.asset-card-click-target) div[data-testid="stCheckbox"] {
            position: relative;
            z-index: 20;
        }
        div[data-testid="stVerticalBlockBorderWrapper"]:has(.asset-card-click-target) div[data-testid="stCheckbox"] * {
            position: relative;
            z-index: 21;
        }
        .asset-card-title {
            color: #0f172a;
            font-size: 1rem;
            font-weight: 900;
            line-height: 1.22;
            margin-bottom: 0.25rem;
            padding-right: 0.25rem;
        }
        .asset-card-ticker {
            color: #ef6f61;
            font-size: 0.78rem;
            font-weight: 850;
            letter-spacing: 0.04em;
            margin: 0.15rem 0 0.55rem;
        }
        .asset-card-copy {
            color: #64748b;
            font-size: 0.82rem;
            line-height: 1.45;
            min-height: 3.55rem;
        }
        .asset-card-status {
            border-radius: 999px;
            display: inline-block;
            font-size: 0.72rem;
            font-weight: 850;
            margin: 0.75rem 0 0.5rem;
            padding: 0.25rem 0.62rem;
        }
        .asset-card-status.selected {
            background: rgba(239, 111, 97, 0.14);
            color: #be3f34;
        }
        .asset-card-status.idle {
            background: #f1f5f9;
            color: #64748b;
        }
        .portfolio-config-summary {
            background: linear-gradient(180deg, #ffffff 0%, #fff7f5 100%);
            border: 1px solid rgba(239, 111, 97, 0.22);
            border-radius: 18px;
            box-shadow: 0 10px 28px rgba(15, 23, 42, 0.08);
            margin: 0.9rem 0 1.2rem;
            padding: 18px;
        }
        .portfolio-config-title {
            color: #0f172a;
            font-size: 1.05rem;
            font-weight: 900;
            margin-bottom: 0.35rem;
        }
        .portfolio-config-line {
            color: #475569;
            font-size: 0.88rem;
            line-height: 1.45;
            margin: 0.18rem 0;
        }
        .module-pill {
            background: #f8fafc;
            border: 1px solid #e2e8f0;
            border-radius: 999px;
            color: #475569;
            display: inline-block;
            font-size: 0.76rem;
            font-weight: 750;
            margin: 0.18rem 0.2rem 0 0;
            padding: 0.22rem 0.55rem;
        }
        .home-header {
            margin: 0.6rem auto 1rem 0;
            max-width: 720px;
            text-align: left;
        }
        .home-header h1 {
            color: #0f172a;
            font-size: 2rem;
            font-weight: 900;
            letter-spacing: 0;
            line-height: 1.12;
            margin: 0 0 0.35rem;
        }
        .home-header p {
            color: #64748b;
            font-size: 0.98rem;
            line-height: 1.5;
            margin: 0 auto 0 0;
            max-width: 720px;
        }
        .step-label {
            color: #ef6f61;
            font-size: 0.76rem;
            font-weight: 900;
            letter-spacing: 0.08em;
            margin: 1.1rem 0 0.45rem;
            text-transform: uppercase;
        }
        .choice-card-title {
            color: #0f172a;
            font-size: 1rem;
            font-weight: 900;
            line-height: 1.25;
        }
        .choice-card-copy {
            color: #64748b;
            font-size: 0.84rem;
            line-height: 1.42;
            margin-top: 0.3rem;
            min-height: 2.35rem;
        }
        .module-card-title {
            color: #0f172a;
            font-size: 0.82rem;
            font-weight: 850;
            line-height: 1.25;
            min-height: 1.9rem;
        }
        .module-card-code {
            color: #ef6f61;
            font-size: 0.72rem;
            font-weight: 900;
            letter-spacing: 0.06em;
            margin-bottom: 0.15rem;
            text-transform: uppercase;
        }
        .bottom-action-bar {
            background: #ffffff;
            border: 1px solid rgba(15, 23, 42, 0.08);
            border-radius: 18px;
            box-shadow: 0 8px 22px rgba(15, 23, 42, 0.06);
            margin: 1.5rem 0 0.8rem;
            padding: 0.85rem 1rem;
        }
        .bottom-action-title {
            color: #475569;
            font-size: 0.78rem;
            font-weight: 850;
            letter-spacing: 0.06em;
            margin-bottom: 0.6rem;
            text-transform: uppercase;
        }
        .active-app-shell {
            background: #ffffff;
            border: 1px solid rgba(239, 111, 97, 0.22);
            border-radius: 20px;
            box-shadow: 0 12px 34px rgba(15, 23, 42, 0.08);
            margin: 0.8rem 0 1rem;
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
        .module-nav-shell {
            background: #ffffff;
            border: 1px solid rgba(15, 23, 42, 0.08);
            border-radius: 18px;
            box-shadow: 0 8px 24px rgba(15, 23, 42, 0.06);
            margin: 0.55rem 0 1rem;
            padding: 0.85rem;
        }
        .module-nav-title {
            color: #475569;
            font-size: 0.76rem;
            font-weight: 900;
            letter-spacing: 0.08em;
            margin-bottom: 0.6rem;
            text-transform: uppercase;
        }
        .module-nav-shell a {
            border: 1px solid #e2e8f0;
            border-radius: 999px;
            color: #334155;
            font-size: 0.82rem;
            font-weight: 800;
            padding: 0.35rem 0.65rem;
        }
        .module-nav-shell a:hover {
            border-color: #ef6f61;
            color: #be3f34;
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
        div[data-testid="stHorizontalBlock"] div[data-testid="stVerticalBlockBorderWrapper"] {
            border-radius: 16px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def sanitize_text(text):
    if text is None:
        return ""
    return str(text).replace("<", "").replace(">", "")


def display_horizon(value: object) -> str:
    return str(value or "1 año").replace("anos", "años").replace("ano", "año")


def summary_chip(label: str, value: str):
    st.markdown(
        f"""
        <div class="summary-chip">
            <div class="summary-chip-label">{sanitize_text(label)}</div>
            <div class="summary-chip-value">{sanitize_text(value)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def get_dates_from_horizon(selected_horizon: str) -> tuple[object, object]:
    fecha_fin_ref = pd.to_datetime(DEFAULT_END_DATE)
    offset_by_horizon = {
        "6 meses": pd.DateOffset(months=6),
        "1 año": pd.DateOffset(years=1),
        "1 ano": pd.DateOffset(years=1),
        "2 años": pd.DateOffset(years=2),
        "2 anos": pd.DateOffset(years=2),
        "5 años": pd.DateOffset(years=5),
        "5 anos": pd.DateOffset(years=5),
    }
    start_date = (fecha_fin_ref - offset_by_horizon.get(selected_horizon, pd.DateOffset(years=1))).date()
    return start_date, fecha_fin_ref.date()


def asset_label(asset_name: str) -> str:
    ticker = AVAILABLE_ASSETS.get(asset_name, asset_name)
    return f"{asset_name} - {ticker}"


def render_saved_portfolio_summary(config: dict) -> None:
    selected_assets = config.get("selected_asset_names", [])
    selected_tickers = config.get("selected_tickers", [])
    selected_weights = config.get("selected_weights", {})
    selected_modules = config.get("selected_modules", [])
    module_count = len(selected_modules)

    st.markdown(
        f"""
        <div class="portfolio-config-summary">
            <div class="portfolio-config-title">{sanitize_text(config.get("portfolio_name"))}</div>
            <div class="portfolio-config-line">
                <strong>{len(selected_tickers)} activos</strong> - Horizonte {sanitize_text(display_horizon(config.get("selected_horizon")))} - {module_count} módulos seleccionados
            </div>
            <div class="portfolio-config-line">Configuración guardada para esta sesión.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    weight_rows = [
        {
            "Activo": selected_assets[index] if index < len(selected_assets) else ticker,
            "Ticker": ticker,
            "Peso": f"{float(selected_weights.get(ticker, 0.0)):.2%}",
        }
        for index, ticker in enumerate(selected_tickers)
    ]
    if weight_rows:
        render_table(pd.DataFrame(weight_rows), hide_index=True)

    if module_count > 4:
        st.caption(f"{module_count} módulos seleccionados.")
        with st.expander("Ver módulos seleccionados", expanded=False):
            st.markdown("\n".join(f"- {sanitize_text(module)}" for module in selected_modules))
    elif selected_modules:
        module_html = "".join(
            f'<span class="module-pill">{sanitize_text(module)}</span>'
            for module in selected_modules
        )
        st.markdown(module_html, unsafe_allow_html=True)


def _render_home_header() -> None:
    st.markdown(
        """
        <div class="home-header">
            <h1>RiskLab USTA</h1>
            <p>Aplicación interactiva para la construcción y análisis de portafolios financieros.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _usage_guide_body() -> None:
    st.markdown(
        """
        1. **Cómo crear un portafolio:** inicia una configuración nueva desde la bienvenida o desde tus portafolios guardados.
        2. **Cómo seleccionar activos:** marca **Incluir** en los activos que deseas incorporar al portafolio.
        3. **Cómo revisar la ficha de cada activo:** haz clic directamente sobre la tarjeta del activo para abrir su descripción.
        4. **Cómo configurar pesos y horizonte:** elige el periodo de análisis y define pesos iguales o personalizados.
        5. **Cómo guardar el portafolio:** confirma activos, pesos, horizonte y módulos para conservar la configuración.
        6. **Cómo continuar un portafolio existente:** selecciona una tarjeta de portafolio guardado y continúa el análisis.
        7. **Cómo acceder a los módulos de análisis:** al guardar, los módulos seleccionados aparecerán en el panel derecho.
        """
    )


def _render_usage_guide_dialog() -> None:
    dialog = getattr(st, "dialog", None)
    if dialog is not None:
        @dialog("Guía de uso")
        def _guide_dialog():
            _usage_guide_body()

        _guide_dialog()
    else:
        st.session_state["show_usage_guide"] = True


def _render_usage_guide_button() -> None:
    help_cols = st.columns([4, 1])
    with help_cols[1]:
        if st.button("Guía de uso", use_container_width=True):
            _render_usage_guide_dialog()

    if st.session_state.get("show_usage_guide"):
        with st.container(border=True):
            st.markdown("### Guía de uso")
            _usage_guide_body()
            if st.button("Cerrar guía", use_container_width=False):
                st.session_state.pop("show_usage_guide", None)
                st.rerun()


def _render_welcome_step() -> None:
    _render_home_header()
    st.markdown(
        """
        <div class="insight-box">
            RiskLab USTA permite construir un portafolio personalizado y analizar su comportamiento mediante herramientas de riesgo, rendimiento y apoyo a la toma de decisiones. La aplicación integra módulos de análisis técnico, rendimientos, volatilidad, CAPM, VaR/CVaR, optimización de portafolios y contexto macrofinanciero, con el fin de ofrecer una lectura estructurada y aplicada del desempeño de los activos seleccionados.
        </div>
        """,
        unsafe_allow_html=True,
    )
    _render_usage_guide_button()
    if st.button("Comenzar configuración", type="primary", use_container_width=False):
        _clear_active_asset_info()
        st.session_state[ONBOARDING_STEP_SESSION_KEY] = "assets"
        st.rerun()


def _portfolio_option_label(portfolio: dict) -> str:
    name = portfolio.get("portfolio_name", "Portafolio sin nombre")
    tickers = portfolio.get("selected_tickers") or []
    updated_at = str(portfolio.get("updated_at", ""))[:10]
    suffix = f" - {updated_at}" if updated_at else ""
    return f"{name} ({len(tickers)} activos){suffix}"


def _portfolio_saved_date(portfolio: dict) -> str:
    raw_date = portfolio.get("updated_at") or portfolio.get("created_at")
    if not raw_date:
        return "No disponible"
    return str(raw_date).replace("T", " ")[:19]


def _start_new_portfolio() -> None:
    clear_active_portfolio_config()
    st.session_state[ONBOARDING_STEP_SESSION_KEY] = "assets"
    st.session_state[SHOW_PORTFOLIO_CONFIG_SESSION_KEY] = True
    st.session_state.pop("active_asset_info", None)
    st.session_state.pop("show_asset_modal", None)
    for ticker in AVAILABLE_ASSETS.values():
        st.session_state.pop(f"asset_selected_{ticker}", None)
        st.session_state.pop(f"custom_weight_{ticker}", None)
    for module in MODULE_OPTIONS:
        st.session_state.pop(f"module_selected_{module.split()[0]}", None)
    st.session_state.pop("portfolio_name_input", None)
    st.session_state.pop("portfolio_weight_mode", None)


def _select_saved_portfolio(portfolio_id: str) -> None:
    if st.session_state.get("selected_saved_portfolio_id") != portfolio_id:
        _clear_saved_portfolio_delete_state()
    st.session_state["selected_saved_portfolio_id"] = portfolio_id


def _clear_saved_portfolio_delete_state() -> None:
    st.session_state.pop("saved_portfolio_delete_options_id", None)
    for key in list(st.session_state.keys()):
        if str(key).startswith("confirm_delete_portfolio_"):
            st.session_state.pop(key, None)


def render_saved_portfolio_card(portfolio: dict, is_selected: bool, index: int) -> None:
    portfolio_id = portfolio["portfolio_id"]
    tickers = portfolio.get("selected_tickers") or []
    ticker_text = ", ".join(str(ticker) for ticker in tickers) or "Sin activos"
    card_class = "saved-portfolio-card selected" if is_selected else "saved-portfolio-card"
    state_class = "selected" if is_selected else "idle"
    state_label = "Seleccionado" if is_selected else "Disponible"

    with st.container(key=f"saved_portfolio_click_{index}"):
        st.markdown(
            f"""
            <div class="saved-portfolio-link">
            <div class="{card_class}">
                <div class="saved-portfolio-name">{sanitize_text(portfolio.get("portfolio_name", "Portafolio sin nombre"))}</div>
                <div class="saved-portfolio-state {state_class}">{state_label}</div>
                <div class="saved-portfolio-meta"><strong>Activos/tickers:</strong> {sanitize_text(ticker_text)}</div>
                <div class="saved-portfolio-meta"><strong>Horizonte:</strong> {sanitize_text(display_horizon(portfolio.get("selected_horizon", "1 año")))}</div>
                <div class="saved-portfolio-meta"><strong>Módulos:</strong> {len(portfolio.get("selected_modules") or [])}</div>
                <div class="saved-portfolio-meta"><strong>Última actualización:</strong> {sanitize_text(_portfolio_saved_date(portfolio))}</div>
            </div>
        </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button(
            f"Seleccionar {portfolio.get('portfolio_name', 'portafolio')}",
            key=f"select_saved_portfolio_{index}",
            use_container_width=True,
        ):
            _select_saved_portfolio(portfolio_id)
            st.rerun()


def render_selected_saved_portfolio_actions(portfolio_id: str, index: int) -> None:
    delete_options_active = st.session_state.get("saved_portfolio_delete_options_id") == portfolio_id
    action_cols = st.columns([1.25, 1, 0.75, 2])
    with action_cols[0]:
        if st.button(
            "Continuar con este portafolio",
            type="primary",
            key=f"continue_saved_portfolio_{index}",
            use_container_width=True,
        ):
            config = load_user_portfolio(portfolio_id)
            if config is None:
                st.error("No fue posible cargar el portafolio seleccionado.")
                return
            load_portfolio_config(config)
            st.session_state[SHOW_PORTFOLIO_CONFIG_SESSION_KEY] = False
            st.session_state[ONBOARDING_STEP_SESSION_KEY] = "ready"
            st.rerun()
    with action_cols[1]:
        if not delete_options_active:
            if st.button("Eliminar", key=f"show_delete_saved_portfolio_{index}", use_container_width=True):
                st.session_state["saved_portfolio_delete_options_id"] = portfolio_id
                st.rerun()
        else:
            confirm_delete = st.checkbox(
                "Confirmar eliminaciÃ³n de este portafolio",
                key=f"confirm_delete_portfolio_{index}",
            )
            if st.button(
                "Eliminar portafolio",
                key=f"delete_saved_portfolio_{index}",
                use_container_width=True,
            ):
                if not confirm_delete:
                    st.warning("Confirma la eliminaciÃ³n antes de continuar.")
                    return
                delete_user_portfolio(portfolio_id)
                st.session_state.pop("selected_saved_portfolio_id", None)
                _clear_saved_portfolio_delete_state()
                st.success("Portafolio eliminado.")
                st.rerun()
    with action_cols[2]:
        if st.button("Cerrar", key=f"close_saved_portfolio_{index}", use_container_width=True):
            st.session_state.pop("selected_saved_portfolio_id", None)
            _clear_saved_portfolio_delete_state()
            st.rerun()


def _render_portfolio_choice_step() -> None:
    portfolios = list_user_portfolios()
    if not portfolios:
        st.markdown(
            """
            <div class="home-header">
                <h1>Tus portafolios guardados</h1>
                <p>Aún no tienes portafolios guardados. Crea uno nuevo para comenzar el análisis.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("Crear nuevo portafolio", type="primary", use_container_width=False):
            _start_new_portfolio()
            st.rerun()
        return

    st.markdown(
        """
        <div class="home-header">
            <h1>Tus portafolios guardados</h1>
            <p>Selecciona uno de tus portafolios guardados para continuar el análisis o crea un nuevo portafolio desde cero.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    valid_ids = [portfolio["portfolio_id"] for portfolio in portfolios]
    selected_portfolio_id = st.session_state.get("selected_saved_portfolio_id")
    if selected_portfolio_id not in valid_ids:
        selected_portfolio_id = None
        st.session_state.pop("selected_saved_portfolio_id", None)
        _clear_saved_portfolio_delete_state()

    for index, portfolio in enumerate(portfolios):
        portfolio_id = portfolio["portfolio_id"]
        is_selected = portfolio_id == selected_portfolio_id
        render_saved_portfolio_card(portfolio, is_selected, index)
        if is_selected:
            render_selected_saved_portfolio_actions(portfolio_id, index)

    st.markdown("---")
    if st.button("Crear nuevo portafolio", use_container_width=False):
        _start_new_portfolio()
        st.rerun()


def _asset_info_body(asset_name: str) -> None:
    ticker = AVAILABLE_ASSETS.get(asset_name, asset_name)
    details = ASSET_DETAILS.get(asset_name, {})
    complementary_fields = [
        ("Mercado/país", details.get("market")),
        ("Sector", details.get("sector")),
        ("Rol dentro del portafolio", details.get("role")),
        ("Tipo de exposición", details.get("exposure")),
        ("Relevancia financiera", details.get("relevance")),
    ]
    for label, value in complementary_fields:
        if value:
            st.caption(f"{label}: {value}")
    st.caption(f"Ticker de referencia: {ticker}")
    if st.button("Cerrar", key=f"close_asset_info_{ticker}"):
        _clear_active_asset_info()
        st.rerun()


def _clear_active_asset_info() -> None:
    st.session_state.pop("active_asset_info", None)
    st.session_state.pop("show_asset_modal", None)


def _render_asset_info_dialog() -> None:
    if not st.session_state.get("show_asset_modal"):
        return
    asset_name = st.session_state.get("active_asset_info")
    if not asset_name:
        return

    dialog = getattr(st, "dialog", None)
    if dialog is not None:
        @dialog("Información del activo")
        def _asset_dialog():
            _asset_info_body(asset_name)

        _asset_dialog()
    else:
        with st.expander("Información del activo", expanded=True):
            _asset_info_body(asset_name)


def _set_active_asset_info(asset_name: str) -> None:
    _sync_selected_assets_session_state()
    st.session_state["active_asset_info"] = asset_name
    st.session_state["show_asset_modal"] = True


def _on_asset_checkbox_change(asset_name: str, ticker: str) -> None:
    _sync_selected_assets_session_state()


def _sync_asset_checkbox_defaults(selected_asset_names: list[str]) -> None:
    for asset_name, ticker in AVAILABLE_ASSETS.items():
        key = f"asset_selected_{ticker}"
        if key not in st.session_state:
            st.session_state[key] = asset_name in selected_asset_names


def _sync_selected_assets_session_state() -> None:
    selected_names = [
        asset_name
        for asset_name, ticker in AVAILABLE_ASSETS.items()
        if st.session_state.get(f"asset_selected_{ticker}", False)
    ]
    selected_tickers = [AVAILABLE_ASSETS[name] for name in selected_names]
    st.session_state["selected_asset_names"] = selected_names
    st.session_state["selected_tickers"] = selected_tickers
    if st.session_state.get("portfolio_weight_mode", "Pesos iguales") == "Pesos iguales":
        equal_weight = 1 / len(selected_tickers) if selected_tickers else 0.0
        st.session_state["selected_weights"] = {
            ticker: equal_weight for ticker in selected_tickers
        }


def _set_all_asset_checkboxes(selected: bool) -> None:
    for ticker in AVAILABLE_ASSETS.values():
        st.session_state[f"asset_selected_{ticker}"] = selected
    _sync_selected_assets_session_state()


def _request_all_asset_checkboxes(selected: bool) -> None:
    st.session_state["asset_bulk_selection_requested"] = selected
    _clear_active_asset_info()


def _apply_pending_asset_bulk_selection() -> None:
    if "asset_bulk_selection_requested" not in st.session_state:
        return

    selected = bool(st.session_state.pop("asset_bulk_selection_requested"))
    _set_all_asset_checkboxes(selected)


def _render_asset_card_selector(selected_asset_names: list[str]) -> list[str]:
    _sync_asset_checkbox_defaults(selected_asset_names)
    _apply_pending_asset_bulk_selection()

    st.markdown("### Selecciona los activos del portafolio")
    selected_names: list[str] = []
    columns = st.columns(3)

    for index, (asset_name, ticker) in enumerate(AVAILABLE_ASSETS.items()):
        key = f"asset_selected_{ticker}"
        is_selected = bool(st.session_state.get(key, False))
        status_class = "selected" if is_selected else "idle"
        status_label = "Seleccionado" if is_selected else "Disponible"

        with columns[index % len(columns)]:
            with st.container(border=True, key=f"asset_card_shell_{index}"):
                st.markdown(
                    f"""<div
                    class="asset-card-click-target {status_class}"
                    aria-label="{sanitize_text(asset_name)}"
                    ></div>
                    <div class="asset-card-title">{sanitize_text(asset_name)}</div>
                    <div class="asset-card-ticker">{sanitize_text(ticker)}</div>
                    <div class="asset-card-copy">{sanitize_text(ASSET_DESCRIPTIONS.get(asset_name, "Activo del universo disponible de RiskLab."))}</div>
                    <div class="asset-card-status {status_class}">{status_label}</div>
                    """,
                    unsafe_allow_html=True,
                )
                st.button(
                    " ",
                    key=f"asset_card_click_{ticker}",
                    use_container_width=True,
                    on_click=_set_active_asset_info,
                    args=(asset_name,),
                )
                st.checkbox(
                    "Incluir",
                    key=key,
                    on_change=_on_asset_checkbox_change,
                    args=(asset_name, ticker),
                )

        if bool(st.session_state.get(key, False)):
            selected_names.append(asset_name)

    selected_count = len(selected_names)
    all_selected = selected_count == len(AVAILABLE_ASSETS)
    selector_cols = st.columns([1, 1, 4])
    with selector_cols[0]:
        if st.button(
            "Deseleccionar todos" if all_selected else "Seleccionar todos",
            use_container_width=True,
        ):
            _request_all_asset_checkboxes(not all_selected)
            st.rerun()
    with selector_cols[1]:
        if selected_count and st.button("Limpiar selección", use_container_width=True):
            _request_all_asset_checkboxes(False)
            st.rerun()

    _sync_selected_assets_session_state()
    _render_asset_info_dialog()
    return selected_names


def _render_weight_editor(selected_names: list[str], current_config: dict) -> tuple[dict[str, float], float]:
    st.markdown("#### Pesos personalizados")
    st.caption("Ingresa pesos en escala decimal. Ejemplo: 0.25 equivale a 25%.")
    custom_weights: dict[str, float] = {}

    for row_start in range(0, len(selected_names), 3):
        row_columns = st.columns(3)
        for column, asset_name in zip(row_columns, selected_names[row_start : row_start + 3]):
            ticker = AVAILABLE_ASSETS[asset_name]
            default_weight = float(current_config.get("selected_weights", {}).get(ticker, 0.0))
            with column:
                with st.container(border=True):
                    st.markdown(f"**{asset_name}**")
                    st.caption(ticker)
                    custom_weights[ticker] = st.number_input(
                        "Peso",
                        min_value=0.0,
                        max_value=1.0,
                        value=min(max(default_weight, 0.0), 1.0),
                        step=0.01,
                        format="%.4f",
                        key=f"custom_weight_{ticker}",
                    )

    weights_sum = sum(custom_weights.values())
    if abs(weights_sum - 1.0) <= 0.01:
        st.success(f"Suma de pesos: {weights_sum:.4f}. Lista para guardar.")
    else:
        st.warning(f"Suma de pesos: {weights_sum:.4f}. Debe ser aproximadamente 1.0000.")

    return custom_weights, weights_sum


def _set_module_default(module: str, selected_modules: list[str]) -> None:
    key = f"module_selected_{module.split()[0]}"
    if key not in st.session_state:
        st.session_state[key] = module in selected_modules


def _render_module_selector(current_modules: list[str]) -> list[str]:
    st.markdown('<div class="step-label">Módulos de análisis</div>', unsafe_allow_html=True)
    default_modules = current_modules or MODULE_OPTIONS
    selected_modules: list[str] = []

    for module in MODULE_OPTIONS:
        _set_module_default(module, default_modules)

    for row_start in range(0, len(MODULE_OPTIONS), 2):
        columns = st.columns(2)
        for column, module in zip(columns, MODULE_OPTIONS[row_start : row_start + 2]):
            module_code, module_name = module.split(" ", maxsplit=1)
            key = f"module_selected_{module_code}"
            is_selected = bool(st.session_state.get(key, False))
            with column:
                with st.container(border=True):
                    st.markdown(
                        f"""
                        <div class="module-card-code">{sanitize_text(module_code)}</div>
                        <div class="module-card-title">{sanitize_text(module_name)}</div>
                        <div class="asset-card-copy">{sanitize_text(MODULE_DESCRIPTIONS.get(module_code, "Módulo de análisis financiero."))}</div>
                        <div class="asset-card-status {'selected' if is_selected else 'idle'}">{'Seleccionado' if is_selected else 'No seleccionado'}</div>
                        """,
                        unsafe_allow_html=True,
                    )
                    st.checkbox("Incluir", key=key)
            if bool(st.session_state.get(key, False)):
                selected_modules.append(module)

    return selected_modules


def _reset_home_portfolio_config() -> None:
    reset_portfolio_config()
    st.session_state[SHOW_PORTFOLIO_CONFIG_SESSION_KEY] = True
    st.session_state[ONBOARDING_STEP_SESSION_KEY] = "welcome"
    st.session_state.pop(PORTFOLIO_SAVE_MESSAGE_SESSION_KEY, None)
    st.session_state.pop("active_asset_info", None)
    st.session_state.pop("show_asset_modal", None)
    for ticker in AVAILABLE_ASSETS.values():
        st.session_state.pop(f"asset_selected_{ticker}", None)
        st.session_state.pop(f"custom_weight_{ticker}", None)
    for module in MODULE_OPTIONS:
        st.session_state.pop(f"module_selected_{module.split()[0]}", None)
    st.session_state.pop("portfolio_name_input", None)
    st.session_state.pop("portfolio_weight_mode", None)


def _hide_portfolio_config(message: str) -> None:
    st.session_state[SHOW_PORTFOLIO_CONFIG_SESSION_KEY] = False
    st.session_state[ONBOARDING_STEP_SESSION_KEY] = "ready"
    st.session_state[PORTFOLIO_SAVE_MESSAGE_SESSION_KEY] = message
    st.rerun()


def _show_transient_message(message: str) -> None:
    toast = getattr(st, "toast", None)
    if toast is not None:
        toast(message)
    else:
        st.caption(message)


def _logout_current_user() -> None:
    clear_active_portfolio_config()
    st.session_state.pop("active_portfolio_user", None)
    st.session_state.pop(AUTH_SESSION_KEY, None)
    st.session_state.pop(AUTH_USER_SESSION_KEY, None)
    st.session_state[SHOW_PORTFOLIO_CONFIG_SESSION_KEY] = True
    st.session_state[ONBOARDING_STEP_SESSION_KEY] = "welcome"


def _show_portfolio_config() -> None:
    st.session_state[SHOW_PORTFOLIO_CONFIG_SESSION_KEY] = True
    st.session_state[ONBOARDING_STEP_SESSION_KEY] = "assets"


def _should_show_portfolio_config() -> bool:
    if not is_portfolio_config_ready():
        return True
    return bool(st.session_state.get(SHOW_PORTFOLIO_CONFIG_SESSION_KEY, False))


def _render_active_portfolio_panel(config: dict) -> None:
    selected_tickers = config.get("selected_tickers", [])
    selected_modules = config.get("selected_modules", [])
    st.markdown(
        f"""
        <div class="active-app-shell">
            <div class="active-app-title">{sanitize_text(config.get("portfolio_name", "Portafolio activo"))}</div>
            <div class="active-app-meta">
                {len(selected_tickers)} activos - Horizonte {sanitize_text(display_horizon(config.get("selected_horizon")))} - {len(selected_modules)} módulos - Benchmark {sanitize_text(GLOBAL_BENCHMARK)}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_portfolio_options_panel(config: dict) -> None:
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
        if st.button("Restablecer", key="home_reset_active", use_container_width=True):
            _reset_home_portfolio_config()
            st.rerun()
        if st.button("Cerrar sesión", key="home_logout_active", use_container_width=True):
            _logout_current_user()
            st.rerun()

        _render_selected_module_navigation(config)


def _module_short_label(module: str) -> str:
    code, name = module.split(" ", maxsplit=1)
    return f"{code} {name}"


def _render_selected_module_navigation(config: dict) -> None:
    selected_modules = [
        module
        for module in config.get("selected_modules", [])
        if module in MODULE_PAGE_LINKS
    ]
    if not selected_modules:
        return

    st.markdown(
        """
        <div class="portfolio-options-divider"></div>
        <div class="module-nav-list">
            <div class="module-nav-title">Módulos seleccionados</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    for module in selected_modules:
        st.page_link(
            MODULE_PAGE_LINKS[module],
            label=_module_short_label(module),
            use_container_width=True,
        )


def _selected_names_from_checkboxes() -> list[str]:
    return [
        asset_name
        for asset_name, ticker in AVAILABLE_ASSETS.items()
        if st.session_state.get(f"asset_selected_{ticker}")
    ]


def _current_onboarding_asset_names(current_config: dict) -> list[str]:
    if is_portfolio_config_ready():
        return current_config.get("selected_asset_names", [])
    return []


def _selected_weights_for_assets(selected_names: list[str], current_config: dict) -> tuple[dict[str, float], float]:
    selected_tickers = [AVAILABLE_ASSETS[name] for name in selected_names]
    if not selected_tickers:
        return {}, 0.0

    weight_mode = st.radio(
        "Asignación de pesos",
        ["Pesos iguales", "Pesos personalizados"],
        horizontal=True,
        key="portfolio_weight_mode",
    )
    if weight_mode == "Pesos personalizados":
        return _render_weight_editor(selected_names, current_config)

    equal_weight = 1 / len(selected_tickers)
    st.success(f"Pesos iguales calculados automáticamente: {equal_weight:.2%} por activo.")
    return {ticker: equal_weight for ticker in selected_tickers}, 1.0


def _render_assets_step(current_config: dict) -> None:
    _render_usage_guide_button()
    st.markdown('<div class="step-label">Paso 1 - Activos</div>', unsafe_allow_html=True)
    name_col, horizon_col = st.columns([1.7, 1])
    with name_col:
        st.text_input(
            "Nombre del portafolio",
            value=current_config.get("portfolio_name") or "Portafolio RiskLab USTA",
            key="portfolio_name_input",
        )
    with horizon_col:
        current_horizon = current_config.get("selected_horizon", "1 año")
        if current_horizon in {"1 ano", "2 anos", "5 anos"}:
            current_horizon = display_horizon(current_horizon)
        st.selectbox(
            "Horizonte",
            HORIZON_OPTIONS,
            index=HORIZON_OPTIONS.index(current_horizon) if current_horizon in HORIZON_OPTIONS else 1,
            key="portfolio_horizon_select",
        )

    selected_names = _render_asset_card_selector(_current_onboarding_asset_names(current_config))
    if not selected_names:
        st.warning("Selecciona al menos 1 activo para continuar.")

    action_cols = st.columns([1, 1, 4])
    with action_cols[0]:
        if st.button("Volver", use_container_width=True):
            _clear_active_asset_info()
            st.session_state[ONBOARDING_STEP_SESSION_KEY] = "welcome"
            st.rerun()
    with action_cols[1]:
        if st.button("Continuar", type="primary", use_container_width=True):
            if not _selected_names_from_checkboxes():
                st.error("Selecciona al menos 1 activo para continuar.")
                return
            _clear_active_asset_info()
            st.session_state[ONBOARDING_STEP_SESSION_KEY] = "modules"
            st.rerun()


def _render_modules_step(current_config: dict) -> None:
    selected_names = _selected_names_from_checkboxes()
    if not selected_names:
        st.warning("Selecciona al menos 1 activo antes de elegir módulos.")
        st.session_state[ONBOARDING_STEP_SESSION_KEY] = "assets"
        st.rerun()

    st.markdown('<div class="step-label">Paso 2 - Módulos</div>', unsafe_allow_html=True)
    st.markdown("### Selecciona los módulos de análisis")
    st.caption("Los módulos seleccionados aparecerán en el panel derecho de la aplicación.")
    selected_modules = _render_module_selector(current_config.get("selected_modules") or MODULE_OPTIONS)

    selected_weights, weights_sum = _selected_weights_for_assets(selected_names, current_config)
    selected_tickers = [AVAILABLE_ASSETS[name] for name in selected_names]

    action_cols = st.columns([1, 1, 3])
    with action_cols[0]:
        if st.button("Volver a activos", use_container_width=True):
            st.session_state[ONBOARDING_STEP_SESSION_KEY] = "assets"
            st.rerun()
    with action_cols[1]:
        if st.button("Guardar configuración", type="primary", use_container_width=True):
            portfolio_name = st.session_state.get("portfolio_name_input", "").strip() or "Portafolio RiskLab USTA"
            selected_horizon = st.session_state.get("portfolio_horizon_select", "1 año")
            if not selected_modules:
                st.error("Selecciona al menos un módulo de análisis.")
                return
            if not selected_tickers:
                st.error("Selecciona al menos 1 activo para guardar el portafolio.")
                return
            if st.session_state.get("portfolio_weight_mode") == "Pesos personalizados":
                if any(weight < 0 for weight in selected_weights.values()):
                    st.error("Los pesos personalizados deben ser no negativos.")
                    return
                if abs(weights_sum - 1.0) > 0.01:
                    st.warning(f"Los pesos deben sumar aproximadamente 1. Suma actual: {weights_sum:.4f}.")
                    return

            save_portfolio_config(
                {
                    "portfolio_name": portfolio_name,
                    "selected_tickers": selected_tickers,
                    "selected_asset_names": selected_names,
                    "selected_weights": selected_weights,
                    "selected_horizon": selected_horizon,
                    "selected_modules": selected_modules,
                }
            )
            _hide_portfolio_config("Configuración del portafolio guardada.")


def _render_portfolio_builder() -> None:
    current_config = get_portfolio_config()
    step = st.session_state.get(ONBOARDING_STEP_SESSION_KEY)
    if step not in {"portfolio_choice", "welcome", "assets", "modules", "ready"}:
        step = (
            "assets"
            if is_portfolio_config_ready()
            else "portfolio_choice"
            if has_saved_portfolios()
            else "welcome"
        )
        st.session_state[ONBOARDING_STEP_SESSION_KEY] = step

    if step == "portfolio_choice":
        _render_portfolio_choice_step()
        return

    if step == "welcome":
        _render_welcome_step()
        return

    with st.container(border=True):
        if step == "assets":
            _render_assets_step(current_config)
        else:
            _render_modules_step(current_config)

    if is_portfolio_config_ready():
        render_saved_portfolio_summary(get_portfolio_config())

def render_portfolio_configurator() -> None:
    _render_portfolio_builder()


def weighted_portfolio_returns(returns: pd.DataFrame, tickers: list[str], weights_by_ticker: dict) -> pd.Series:
    weights = pd.Series(
        {ticker: float(weights_by_ticker.get(ticker, 0.0)) for ticker in tickers},
        dtype=float,
    )
    if weights.sum() <= 0:
        return equal_weight_portfolio(returns)

    weights = weights / weights.sum()
    return returns.loc[:, tickers].mul(weights, axis=1).sum(axis=1)


inject_ui_css()
_inject_home_styles()


# ---------------------------------------------------------
# Cache de datos
# ---------------------------------------------------------
@st.cache_data(show_spinner=False, ttl=3600)
def get_market_data(tickers, start, end):
    """Obtiene el bundle de mercado exclusivamente desde el backend FastAPI.
    No debe usarse el flujo directo de yfinance para el bundle principal del dashboard.
    """
    return fetch_market_bundle_from_backend(tickers=tickers, start=start, end=end)


def is_market_bundle_empty(market_data) -> bool:
    if market_data is None:
        return True
    close = market_data.get("close", pd.DataFrame())
    returns = market_data.get("returns", pd.DataFrame())
    return close.empty or returns.empty


def market_data_diagnostics(horizonte, start_date, end_date, market_data, stage: str) -> dict:
    close = None if market_data is None else market_data.get("close")
    returns = None if market_data is None else market_data.get("returns")
    close_diag = pd.DataFrame() if close is None else close.copy()
    returns_diag = pd.DataFrame() if returns is None else returns.copy()
    metadata = {} if market_data is None else market_data.get("metadata", {})
    backend_call = last_backend_call()
    calendar_diagnostics = metadata.get("calendar_diagnostics", {})
    diagnostic_warnings = []

    def ensure_datetime_index(frame: pd.DataFrame, label: str) -> pd.DataFrame:
        if frame.empty:
            return frame

        original_index_type = type(frame.index).__name__
        if isinstance(frame.index, pd.DatetimeIndex):
            return frame.sort_index()

        normalized = frame.copy()
        date_col = next((col for col in ("Date", "date") if col in normalized.columns), None)
        if date_col is not None:
            normalized[date_col] = pd.to_datetime(normalized[date_col], errors="coerce")
            normalized = normalized.dropna(subset=[date_col]).set_index(date_col)
            normalized = normalized.drop(columns=[c for c in normalized.columns if str(c).lower() in RESERVED_PRICE_COLUMNS], errors="ignore")
            normalized = normalized.apply(pd.to_numeric, errors="coerce").dropna(axis=1, how="all")
            return normalized.sort_index()

        if isinstance(normalized.index, pd.RangeIndex):
            diagnostic_warnings.append(
                f"{label}: indice no es datetime; se omite filtro por fechas."
            )
            return normalized

        diagnostic_warnings.append(
            f"{label}: indice no es datetime; se omite filtro por fechas."
        )
        return normalized

    close = ensure_datetime_index(close_diag, "close")
    returns = ensure_datetime_index(returns_diag, "returns")
    portfolio_returns_diag = (
        equal_weight_portfolio(returns)
        if not returns.empty
        else pd.Series(dtype=float)
    )
    close_filter_note = None
    if not close.empty and isinstance(close.index, pd.DatetimeIndex):
        close_after_filter = close.loc[
            (close.index >= pd.to_datetime(start_date))
            & (close.index <= pd.to_datetime(end_date))
        ]
    else:
        close_after_filter = None
        if not close.empty:
            close_filter_note = "indice no es datetime; se omite filtro por fechas"
            diagnostic_warnings.append(f"close: {close_filter_note}.")
    business_days = pd.bdate_range(start=start_date, end=end_date)
    if close.empty and len(business_days) == 0:
        empty_reason = "El rango no contiene dias habiles."
    elif close.empty and backend_call.get("status_code") == 404:
        empty_reason = "El backend no encontro precios para ese rango/tickers."
    elif close.empty:
        empty_reason = "La respuesta llego sin precios o el DataFrame quedo vacio al estandarizar."
    else:
        empty_reason = None

    return {
        "etapa": stage,
        "api_base_url_efectivo": backend_base_url(),
        "endpoint": backend_call.get("path") or "/market/bundle",
        "endpoint_llamado": backend_call.get("path") or "/market/bundle",
        "url_llamada": backend_call.get("url"),
        "status_code": backend_call.get("status_code"),
        "index_type_close": type(close.index).__name__,
        "index_type_returns": type(returns.index).__name__,
        "warning": "; ".join(diagnostic_warnings) if diagnostic_warnings else None,
        "horizonte": horizonte,
        "start_date_usuario": str(start_date),
        "end_date_usuario": str(end_date),
        "end_date_enviado_api": str(end_date),
        "end_date_enviado_yfinance": yfinance_exclusive_end(str(end_date)),
        "close.shape": tuple(close.shape),
        "close.index.min": close.index.min() if not close.empty else None,
        "close.index.max": close.index.max() if not close.empty else None,
        "returns.shape": tuple(returns.shape),
        "returns.index.min": returns.index.min() if not returns.empty else None,
        "returns.index.max": returns.index.max() if not returns.empty else None,
        "returns.columns": list(returns.columns),
        "close.columns": list(close.columns),
        "len_valid_tickers": len([ticker for ticker in close.columns if ticker in returns.columns]),
        "na_por_activo_retornos": returns.isna().sum().to_dict() if not returns.empty else {},
        "na_por_activo_retornos_antes_fill": calendar_diagnostics.get(
            "na_por_activo_retornos_antes_fill",
            {},
        ),
        "na_por_activo_retornos_despues_fill": calendar_diagnostics.get(
            "na_por_activo_retornos_despues_fill",
            returns.isna().sum().to_dict() if not returns.empty else {},
        ),
        "tickers_con_nan_retornos": (
            returns.isna().sum()[returns.isna().sum() > 0].to_dict()
            if not returns.empty
            else {}
        ),
        "start_effective_aplicado": calendar_diagnostics.get("start_effective"),
        "obs_count_final": int(portfolio_returns_diag.dropna().shape[0]),
        "portfolio_returns.dropna.shape[0]": int(portfolio_returns_diag.dropna().shape[0]),
        "metadata_backend": {
            "missing_tickers": metadata.get("missing_tickers", []),
            "last_available_date": metadata.get("last_available_date"),
            "calendar_diagnostics": calendar_diagnostics,
        },
        "df_shape": tuple(close.shape),
        "df_index_max": close.index.max() if not close.empty else None,
        "fechas_en_precios_union_calendarios": int(close.shape[0]),
        "observaciones_efectivas_retornos_usados": int(portfolio_returns_diag.dropna().shape[0]),
        "shape_df_antes_filtro_fechas": tuple(close.shape),
        "shape_df_despues_filtro_fechas": (
            tuple(close_after_filter.shape)
            if close_after_filter is not None
            else close_filter_note
        ),
        "shape_returns": tuple(returns.shape),
        "tickers_df_vacio": metadata.get("missing_tickers", []),
        "shapes_ohlcv_por_ticker": metadata.get("ohlcv_shapes", {}),
        "ultimo_dia_disponible": metadata.get("last_available_date"),
        "explicacion_si_vacio": empty_reason,
    }


def load_market_data_with_business_day_fallback(tickers, start_date, end_date, horizonte):
    attempts = []
    today = pd.Timestamp.today().normalize().date()
    candidate_end = min(end_date, today)

    for label, current_end in [
        ("rango seleccionado", end_date),
        ("end ajustado a hoy", candidate_end),
        ("end ajustado a ultimo dia habil previo", (pd.Timestamp(candidate_end) - pd.offsets.BDay(1)).date()),
    ]:
        if attempts and current_end == attempts[-1]["end_date"]:
            continue

        data = get_market_data(
            tickers=tickers,
            start=str(start_date),
            end=str(current_end),
        )
        attempts.append({"label": label, "end_date": current_end, "data": data})
        if not is_market_bundle_empty(data):
            return data, current_end, attempts

    return attempts[-1]["data"], attempts[-1]["end_date"], attempts


# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------
def default_visible_trace_tickers(tickers: list[str]) -> list[str]:
    preferred = ["BP.L", "CA.PA"]
    if all(ticker in tickers for ticker in preferred):
        return preferred
    return tickers[:2]

portfolio_config = get_portfolio_config()
default_portfolio_config = get_default_portfolio_config()

portfolio_config = get_portfolio_config()
if _should_show_portfolio_config():
    render_portfolio_configurator()
    st.stop()

save_message = st.session_state.pop(PORTFOLIO_SAVE_MESSAGE_SESSION_KEY, None)
if save_message:
    _show_transient_message(save_message)

_render_active_portfolio_panel(portfolio_config)

selected_tickers = portfolio_config.get("selected_tickers") or default_portfolio_config["selected_tickers"]
selected_weights = portfolio_config.get("selected_weights") or default_portfolio_config["selected_weights"]
horizonte = portfolio_config.get("selected_horizon") or default_portfolio_config["selected_horizon"]
start_date, end_date = get_dates_from_horizon(horizonte)
# ---------------------------------------------------------
# Validaciones
# ---------------------------------------------------------
if start_date >= end_date:
    st.error("La fecha inicial debe ser menor que la fecha final.")
    st.stop()

if not selected_tickers:
    st.warning("Selecciona al menos un activo para continuar.")
    st.stop()


# Periodo y contexto se muestran de forma compacta en el resumen ejecutivo.


# ---------------------------------------------------------
# Descarga de datos
# ---------------------------------------------------------
try:
    with st.spinner("Descargando datos de mercado..."):
        market_data, effective_end_date, market_attempts = load_market_data_with_business_day_fallback(
            tickers=selected_tickers,
            start_date=start_date,
            end_date=end_date,
            horizonte=horizonte,
        )
except Exception as e:
    st.error(data_error_message(f"Ocurrio un error al descargar los datos: {e}"))
    st.stop()

if market_data is None:
    st.error("No se recibieron datos del mercado.")
    st.stop()

if "close" not in market_data or market_data["close"].empty:
    metadata = market_data.get("metadata", {})
    last_available = metadata.get("last_available_date") or "no disponible"
    st.error(
        data_error_message(
            f"Sin datos para el rango {start_date}{effective_end_date}. "
            f"Último día disponible: {last_available}."
        )
    )
    st.stop()

if "returns" not in market_data or market_data["returns"].empty:
    st.error("No fue posible calcular rendimientos con los datos descargados.")
    st.stop()


normalized_close, _ = normalize_market_frame(market_data["close"], "close", stop_on_invalid=True)
normalized_returns, _ = normalize_market_frame(market_data["returns"], "returns", stop_on_invalid=True)
market_data = {
    **market_data,
    "close": normalized_close,
    "returns": normalized_returns,
}


# ---------------------------------------------------------
# Variables principales
# ---------------------------------------------------------
close_prices = market_data["close"]
returns = market_data["returns"]
missing_tickers = market_data.get("metadata", {}).get("missing_tickers", [])
valid_tickers = [
    ticker
    for ticker in close_prices.columns
    if ticker in returns.columns
    and ticker in selected_tickers
    and str(ticker).lower() not in RESERVED_PRICE_COLUMNS
]
dropped_tickers = [
    ticker
    for ticker in selected_tickers
    if ticker not in valid_tickers
]

if missing_tickers:
    st.warning(
        "Sin datos para estos tickers en el rango seleccionado; se excluyen del análisis: "
        + ", ".join(missing_tickers)
    )

if dropped_tickers:
    st.warning(
        "Estos tickers no tienen precios y retornos alineados suficientes; se excluyen del portafolio: "
        + ", ".join(dropped_tickers)
    )

if not valid_tickers:
    st.error("No quedan activos con precios y retornos validos para calcular el portafolio.")
    st.stop()

close_prices = close_prices.loc[:, valid_tickers]
returns = returns.loc[:, valid_tickers]
market_data = {
    **market_data,
    "close": close_prices,
    "returns": returns,
}

if len(valid_tickers) == 1:
    st.warning("Solo queda un activo valido; los KPIs se calculan sobre ese activo.")

if effective_end_date != end_date:
    st.info(f"Se ajusto la fecha final efectiva a {effective_end_date} para encontrar datos disponibles.")

portfolio_returns = weighted_portfolio_returns(returns, valid_tickers, selected_weights)
if portfolio_returns.empty:
    st.error("No fue posible calcular retornos efectivos del portafolio con los activos validos.")
    st.stop()

ann_return = annualize_return(portfolio_returns)
ann_vol = annualize_volatility(portfolio_returns)
obs_count = int(portfolio_returns.dropna().shape[0])
asset_count = len(valid_tickers)

ret_delta = "Sesgo positivo" if ann_return > 0 else "Sesgo negativo" if ann_return < 0 else "Sin sesgo"
ret_delta_type = "pos" if ann_return > 0 else "neg" if ann_return < 0 else "neu"

vol_delta = "Mayor dispersion" if ann_vol > 0.20 else "Dispersion moderada"
vol_delta_type = "neg" if ann_vol > 0.20 else "neu"


main_col, options_col = st.columns([3.2, 1.05], gap="large")

with main_col:
    # ---------------------------------------------------------
    # Resumen ejecutivo
    # ---------------------------------------------------------
    st.markdown("### Resumen corto")

    info_col1, info_col2, info_col3 = st.columns([1.6, 1, 0.9])

    with info_col1:
        summary_chip("Activo(s)", ", ".join(valid_tickers))

    with info_col2:
        summary_chip("Periodo", f"{start_date} a {effective_end_date}")

    with info_col3:
        summary_chip("Benchmark", GLOBAL_BENCHMARK)


    # ---------------------------------------------------------
    # Grafico principal
    # ---------------------------------------------------------
    st.markdown("### Precios normalizados (base 100)")

    # Solo para visualizacion: suaviza huecos por calendarios bursatiles distintos.
    close_prices_chart = close_prices.ffill()
    close_prices_chart = close_prices_chart.copy()

    date_col = next((col for col in ("Date", "date", "index") if col in close_prices_chart.columns), None)
    if date_col is not None:
        parsed_dates = pd.to_datetime(close_prices_chart[date_col], errors="coerce")
        if parsed_dates.notna().all():
            close_prices_chart = close_prices_chart.drop(columns=[date_col])
            close_prices_chart.index = parsed_dates
        else:
            close_prices_chart = close_prices_chart.drop(columns=[date_col])

    if isinstance(close_prices_chart.index, pd.DatetimeIndex):
        close_prices_chart = close_prices_chart[~close_prices_chart.index.isna()]
    else:
        close_prices_chart = close_prices_chart.iloc[0:0]

    close_prices_chart = close_prices_chart.apply(pd.to_numeric, errors="coerce")
    close_prices_chart = close_prices_chart.dropna(axis=1, how="all")
    if close_prices_chart.empty or close_prices_chart.dropna(how="all").empty:
        st.warning("No hay datos numericos para graficar base 100")
    else:
        fig_norm = plot_normalized_prices(close_prices_chart)
        initially_visible_tickers = default_visible_trace_tickers(list(close_prices_chart.columns))
        for trace in fig_norm.data:
            if trace.name not in initially_visible_tickers:
                trace.visible = "legendonly"
        st.plotly_chart(fig_norm, width="stretch")
    st.caption("Base 100 para comparar desempeño relativo entre los activos seleccionados.")


    # ---------------------------------------------------------
    # Últimos precios
    # ---------------------------------------------------------
    st.markdown("### Últimos precios del portafolio")
    asset_names_by_ticker = dict(
        zip(
            portfolio_config.get("selected_tickers", []),
            portfolio_config.get("selected_asset_names", []),
        )
    )
    latest_prices = close_prices.tail(1)
    if latest_prices.empty:
        st.warning("No hay precios recientes disponibles para el portafolio.")
    else:
        latest_date = latest_prices.index[-1]
        latest_prices_table = pd.DataFrame(
            [
                {
                    "Activo": asset_names_by_ticker.get(ticker, ticker),
                    "Ticker": ticker,
                    "Fecha": latest_date.date() if hasattr(latest_date, "date") else latest_date,
                    "Último precio": latest_prices[ticker].iloc[-1],
                }
                for ticker in valid_tickers
                if ticker in latest_prices.columns
            ]
        )
        render_table(latest_prices_table.style.format({"Último precio": "{:.2f}"}))

with options_col:
    _render_portfolio_options_panel(portfolio_config)

st.stop()


# ---------------------------------------------------------
# Resumen del portafolio
# ---------------------------------------------------------
st.markdown("### Resumen rápido del portafolio configurado")
render_section(
    "Métricas descriptivas del portafolio",
    "Este resumen concentra medidas básicas que ayudan a caracterizar retorno medio y dispersión del portafolio construido con ponderaciones equivalentes.",
)

summary = pd.DataFrame(
    {
        "Métrica": [
            "Rendimiento anualizado",
            "Volatilidad anualizada",
            "Promedio diario",
            "Desviación estándar diaria",
        ],
        "Valor": [
            f"{ann_return:.2%}",
            f"{ann_vol:.2%}",
            f"{portfolio_returns.mean():.4%}",
            f"{portfolio_returns.std(ddof=1):.4%}",
        ],
    }
)

render_table(summary)

with st.expander("Cómo interpretar las métricas del resumen estadístico", expanded=False):
    st.markdown(
        """
        ### Rendimiento anualizado
        Indica el desempeño del portafolio llevado a una escala anual. Sirve para entender si, durante el periodo analizado, el portafolio generó una rentabilidad positiva o negativa. No debe interpretarse como una predicción futura, sino como una medida histórica del comportamiento observado.

        ### Volatilidad anualizada
        Mide qué tanto fluctuaron los retornos del portafolio en escala anual. Representa el nivel de variabilidad o incertidumbre asociado al desempeño del portafolio. Una mayor volatilidad implica movimientos más fuertes y, por tanto, mayor riesgo.

        ### Promedio diario
        Resume el retorno promedio que obtuvo el portafolio en cada día bursátil. Permite ver la tendencia diaria general del portafolio, aunque no significa que todos los días hayan tenido ese mismo comportamiento.

        ### Desviación estándar diaria
        Mide qué tanto se alejaron los retornos diarios respecto al promedio diario. Sirve para identificar si los movimientos diarios fueron estables o si presentaron variaciones importantes de un día a otro.
        """
    )


# ---------------------------------------------------------
# Últimos precios
# ---------------------------------------------------------
st.markdown("### Últimos precios disponibles")
with st.expander("Últimos precios", expanded=False):
    render_table(close_prices.tail(10).style.format("{:.2f}"), hide_index=False)


# ---------------------------------------------------------
# Interpretacion
# ---------------------------------------------------------
st.markdown("### Interpretación general")

st.info(
    "La portada resume activos, periodo, rendimiento, volatilidad y evolución relativa para una primera lectura del portafolio configurado."
)
render_explanation_expander(
    "Cómo interpretar la portada",
    [
        "Esta portada resume el universo de activos (Seven & i Holdings (3382.T), Couche-Tard (ATD.TO), FEMSA (FEMSAUBD.MX), BP (BP.L) y Carrefour (CA.PA)), el horizonte y un primer perfil riesgo-retorno.",
        "Los KPIs muestran una lectura agregada del portafolio configurado: rendimiento anualizado y volatilidad anualizada.",
        "El gráfico base 100 permite ver rápidamente qué activo lidera y cuál es más inestable en el periodo elegido.",
        "Si quieres profundizar, los módulos M1-M10 separan el análisis en técnica, rendimientos, volatilidad, riesgo, optimización, decisión y modelos financieros avanzados.",
    ],
)


# ---------------------------------------------------------
# Estructura de la aplicación
# ---------------------------------------------------------
st.markdown("### Estructura de la aplicación")
with st.expander("Ver módulos de la aplicación", expanded=False):
    st.markdown(
        """
        - **Contextualización:** lectura cualitativa y rol de los activos en el portafolio.
        - **M1. Análisis técnico:** indicadores y gráficos por activo.
        - **M2. Rendimientos:** estadística descriptiva y pruebas de normalidad.
        - **M3. Modelos GARCH:** comparación de modelos de volatilidad.
        - **M4. CAPM y Beta:** beta, CAPM y benchmark local.
        - **M5. VaR/CVaR:** riesgo del portafolio con 3 metodos.
        - **M6. Optimización Markowitz:** frontera eficiente y portafolios óptimos.
        - **M7. Señales:** alertas automáticas de trading.
        - **M8. Macro y Benchmark:** contexto macro y comparación contra índice global.
        - **M9. Panel de decisión:** integración final para postura de acción.
        - **M10. Modelos financieros:** modelos avanzados consumidos desde backend, iniciando con volatilidad EWMA.
        """
    )



