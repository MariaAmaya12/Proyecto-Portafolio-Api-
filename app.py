import streamlit as st
import pandas as pd

from src.config import (
    APP_TITLE,
    ASSET_TICKERS,
    DEFAULT_END_DATE,
    DEFAULT_START_DATE,
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
from src.ui_navigation import render_sidebar_navigation
from src.ui_style import apply_global_typography, render_page_title

RESERVED_PRICE_COLUMNS = {"index", "date"}


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
                f"{label}: índice RangeIndex sin columna Date/date. "
                "No se pueden recuperar fechas reales; se detienen los cálculos para evitar fechas 1970."
            )
            if stop_on_invalid:
                st.error(message)
                st.stop()
            return normalized, message
        else:
            parsed_index = pd.to_datetime(normalized.index, errors="coerce")
            if pd.isna(parsed_index).any():
                message = f"{label}: índice {original_index_type} no contiene fechas válidas."
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
        message = f"{label}: no quedaron columnas numéricas válidas después de normalizar."
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
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)
apply_global_typography()


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


def sanitize_text(text):
    if text is None:
        return ""
    return str(text).replace("<", "").replace(">", "")


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


inject_ui_css()
render_sidebar_navigation()


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
                f"{label}: índice no es datetime; se omite filtro por fechas."
            )
            return normalized

        diagnostic_warnings.append(
            f"{label}: índice no es datetime; se omite filtro por fechas."
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
            close_filter_note = "índice no es datetime; se omite filtro por fechas"
            diagnostic_warnings.append(f"close: {close_filter_note}.")
    business_days = pd.bdate_range(start=start_date, end=end_date)
    if close.empty and len(business_days) == 0:
        empty_reason = "El rango no contiene días hábiles."
    elif close.empty and backend_call.get("status_code") == 404:
        empty_reason = "El backend no encontró precios para ese rango/tickers."
    elif close.empty:
        empty_reason = "La respuesta llegó sin precios o el DataFrame quedó vacío al estandarizar."
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
        ("end ajustado a último día hábil previo", (pd.Timestamp(candidate_end) - pd.offsets.BDay(1)).date()),
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
all_tickers = list(ASSET_TICKERS.values())
ASSET_DISPLAY_LABELS = {
    "CA.PA": "Carrefour (CA.PA)",
    "3382.T": "Seven & i Holdings (3382.T)",
    "ATD.TO": "Alimentation Couche-Tard (ATD.TO)",
    "FEMSAUBD.MX": "FEMSA (FEMSAUBD.MX)",
    "BP.L": "BP (BP.L)",
}
default_home_tickers = [
    ticker
    for ticker in ["3382.T", "ATD.TO", "FEMSAUBD.MX", "BP.L", "CA.PA"]
    if ticker in all_tickers
]
if "home_selected_tickers" not in st.session_state:
    st.session_state["home_selected_tickers"] = default_home_tickers


def asset_display_label(ticker: str) -> str:
    return ASSET_DISPLAY_LABELS.get(ticker, ticker)


def toggle_home_ticker(ticker: str) -> None:
    selected = set(st.session_state.get("home_selected_tickers", default_home_tickers))
    if ticker in selected:
        selected.remove(ticker)
    else:
        selected.add(ticker)

    st.session_state["home_selected_tickers"] = [
        item for item in all_tickers if item in selected
    ]


def default_visible_trace_tickers(tickers: list[str]) -> list[str]:
    preferred = ["BP.L", "CA.PA"]
    if all(ticker in tickers for ticker in preferred):
        return preferred
    return tickers[:2]

with st.sidebar:
    horizonte = st.selectbox(
        "Horizonte de análisis",
        [
            "1 mes",
            "Trimestre",
            "Semestre",
            "1 año",
            "2 años",
            "3 años",
            "5 años",
            "Personalizado",
        ],
        index=3,
    )

    fecha_fin_ref = pd.to_datetime(DEFAULT_END_DATE)

    if horizonte == "1 mes":
        start_date = (fecha_fin_ref - pd.DateOffset(months=1)).date()
        end_date = fecha_fin_ref.date()
    elif horizonte == "Trimestre":
        start_date = (fecha_fin_ref - pd.DateOffset(months=3)).date()
        end_date = fecha_fin_ref.date()
    elif horizonte == "Semestre":
        start_date = (fecha_fin_ref - pd.DateOffset(months=6)).date()
        end_date = fecha_fin_ref.date()
    elif horizonte == "1 año":
        start_date = (fecha_fin_ref - pd.DateOffset(years=1)).date()
        end_date = fecha_fin_ref.date()
    elif horizonte == "2 años":
        start_date = (fecha_fin_ref - pd.DateOffset(years=2)).date()
        end_date = fecha_fin_ref.date()
    elif horizonte == "3 años":
        start_date = (fecha_fin_ref - pd.DateOffset(years=3)).date()
        end_date = fecha_fin_ref.date()
    elif horizonte == "5 años":
        start_date = (fecha_fin_ref - pd.DateOffset(years=5)).date()
        end_date = fecha_fin_ref.date()
    else:
        start_date = st.date_input("Fecha inicial", value=DEFAULT_START_DATE)
        end_date = st.date_input("Fecha final", value=DEFAULT_END_DATE)

    st.divider()
    if st.button("Actualizar datos", key="home_refresh_data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    st.divider()
    st.subheader("Selección de activos")

    if st.button("Restablecer selección", use_container_width=True):
        st.session_state["home_selected_tickers"] = default_home_tickers
        st.rerun()

    for ticker in all_tickers:
        is_selected = ticker in st.session_state["home_selected_tickers"]
        if st.button(
            asset_display_label(ticker),
            key=f"home_asset_chip_{ticker}",
            type="primary" if is_selected else "secondary",
            use_container_width=True,
        ):
            toggle_home_ticker(ticker)
            st.rerun()

    selected_tickers = st.session_state["home_selected_tickers"]
    st.caption(f"{len(selected_tickers)} de {len(all_tickers)} activos seleccionados.")




# ---------------------------------------------------------
# Validaciones
# ---------------------------------------------------------
if start_date >= end_date:
    st.error("La fecha inicial debe ser menor que la fecha final.")
    st.stop()

if not selected_tickers:
    st.warning("Selecciona al menos un activo para continuar.")
    st.stop()


# ---------------------------------------------------------
# Encabezado principal
# ---------------------------------------------------------
render_page_title(
    APP_TITLE,
)

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
    st.error(data_error_message(f"Ocurrió un error al descargar los datos: {e}"))
    st.stop()

if market_data is None:
    st.error("No se recibieron datos del mercado.")
    st.stop()

if "close" not in market_data or market_data["close"].empty:
    metadata = market_data.get("metadata", {})
    last_available = metadata.get("last_available_date") or "no disponible"
    st.error(
        data_error_message(
            f"Sin datos para el rango {start_date}–{effective_end_date}. "
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
    st.error("No quedan activos con precios y retornos válidos para calcular el portafolio.")
    st.stop()

close_prices = close_prices.loc[:, valid_tickers]
returns = returns.loc[:, valid_tickers]
market_data = {
    **market_data,
    "close": close_prices,
    "returns": returns,
}

if len(valid_tickers) == 1:
    st.warning("Solo queda un activo válido; los KPIs se calculan sobre ese activo.")

if effective_end_date != end_date:
    st.info(f"Se ajustó la fecha final efectiva a {effective_end_date} para encontrar datos disponibles.")

portfolio_returns = equal_weight_portfolio(returns)
if portfolio_returns.empty:
    st.error("No fue posible calcular retornos efectivos del portafolio con los activos válidos.")
    st.stop()

ann_return = annualize_return(portfolio_returns)
ann_vol = annualize_volatility(portfolio_returns)
obs_count = int(portfolio_returns.dropna().shape[0])
asset_count = len(valid_tickers)

ret_delta = "Sesgo positivo" if ann_return > 0 else "Sesgo negativo" if ann_return < 0 else "Sin sesgo"
ret_delta_type = "pos" if ann_return > 0 else "neg" if ann_return < 0 else "neu"

vol_delta = "Mayor dispersión" if ann_vol > 0.20 else "Dispersión moderada"
vol_delta_type = "neg" if ann_vol > 0.20 else "neu"


# ---------------------------------------------------------
# Resumen ejecutivo
# ---------------------------------------------------------
st.markdown("### Resumen ")
render_section(
    
    "Este bloque resume los activos seleccionados, la ventana temporal analizada y el comportamiento agregado del portafolio equiponderado.",
)

info_col1, info_col2, info_col3 = st.columns([1.6, 1, 0.9])

with info_col1:
    summary_chip("Activo(s)", ", ".join(valid_tickers))

with info_col2:
    summary_chip("Periodo", f"{start_date} a {effective_end_date}")

with info_col3:
    summary_chip("Benchmark", GLOBAL_BENCHMARK)


# ---------------------------------------------------------
# Métricas principales
# ---------------------------------------------------------
st.markdown("### KPIs del portafolio")

metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)

with metric_col1:
    kpi_card(
        "Número de activos",
        str(asset_count),
        caption="Activos incluidos en el análisis",
    )

with metric_col2:
    kpi_card(
        "Observaciones",
        str(obs_count),
        caption="Días efectivos usados para retorno y riesgo (con precios forward-filled en días sin trading)",
    )

with metric_col3:
    kpi_card(
        "Rendimiento anualizado",
        f"{ann_return:.2%}",
        delta=ret_delta,
        delta_type=ret_delta_type,
        caption="Retorno estimado del portafolio equiponderado",
    )

with metric_col4:
    kpi_card(
        "Volatilidad anualizada",
        f"{ann_vol:.2%}",
        delta=vol_delta,
        delta_type=vol_delta_type,
        caption="Riesgo agregado estimado del portafolio",
    )

render_explanation_expander(
    "Cómo interpretar los KPIs del portafolio",
    [
        "Número de activos: indica cuántos activos están incluidos en el análisis actual.",
        "Observaciones: corresponde a los días efectivos disponibles para calcular retornos y riesgo.",
        "Rendimiento anualizado: resume el retorno estimado del portafolio equiponderado llevado a escala anual.",
        "Volatilidad anualizada: mide la dispersión anualizada de los retornos; valores más altos indican mayor variabilidad del portafolio.",
        "Estos KPIs son una lectura inicial y deben complementarse con los módulos de rendimientos, GARCH, VaR/CVaR, Markowitz y benchmark.",
    ],
)


# ---------------------------------------------------------
# Gráfico principal
# ---------------------------------------------------------
st.markdown("### Precios normalizados (base 100)")
render_section(
    "Evolución comparativa de los activos",
    "El gráfico permite comparar visualmente la trayectoria relativa de precios desde una base común.",
)

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
    st.warning("No hay datos numéricos para graficar base 100")
else:
    fig_norm = plot_normalized_prices(close_prices_chart)
    initially_visible_tickers = default_visible_trace_tickers(list(close_prices_chart.columns))
    for trace in fig_norm.data:
        if trace.name not in initially_visible_tickers:
            trace.visible = "legendonly"
    st.plotly_chart(fig_norm, width="stretch")
st.caption(
    "Nota visual: la continuidad de las líneas se suaviza solo para la visualización, "
    "debido a calendarios bursátiles distintos entre mercados. Esto no afecta cálculos financieros ni métricas del dashboard."
)

render_explanation_expander(
    "Cómo interpretar el gráfico base 100",
    [
        "Todas las líneas arrancan en 100 (misma base) para comparar desempeño relativo, no precios reales.",
        "Si un activo como BP (BP.L) termina en 120, significa que subió aprox. +20% desde el inicio del periodo seleccionado.",
        "Si un activo como Seven & i Holdings (3382.T) termina en 90, significa que cayó aprox. −10% desde el inicio del periodo.",
        "La curva que esté más arriba al final del periodo fue la de mejor rendimiento relativo en esa ventana.",
        "Una serie con subidas y bajadas más fuertes (por ejemplo, Carrefour (CA.PA) si presenta picos pronunciados) suele indicar mayor volatilidad.",
        "Nota: cuando comparas mercados distintos (Tokio, Toronto, México, Londres, París), puede haber ajustes visuales por calendario; esto no cambia las métricas.",
    ],
)


# ---------------------------------------------------------
# Resumen del portafolio
# ---------------------------------------------------------
st.markdown("### Resumen rápido del portafolio equiponderado")
render_section(
    "Métricas descriptivas del portafolio",
    "Este resumen concentra medidas básicas que ayudan a caracterizar retorno medio y dispersión del portafolio construido con pesos iguales.",
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

with st.expander("Resumen estadístico del portafolio", expanded=False):
    render_table(summary)


# ---------------------------------------------------------
# Últimos precios
# ---------------------------------------------------------
st.markdown("### Últimos precios disponibles")
with st.expander("Últimos precios", expanded=False):
    render_table(close_prices.tail(10).style.format("{:.2f}"), hide_index=False)


# ---------------------------------------------------------
# Interpretación
# ---------------------------------------------------------
st.markdown("### Interpretación general")

st.info(
    "La portada resume activos, periodo, rendimiento, volatilidad y evolución relativa para una primera lectura del portafolio equiponderado."
)
render_explanation_expander(
    "Cómo interpretar la portada",
    [
        "Esta portada resume el universo de activos (Seven & i Holdings (3382.T), Couche-Tard (ATD.TO), FEMSA (FEMSAUBD.MX), BP (BP.L) y Carrefour (CA.PA)), el horizonte y un primer perfil riesgo–retorno.",
        "Los KPIs muestran una lectura agregada del portafolio equiponderado: rendimiento anualizado y volatilidad anualizada.",
        "El gráfico base 100 permite ver rápidamente qué activo lidera y cuál es más inestable en el periodo elegido.",
        "Si quieres profundizar, los módulos M1–M9 separan el análisis en técnica, rendimientos, volatilidad, riesgo, optimización y decisión.",
    ],
)


# ---------------------------------------------------------
# Estructura del dashboard
# ---------------------------------------------------------
st.markdown("### Estructura del dashboard")
with st.expander("Ver módulos del dashboard", expanded=False):
    st.markdown(
        """
        - **Contextualización:** lectura cualitativa y rol de los activos en el portafolio.
        - **M1. Análisis técnico:** indicadores y gráficos por activo.
        - **M2. Rendimientos:** estadística descriptiva y pruebas de normalidad.
        - **M3. Modelos GARCH:** comparación de modelos de volatilidad.
        - **M4. CAPM y Beta:** beta, CAPM y benchmark local.
        - **M5. VaR/CVaR:** riesgo del portafolio con 3 métodos.
        - **M6. Optimización Markowitz:** frontera eficiente y portafolios óptimos.
        - **M7. Señales:** alertas automáticas de trading.
        - **M8. Macro y Benchmark:** contexto macro y comparación contra índice global.
        - **M9. Panel de decisión:** integración final para postura de acción.
        """
    )
