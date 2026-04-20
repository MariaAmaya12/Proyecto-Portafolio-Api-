import streamlit as st
import pandas as pd
import streamlit.components.v1 as components

from src.config import (
    APP_TITLE,
    ASSET_TICKERS,
    DEFAULT_END_DATE,
    DEFAULT_START_DATE,
    GLOBAL_BENCHMARK,
    ensure_project_dirs,
)
from src.download import data_error_message, load_market_bundle
from src.preprocess import (
    equal_weight_portfolio,
    annualize_return,
    annualize_volatility,
)
from src.plots import plot_normalized_prices
from src.ui_navigation import render_sidebar_navigation
from src.ui_style import apply_global_typography, render_page_title


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


def section_intro(title: str, subtitle: str):
    st.markdown(
        f"""
        <div class="section-box">
            <div class="section-title">{title}</div>
            <div class="section-subtitle">{subtitle}</div>
        </div>
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


def insight_box(title: str, items, tone: str = "info"):
    if isinstance(items, str):
        body = f"<div>{sanitize_text(items)}</div>"
    else:
        body = "<ul>" + "".join(f"<li>{sanitize_text(item)}</li>" for item in items) + "</ul>"

    st.markdown(
        f"""
        <div class="insight-box {sanitize_text(tone)}">
            <div class="insight-title">{sanitize_text(title)}</div>
            {body}
        </div>
        """,
        unsafe_allow_html=True,
    )


def kpi_card(title, value, delta=None, delta_type="neu", caption=""):
    title = sanitize_text(title)
    value = sanitize_text(value)
    delta = sanitize_text(delta) if delta is not None else ""
    caption = sanitize_text(caption)

    delta_html = ""
    if delta:
        delta_html = f'<div class="kpi-delta {delta_type}">{delta}</div>'

    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            body {{
                margin: 0;
                padding: 0;
                background: transparent;
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            }}

            .kpi-card {{
                background: linear-gradient(180deg, #f4f8ff 0%, #eaf3ff 100%);
                border: 1px solid #c9ddfc;
                border-radius: 18px;
                padding: 18px;
                box-shadow: 0 8px 22px rgba(37, 99, 235, 0.08);
                min-height: 156px;
                height: 156px;
                box-sizing: border-box;
                display: grid;
                grid-template-rows: auto auto 30px 1fr;
                align-items: start;
            }}

            .kpi-label {{
                font-size: 0.88rem;
                font-weight: 600;
                color: #274c77;
                line-height: 1.25;
                margin-bottom: 0.45rem;
                min-height: 1.1rem;
            }}

            .kpi-value {{
                font-size: 1.78rem;
                font-weight: 800;
                color: #0f3d75;
                line-height: 1.1;
                margin-bottom: 0.35rem;
                word-break: break-word;
            }}

            .kpi-delta-slot {{
                min-height: 30px;
                display: flex;
                align-items: flex-start;
            }}

            .kpi-delta {{
                display: inline-block;
                width: fit-content;
                font-size: 0.80rem;
                font-weight: 700;
                padding: 0.28rem 0.55rem;
                border-radius: 999px;
                margin-top: 0.10rem;
            }}

            .kpi-delta.pos {{
                background-color: #dcfce7;
                border: 1px solid #bbf7d0;
                color: #166534;
            }}

            .kpi-delta.neg {{
                background-color: #fee2e2;
                border: 1px solid #fecaca;
                color: #991b1b;
            }}

            .kpi-delta.neu {{
                background-color: #e0ecff;
                border: 1px solid #c9ddfc;
                color: #274c77;
            }}

            .kpi-caption {{
                font-size: 0.78rem;
                color: #5f6f86;
                line-height: 1.35;
                align-self: end;
            }}
        </style>
    </head>
    <body>
        <div class="kpi-card">
            <div class="kpi-label">{title}</div>
            <div class="kpi-value">{value}</div>
            <div class="kpi-delta-slot">{delta_html}</div>
            <div class="kpi-caption">{caption}</div>
        </div>
    </body>
    </html>
    """
    components.html(html, height=172)


inject_ui_css()
render_sidebar_navigation()


# ---------------------------------------------------------
# Cache de datos
# ---------------------------------------------------------
@st.cache_data(show_spinner=False)
def get_market_data(tickers, start, end):
    return load_market_bundle(tickers=tickers, start=start, end=end)


# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------
all_tickers = list(ASSET_TICKERS.values())
ASSET_DISPLAY_LABELS = {
    "CA.PA": "Carrefour (CA.PA)",
    "3382.T": "Honda (3382.T)",
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


def toggle_home_ticker(ticker: str) -> None:
    selected = set(st.session_state.get("home_selected_tickers", default_home_tickers))
    if ticker in selected:
        selected.remove(ticker)
    else:
        selected.add(ticker)

    st.session_state["home_selected_tickers"] = [
        item for item in all_tickers if item in selected
    ]


def asset_display_label(ticker: str) -> str:
    return ASSET_DISPLAY_LABELS.get(ticker, ticker)

with st.sidebar:
    st.header("Parámetros")

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
    st.subheader("Selección de activos")
    st.caption("Activos del portafolio")

    asset_cols = st.columns(2)
    for index, ticker in enumerate(all_tickers):
        is_selected = ticker in st.session_state["home_selected_tickers"]
        with asset_cols[index % 2]:
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
    if st.button("Restablecer selección", use_container_width=True):
        st.session_state["home_selected_tickers"] = default_home_tickers
        st.rerun()

    st.caption("Estos activos alimentan los KPIs, el gráfico normalizado y el resumen de la portada.")


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
    "Vista ejecutiva para analizar riesgo, rendimiento y comportamiento relativo del portafolio.",
)

# Periodo y contexto se muestran de forma compacta en el resumen ejecutivo.


# ---------------------------------------------------------
# Descarga de datos
# ---------------------------------------------------------
try:
    with st.spinner("Descargando datos de mercado..."):
        market_data = get_market_data(
            tickers=selected_tickers,
            start=str(start_date),
            end=str(end_date),
        )
except Exception as e:
    st.error(data_error_message(f"Ocurrió un error al descargar los datos: {e}"))
    st.stop()

if market_data is None:
    st.error("No se recibieron datos del mercado.")
    st.stop()

if "close" not in market_data or market_data["close"].empty:
    st.error(data_error_message("No fue posible descargar precios. Verifica conexión, fechas o tickers."))
    st.stop()

if "returns" not in market_data or market_data["returns"].empty:
    st.error("No fue posible calcular rendimientos con los datos descargados.")
    st.stop()


# ---------------------------------------------------------
# Variables principales
# ---------------------------------------------------------
close_prices = market_data["close"]
returns = market_data["returns"]
portfolio_returns = equal_weight_portfolio(returns)

ann_return = annualize_return(portfolio_returns)
ann_vol = annualize_volatility(portfolio_returns)
obs_count = int(close_prices.shape[0])
asset_count = len(selected_tickers)

ret_delta = "Sesgo positivo" if ann_return > 0 else "Sesgo negativo" if ann_return < 0 else "Sin sesgo"
ret_delta_type = "pos" if ann_return > 0 else "neg" if ann_return < 0 else "neu"

vol_delta = "Mayor dispersión" if ann_vol > 0.20 else "Dispersión moderada"
vol_delta_type = "neg" if ann_vol > 0.20 else "neu"


# ---------------------------------------------------------
# Resumen ejecutivo
# ---------------------------------------------------------
st.markdown("### Resumen ejecutivo")
section_intro(
    "Lectura general del dashboard",
    "Este bloque resume los activos seleccionados, la ventana temporal analizada y el comportamiento agregado del portafolio equiponderado.",
)

info_col1, info_col2, info_col3 = st.columns([1.6, 1, 0.9])

with info_col1:
    summary_chip("Activo(s)", ", ".join(selected_tickers))

with info_col2:
    summary_chip("Periodo", f"{start_date} a {end_date}")

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
        caption="Número de precios disponibles en la muestra",
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


# ---------------------------------------------------------
# Gráfico principal
# ---------------------------------------------------------
st.markdown("### Precios normalizados (base 100)")
section_intro(
    "Evolución comparativa de los activos",
    "El gráfico permite comparar visualmente la trayectoria relativa de precios desde una base común.",
)

# Solo para visualizacion: suaviza huecos por calendarios bursatiles distintos.
close_prices_chart = close_prices.ffill()
fig_norm = plot_normalized_prices(close_prices_chart)
st.plotly_chart(fig_norm, width="stretch")
st.caption(
    "Nota visual: la continuidad de las líneas se suaviza solo para la visualización, "
    "debido a calendarios bursátiles distintos entre mercados. Esto no afecta cálculos financieros ni métricas del dashboard."
)

insight_box(
    "Cómo leer el gráfico (Base 100)",
    [
        "Todas las líneas arrancan en 100 (misma base) para comparar desempeño relativo, no precios reales.",
        "Si un activo como BP (BP.L) termina en 120, significa que subió aprox. +20% desde el inicio del periodo seleccionado.",
        "Si un activo como Honda (3382.T) termina en 90, significa que cayó aprox. −10% desde el inicio del periodo.",
        "La curva que esté más arriba al final del periodo fue la de mejor rendimiento relativo en esa ventana.",
        "Una serie con subidas y bajadas más fuertes (por ejemplo, Carrefour (CA.PA) si presenta picos pronunciados) suele indicar mayor volatilidad.",
        "Nota: cuando comparas mercados distintos (Tokio, Toronto, México, Londres, París), puede haber ajustes visuales por calendario; esto no cambia las métricas.",
    ],
)


# ---------------------------------------------------------
# Resumen del portafolio
# ---------------------------------------------------------
st.markdown("### Resumen rápido del portafolio equiponderado")
section_intro(
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

with st.expander("Resumen estadístico del portafolio"):
    st.dataframe(summary, width="stretch", hide_index=True)


# ---------------------------------------------------------
# Últimos precios
# ---------------------------------------------------------
st.markdown("### Últimos precios disponibles")
with st.expander("Últimos precios"):
    st.dataframe(
        close_prices.tail(10).style.format("{:.2f}"),
        width="stretch",
    )


# ---------------------------------------------------------
# Interpretación
# ---------------------------------------------------------
st.markdown("### Interpretación general")

insight_box(
    "Lectura rápida de la portada",
    [
        "Esta portada resume el universo de activos (Honda (3382.T), Couche-Tard (ATD.TO), FEMSA (FEMSAUBD.MX), BP (BP.L) y Carrefour (CA.PA)), el horizonte y un primer perfil riesgo–retorno.",
        "Los KPIs muestran una lectura agregada del portafolio equiponderado: rendimiento anualizado y volatilidad anualizada.",
        "El gráfico base 100 permite ver rápidamente qué activo lidera y cuál es más inestable en el periodo elegido.",
        "Si quieres profundizar, los módulos M1–M9 separan el análisis en técnica, rendimientos, volatilidad, riesgo, optimización y decisión.",
    ],
    tone="positive",
)


# ---------------------------------------------------------
# Estructura del dashboard
# ---------------------------------------------------------
st.markdown("### Estructura del dashboard")
with st.expander("Ver módulos del dashboard"):
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
