import streamlit as st
import pandas as pd
import streamlit.components.v1 as components

from src.config import (
    APP_TITLE,
    APP_SUBTITLE,
    ASSET_TICKERS,
    DEFAULT_END_DATE,
    DEFAULT_START_DATE,
    GLOBAL_BENCHMARK,
    ensure_project_dirs,
)
from src.download import load_market_bundle
from src.preprocess import (
    equal_weight_portfolio,
    annualize_return,
    annualize_volatility,
)
from src.plots import plot_normalized_prices


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
                background: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%);
                border: 1px solid rgba(15, 23, 42, 0.08);
                border-radius: 18px;
                padding: 18px 18px 14px 18px;
                box-shadow: 0 4px 14px rgba(15, 23, 42, 0.06);
                min-height: 126px;
                box-sizing: border-box;
                display: flex;
                flex-direction: column;
                justify-content: space-between;
            }}

            .kpi-label {{
                font-size: 0.88rem;
                font-weight: 600;
                color: #475569;
                margin-bottom: 0.35rem;
            }}

            .kpi-value {{
                font-size: 1.85rem;
                font-weight: 800;
                color: #0f172a;
                line-height: 1.1;
                margin-bottom: 0.45rem;
                word-break: break-word;
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
                background-color: rgba(22, 163, 74, 0.10);
                color: #15803d;
            }}

            .kpi-delta.neg {{
                background-color: rgba(220, 38, 38, 0.10);
                color: #b91c1c;
            }}

            .kpi-delta.neu {{
                background-color: rgba(100, 116, 139, 0.12);
                color: #475569;
            }}

            .kpi-caption {{
                font-size: 0.78rem;
                color: #64748b;
                margin-top: 0.65rem;
                line-height: 1.35;
            }}
        </style>
    </head>
    <body>
        <div class="kpi-card">
            <div>
                <div class="kpi-label">{title}</div>
                <div class="kpi-value">{value}</div>
                {delta_html}
            </div>
            <div class="kpi-caption">{caption}</div>
        </div>
    </body>
    </html>
    """
    components.html(html, height=150)


inject_ui_css()


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

with st.sidebar:
    st.header("Configuración general")

    horizonte = st.selectbox(
        "Horizonte de análisis",
        [
            "1 mes",
            "Trimestre",
            "Semestre",
            "1 año",
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
    st.subheader("Modo de visualización")
    modo = st.radio(
        "Selecciona el nivel de detalle",
        ["General", "Estadístico"],
        index=0,
    )

    st.divider()
    st.subheader("Opciones de visualización")
    mostrar_tablas = st.checkbox("Mostrar tablas completas", value=False)

    with st.expander("Filtros secundarios"):
        selected_tickers = st.multiselect(
            "Selecciona activos",
            options=all_tickers,
            default=all_tickers[: min(4, len(all_tickers))],
            help="Escoge uno o varios activos para construir el análisis.",
        )

        mostrar_estructura = st.checkbox("Mostrar estructura del dashboard", value=True)
        mostrar_ultimos_precios = st.checkbox("Mostrar últimos precios", value=True)

    st.divider()
    st.info(
        "La navegación entre módulos se realiza desde la barra lateral de Streamlit usando la carpeta `pages/`."
    )


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
st.title(APP_TITLE)
st.caption(APP_SUBTITLE)

if modo == "General":
    st.markdown(
        """
        Dashboard de análisis financiero enfocado en gestión del riesgo y portafolios.
        Integra módulos de análisis técnico, rendimientos, volatilidad, CAPM, VaR/CVaR,
        optimización de Markowitz, señales y benchmark.
        """
    )
else:
    st.markdown(
        """
        Panel principal de síntesis para el proyecto integrador. Resume la estructura del portafolio,
        el comportamiento agregado de precios normalizados y métricas básicas del portafolio equiponderado
        como punto de entrada al resto de módulos cuantitativos y de decisión.
        """
    )

st.caption(f"Periodo analizado: {start_date} a {end_date}")


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
    st.error(f"Ocurrió un error al descargar los datos: {e}")
    st.stop()

if market_data is None:
    st.error("No se recibieron datos del mercado.")
    st.stop()

if "close" not in market_data or market_data["close"].empty:
    st.error("No fue posible descargar precios. Verifica conexión, fechas o tickers.")
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

info_col1, info_col2 = st.columns([2, 1])

with info_col1:
    st.markdown(
        f"""
        **Activos seleccionados:** {', '.join(selected_tickers)}  
        **Periodo analizado:** {start_date} → {end_date}
        """
    )

with info_col2:
    st.markdown(f"**Benchmark global:** {GLOBAL_BENCHMARK}")


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

fig_norm = plot_normalized_prices(close_prices)
st.plotly_chart(fig_norm, width="stretch")

if modo == "General":
    st.info(
        """
        **Cómo interpretar este gráfico**
        - Todos los activos comienzan en una base comparable de 100.
        - Si una serie sube más que otra, su desempeño relativo fue mejor en la ventana elegida.
        - La separación entre curvas permite identificar activos más estables, más volátiles o más rentables.
        """
    )
else:
    st.info(
        """
        El gráfico de precios normalizados permite contrastar trayectorias relativas sin sesgo por nivel de precio inicial.
        Es útil para identificar divergencias de desempeño, episodios de mayor volatilidad relativa y cambios en la dispersión del conjunto analizado.
        """
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

if mostrar_tablas:
    st.dataframe(summary, width="stretch", hide_index=True)
else:
    with st.expander("Ver resumen estadístico"):
        st.dataframe(summary, width="stretch", hide_index=True)


# ---------------------------------------------------------
# Últimos precios
# ---------------------------------------------------------
if mostrar_ultimos_precios:
    st.markdown("### Últimos precios disponibles")
    if mostrar_tablas:
        st.dataframe(
            close_prices.tail(10).style.format("{:.2f}"),
            width="stretch",
        )
    else:
        with st.expander("Ver últimos precios"):
            st.dataframe(
                close_prices.tail(10).style.format("{:.2f}"),
                width="stretch",
            )


# ---------------------------------------------------------
# Interpretación
# ---------------------------------------------------------
st.markdown("### Interpretación general")

if modo == "General":
    st.success(
        """
        **Lectura sencilla**

        - Esta página sirve como puerta de entrada al dashboard.
        - Resume rápidamente qué activos están siendo analizados, cómo evolucionaron sus precios y cuál fue el comportamiento básico del portafolio equiponderado.
        - A partir de aquí, los demás módulos profundizan en técnica, rendimientos, riesgo, CAPM, optimización y decisión.
        """
    )
else:
    st.info(
        """
        **Lectura técnica**

        - La portada resume el universo de análisis y ofrece una primera caracterización del portafolio equiponderado.
        - El rendimiento anualizado y la volatilidad anualizada permiten una lectura preliminar del perfil riesgo-retorno.
        - El gráfico base 100 sirve como referencia visual para contrastar desempeño relativo antes de entrar a los módulos especializados.
        """
    )


# ---------------------------------------------------------
# Estructura del dashboard
# ---------------------------------------------------------
if mostrar_estructura:
    st.markdown("### Estructura del dashboard")
    st.markdown(
        """
        - **00 Contextualización:** lectura cualitativa y rol de los activos en el portafolio.
        - **01 Técnico:** indicadores y gráficos por activo.
        - **02 Rendimientos:** estadística descriptiva y pruebas de normalidad.
        - **03 GARCH:** comparación de modelos de volatilidad.
        - **04 CAPM:** beta, CAPM y benchmark local.
        - **05 VaR/CVaR:** riesgo del portafolio con 3 métodos.
        - **06 Markowitz:** frontera eficiente y portafolios óptimos.
        - **07 Señales:** alertas automáticas de trading.
        - **08 Macro y benchmark:** contexto macro y comparación contra índice global.
        - **09 Panel de decisión:** integración final para postura de acción.
        """
    )