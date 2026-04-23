import streamlit as st
import streamlit.components.v1 as components
import pandas as pd

from src.config import (
    ASSETS,
    GLOBAL_BENCHMARK,
    DEFAULT_START_DATE,
    DEFAULT_END_DATE,
    ensure_project_dirs,
)
from src.api.backend_client import BackendAPIError, friendly_error_message
from src.download import data_error_message
from src.preprocess import equal_weight_portfolio
from src.api.macro import macro_snapshot
from src.benchmark import benchmark_summary
from src.plots import plot_benchmark_base100
from src.services.market_data_client import MarketDataClient
from src.ui_navigation import render_sidebar_navigation
from src.ui_style import apply_global_typography, render_page_title

ensure_project_dirs()
apply_global_typography()
render_sidebar_navigation()


# ==============================
# Estilos UI
# ==============================
def inject_kpi_cards_css():
    st.markdown(
        """
        <style>
        .section-intro-box {
            background: #ffffff;
            border: 1px solid rgba(15, 23, 42, 0.08);
            border-radius: 18px;
            padding: 16px 18px;
            box-shadow: 0 4px 14px rgba(15, 23, 42, 0.06);
            margin-bottom: 0.75rem;
        }

        .section-intro-title {
            font-size: 1rem;
            font-weight: 700;
            color: #0f172a;
            margin-bottom: 0.2rem;
        }

        .section-intro-subtitle {
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
        <div class="section-intro-box">
            <div class="section-intro-title">{title}</div>
            <div class="section-intro-subtitle">{subtitle}</div>
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
                min-height: 124px;
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
                letter-spacing: 0.2px;
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
                background-color: rgba(100, 116, 139, 0.10);
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

    components.html(html, height=145)


inject_kpi_cards_css()

render_page_title(
    "Módulo 8 - Contexto macro y benchmark",
    "Compara el desempeño del portafolio frente a un benchmark global y contextualiza los resultados con variables macroeconómicas.",
)

# ==============================
# Sidebar
# ==============================
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
        start_date = st.date_input("Fecha inicial", value=DEFAULT_START_DATE, key="bm_start")
        end_date = st.date_input("Fecha final", value=DEFAULT_END_DATE, key="bm_end")

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

    mostrar_estado_macro = False
    mostrar_interpretacion_tecnica = False

    with st.expander("Filtros secundarios"):
        if modo == "Estadístico":
            mostrar_interpretacion_tecnica = st.checkbox(
                "Mostrar interpretación técnica",
                value=True,
            )
            mostrar_estado_macro = st.checkbox(
                "Mostrar estado de carga macro",
                value=False,
            )
        else:
            mostrar_estado_macro = st.checkbox(
                "Mostrar estado de carga macro",
                value=False,
            )

# ==============================
# Resumen del módulo
# ==============================
st.markdown("### Resumen del módulo")
if modo == "General":
    st.write(
        """
        Este módulo permite comparar si el portafolio tuvo un comportamiento mejor, similar o peor
        que su índice de referencia. Además, muestra variables macroeconómicas que ayudan a entender
        el entorno financiero del periodo analizado.
        """
    )
else:
    st.write(
        """
        Este módulo evalúa el desempeño relativo del portafolio frente a un benchmark mediante métricas
        como Alpha de Jensen, Tracking Error, Information Ratio y máximo drawdown, incorporando además
        variables macroeconómicas relevantes para contextualizar el análisis.
        """
    )

st.caption(f"Periodo analizado: {start_date} a {end_date}")

# ==============================
# Datos
# ==============================
tickers = [meta["ticker"] for meta in ASSETS.values()] + [GLOBAL_BENCHMARK]
market_client = MarketDataClient()
try:
    bundle = market_client.fetch_bundle(tickers=tickers, start=str(start_date), end=str(end_date))
except BackendAPIError as exc:
    st.error(friendly_error_message(exc, "No fue posible obtener datos de mercado desde el backend."))
    if exc.technical_detail:
        st.caption(exc.technical_detail)
    st.stop()

missing_tickers = market_client.missing_tickers(bundle)
if missing_tickers:
    st.warning(
        "Sin datos para estos tickers en el rango seleccionado; se excluyen del analisis: "
        + ", ".join(missing_tickers)
    )

returns = bundle["returns"].dropna()

if returns.empty or GLOBAL_BENCHMARK not in returns.columns:
    st.error(data_error_message("No fue posible construir benchmark global."))
    st.stop()

portfolio_returns = equal_weight_portfolio(
    returns[[c for c in returns.columns if c != GLOBAL_BENCHMARK]]
)
benchmark_returns = returns[GLOBAL_BENCHMARK]

macro = macro_snapshot()
rf_annual = (
    macro["risk_free_rate_pct"] / 100
    if macro["risk_free_rate_pct"] == macro["risk_free_rate_pct"]
    else 0.03
)

summary_df, extras_df, cum_port, cum_bench = benchmark_summary(
    portfolio_returns=portfolio_returns,
    benchmark_returns=benchmark_returns,
    rf_annual=rf_annual,
)

# ==============================
# Contexto macro
# ==============================
if "source" in macro and macro["source"]:
    st.caption(f"Fuente macro: {macro['source']}")

if "last_updated" in macro and macro["last_updated"]:
    st.caption(f"Última actualización: {macro['last_updated']}")

if mostrar_estado_macro:
    with st.expander("Ver estado de carga de variables macro"):
        if macro["inflation_yoy"] != macro["inflation_yoy"]:
            st.warning("No se pudo obtener inflación desde API. Usando fallback o valor no disponible.")

        if macro["usdcop_market"] != macro["usdcop_market"]:
            st.warning("No se pudo obtener USD/COP spot desde API. Usando fallback o valor no disponible.")

        if macro["cop_per_usd"] != macro["cop_per_usd"]:
            st.warning("No se pudo obtener USD/COP promedio anual desde API. Usando fallback o valor no disponible.")

# ==============================
# KPIs macroeconómicos
# ==============================
st.markdown("### Indicadores macroeconómicos")
section_intro(
    "Contexto macro relevante",
    "Estas variables ayudan a interpretar el entorno financiero en el que se evaluó el portafolio y su benchmark.",
)

rf_pct = macro["risk_free_rate_pct"] if macro["risk_free_rate_pct"] == macro["risk_free_rate_pct"] else None
inflation_yoy = macro["inflation_yoy"] if macro["inflation_yoy"] == macro["inflation_yoy"] else None
usdcop_market = macro["usdcop_market"] if macro["usdcop_market"] == macro["usdcop_market"] else None
cop_per_usd = macro["cop_per_usd"] if macro["cop_per_usd"] == macro["cop_per_usd"] else None

infl_delta = None
infl_delta_type = "neu"
if inflation_yoy is not None:
    if inflation_yoy > 0.05:
        infl_delta = "Inflación alta"
        infl_delta_type = "neg"
    else:
        infl_delta = "Inflación moderada"
        infl_delta_type = "neu"

fx_delta = None
fx_delta_type = "neu"
if usdcop_market is not None and cop_per_usd is not None:
    if usdcop_market > cop_per_usd:
        fx_delta = "Spot sobre promedio"
        fx_delta_type = "neg"
    elif usdcop_market < cop_per_usd:
        fx_delta = "Spot bajo promedio"
        fx_delta_type = "pos"

col1, col2, col3, col4 = st.columns(4)

with col1:
    kpi_card(
        "Tasa libre de riesgo (%)",
        f"{rf_pct:.2f}" if rf_pct is not None else "N/D",
        caption="Usada en métricas relativas de desempeño",
    )

with col2:
    kpi_card(
        "Inflación interanual",
        f"{inflation_yoy:.2%}" if inflation_yoy is not None else "N/D",
        delta=infl_delta,
        delta_type=infl_delta_type,
        caption="Variación interanual del nivel de precios",
    )

with col3:
    kpi_card(
        "USD/COP (spot)",
        f"{usdcop_market:.2f}" if usdcop_market is not None else "N/D",
        delta=fx_delta,
        delta_type=fx_delta_type,
        caption="Tasa de cambio observada en mercado",
    )

with col4:
    kpi_card(
        "USD/COP (promedio anual)",
        f"{cop_per_usd:.2f}" if cop_per_usd is not None else "N/D",
        caption="Promedio anual de referencia",
    )

# ==============================
# Comparación visual
# ==============================
st.markdown("### Comparación visual")
section_intro(
    "Portafolio vs benchmark",
    "El gráfico base 100 permite comparar el desempeño acumulado del portafolio frente a su índice de referencia.",
)

st.plotly_chart(plot_benchmark_base100(cum_port, cum_bench), width="stretch")

if modo == "General":
    st.info(
        """
        **Cómo leer este gráfico**

        - Si la línea del portafolio termina por encima del benchmark, hubo mejor desempeño acumulado.
        - Si ambas líneas se mueven muy parecido, el portafolio siguió de cerca al índice de referencia.
        - Caídas pronunciadas indican periodos de pérdida acumulada y mayor presión de riesgo.
        """
    )
else:
    with st.expander("Ver interpretación técnica del gráfico"):
        st.write(
            """
            El gráfico base 100 permite comparar trayectorias acumuladas normalizadas del portafolio
            y del benchmark. La separación entre ambas curvas refleja desempeño relativo, mientras que
            la amplitud de las caídas ayuda a identificar episodios de drawdown y sensibilidad a choques
            de mercado.
            """
        )

# ==============================
# KPIs de desempeño relativo
# ==============================
st.markdown("### KPIs de desempeño relativo")
section_intro(
    "Resumen ejecutivo del benchmark",
    "Estas métricas resumen retorno acumulado, generación de alpha y grado de desviación frente al benchmark.",
)

try:
    ret_port = float(summary_df.loc[summary_df["serie"] == "Portafolio", "ret_acumulado"].iloc[0])
except Exception:
    ret_port = None

try:
    ret_bench = float(summary_df.loc[summary_df["serie"] == "Benchmark", "ret_acumulado"].iloc[0])
except Exception:
    ret_bench = None

try:
    alpha_jensen = float(extras_df.loc[extras_df["métrica"] == "Alpha de Jensen", "valor"].iloc[0])
except Exception:
    alpha_jensen = None

try:
    tracking_error = float(extras_df.loc[extras_df["métrica"] == "Tracking Error", "valor"].iloc[0])
except Exception:
    tracking_error = None

ret_delta = None
ret_delta_type = "neu"
if ret_port is not None and ret_bench is not None:
    diff_ret = ret_port - ret_bench
    ret_delta = f"Diferencia: {diff_ret:.2%}"
    ret_delta_type = "pos" if diff_ret > 0 else "neg" if diff_ret < 0 else "neu"

alpha_delta = None
alpha_delta_type = "neu"
if alpha_jensen is not None:
    if alpha_jensen > 0:
        alpha_delta = "Alpha positivo"
        alpha_delta_type = "pos"
    elif alpha_jensen < 0:
        alpha_delta = "Alpha negativo"
        alpha_delta_type = "neg"

te_delta = None
te_delta_type = "neu"
if tracking_error is not None:
    if tracking_error > 0.05:
        te_delta = "Alta desviación"
        te_delta_type = "neg"
    else:
        te_delta = "Desviación moderada"
        te_delta_type = "neu"

c1, c2, c3, c4 = st.columns(4)

with c1:
    kpi_card(
        "Retorno acumulado portafolio",
        f"{ret_port:.2%}" if ret_port is not None else "N/D",
        delta=ret_delta,
        delta_type=ret_delta_type,
        caption="Desempeño acumulado del portafolio",
    )

with c2:
    kpi_card(
        "Retorno acumulado benchmark",
        f"{ret_bench:.2%}" if ret_bench is not None else "N/D",
        caption="Desempeño acumulado del benchmark global",
    )

with c3:
    kpi_card(
        "Alpha de Jensen",
        f"{alpha_jensen:.4f}" if alpha_jensen is not None else "N/D",
        delta=alpha_delta,
        delta_type=alpha_delta_type,
        caption="Exceso de desempeño ajustado por riesgo",
    )

with c4:
    kpi_card(
        "Tracking Error",
        f"{tracking_error:.4f}" if tracking_error is not None else "N/D",
        delta=te_delta,
        delta_type=te_delta_type,
        caption="Desviación del portafolio frente al benchmark",
    )

# ==============================
# Tablas
# ==============================
st.markdown("### Tablas de resultados")
if mostrar_tablas:
    st.subheader("Desempeño: portafolio vs benchmark")
    st.dataframe(summary_df, width="stretch")

    st.subheader("Métricas adicionales")
    st.dataframe(extras_df, width="stretch")
else:
    with st.expander("Ver tablas completas de resultados"):
        st.subheader("Desempeño: portafolio vs benchmark")
        st.dataframe(summary_df, width="stretch")

        st.subheader("Métricas adicionales")
        st.dataframe(extras_df, width="stretch")

# ==============================
# Interpretación
# ==============================
if modo == "General":
    st.markdown("### Interpretación")
    st.success(
        """
        **Lectura sencilla de resultados**

        - El benchmark sirve como punto de comparación para saber si el portafolio realmente agregó valor.
        - Un alpha positivo sugiere mejor desempeño que el esperado según su riesgo de mercado.
        - Un tracking error alto indica que el portafolio se aleja bastante del benchmark.
        - Un drawdown alto indica que en algún momento hubo una caída acumulada fuerte.
        """
    )
else:
    st.markdown("### Interpretación técnica")
    if mostrar_interpretacion_tecnica:
        st.info(
            """
            **Interpretación del benchmark y del contexto macro**

            - El gráfico base 100 permite comparar visualmente la trayectoria acumulada del portafolio frente al benchmark.
            - Un **Alpha de Jensen** positivo sugiere que el portafolio obtuvo un desempeño superior al explicado por su nivel de riesgo sistemático.
            - Un **Tracking Error** alto indica mayor desviación frente al benchmark, mientras que uno bajo sugiere un comportamiento más cercano al índice de referencia.
            - El **Information Ratio** resume cuánto retorno activo genera el portafolio por unidad de riesgo activo.
            - El **máximo drawdown** muestra la peor caída acumulada desde un máximo previo, y es clave para evaluar pérdidas severas.
            - El contexto macroeconómico, en particular la tasa libre de riesgo, la inflación y la tasa de cambio, ayuda a interpretar el entorno financiero en el que se evalúa el portafolio.
            """
        )
