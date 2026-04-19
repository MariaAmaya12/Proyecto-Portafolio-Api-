import streamlit as st
import streamlit.components.v1 as components
import pandas as pd

from src.config import ASSETS, DEFAULT_START_DATE, DEFAULT_END_DATE, get_ticker, ensure_project_dirs
from src.download import data_error_message, download_single_ticker
from src.returns_analysis import (
    compute_return_series,
    descriptive_stats,
    normality_tests,
    qq_plot_data,
    stylized_facts_comment,
)
from src.plots import plot_histogram_with_normal, plot_qq, plot_box

ensure_project_dirs()


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

st.title("Módulo 2 - Rendimientos y propiedades empíricas")
st.caption("Analiza la distribución de rendimientos del activo y sus principales propiedades estadísticas.")

# ==============================
# Sidebar
# ==============================
with st.sidebar:
    st.header("Parámetros")
    asset_name = st.selectbox("Activo", list(ASSETS.keys()), index=0)

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
        start_date = st.date_input("Fecha inicial", value=DEFAULT_START_DATE, key="ret_start")
        end_date = st.date_input("Fecha final", value=DEFAULT_END_DATE, key="ret_end")

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
    mostrar_qq = st.checkbox("Mostrar gráfico Q-Q", value=(modo == "Estadístico"))

    with st.expander("Filtros secundarios"):
        return_type = st.radio(
            "Tipo de retorno",
            ["simple_return", "log_return"],
            index=1,
        )

# ==============================
# Datos
# ==============================
ticker = get_ticker(asset_name)
df = download_single_ticker(ticker=ticker, start=str(start_date), end=str(end_date))

if df.empty:
    st.error(data_error_message("No se pudieron descargar datos."))
    st.stop()

price_series = df["Adj Close"] if "Adj Close" in df.columns else df["Close"]
ret_df = compute_return_series(price_series)

if ret_df.empty or return_type not in ret_df.columns:
    st.error("No fue posible calcular la serie de rendimientos.")
    st.stop()

series = ret_df[return_type].dropna()

if series.empty:
    st.error("La serie de rendimientos está vacía.")
    st.stop()

desc_df = descriptive_stats(series)
norm_df = normality_tests(series)
qq_df = qq_plot_data(series)

# ==============================
# Resumen
# ==============================
st.markdown("### Resumen del módulo")
if modo == "General":
    st.write(
        f"""
        Este módulo analiza los rendimientos de **{asset_name} ({ticker})** para entender su comportamiento,
        dispersión y qué tan lejos están de una distribución normal.
        """
    )
else:
    st.write(
        f"""
        Este módulo evalúa los rendimientos de **{asset_name} ({ticker})** mediante estadísticos descriptivos,
        pruebas de normalidad y herramientas gráficas como histograma, boxplot y gráfico Q-Q.
        """
    )

st.caption(f"Periodo analizado: {start_date} a {end_date}")

# ==============================
# KPIs
# ==============================
st.markdown("### KPIs de rendimientos")
section_intro(
    "Resumen ejecutivo de la distribución",
    "Aquí se sintetizan la rentabilidad media, la volatilidad y los extremos observados en la serie de rendimientos seleccionada.",
)

mean_ret = series.mean()
vol_ret = series.std(ddof=1)
min_ret = series.min()
max_ret = series.max()
last_ret = series.iloc[-1] if len(series) > 0 else None

prom_delta = None
prom_delta_type = "neu"
if mean_ret > 0:
    prom_delta = "Sesgo positivo"
    prom_delta_type = "pos"
elif mean_ret < 0:
    prom_delta = "Sesgo negativo"
    prom_delta_type = "neg"

vol_delta = None
vol_delta_type = "neu"
if vol_ret >= series.abs().median():
    vol_delta = "Alta dispersión"
    vol_delta_type = "neg"
else:
    vol_delta = "Dispersión moderada"
    vol_delta_type = "pos"

min_delta = None
min_delta_type = "neg"
if last_ret is not None:
    min_delta = f"Último: {last_ret:.4%}"

max_delta = None
max_delta_type = "pos"
if last_ret is not None:
    max_delta = f"Último: {last_ret:.4%}"

col1, col2, col3, col4 = st.columns(4)

with col1:
    kpi_card(
        "Promedio",
        f"{mean_ret:.4%}",
        delta=prom_delta,
        delta_type=prom_delta_type,
        caption=f"Media de la serie de {return_type}",
    )

with col2:
    kpi_card(
        "Volatilidad",
        f"{vol_ret:.4%}",
        delta=vol_delta,
        delta_type=vol_delta_type,
        caption="Desviación estándar muestral de rendimientos",
    )

with col3:
    kpi_card(
        "Mínimo",
        f"{min_ret:.4%}",
        delta=min_delta,
        delta_type="neg",
        caption="Peor rendimiento observado en el periodo",
    )

with col4:
    kpi_card(
        "Máximo",
        f"{max_ret:.4%}",
        delta=max_delta,
        delta_type="pos",
        caption="Mejor rendimiento observado en el periodo",
    )

# ==============================
# Tablas principales
# ==============================
st.markdown("### Resumen estadístico")
section_intro(
    "Estadísticos y normalidad",
    "Estas tablas permiten contrastar medidas descriptivas de la serie y evaluar si la distribución se aparta de una normal teórica.",
)

if mostrar_tablas:
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Estadísticos descriptivos")
        st.dataframe(desc_df, width="stretch")
    with col2:
        st.markdown("#### Pruebas de normalidad")
        st.dataframe(norm_df, width="stretch")
else:
    with st.expander("Ver tablas estadísticas"):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Estadísticos descriptivos")
            st.dataframe(desc_df, width="stretch")
        with col2:
            st.markdown("#### Pruebas de normalidad")
            st.dataframe(norm_df, width="stretch")

# ==============================
# Gráficos principales
# ==============================
st.markdown("### Distribución de rendimientos")
section_intro(
    "Forma de la distribución",
    "Los gráficos permiten visualizar la dispersión, la asimetría y la presencia de valores extremos en los rendimientos.",
)

col3, col4 = st.columns(2)
with col3:
    st.plotly_chart(plot_histogram_with_normal(series), width="stretch")
with col4:
    st.plotly_chart(plot_box(series), width="stretch")

if modo == "General":
    st.info(
        """
        **Cómo leer estos gráficos**

        - El histograma muestra cómo se distribuyen los rendimientos del activo.
        - La curva normal sirve como referencia para comparar si la distribución real se parece o no a una normal.
        - El boxplot resume dispersión, mediana y posibles valores extremos.
        """
    )
else:
    with st.expander("Ver interpretación técnica de la distribución"):
        st.write(
            """
            El histograma permite contrastar la forma empírica de la distribución con la referencia normal teórica,
            mientras que el boxplot resume posición, dispersión, asimetría y presencia de valores atípicos.
            """
        )

# ==============================
# Gráfico Q-Q
# ==============================
if mostrar_qq:
    st.markdown("### Gráfico Q-Q")
    section_intro(
        "Contraste visual con normalidad",
        "El gráfico Q-Q permite verificar si los cuantiles observados siguen la forma esperada bajo una distribución normal.",
    )

    st.plotly_chart(plot_qq(qq_df), width="stretch")

    if modo == "General":
        st.info(
            """
            El gráfico Q-Q ayuda a ver si los rendimientos siguen una forma parecida a la distribución normal.
            Si los puntos se alejan mucho de la diagonal, la normalidad es cuestionable.
            """
        )
    else:
        with st.expander("Ver interpretación técnica del gráfico Q-Q"):
            st.write(
                """
                El gráfico Q-Q compara cuantiles muestrales frente a cuantiles teóricos normales. Desviaciones
                sistemáticas respecto a la recta de 45° sugieren asimetría, colas pesadas o no normalidad.
                """
            )

# ==============================
# Interpretación
# ==============================
st.markdown("### Interpretación")

if modo == "General":
    st.success(
        """
        **Lectura sencilla**

        - Este módulo muestra si los rendimientos son estables o si presentan variaciones fuertes.
        - También ayuda a detectar si el comportamiento del activo se parece a una distribución normal o si tiene eventos extremos.
        - Si hay mucha dispersión o colas pronunciadas, el riesgo puede ser mayor de lo que sugiere una aproximación normal simple.
        """
    )
else:
    st.info(stylized_facts_comment(series))

# ==============================
# Datos recientes
# ==============================
st.markdown("### Últimos rendimientos")
if mostrar_tablas:
    st.dataframe(ret_df.tail(15), width="stretch")
else:
    with st.expander("Ver últimos rendimientos"):
        st.dataframe(ret_df.tail(15), width="stretch")
