import streamlit as st
import pandas as pd

from src.config import ASSETS, DEFAULT_START_DATE, DEFAULT_END_DATE, get_ticker, ensure_project_dirs
from src.download import download_single_ticker
from src.returns_analysis import (
    compute_return_series,
    descriptive_stats,
    normality_tests,
    qq_plot_data,
    stylized_facts_comment,
)
from src.plots import plot_histogram_with_normal, plot_qq, plot_box

ensure_project_dirs()

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
    st.error("No se pudieron descargar datos.")
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

mean_ret = series.mean()
vol_ret = series.std(ddof=1)
min_ret = series.min()
max_ret = series.max()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Promedio", f"{mean_ret:.4%}")
col2.metric("Volatilidad", f"{vol_ret:.4%}")
col3.metric("Mínimo", f"{min_ret:.4%}")
col4.metric("Máximo", f"{max_ret:.4%}")

# ==============================
# Tablas principales
# ==============================
st.markdown("### Resumen estadístico")

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