import streamlit as st
import pandas as pd

from src.config import ASSETS, DEFAULT_START_DATE, DEFAULT_END_DATE, get_ticker, ensure_project_dirs
from src.download import download_single_ticker
from src.indicators import compute_all_indicators
from src.plots import (
    plot_price_and_mas,
    plot_bollinger,
    plot_rsi,
    plot_macd,
    plot_stochastic,
)

ensure_project_dirs()

st.title("Módulo 1 - Análisis técnico")
st.caption("Explora tendencia, momentum y señales técnicas del activo seleccionado.")

# ==============================
# Sidebar
# ==============================
with st.sidebar:
    st.header("Parámetros técnicos")
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
        start_date = st.date_input("Fecha inicial", value=DEFAULT_START_DATE, key="tec_start")
        end_date = st.date_input("Fecha final", value=DEFAULT_END_DATE, key="tec_end")

    st.divider()
    st.subheader("Modo de visualización")
    modo = st.radio(
        "Selecciona el nivel de detalle",
        ["General", "Estadístico"],
        index=0,
    )

    st.divider()
    st.subheader("Opciones de visualización")
    mostrar_tabla = st.checkbox("Mostrar tabla técnica", value=False)

    mostrar_indicadores_avanzados = False
    if modo == "Estadístico":
        mostrar_indicadores_avanzados = st.checkbox("Mostrar indicadores avanzados", value=True)

    with st.expander("Filtros secundarios"):
        sma_window = st.slider("Ventana SMA", min_value=5, max_value=60, value=20)
        ema_window = st.slider("Ventana EMA", min_value=5, max_value=60, value=20)
        rsi_window = st.slider("Ventana RSI", min_value=5, max_value=30, value=14)
        bb_window = st.slider("Ventana Bollinger", min_value=10, max_value=60, value=20)
        stoch_window = st.slider("Ventana Estocástico", min_value=5, max_value=30, value=14)

# ==============================
# Datos
# ==============================
ticker = get_ticker(asset_name)
df = download_single_ticker(ticker=ticker, start=str(start_date), end=str(end_date))

if df.empty:
    st.error("No se pudieron descargar datos del activo seleccionado.")
    st.stop()

ind = compute_all_indicators(
    df,
    sma_window=sma_window,
    ema_window=ema_window,
    rsi_window=rsi_window,
    bb_window=bb_window,
    stoch_window=stoch_window,
)

if ind.empty:
    st.error("No fue posible calcular indicadores técnicos.")
    st.stop()

# ==============================
# Resumen
# ==============================
st.markdown("### Resumen del módulo")
if modo == "General":
    st.write(
        f"""
        Este módulo permite analizar el comportamiento reciente de **{asset_name} ({ticker})** mediante
        herramientas de análisis técnico orientadas a identificar tendencia, zonas de sobrecompra o sobreventa
        y posibles cambios de momentum.
        """
    )
else:
    st.write(
        f"""
        Este módulo calcula indicadores técnicos sobre **{asset_name} ({ticker})**, incluyendo medias móviles,
        Bandas de Bollinger, RSI, MACD y oscilador estocástico, para evaluar tendencia, dispersión y señales
        de momentum del activo.
        """
    )

st.caption(f"Periodo analizado: {start_date} a {end_date}")

# ==============================
# KPIs
# ==============================
st.markdown("### KPIs del activo")

close_now = float(ind["Close"].iloc[-1]) if "Close" in ind.columns else None
rsi_now = float(ind[f"RSI_{rsi_window}"].iloc[-1]) if f"RSI_{rsi_window}" in ind.columns else None
sma_now = float(ind[f"SMA_{sma_window}"].iloc[-1]) if f"SMA_{sma_window}" in ind.columns else None
ema_now = float(ind[f"EMA_{ema_window}"].iloc[-1]) if f"EMA_{ema_window}" in ind.columns else None

col1, col2, col3, col4 = st.columns(4)
col1.metric("Precio actual", f"{close_now:,.2f}" if close_now is not None else "N/D")
col2.metric(f"RSI ({rsi_window})", f"{rsi_now:.2f}" if rsi_now is not None else "N/D")
col3.metric(f"SMA ({sma_window})", f"{sma_now:,.2f}" if sma_now is not None else "N/D")
col4.metric(f"EMA ({ema_window})", f"{ema_now:,.2f}" if ema_now is not None else "N/D")

# ==============================
# Gráfico principal
# ==============================
st.markdown("### Tendencia del precio")
st.plotly_chart(
    plot_price_and_mas(ind, sma_col=f"SMA_{sma_window}", ema_col=f"EMA_{ema_window}"),
    width="stretch",
)

if modo == "General":
    st.info(
        """
        **Cómo leer este gráfico**

        - La serie principal muestra la evolución del precio.
        - La **SMA** suaviza el movimiento del precio y ayuda a ver la tendencia general.
        - La **EMA** responde más rápido a cambios recientes, por lo que puede anticipar giros de corto plazo.
        """
    )
else:
    with st.expander("Ver interpretación técnica de medias móviles"):
        st.write(
            """
            La comparación entre precio, SMA y EMA permite evaluar tendencia y velocidad de ajuste.
            Cuando la EMA se separa o cruza la SMA, puede sugerir cambios recientes en el momentum
            del activo. Una SMA estable ayuda a identificar la dirección tendencial dominante.
            """
        )

# ==============================
# Indicadores esenciales
# ==============================
st.markdown("### Indicadores esenciales")

col1, col2 = st.columns(2)

with col1:
    st.plotly_chart(
        plot_rsi(ind, rsi_col=f"RSI_{rsi_window}"),
        width="stretch",
    )
    if modo == "General":
        st.info("RSI mayor a 70 puede sugerir sobrecompra; menor a 30, sobreventa.")
    else:
        with st.expander("Ver interpretación técnica del RSI"):
            st.write(
                """
                El RSI resume la fuerza relativa de los movimientos recientes del precio. Niveles altos
                pueden sugerir sobrecompra y niveles bajos sobreventa, aunque siempre deben interpretarse
                junto con la tendencia del activo.
                """
            )

with col2:
    st.plotly_chart(
        plot_bollinger(ind),
        width="stretch",
    )
    if modo == "General":
        st.info("Las Bandas de Bollinger muestran dispersión alrededor de la media móvil.")
    else:
        with st.expander("Ver interpretación técnica de Bollinger"):
            st.write(
                """
                Las Bandas de Bollinger permiten evaluar dispersión y episodios de mayor o menor volatilidad.
                Expansiones de bandas sugieren aumento de volatilidad, mientras que contracciones pueden anticipar
                periodos de compresión del precio.
                """
            )

# ==============================
# Indicadores avanzados
# ==============================
if modo == "Estadístico" and mostrar_indicadores_avanzados:
    st.markdown("### Indicadores avanzados")

    col3, col4 = st.columns(2)

    with col3:
        st.plotly_chart(plot_macd(ind), width="stretch")
        with st.expander("Ver interpretación técnica del MACD"):
            st.write(
                """
                El MACD captura diferencias entre medias exponenciales y ayuda a identificar cambios de momentum.
                Los cruces entre la línea MACD y la señal, así como la dinámica del histograma, pueden sugerir
                aceleración o desaceleración del movimiento del precio.
                """
            )

    with col4:
        st.plotly_chart(plot_stochastic(ind), width="stretch")
        with st.expander("Ver interpretación técnica del oscilador estocástico"):
            st.write(
                """
                El oscilador estocástico compara el cierre actual con el rango reciente del precio. Valores altos
                y bajos pueden señalar zonas extremas, aunque su lectura es más útil cuando se combina con tendencia
                y otros indicadores de confirmación.
                """
            )

# ==============================
# Lectura ejecutiva
# ==============================
st.markdown("### Interpretación")

if modo == "General":
    st.success(
        """
        **Lectura sencilla**

        - Este módulo ayuda a ver si el activo mantiene tendencia, si viene perdiendo fuerza o si está en una zona extrema.
        - RSI y Bollinger dan señales rápidas de posible sobrecompra, sobreventa o cambios de volatilidad.
        - Las medias móviles ayudan a distinguir si el comportamiento reciente es consistente con una tendencia.
        """
    )
else:
    st.info(
        """
        **Lectura técnica**

        - El análisis técnico se apoya en indicadores de tendencia, momentum y dispersión.
        - SMA y EMA resumen trayectoria y sensibilidad reciente del precio.
        - RSI, MACD y estocástico ayudan a evaluar agotamiento o aceleración del movimiento.
        - Bollinger aporta contexto sobre volatilidad relativa y compresión/expansión del rango.
        """
    )

# ==============================
# Tabla técnica
# ==============================
st.markdown("### Datos recientes")
if mostrar_tabla:
    st.dataframe(ind.tail(15), width="stretch")
else:
    with st.expander("Ver últimos datos con indicadores"):
        st.dataframe(ind.tail(15), width="stretch")