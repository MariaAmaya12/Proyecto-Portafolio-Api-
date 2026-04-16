import streamlit as st
import streamlit.components.v1 as components
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
section_intro(
    "Resumen ejecutivo del activo",
    "Aquí se resumen los indicadores técnicos más importantes para una lectura rápida de tendencia, momentum y nivel actual del precio.",
)

close_now = float(ind["Close"].iloc[-1]) if "Close" in ind.columns else None
rsi_now = float(ind[f"RSI_{rsi_window}"].iloc[-1]) if f"RSI_{rsi_window}" in ind.columns else None
sma_now = float(ind[f"SMA_{sma_window}"].iloc[-1]) if f"SMA_{sma_window}" in ind.columns else None
ema_now = float(ind[f"EMA_{ema_window}"].iloc[-1]) if f"EMA_{ema_window}" in ind.columns else None
close_prev = float(ind["Close"].iloc[-2]) if "Close" in ind.columns and len(ind) > 1 else None

precio_delta = None
precio_delta_type = "neu"
if close_now is not None and close_prev is not None and close_prev != 0:
    price_change = (close_now / close_prev) - 1
    precio_delta = f"{price_change:.2%} vs sesión previa"
    precio_delta_type = "pos" if price_change > 0 else "neg" if price_change < 0 else "neu"

rsi_delta = None
rsi_delta_type = "neu"
if rsi_now is not None:
    if rsi_now >= 70:
        rsi_delta = "Sobrecompra"
        rsi_delta_type = "neg"
    elif rsi_now <= 30:
        rsi_delta = "Sobreventa"
        rsi_delta_type = "pos"
    else:
        rsi_delta = "Zona neutral"
        rsi_delta_type = "neu"

sma_delta = None
sma_delta_type = "neu"
if close_now is not None and sma_now is not None and sma_now != 0:
    dist_sma = (close_now / sma_now) - 1
    sma_delta = f"{dist_sma:.2%} vs SMA"
    sma_delta_type = "pos" if dist_sma > 0 else "neg" if dist_sma < 0 else "neu"

ema_delta = None
ema_delta_type = "neu"
if close_now is not None and ema_now is not None and ema_now != 0:
    dist_ema = (close_now / ema_now) - 1
    ema_delta = f"{dist_ema:.2%} vs EMA"
    ema_delta_type = "pos" if dist_ema > 0 else "neg" if dist_ema < 0 else "neu"

col1, col2, col3, col4 = st.columns(4)

with col1:
    kpi_card(
        "Precio actual",
        f"{close_now:,.2f}" if close_now is not None else "N/D",
        delta=precio_delta,
        delta_type=precio_delta_type,
        caption="Último precio de cierre disponible",
    )

with col2:
    kpi_card(
        f"RSI ({rsi_window})",
        f"{rsi_now:.2f}" if rsi_now is not None else "N/D",
        delta=rsi_delta,
        delta_type=rsi_delta_type,
        caption="Momentum reciente del activo",
    )

with col3:
    kpi_card(
        f"SMA ({sma_window})",
        f"{sma_now:,.2f}" if sma_now is not None else "N/D",
        delta=sma_delta,
        delta_type=sma_delta_type,
        caption="Promedio móvil simple para tendencia",
    )

with col4:
    kpi_card(
        f"EMA ({ema_window})",
        f"{ema_now:,.2f}" if ema_now is not None else "N/D",
        delta=ema_delta,
        delta_type=ema_delta_type,
        caption="Promedio móvil exponencial más sensible",
    )

# ==============================
# Gráfico principal
# ==============================
st.markdown("### Tendencia del precio")
section_intro(
    "Precio y medias móviles",
    "Este gráfico permite comparar la trayectoria del precio con sus referencias de tendencia de corto y mediano plazo.",
)

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
section_intro(
    "Momentum y dispersión",
    "Aquí se muestran señales clave para identificar zonas extremas, fuerza relativa y cambios en la volatilidad implícita del precio.",
)

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
    section_intro(
        "Señales complementarias",
        "Estos indicadores amplían la lectura de momentum y posibles cambios de dirección mediante señales adicionales de aceleración o agotamiento.",
    )

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