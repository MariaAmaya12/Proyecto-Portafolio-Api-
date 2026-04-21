import streamlit as st
import pandas as pd
from pydantic import ValidationError

from src.config import ASSETS, DEFAULT_START_DATE, DEFAULT_END_DATE, get_ticker, ensure_project_dirs
from src.download import data_error_message, download_single_ticker
from src.indicators import compute_all_indicators
from src.plots import (
    plot_price_and_mas,
    plot_bollinger,
    plot_rsi,
    plot_macd,
    plot_stochastic,
)
from src.ui_navigation import render_sidebar_navigation
from src.ui_style import apply_global_typography, render_page_title
from src.signal import compute_signal
from src.ticker_validation import (
    PORTFOLIO_VALIDATION_MESSAGE,
    asset_name_for_ticker,
    validate_portfolio_ticker,
)

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

        .asset-focus {
            align-items: center;
            background: linear-gradient(180deg, #eef6ff 0%, #f8fbff 100%);
            border: 1px solid rgba(37, 99, 235, 0.18);
            border-radius: 14px;
            box-shadow: 0 4px 14px rgba(37, 99, 235, 0.07);
            display: flex;
            justify-content: space-between;
            gap: 1rem;
            margin: -0.15rem 0 1rem;
            padding: 16px 18px;
        }

        .asset-focus-label {
            color: #1e3a8a;
            font-size: 0.78rem;
            font-weight: 850;
            letter-spacing: 0.02em;
            margin-bottom: 0.18rem;
            text-transform: uppercase;
        }

        .asset-focus-name {
            color: #0f172a;
            font-size: clamp(1.25rem, 2.3vw, 1.75rem);
            font-weight: 900;
            line-height: 1.12;
        }

        .asset-focus-meta {
            color: #475569;
            font-size: 0.88rem;
            font-weight: 700;
            line-height: 1.35;
            text-align: right;
            white-space: nowrap;
        }

        @media (max-width: 760px) {
            .asset-focus {
                align-items: flex-start;
                flex-direction: column;
            }

            .asset-focus-meta {
                text-align: left;
                white-space: normal;
            }
        }

        .kpi-card {
            background: #eef6ff;
            border: 1px solid rgba(37, 99, 235, 0.18);
            border-radius: 14px;
            padding: 20px 18px 18px;
            box-shadow: 0 4px 14px rgba(37, 99, 235, 0.08);
            min-height: 168px;
            height: 100%;
        }

        .kpi-label {
            color: #1e3a8a;
            font-size: 0.9rem;
            font-weight: 800;
            line-height: 1.2;
            margin-bottom: 0.35rem;
        }

        .kpi-value {
            color: #0f172a;
            font-size: clamp(1.7rem, 2.6vw, 2.25rem);
            font-weight: 900;
            line-height: 1.1;
            overflow-wrap: anywhere;
            margin-bottom: 0.55rem;
        }

        .kpi-delta {
            display: inline-block;
            width: fit-content;
            font-size: 0.78rem;
            font-weight: 800;
            padding: 0.24rem 0.55rem;
            border-radius: 999px;
            margin-bottom: 0.45rem;
        }

        .kpi-delta.pos {
            background-color: rgba(22, 163, 74, 0.12);
            color: #15803d;
        }

        .kpi-delta.neg {
            background-color: rgba(220, 38, 38, 0.12);
            color: #b91c1c;
        }

        .kpi-delta.neu {
            background-color: rgba(100, 116, 139, 0.13);
            color: #475569;
        }

        .kpi-caption {
            color: #475569;
            font-size: 0.82rem;
            font-weight: 650;
            line-height: 1.3;
            overflow-wrap: anywhere;
        }

        .kpi-spacer {
            height: 1.65rem;
            margin-bottom: 0.45rem;
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

    delta_html = (
        f'<div class="kpi-delta {delta_type}">{delta}</div>'
        if delta
        else '<div class="kpi-spacer"></div>'
    )

    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">{title}</div>
            <div class="kpi-value">{value}</div>
            {delta_html}
            <div class="kpi-caption">{caption}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def fmt_number(value, decimals: int = 2) -> str:
    if value is None or pd.isna(value):
        return "N/D"
    return f"{value:,.{decimals}f}"


def fmt_pct(value) -> str:
    if value is None or pd.isna(value):
        return "N/D"
    return f"{value:.2%}"


def latest_value(df: pd.DataFrame, column: str):
    if column not in df.columns:
        return None
    clean = df[column].dropna()
    if clean.empty:
        return None
    return float(clean.iloc[-1])


def delta_type(value) -> str:
    if value is None or pd.isna(value):
        return "neu"
    return "pos" if value > 0 else "neg" if value < 0 else "neu"


def rsi_state(value) -> tuple[str, str]:
    if value is None or pd.isna(value):
        return "Sin señal", "neu"
    if value > 70:
        return "Sobrecompra", "neg"
    if value < 30:
        return "Sobreventa", "pos"
    return "Zona neutral", "neu"


def signal_badge_type(signal: str) -> str:
    if signal == "Alcista":
        return "pos"
    if signal == "Bajista":
        return "neg"
    return "neu"


def signal_detail(signal: str) -> str:
    if signal == "Alcista":
        return "Precio y RSI favorables"
    if signal == "Bajista":
        return "Precio y RSI débiles"
    return "Señal sin confirmación"


def ema_vs_sma_text(ema_now, sma_now) -> str:
    if any(value is None or pd.isna(value) for value in [ema_now, sma_now]):
        return "no hay datos suficientes para comparar EMA y SMA."
    if ema_now > sma_now:
        return "la EMA está por encima de la SMA, lo que sugiere que el precio reciente viene mejorando frente a la tendencia suavizada."
    if ema_now < sma_now:
        return "la EMA está por debajo de la SMA, lo que sugiere pérdida de impulso reciente frente a la tendencia suavizada."
    return "la EMA y la SMA están prácticamente alineadas, sin ventaja clara del impulso reciente."


def explain_moving_averages(close_now, sma_now, ema_now) -> str:
    if any(value is None or pd.isna(value) for value in [close_now, sma_now, ema_now]):
        today = "faltan datos suficientes para una lectura completa de medias móviles."
    elif close_now > sma_now and close_now > ema_now:
        today = "el precio está sobre SMA y EMA, una lectura favorable de tendencia."
    elif close_now < sma_now and close_now < ema_now:
        today = "el precio está por debajo de ambas medias, lo que sugiere debilidad técnica."
    else:
        today = "el precio está entre medias o sin confirmación clara, por lo que la lectura es mixta."
    return f"""
- **Qué mide:** SMA y EMA resumen la dirección del precio; la EMA reacciona más rápido a cambios recientes.
- **Cómo se interpreta:** precio sobre las medias sugiere fortaleza; precio bajo las medias sugiere debilidad.
- **EMA vs SMA:** {ema_vs_sma_text(ema_now, sma_now)}
- **Qué significa hoy:** {today}
"""


def explain_rsi(rsi_now) -> str:
    if rsi_now is None or pd.isna(rsi_now):
        today = "no hay datos suficientes para calcular una lectura actual."
    elif rsi_now > 70:
        today = "el RSI está en zona alta, con posible agotamiento relativo."
    elif rsi_now < 30:
        today = "el RSI está en zona baja, con presión bajista reciente."
    elif 45 <= rsi_now <= 55:
        today = "el RSI está cerca de 50, en una zona neutral sin señal extrema."
    elif rsi_now > 55:
        today = "el momentum reciente es moderadamente positivo, sin llegar a sobrecompra."
    else:
        today = "el momentum reciente es moderadamente débil, sin llegar a sobreventa."
    return f"""
- **Qué mide:** intensidad de los movimientos recientes del precio.
- **Cómo se interpreta:** cerca de 70 refleja presión compradora extendida; cerca de 50 indica momentum neutral; cerca de 30 refleja presión vendedora extendida.
- **Qué significa hoy:** {today}
"""


def explain_bollinger(close_now, bb_low, bb_mid, bb_up) -> str:
    if any(value is None or pd.isna(value) for value in [close_now, bb_low, bb_mid, bb_up]):
        today = "faltan datos suficientes para ubicar el precio frente a las bandas."
    elif close_now >= bb_up * 0.98:
        today = "el precio está cerca de la banda superior, con presión alcista relativa."
    elif close_now <= bb_low * 1.02:
        today = "el precio está cerca de la banda inferior, con debilidad reciente."
    elif abs(close_now / bb_mid - 1) <= 0.02:
        today = "el precio está dentro de las bandas y alrededor de la media, sin extensión extrema."
    elif close_now > bb_mid:
        today = "el precio está dentro de las bandas y por encima de la media, con sesgo constructivo."
    else:
        today = "el precio está dentro de las bandas y por debajo de la media, con lectura prudente."
    return f"""
- **Qué mide:** volatilidad relativa y rango probable del precio alrededor de una media móvil.
- **Cómo se interpreta:** bandas en expansión indican mayor volatilidad; bandas comprimidas sugieren menor rango. Tocar extremos puede señalar presión alcista o bajista relativa.
- **Qué significa hoy:** {today}
"""


def explain_macd(macd_now, signal_now, hist_now) -> str:
    if any(value is None or pd.isna(value) for value in [macd_now, signal_now, hist_now]):
        today = "faltan datos suficientes para leer la señal actual."
    elif macd_now > signal_now and hist_now > 0:
        today = "el MACD está sobre su señal y el histograma es positivo, una lectura de momentum favorable."
    elif macd_now < signal_now and hist_now < 0:
        today = "el MACD está bajo su señal y el histograma es negativo, una lectura de debilidad."
    else:
        today = "el MACD está cerca del cruce y sin confirmación fuerte."
    return f"""
- **Qué mide:** aceleración o pérdida de momentum a partir de medias exponenciales.
- **Cómo se interpreta:** MACD sobre la señal favorece lectura alcista; debajo, lectura bajista. Histograma positivo confirma impulso favorable; negativo confirma debilidad.
- **Qué significa hoy:** {today}
"""


def explain_stochastic(k_now, d_now) -> str:
    if any(value is None or pd.isna(value) for value in [k_now, d_now]):
        today = "faltan datos suficientes para evaluar el oscilador."
    elif 35 <= k_now <= 65 and 35 <= d_now <= 65:
        today = "el estocástico está en zona media, sin señal extrema clara."
    elif k_now > d_now:
        today = "%K está por encima de %D, lo que sugiere mejora de corto plazo."
    else:
        today = "%K está por debajo de %D, con pérdida de impulso reciente."
    return f"""
- **Qué mide:** ubicación del cierre frente al rango reciente de precios.
- **Cómo se interpreta:** lecturas sobre 80 son altas y bajo 20 son bajas; cruces entre %K y %D ayudan a leer timing e impulso de corto plazo.
- **Qué significa hoy:** {today}
"""


PLOT_CONFIG = {"displayModeBar": False, "responsive": True}


def prepare_price_fig(fig):
    fig.update_layout(height=520, margin=dict(l=42, r=24, t=56, b=42))
    return fig


def prepare_rsi_fig(fig):
    fig.add_hrect(y0=70, y1=100, fillcolor="#FEE2E2", opacity=0.18, line_width=0)
    fig.add_hrect(y0=0, y1=30, fillcolor="#DBEAFE", opacity=0.18, line_width=0)
    fig.add_hline(y=70, line_dash="dash", line_color="#DC2626", line_width=1.6)
    fig.add_hline(y=30, line_dash="dash", line_color="#2563EB", line_width=1.6)
    fig.add_hline(y=50, line_dash="dot", line_color="rgba(100,116,139,0.55)", line_width=1.1)
    fig.update_layout(height=430, margin=dict(l=42, r=24, t=52, b=42))
    return fig


def prepare_bollinger_fig(fig):
    for trace in fig.data:
        if trace.name == "Close":
            trace.line.color = "#1D4ED8"
            trace.line.width = 2.4
        elif trace.name == "BB_up":
            trace.line.color = "#06B6D4"
            trace.line.width = 1.8
        elif trace.name == "BB_mid":
            trace.line.color = "#94A3B8"
            trace.line.width = 1.7
            trace.line.dash = "dot"
        elif trace.name == "BB_low":
            trace.line.color = "#F87171"
            trace.line.width = 1.8
    fig.update_layout(height=470, margin=dict(l=42, r=24, t=52, b=42))
    return fig


def prepare_macd_fig(fig):
    for trace in fig.data:
        if trace.name == "MACD":
            trace.line.color = "#2563EB"
            trace.line.width = 2.4
        elif trace.name == "MACD_signal":
            trace.line.color = "#D97706"
            trace.line.width = 2.2
        elif trace.name == "MACD_hist":
            trace.marker.color = "#93C5FD"
            trace.opacity = 0.65
    fig.add_hline(y=0, line_color="rgba(100,116,139,0.7)", line_width=1)
    fig.update_layout(height=430, margin=dict(l=42, r=24, t=52, b=42))
    return fig


def prepare_stochastic_fig(fig):
    for trace in fig.data:
        if trace.name == "%K":
            trace.line.color = "#2563EB"
            trace.line.width = 2.5
        elif trace.name == "%D":
            trace.line.color = "#D97706"
            trace.line.width = 2.2
    fig.add_hline(y=80, line_dash="dash", line_color="#DC2626", line_width=1.3)
    fig.add_hline(y=20, line_dash="dash", line_color="#2563EB", line_width=1.3)
    fig.update_layout(height=430, margin=dict(l=42, r=24, t=52, b=42))
    return fig


inject_kpi_cards_css()

render_page_title(
    "Módulo 1 - Análisis técnico",
    "Explora tendencia, momentum y señales técnicas del activo seleccionado.",
)

# ==============================
# Sidebar
# ==============================
default_start = pd.to_datetime(DEFAULT_START_DATE).date()
default_end = pd.to_datetime(DEFAULT_END_DATE).date()

with st.sidebar:
    st.header("Parámetros técnicos")
    asset_name = st.selectbox("Activo", list(ASSETS.keys()), index=0)

    horizonte = st.selectbox(
        "Horizonte",
        ["1 mes", "Trimestre", "Semestre", "1 año", "2 años", "3 años", "5 años", "Personalizado"],
        index=3,
    )

    fecha_fin_ref = default_end
    if horizonte == "1 mes":
        start_date = (pd.Timestamp(fecha_fin_ref) - pd.DateOffset(months=1)).date()
        end_date = fecha_fin_ref
    elif horizonte == "Trimestre":
        start_date = (pd.Timestamp(fecha_fin_ref) - pd.DateOffset(months=3)).date()
        end_date = fecha_fin_ref
    elif horizonte == "Semestre":
        start_date = (pd.Timestamp(fecha_fin_ref) - pd.DateOffset(months=6)).date()
        end_date = fecha_fin_ref
    elif horizonte == "1 año":
        start_date = (pd.Timestamp(fecha_fin_ref) - pd.DateOffset(years=1)).date()
        end_date = fecha_fin_ref
    elif horizonte == "2 años":
        start_date = (pd.Timestamp(fecha_fin_ref) - pd.DateOffset(years=2)).date()
        end_date = fecha_fin_ref
    elif horizonte == "3 años":
        start_date = (pd.Timestamp(fecha_fin_ref) - pd.DateOffset(years=3)).date()
        end_date = fecha_fin_ref
    elif horizonte == "5 años":
        start_date = (pd.Timestamp(fecha_fin_ref) - pd.DateOffset(years=5)).date()
        end_date = fecha_fin_ref
    else:
        start_date = default_start
        end_date = default_end

    sma_window = 20
    ema_window = 20
    rsi_window = 14
    bb_window = 20
    stoch_window = 14
    usar_ticker_manual = False
    manual_ticker = ""
    ticker = get_ticker(asset_name)
    asset_label = asset_name

    with st.expander("Ajustes avanzados", expanded=True):
        usar_ticker_manual = st.checkbox("Ingresar ticker manual", value=False)
        if usar_ticker_manual:
            manual_ticker = st.text_input(
                "Ticker manual",
                value=get_ticker(asset_name),
                help="Solo se permiten tickers de los activos oficiales del portafolio.",
            )
            st.caption("Tickers habilitados: 3382.T, ATD.TO, FEMSAUBD.MX, BP.L y CA.PA.")
            try:
                ticker_input = validate_portfolio_ticker(manual_ticker)
            except ValidationError:
                st.error(PORTFOLIO_VALIDATION_MESSAGE)
                st.stop()

            ticker = ticker_input.ticker
            asset_label = asset_name_for_ticker(ticker) or asset_label

        if horizonte == "Personalizado":
            start_date = st.date_input("Fecha inicial", value=default_start, key="tec_start")
            end_date = st.date_input("Fecha final", value=default_end, key="tec_end")
        else:
            st.caption(f"Rango activo: {start_date} a {end_date}")

        if "df_prices" in st.session_state:
            csv_bytes = st.session_state["df_prices"].to_csv(index=False).encode("utf-8")
            st.download_button(
                "Exportar CSV de precios",
                data=csv_bytes,
                file_name=f"{ticker}_prices.csv",
                mime="text/csv",
                key="download_csv_sidebar",
            )

        st.markdown("**Ventanas de indicadores**")
        sma_window = st.slider("Ventana SMA", min_value=5, max_value=60, value=sma_window)
        ema_window = st.slider("Ventana EMA", min_value=5, max_value=60, value=ema_window)
        rsi_window = st.slider("Ventana RSI", min_value=5, max_value=30, value=rsi_window)
        bb_window = st.slider("Ventana Bollinger", min_value=10, max_value=60, value=bb_window)
        stoch_window = st.slider("Ventana Estocástico", min_value=5, max_value=30, value=stoch_window)

if start_date >= end_date:
    st.error("La fecha inicial debe ser anterior a la fecha final.")
    st.stop()

# ==============================
# Datos
# ==============================
try:
    df_prices = download_single_ticker(ticker=ticker, start=str(start_date), end=str(end_date))
    st.session_state["df_prices"] = df_prices
except Exception as exc:
    st.error(f"Error al descargar datos: {exc}")
    st.stop()

df = st.session_state["df_prices"]

if df.empty:
    st.error(data_error_message("No se pudieron descargar datos del activo seleccionado."))
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

rsi_col = f"RSI_{rsi_window}"
sma_col = f"SMA_{sma_window}"
ema_col = f"EMA_{ema_window}"
required_cols = [
    "Close",
    sma_col,
    ema_col,
    rsi_col,
    "BB_low",
    "BB_mid",
    "BB_up",
    "MACD",
    "MACD_signal",
    "MACD_hist",
    "%K",
    "%D",
]
missing_cols = [col for col in required_cols if col not in ind.columns]
if missing_cols:
    st.error(f"Faltan columnas técnicas para construir el análisis: {', '.join(missing_cols)}")
    st.stop()

chart_df = ind.dropna(subset=required_cols).copy()
if chart_df.empty:
    st.error("No hay suficientes datos completos para calcular todos los indicadores con las ventanas seleccionadas.")
    st.stop()

# ==============================
# Valores actuales
# ==============================
close_now = latest_value(chart_df, "Close")
close_prev = float(chart_df["Close"].iloc[-2]) if len(chart_df) > 1 else None
rsi_now = latest_value(chart_df, rsi_col)
sma_now = latest_value(chart_df, sma_col)
ema_now = latest_value(chart_df, ema_col)
bb_low_now = latest_value(chart_df, "BB_low")
bb_mid_now = latest_value(chart_df, "BB_mid")
bb_up_now = latest_value(chart_df, "BB_up")
macd_now = latest_value(chart_df, "MACD")
macd_signal_now = latest_value(chart_df, "MACD_signal")
macd_hist_now = latest_value(chart_df, "MACD_hist")
stoch_k_now = latest_value(chart_df, "%K")
stoch_d_now = latest_value(chart_df, "%D")

price_change = None
if close_now is not None and close_prev is not None and close_prev != 0:
    price_change = (close_now / close_prev) - 1

rsi_label, rsi_kind = rsi_state(rsi_now)
technical_signal = compute_signal(close_now, sma_now, ema_now, rsi_now)
technical_signal_kind = signal_badge_type(technical_signal)
technical_signal_detail = signal_detail(technical_signal)

# ==============================
# Encabezado
# ==============================
st.markdown("### Resumen del módulo")
section_intro(
    "Lectura técnica unificada",
    (
        "Este módulo permite evaluar la situación técnica del activo seleccionado a partir de tendencia, "
        "momentum y volatilidad. La lectura combina precio, medias móviles, RSI, Bandas de Bollinger, "
        "MACD y oscilador estocástico para identificar si el activo muestra fortaleza, debilidad o "
        "señales de cautela en el periodo analizado."
    ),
)
st.markdown(
    f"""
    <div class="asset-focus">
        <div>
            <div class="asset-focus-label">Activo seleccionado</div>
            <div class="asset-focus-name">{sanitize_text(asset_label)} ({sanitize_text(ticker)})</div>
        </div>
        <div class="asset-focus-meta">
            Periodo analizado<br>{start_date} a {end_date}
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ==============================
# KPIs ejecutivos
# ==============================
st.markdown("### KPIs ejecutivos")
col1, col2, col3, col4 = st.columns(4)

with col1:
    kpi_card(
        "Precio actual",
        fmt_number(close_now),
        delta="Cierre",
        delta_type="neu",
        caption="Último cierre disponible",
    )

with col2:
    kpi_card(
        "Variación previa",
        fmt_pct(price_change),
        delta="vs sesión previa" if price_change is not None else None,
        delta_type=delta_type(price_change),
        caption="Cambio frente al cierre anterior",
    )

with col3:
    kpi_card(
        "RSI (14)" if rsi_window == 14 else f"RSI ({rsi_window})",
        fmt_number(rsi_now),
        delta=rsi_label,
        delta_type=rsi_kind,
        caption="Momentum reciente",
    )

with col4:
    kpi_card(
        "Señal técnica",
        technical_signal,
        delta=technical_signal_detail,
        delta_type=technical_signal_kind,
        caption="Precio vs SMA/EMA + RSI",
    )

with st.expander("¿Cómo se calcula la señal técnica?", expanded=False):
    st.markdown(
        """
**Criterio de la señal**  
- Alcista ↗: precio > SMA y EMA **y** RSI ≥ 55  
- Bajista ↘: precio < SMA y EMA **y** RSI ≤ 45  
- Neutral ↔: resto de casos
"""
    )

# ==============================
# Bloque principal
# ==============================
st.markdown("### Tendencia del precio")
section_intro(
    "Precio, SMA y EMA",
    "Compara la trayectoria del precio con dos referencias de tendencia: una media simple y una media exponencial.",
)

st.plotly_chart(
    prepare_price_fig(plot_price_and_mas(chart_df, sma_col=sma_col, ema_col=ema_col)),
    width="stretch",
    config=PLOT_CONFIG,
)

st.info(explain_moving_averages(close_now, sma_now, ema_now))

# ==============================
# Indicadores esenciales
# ==============================
st.markdown("### Indicadores esenciales")
section_intro(
    "Momentum y volatilidad",
    "RSI y Bollinger ayudan a detectar zonas extremas y cambios en el rango reciente del precio.",
)

st.plotly_chart(
    prepare_rsi_fig(plot_rsi(chart_df, rsi_col=rsi_col)),
    width="stretch",
    config=PLOT_CONFIG,
)
st.info(explain_rsi(rsi_now))

st.plotly_chart(
    prepare_bollinger_fig(plot_bollinger(chart_df)),
    width="stretch",
    config=PLOT_CONFIG,
)
st.info(explain_bollinger(close_now, bb_low_now, bb_mid_now, bb_up_now))
with st.expander("¿Cómo leer las Bandas de Bollinger?", expanded=False):
    st.markdown(
        """
- **Close:** precio de cierre del activo.
- **BB_up:** banda superior; marca una zona alta frente al rango reciente.
- **BB_mid:** media móvil central; sirve como referencia de equilibrio.
- **BB_low:** banda inferior; marca una zona baja frente al rango reciente.
- **Interpretación:** acercarse a la banda superior sugiere presión alcista relativa; acercarse a la inferior sugiere debilidad o presión vendedora.
- **Volatilidad:** bandas en expansión indican mayor dispersión del precio; bandas comprimidas muestran menor rango y posible pausa antes de un nuevo movimiento.
"""
    )

# ==============================
# Bloque avanzado
# ==============================
st.markdown("### Bloque avanzado")
section_intro(
    "Señales complementarias de momentum",
    "MACD y estocástico agregan confirmación sobre aceleración, cruces y posibles zonas extendidas.",
)

st.plotly_chart(
    prepare_macd_fig(plot_macd(chart_df)),
    width="stretch",
    config=PLOT_CONFIG,
)
st.info(explain_macd(macd_now, macd_signal_now, macd_hist_now))
with st.expander("¿Cómo leer el MACD?", expanded=False):
    st.markdown(
        """
- **MACD:** mide diferencia entre medias exponenciales y resume cambios de momentum.
- **MACD_signal:** línea suavizada usada para confirmar cruces.
- **MACD_hist:** distancia entre MACD y señal; muestra aceleración o pérdida de impulso.
- **Cruces:** MACD sobre la señal favorece lectura alcista; por debajo, lectura bajista.
- **Histograma:** positivo confirma impulso favorable; negativo confirma debilidad.
- **Línea cero:** cruzar por encima de cero refuerza momentum positivo; cruzar por debajo refuerza cautela.
"""
    )

st.plotly_chart(
    prepare_stochastic_fig(plot_stochastic(chart_df)),
    width="stretch",
    config=PLOT_CONFIG,
)
st.info(explain_stochastic(stoch_k_now, stoch_d_now))
with st.expander("¿Cómo leer el oscilador estocástico?", expanded=False):
    st.markdown(
        """
- **%K:** línea rápida; ubica el cierre frente al rango reciente.
- **%D:** línea suavizada de %K; ayuda a confirmar la señal.
- **Zonas:** sobre 80 indica zona alta; bajo 20 indica zona baja.
- **Cruces:** %K sobre %D sugiere mejora de corto plazo; %K bajo %D sugiere pérdida de impulso.
- **Timing:** es útil para leer impulso táctico, especialmente cuando confirma la tendencia observada en precio y medias.
"""
    )

# ==============================
# Datos recientes
# ==============================
st.markdown("### Datos recientes")
with st.expander("Ver tabla técnica reciente", expanded=False):
    table_cols = [col for col in required_cols if col in chart_df.columns]
    st.dataframe(chart_df[table_cols].tail(15), width="stretch")
