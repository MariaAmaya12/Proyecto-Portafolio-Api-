import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from src.config import ASSETS, DEFAULT_START_DATE, DEFAULT_END_DATE, ensure_project_dirs
from src.download import download_single_ticker
from src.indicators import compute_all_indicators
from src.signals import evaluate_signals

ensure_project_dirs()
st.title("Módulo 7 - Señales y alertas")
st.caption("Resume señales técnicas por activo y las traduce en una lectura operativa más clara.")


SIGNAL_LABELS = {
    "macd_buy": "MACD compra",
    "macd_sell": "MACD venta",
    "rsi_buy": "RSI sobreventa",
    "rsi_sell": "RSI sobrecompra",
    "boll_buy": "Bollinger compra",
    "boll_sell": "Bollinger venta",
    "golden_cross": "Golden cross",
    "death_cross": "Death cross",
    "stoch_buy": "Estocástico compra",
    "stoch_sell": "Estocástico venta",
}


# ==============================
# UI helpers
# ==============================
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


def signal_card(asset_name, ticker, recommendation, semaforo_estado, semaforo_msg, score_buy, score_sell, active_text, level):
    styles = {
        "positive": {
            "bg": "linear-gradient(180deg, #ecfdf5 0%, #f0fdf4 100%)",
            "border": "rgba(22, 163, 74, 0.22)",
            "pill_bg": "rgba(22, 163, 74, 0.14)",
            "pill_color": "#15803d",
        },
        "warning": {
            "bg": "linear-gradient(180deg, #fffbeb 0%, #fefce8 100%)",
            "border": "rgba(234, 179, 8, 0.28)",
            "pill_bg": "rgba(234, 179, 8, 0.16)",
            "pill_color": "#a16207",
        },
        "danger": {
            "bg": "linear-gradient(180deg, #fff1f2 0%, #fef2f2 100%)",
            "border": "rgba(220, 38, 38, 0.22)",
            "pill_bg": "rgba(220, 38, 38, 0.12)",
            "pill_color": "#b91c1c",
        },
    }

    s = styles.get(level, styles["warning"])

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
            .card {{
                background: {s["bg"]};
                border: 1px solid {s["border"]};
                border-radius: 20px;
                padding: 18px;
                min-height: 250px;
                box-sizing: border-box;
                box-shadow: 0 4px 14px rgba(15, 23, 42, 0.06);
            }}
            .asset {{
                font-size: 1rem;
                font-weight: 800;
                color: #0f172a;
                margin-bottom: 0.4rem;
            }}
            .ticker {{
                font-size: 0.82rem;
                color: #64748b;
                margin-bottom: 0.8rem;
            }}
            .pill {{
                display: inline-block;
                padding: 6px 10px;
                border-radius: 999px;
                background: {s["pill_bg"]};
                color: {s["pill_color"]};
                font-size: 0.82rem;
                font-weight: 800;
                margin-bottom: 0.8rem;
            }}
            .subtitle {{
                font-size: 0.82rem;
                font-weight: 700;
                color: #334155;
                margin-top: 0.5rem;
                margin-bottom: 0.2rem;
            }}
            .text {{
                font-size: 0.82rem;
                color: #334155;
                line-height: 1.45;
                margin-bottom: 0.4rem;
            }}
            .scores {{
                display: flex;
                gap: 10px;
                margin-top: 0.6rem;
                margin-bottom: 0.4rem;
            }}
            .score-box {{
                flex: 1;
                background: rgba(255,255,255,0.72);
                border: 1px solid rgba(15, 23, 42, 0.06);
                border-radius: 12px;
                padding: 10px 12px;
            }}
            .score-label {{
                font-size: 0.74rem;
                color: #64748b;
                margin-bottom: 0.15rem;
            }}
            .score-value {{
                font-size: 1.05rem;
                font-weight: 800;
                color: #0f172a;
            }}
        </style>
    </head>
    <body>
        <div class="card">
            <div class="asset">{sanitize_text(asset_name)}</div>
            <div class="ticker">{sanitize_text(ticker)}</div>
            <div class="pill">{sanitize_text(recommendation)}</div>

            <div class="subtitle">Semáforo</div>
            <div class="text"><strong>{sanitize_text(semaforo_estado)}</strong> · {sanitize_text(semaforo_msg)}</div>

            <div class="scores">
                <div class="score-box">
                    <div class="score-label">Score compra</div>
                    <div class="score-value">{score_buy}</div>
                </div>
                <div class="score-box">
                    <div class="score-label">Score venta</div>
                    <div class="score-value">{score_sell}</div>
                </div>
            </div>

            <div class="subtitle">Señales activas</div>
            <div class="text">{sanitize_text(active_text)}</div>
        </div>
    </body>
    </html>
    """
    components.html(html, height=290)


inject_ui_css()


# ==============================
# Helpers lógicos
# ==============================
def normalize_flags(flags: dict) -> dict:
    return {k: bool(v) for k, v in flags.items()}


def active_signals(flags: dict) -> list[str]:
    clean = normalize_flags(flags)
    return [SIGNAL_LABELS.get(k, k) for k, v in clean.items() if v]


def signal_table(flags: dict) -> pd.DataFrame:
    clean = normalize_flags(flags)
    return pd.DataFrame(
        {
            "Señal": [SIGNAL_LABELS.get(k, k) for k in clean.keys()],
            "Activa": ["Sí" if v else "No" for v in clean.values()],
        }
    )


def classify_signal_risk(signal: dict) -> dict:
    recommendation = str(signal.get("recommendation", "")).lower()
    score_buy = int(signal.get("score_buy", 0))
    score_sell = int(signal.get("score_sell", 0))
    reasons = signal.get("reasons", []) or []

    if "compra" in recommendation or score_buy >= score_sell + 2:
        return {
            "estado": "Favorable",
            "ui": "positive",
            "mensaje": (
                "Predominan señales de fortaleza o entrada táctica de corto plazo."
            ),
        }

    if "venta" in recommendation or score_sell >= score_buy + 2:
        return {
            "estado": "Desfavorable",
            "ui": "danger",
            "mensaje": (
                "Predominan señales de deterioro técnico, agotamiento o presión bajista."
            ),
        }

    if len(reasons) == 0 and score_buy == score_sell:
        return {
            "estado": "Neutral",
            "ui": "warning",
            "mensaje": (
                "No hay una ventaja técnica clara; el activo luce en zona de espera."
            ),
        }

    return {
        "estado": "Mixto / Precaución",
        "ui": "warning",
        "mensaje": (
            "Existen señales mixtas o poco concluyentes entre tendencia, momentum y reversión."
        ),
    }


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
        start_date = st.date_input("Fecha inicial", value=DEFAULT_START_DATE, key="sig_start")
        end_date = st.date_input("Fecha final", value=DEFAULT_END_DATE, key="sig_end")

    st.divider()
    st.subheader("Modo de visualización")
    modo = st.radio(
        "Selecciona el nivel de detalle",
        ["General", "Estadístico"],
        index=0,
    )

    st.divider()
    st.subheader("Opciones de visualización")
    mostrar_detalle = st.checkbox("Mostrar detalle por activo", value=False)

    with st.expander("Filtros secundarios"):
        rsi_overbought = st.slider("RSI sobrecompra", min_value=60, max_value=90, value=70)
        rsi_oversold = st.slider("RSI sobreventa", min_value=10, max_value=40, value=30)
        stoch_overbought = st.slider("Estocástico sobrecompra", min_value=60, max_value=95, value=80)
        stoch_oversold = st.slider("Estocástico sobreventa", min_value=5, max_value=40, value=20)


# ==============================
# Introducción
# ==============================
section_intro(
    "Cómo leer este módulo",
    "Cada activo se resume en una tarjeta con recomendación, semáforo técnico, score de compra/venta y señales activas más relevantes.",
)

if modo == "General":
    st.info(
        """
        **Semáforo interpretativo**
        - **Favorable**: predominan señales de entrada o fortaleza técnica.
        - **Neutral / Precaución**: señales mixtas o sin ventaja clara.
        - **Desfavorable**: predominan señales de venta, agotamiento o deterioro técnico.

        Estas señales orientan la lectura táctica, pero no reemplazan el análisis de riesgo, benchmark ni contexto macro.
        """
    )
else:
    st.info(
        """
        Este módulo sintetiza señales provenientes de MACD, RSI, Bandas de Bollinger, cruces de medias y oscilador estocástico.
        La recomendación final agrega evidencia de tendencia, reversión y momentum para construir una lectura técnica por activo.
        """
    )


# ==============================
# Construcción de tarjetas
# ==============================
cards_data = []

for asset_name, meta in ASSETS.items():
    ticker = meta["ticker"]
    df = download_single_ticker(ticker=ticker, start=str(start_date), end=str(end_date))

    if df.empty:
        continue

    ind = compute_all_indicators(df)
    signal = evaluate_signals(
        ind,
        rsi_overbought=rsi_overbought,
        rsi_oversold=rsi_oversold,
        stoch_overbought=stoch_overbought,
        stoch_oversold=stoch_oversold,
    )

    if not signal:
        continue

    flags = normalize_flags(signal["details"])
    active = active_signals(flags)
    semaforo = classify_signal_risk(signal)

    cards_data.append(
        {
            "asset_name": asset_name,
            "ticker": ticker,
            "recommendation": signal.get("recommendation", "Sin recomendación"),
            "semaforo_estado": semaforo["estado"],
            "semaforo_msg": semaforo["mensaje"],
            "score_buy": signal.get("score_buy", 0),
            "score_sell": signal.get("score_sell", 0),
            "active_text": ", ".join(active) if active else "Ninguna",
            "flags": flags,
            "ui": semaforo["ui"],
        }
    )

if not cards_data:
    st.warning("No fue posible construir señales para los activos en la ventana seleccionada.")
    st.stop()


# ==============================
# Grid 2 columnas
# ==============================
st.markdown("### Lectura por activo")

for i in range(0, len(cards_data), 2):
    cols = st.columns(2)
    row_items = cards_data[i:i + 2]

    for col, item in zip(cols, row_items):
        with col:
            signal_card(
                asset_name=item["asset_name"],
                ticker=item["ticker"],
                recommendation=item["recommendation"],
                semaforo_estado=item["semaforo_estado"],
                semaforo_msg=item["semaforo_msg"],
                score_buy=item["score_buy"],
                score_sell=item["score_sell"],
                active_text=item["active_text"],
                level=item["ui"],
            )

            if mostrar_detalle or modo == "Estadístico":
                with st.expander(f"Ver detalle de señales - {item['asset_name']}"):
                    st.dataframe(
                        signal_table(item["flags"]),
                        width="stretch",
                        hide_index=True,
                    )


# ==============================
# Interpretación breve final
# ==============================
favorables = sum(1 for x in cards_data if x["semaforo_estado"] == "Favorable")
desfavorables = sum(1 for x in cards_data if x["semaforo_estado"] == "Desfavorable")
mixtas = len(cards_data) - favorables - desfavorables

st.markdown("### Interpretación breve")

if modo == "General":
    if favorables > desfavorables:
        st.success(
            f"""
            En esta ventana predominan señales **favorables** ({favorables} activos), por lo que la lectura técnica agregada
            sugiere un sesgo más constructivo que defensivo. Aun así, la decisión no debería basarse solo en este módulo:
            conviene contrastar estas señales con riesgo, benchmark y contexto macro.
            """
        )
    elif desfavorables > favorables:
        st.error(
            f"""
            En esta ventana predominan señales **desfavorables** ({desfavorables} activos), lo que sugiere mayor cautela
            táctica. Esto puede reflejar sobrecompra, pérdida de momentum o deterioro técnico en varios activos del portafolio.
            """
        )
    else:
        st.warning(
            f"""
            La lectura agregada es **mixta**: hay señales favorables, desfavorables y neutrales coexistiendo. En este caso,
            el módulo sugiere prudencia y selección más cuidadosa por activo en vez de una lectura uniforme del portafolio.
            """
        )
else:
    st.info(
        f"""
        Desde una perspectiva agregada, el módulo identifica **{favorables} activos con lectura favorable**, 
        **{desfavorables} con lectura desfavorable** y **{mixtas} con lectura intermedia o no concluyente**.

        En términos técnicos, esto sugiere que las señales derivadas de momentum, reversión y tendencia no son homogéneas
        en todo el universo analizado. Por tanto, la interpretación más adecuada es utilizar este bloque como apoyo
        táctico para priorización de activos, no como criterio aislado de asignación.
        """
    )