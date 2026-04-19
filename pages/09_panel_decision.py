from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from src.config import ASSETS, DEFAULT_START_DATE, DEFAULT_END_DATE, ensure_project_dirs
from src.download import data_error_message, load_market_bundle, download_single_ticker
from src.preprocess import equal_weight_vector, equal_weight_portfolio
from src.risk_metrics import risk_comparison_table, validar_serie_para_garch
from src.garch_models import fit_garch_models
from src.benchmark import benchmark_summary
from src.indicators import compute_all_indicators
from src.signals import evaluate_signals

try:
    from src.api.macro import macro_snapshot
except Exception:
    macro_snapshot = None

ensure_project_dirs()


# ==============================
# UI helpers
# ==============================
def sanitize_text(text):
    if text is None:
        return ""
    return str(text).replace("<", "").replace(">", "")


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
            <div class="section-title">{sanitize_text(title)}</div>
            <div class="section-subtitle">{sanitize_text(subtitle)}</div>
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
                margin: 0; padding: 0; background: transparent;
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
                background-color: rgba(234, 179, 8, 0.14);
                color: #a16207;
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


def hero_decision(title, subtitle, level="neutral"):
    styles = {
        "positive": {
            "bg": "linear-gradient(135deg, #ecfdf5 0%, #f0fdf4 100%)",
            "border": "rgba(22, 163, 74, 0.24)",
            "accent": "#15803d",
            "badge": "rgba(22, 163, 74, 0.14)",
        },
        "warning": {
            "bg": "linear-gradient(135deg, #fffbeb 0%, #fefce8 100%)",
            "border": "rgba(234, 179, 8, 0.28)",
            "accent": "#a16207",
            "badge": "rgba(234, 179, 8, 0.16)",
        },
        "danger": {
            "bg": "linear-gradient(135deg, #fff1f2 0%, #fef2f2 100%)",
            "border": "rgba(220, 38, 38, 0.24)",
            "accent": "#b91c1c",
            "badge": "rgba(220, 38, 38, 0.14)",
        },
        "neutral": {
            "bg": "linear-gradient(135deg, #f8fafc 0%, #ffffff 100%)",
            "border": "rgba(100, 116, 139, 0.20)",
            "accent": "#334155",
            "badge": "rgba(100, 116, 139, 0.12)",
        },
    }
    s = styles.get(level, styles["neutral"])

    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            body {{
                margin: 0; padding: 0; background: transparent;
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            }}
            .hero {{
                background: {s["bg"]};
                border: 1px solid {s["border"]};
                border-radius: 24px;
                padding: 24px;
                box-shadow: 0 8px 24px rgba(15, 23, 42, 0.08);
                min-height: 185px;
                box-sizing: border-box;
            }}
            .badge {{
                display: inline-block;
                padding: 7px 12px;
                border-radius: 999px;
                background: {s["badge"]};
                color: {s["accent"]};
                font-size: 0.82rem;
                font-weight: 800;
                margin-bottom: 0.85rem;
            }}
            .title {{
                font-size: 2rem;
                line-height: 1.1;
                font-weight: 800;
                color: #0f172a;
                margin-bottom: 0.6rem;
            }}
            .subtitle {{
                font-size: 0.95rem;
                line-height: 1.55;
                color: #334155;
            }}
        </style>
    </head>
    <body>
        <div class="hero">
            <div class="badge">Decisión integrada del portafolio</div>
            <div class="title">{sanitize_text(title)}</div>
            <div class="subtitle">{sanitize_text(subtitle)}</div>
        </div>
    </body>
    </html>
    """
    components.html(html, height=210)


inject_ui_css()

st.title("Módulo 9 - Panel de decisión")
st.caption("Integra riesgo, volatilidad, señales técnicas y benchmark para producir una postura de acción más clara.")


# ==============================
# Sidebar
# ==============================
with st.sidebar:
    st.header("Parámetros")

    horizonte = st.selectbox(
        "Horizonte de análisis",
        ["1 mes", "Trimestre", "Semestre", "1 año", "3 años", "5 años", "Personalizado"],
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
        start_date = st.date_input("Fecha inicial", value=DEFAULT_START_DATE, key="decision_start")
        end_date = st.date_input("Fecha final", value=DEFAULT_END_DATE, key="decision_end")

    st.divider()
    st.subheader("Modo de visualización")
    modo = st.radio("Selecciona el nivel de detalle", ["General", "Estadístico"], index=0)

    st.divider()
    st.subheader("Opciones de visualización")
    mostrar_detalle = st.checkbox("Mostrar detalle técnico", value=(modo == "Estadístico"))

    with st.expander("Filtros secundarios"):
        alpha = st.selectbox("Nivel de confianza VaR", [0.95, 0.99], index=0, key="decision_alpha")
        n_sim = st.slider(
            "Simulaciones Monte Carlo",
            min_value=5000,
            max_value=50000,
            value=10000,
            step=5000,
            key="decision_nsim",
        )


# ==============================
# Helpers lógicos
# ==============================
def _get_rf_annual() -> float:
    if macro_snapshot is not None:
        try:
            snap = macro_snapshot()
            return float(snap.get("risk_free_rate_pct", 3.0)) / 100.0
        except Exception:
            return 0.03
    return 0.03


def _signal_bucket(recommendation: str) -> str:
    txt = str(recommendation).lower()
    if "compra" in txt:
        return "favorable"
    if "venta" in txt:
        return "desfavorable"
    return "neutral"


def _build_signal_summary(start_date, end_date) -> dict:
    favorables = 0
    neutrales = 0
    desfavorables = 0

    for asset_name, meta in ASSETS.items():
        ticker = meta["ticker"]
        try:
            df = download_single_ticker(ticker=ticker, start=str(start_date), end=str(end_date))
            if df.empty:
                continue

            ind = compute_all_indicators(df)
            signal = evaluate_signals(ind)
            if not signal:
                continue

            bucket = _signal_bucket(signal.get("recommendation", ""))
            if bucket == "favorable":
                favorables += 1
            elif bucket == "desfavorable":
                desfavorables += 1
            else:
                neutrales += 1
        except Exception:
            continue

    if favorables > desfavorables:
        return {
            "lectura": "Favorable",
            "score": 1,
            "ui": "positive",
            "favorables": favorables,
            "neutrales": neutrales,
            "desfavorables": desfavorables,
        }
    if desfavorables > favorables:
        return {
            "lectura": "Desfavorable",
            "score": -1,
            "ui": "danger",
            "favorables": favorables,
            "neutrales": neutrales,
            "desfavorables": desfavorables,
        }
    return {
        "lectura": "Neutral",
        "score": 0,
        "ui": "warning",
        "favorables": favorables,
        "neutrales": neutrales,
        "desfavorables": desfavorables,
    }


def _classify_risk(var_hist: float, persistencia: float, max_dd: float) -> dict:
    dd_abs = abs(max_dd) if pd.notna(max_dd) else np.nan
    puntos = 0

    if pd.notna(var_hist):
        if var_hist >= 0.03:
            puntos += 2
        elif var_hist >= 0.015:
            puntos += 1

    if pd.notna(persistencia):
        if persistencia >= 0.98:
            puntos += 2
        elif persistencia >= 0.90:
            puntos += 1

    if pd.notna(dd_abs):
        if dd_abs >= 0.25:
            puntos += 2
        elif dd_abs >= 0.10:
            puntos += 1

    if puntos >= 5:
        return {
            "nivel": "Alto",
            "score": -1,
            "ui": "danger",
            "mensaje": "El portafolio presenta pérdidas extremas potenciales y/o persistencia de volatilidad suficientemente elevadas como para justificar cautela.",
        }
    if puntos >= 3:
        return {
            "nivel": "Medio",
            "score": 0,
            "ui": "warning",
            "mensaje": "El portafolio muestra un riesgo intermedio: no obliga a salir completamente, pero sí a controlar tamaño de posición y exposición.",
        }
    return {
        "nivel": "Bajo",
        "score": 1,
        "ui": "positive",
        "mensaje": "El perfil de riesgo luce relativamente contenido para la ventana analizada.",
    }


def _classify_benchmark(summary_df: pd.DataFrame, extras_df: pd.DataFrame) -> dict:
    if summary_df.empty:
        return {
            "nivel": "No concluyente",
            "score": 0,
            "ui": "warning",
            "mensaje": "No fue posible construir una comparación robusta frente al benchmark.",
        }

    try:
        port_ret = float(summary_df.loc[summary_df["serie"] == "Portafolio", "ret_anualizado"].iloc[0])
        bench_ret = float(summary_df.loc[summary_df["serie"] == "Benchmark", "ret_anualizado"].iloc[0])

        alpha = np.nan
        if not extras_df.empty and "Alpha de Jensen" in extras_df["métrica"].values:
            alpha = float(extras_df.loc[extras_df["métrica"] == "Alpha de Jensen", "valor"].iloc[0])

        if port_ret > bench_ret and (pd.isna(alpha) or alpha >= 0):
            return {
                "nivel": "Superior",
                "score": 1,
                "ui": "positive",
                "mensaje": "El portafolio presenta una lectura relativa favorable frente al benchmark en la ventana analizada.",
            }

        if port_ret < bench_ret and (pd.isna(alpha) or alpha < 0):
            return {
                "nivel": "Inferior",
                "score": -1,
                "ui": "danger",
                "mensaje": "El portafolio viene rezagado frente al benchmark y no muestra una superioridad relativa consistente.",
            }

        return {
            "nivel": "No concluyente",
            "score": 0,
            "ui": "warning",
            "mensaje": "La comparación relativa no muestra una ventaja consistente; algunas métricas son mixtas y no confirman dominancia clara.",
        }
    except Exception:
        return {
            "nivel": "No concluyente",
            "score": 0,
            "ui": "warning",
            "mensaje": "No fue posible construir una lectura comparativa suficientemente robusta.",
        }


def _final_decision(risk_score: int, signal_score: int, bench_score: int) -> dict:
    total = risk_score + signal_score + bench_score

    if total >= 2:
        return {
            "titulo": "Compra táctica",
            "ui": "positive",
            "mensaje_general": "La lectura integrada favorece una postura compradora o de incremento táctico de exposición.",
            "mensaje_riesgo": "El principal riesgo es que un cambio brusco de mercado revierta la señal técnica y deteriore el perfil de volatilidad.",
            "mensaje_formal": "La combinación de riesgo contenido, sesgo técnico favorable y comparación relativa no adversa respalda una postura de compra táctica dentro de la ventana analizada.",
            "score_total": total,
        }

    if total >= 0:
        return {
            "titulo": "Mantener / compra selectiva",
            "ui": "warning",
            "mensaje_general": "La lectura integrada permite mantener exposición y considerar compras selectivas, pero no justifica una expansión agresiva de riesgo.",
            "mensaje_riesgo": "El principal riesgo es entrar con confirmación incompleta y enfrentar un deterioro posterior en benchmark o volatilidad.",
            "mensaje_formal": "La evidencia agregada no es suficientemente fuerte para una postura agresiva, pero tampoco justifica deshacer exposición. La decisión razonable es mantener y, en todo caso, comprar de forma selectiva.",
            "score_total": total,
        }

    if total == -1:
        return {
            "titulo": "Reducir exposición",
            "ui": "warning",
            "mensaje_general": "La lectura integrada sugiere reducir parcialmente exposición o evitar nuevas compras hasta que mejoren las condiciones.",
            "mensaje_riesgo": "El riesgo central es mantener una posición relativamente alta en un contexto donde la evidencia agregada se ha debilitado.",
            "mensaje_formal": "La combinación de señales no favorece una ampliación de posición. La decisión más consistente es reducir exposición marginalmente y esperar mejor confirmación estadística y técnica.",
            "score_total": total,
        }

    return {
        "titulo": "Venta / postura defensiva",
        "ui": "danger",
        "mensaje_general": "La lectura integrada favorece una postura defensiva: reducir exposición de forma relevante o priorizar salida.",
        "mensaje_riesgo": "El principal riesgo es permanecer sobreexpuesto en un entorno donde coinciden riesgo elevado, deterioro técnico y/o rezago relativo.",
        "mensaje_formal": "La evidencia integrada es adversa. Desde una perspectiva de control de riesgo, la postura más defendible es de venta o reducción sustancial de exposición.",
        "score_total": total,
    }


# ==============================
# Datos base
# ==============================
tickers = [meta["ticker"] for meta in ASSETS.values()]
bundle = load_market_bundle(tickers=tickers, start=str(start_date), end=str(end_date))
returns = bundle["returns"].replace([np.inf, -np.inf], np.nan).dropna()

if returns.empty or len(returns) < 30:
    st.error(data_error_message("No hay suficientes datos para construir el panel de decisión."))
    st.stop()

weights = equal_weight_vector(returns.shape[1])
portfolio_returns = equal_weight_portfolio(returns)
rf_annual = _get_rf_annual()

# Riesgo extremo
risk_table = risk_comparison_table(
    portfolio_returns=portfolio_returns,
    asset_returns_df=returns,
    weights=weights,
    alpha=alpha,
    n_sim=n_sim,
)

var_hist = np.nan
cvar_hist = np.nan
if not risk_table.empty and "Histórico" in risk_table["método"].values:
    row_hist = risk_table.loc[risk_table["método"] == "Histórico"].iloc[0]
    var_hist = float(row_hist["VaR_diario"])
    cvar_hist = float(row_hist["CVaR_diario"])

# GARCH
garch_validation = validar_serie_para_garch(portfolio_returns, min_obs=120, max_null_ratio=0.05)
persistencia = np.nan

if garch_validation["ok"]:
    serie_garch = garch_validation["serie_limpia"] * 100.0
    garch_results = fit_garch_models(serie_garch)
else:
    garch_results = {"comparison": pd.DataFrame(), "diagnostics": pd.DataFrame(), "summary_text": ""}

if not garch_results["diagnostics"].empty:
    persist_row = garch_results["diagnostics"].loc[
        garch_results["diagnostics"]["metrica"] == "persistencia_alpha_mas_beta"
    ]
    if not persist_row.empty:
        persistencia = pd.to_numeric(persist_row["valor"], errors="coerce").iloc[0]

# Benchmark
summary_df = pd.DataFrame()
extras_df = pd.DataFrame()
max_dd = np.nan

try:
    bench_df = download_single_ticker("^GSPC", start=str(start_date), end=str(end_date))

    if not bench_df.empty:
        if "Adj Close" in bench_df.columns:
            benchmark_prices = bench_df["Adj Close"]
        elif "Close" in bench_df.columns:
            benchmark_prices = bench_df["Close"]
        else:
            benchmark_prices = pd.Series(dtype=float)

        benchmark_prices = pd.to_numeric(benchmark_prices, errors="coerce").dropna()
        benchmark_returns = benchmark_prices.pct_change().dropna()

        if not benchmark_returns.empty:
            summary_df, extras_df, _, _ = benchmark_summary(
                portfolio_returns=portfolio_returns,
                benchmark_returns=benchmark_returns,
                rf_annual=rf_annual,
            )
except Exception:
    summary_df = pd.DataFrame()
    extras_df = pd.DataFrame()

if not summary_df.empty:
    try:
        max_dd = float(summary_df.loc[summary_df["serie"] == "Portafolio", "max_drawdown"].iloc[0])
    except Exception:
        max_dd = np.nan

# Señales
signal_summary = _build_signal_summary(start_date, end_date)

# Clasificaciones
risk_view = _classify_risk(var_hist, persistencia, max_dd)
bench_view = _classify_benchmark(summary_df, extras_df)
decision = _final_decision(risk_view["score"], signal_summary["score"], bench_view["score"])


# ==============================
# UI principal
# ==============================
st.markdown("### Resumen del módulo")
if modo == "General":
    st.write(
        """
        Este panel transforma varios resultados técnicos y estadísticos en una postura de acción
        más concreta para el portafolio: comprar, mantener, reducir exposición o vender.
        """
    )
else:
    st.write(
        """
        Este panel sintetiza evidencia de riesgo extremo, persistencia de volatilidad, señales técnicas
        agregadas y desempeño relativo frente al benchmark para producir una postura integrada de decisión.
        """
    )

st.caption(f"Periodo analizado: {start_date} a {end_date}")

hero_text = decision["mensaje_general"] if modo == "General" else decision["mensaje_formal"]
hero_decision(decision["titulo"], hero_text, level=decision["ui"])

st.markdown("### Pilares de la decisión")
section_intro(
    "Qué está empujando la decisión",
    "La postura final se construye combinando riesgo agregado, lectura técnica y comparación frente al benchmark.",
)

c1, c2, c3 = st.columns(3)

with c1:
    kpi_card(
        "Riesgo",
        risk_view["nivel"],
        delta="Pilar 1",
        delta_type="neg" if risk_view["nivel"] == "Alto" else "neu" if risk_view["nivel"] == "Medio" else "pos",
        caption=risk_view["mensaje"],
    )

with c2:
    kpi_card(
        "Técnica",
        signal_summary["lectura"],
        delta="Pilar 2",
        delta_type=signal_summary["ui"] == "positive" and "pos" or signal_summary["ui"] == "danger" and "neg" or "neu",
        caption=(
            f"Favorables: {signal_summary['favorables']} | "
            f"Neutrales: {signal_summary['neutrales']} | "
            f"Desfavorables: {signal_summary['desfavorables']}"
        ),
    )

with c3:
    kpi_card(
        "Benchmark",
        bench_view["nivel"],
        delta="Pilar 3",
        delta_type="pos" if bench_view["nivel"] == "Superior" else "neg" if bench_view["nivel"] == "Inferior" else "neu",
        caption=bench_view["mensaje"],
    )

st.markdown("### Métricas mínimas de soporte")
section_intro(
    "Indicadores clave",
    "Estas métricas respaldan la decisión sin repetir todo el detalle de módulos anteriores.",
)

k1, k2, k3, k4 = st.columns(4)

with k1:
    kpi_card(
        "VaR histórico",
        f"{var_hist:.2%}" if pd.notna(var_hist) else "N/D",
        caption="Pérdida umbral estimada del portafolio",
    )

with k2:
    kpi_card(
        "Persistencia GARCH",
        f"{persistencia:.3f}" if pd.notna(persistencia) else "N/D",
        delta=(
            "Alta persistencia" if pd.notna(persistencia) and persistencia >= 0.90
            else "Media persistencia" if pd.notna(persistencia)
            else None
        ),
        delta_type="neg" if pd.notna(persistencia) and persistencia >= 0.90 else "neu",
        caption="Memoria de la volatilidad del portafolio",
    )

alpha_jensen = np.nan
if not extras_df.empty and "Alpha de Jensen" in extras_df["métrica"].values:
    try:
        alpha_jensen = float(extras_df.loc[extras_df["métrica"] == "Alpha de Jensen", "valor"].iloc[0])
    except Exception:
        alpha_jensen = np.nan

with k3:
    kpi_card(
        "Alpha de Jensen",
        f"{alpha_jensen:.4f}" if pd.notna(alpha_jensen) else "N/D",
        delta="Comparación relativa",
        delta_type="pos" if pd.notna(alpha_jensen) and alpha_jensen > 0 else "neg" if pd.notna(alpha_jensen) and alpha_jensen < 0 else "neu",
        caption="Exceso de desempeño ajustado por riesgo",
    )

balance_signals = signal_summary["favorables"] - signal_summary["desfavorables"]
with k4:
    kpi_card(
        "Balance de señales",
        str(balance_signals),
        delta=(
            "Sesgo comprador" if balance_signals > 0
            else "Sesgo vendedor" if balance_signals < 0
            else "Balance neutro"
        ),
        delta_type="pos" if balance_signals > 0 else "neg" if balance_signals < 0 else "neu",
        caption="Diferencia entre señales favorables y desfavorables",
    )

st.markdown("### Cómo interpretar el resultado")
if modo == "General":
    st.success(
        f"""
        **Decisión sugerida: {decision['titulo']}**

        - Esta salida **sí toma una decisión**: no se queda en un “depende”.
        - La postura surge de combinar el nivel de riesgo del portafolio, la señal técnica agregada y su comparación frente al benchmark.
        - **Qué significa esta postura:** {decision['mensaje_general']}
        - **Qué riesgo asumes si sigues esta decisión:** {decision['mensaje_riesgo']}
        - En otras palabras, este panel no reemplaza el juicio de inversión, pero sí resume si el contexto actual favorece comprar, mantener, reducir o vender.
        """
    )
else:
    st.info(
        f"""
        **Interpretación formal**

        La postura final del panel es **{decision['titulo']}**. Esta clasificación surge de una regla de agregación
        simple pero consistente entre tres dimensiones: **riesgo agregado**, **lectura técnica** y **desempeño relativo
        frente al benchmark**.

        - **Riesgo agregado:** se aproxima mediante VaR histórico, persistencia GARCH y drawdown.
        - **Lectura técnica:** se resume a partir del balance entre señales favorables y desfavorables de los activos.
        - **Benchmark:** se evalúa en términos de retorno relativo y Alpha de Jensen.

        Desde una perspectiva metodológica, la decisión no debe entenderse como una predicción puntual,
        sino como una **postura estadísticamente razonable de acción** bajo la evidencia observada en la ventana analizada.

        **Conclusión formal:** {decision['mensaje_formal']}

        **Riesgo de implementación de la postura:** {decision['mensaje_riesgo']}
        """
    )

if mostrar_detalle:
    with st.expander("Ver detalle técnico del score"):
        detalle_df = pd.DataFrame(
            {
                "Componente": ["Riesgo", "Técnica", "Benchmark", "Score total"],
                "Lectura": [
                    risk_view["nivel"],
                    signal_summary["lectura"],
                    bench_view["nivel"],
                    decision["score_total"],
                ],
                "Score": [
                    risk_view["score"],
                    signal_summary["score"],
                    bench_view["score"],
                    decision["score_total"],
                ],
            }
        )
        st.dataframe(detalle_df, width="stretch", hide_index=True)
