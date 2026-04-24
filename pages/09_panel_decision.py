from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from src.config import ASSETS, DEFAULT_START_DATE, DEFAULT_END_DATE, GLOBAL_BENCHMARK, ensure_project_dirs
from src.api.backend_client import BackendAPIError, friendly_error_message
from src.download import data_error_message
from src.risk_metrics import validar_serie_para_garch
from src.garch_models import fit_garch_models
from src.benchmark import benchmark_summary
from src.services.decision_engine import DecisionEngine
from src.services.market_data_client import MarketDataClient
from src.services.risk_analyzer import RiskAnalyzer
from src.ui_components import kpi_card, render_explanation_expander, render_section, render_table
from src.ui_navigation import render_sidebar_navigation
from src.ui_style import apply_global_typography, render_page_title

try:
    from src.api.macro import macro_snapshot
except Exception:
    macro_snapshot = None

ensure_project_dirs()
apply_global_typography()
render_sidebar_navigation()


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
        .section-intro-box {
            background: #ffffff;
            border: 1px solid rgba(15, 23, 42, 0.08);
            border-radius: 18px;
            padding: 16px 18px;
            box-shadow: 0 4px 14px rgba(15, 23, 42, 0.06);
            margin-bottom: 0.8rem;
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


def hero_decision(title, subtitle, level="neutral"):
    styles = {
        "positive": {
            "bg": "linear-gradient(135deg, #ecfdf5 0%, #f0fdf4 100%)",
            "border": "rgba(22, 163, 74, 0.24)",
            "accent": "#15803d",
            "badge": "rgba(22, 163, 74, 0.14)",
        },
        "warning": {
            "bg": "linear-gradient(135deg, #eff6ff 0%, #f8fbff 100%)",
            "border": "rgba(37, 99, 235, 0.18)",
            "accent": "#1d4ed8",
            "badge": "rgba(37, 99, 235, 0.12)",
        },
        "danger": {
            "bg": "linear-gradient(135deg, #fff1f2 0%, #fef2f2 100%)",
            "border": "rgba(220, 38, 38, 0.24)",
            "accent": "#b91c1c",
            "badge": "rgba(220, 38, 38, 0.14)",
        },
        "neutral": {
            "bg": "linear-gradient(135deg, #eff6ff 0%, #f8fbff 100%)",
            "border": "rgba(37, 99, 235, 0.16)",
            "accent": "#1e3a8a",
            "badge": "rgba(37, 99, 235, 0.10)",
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
                border-radius: 22px;
                padding: 22px;
                box-shadow: 0 8px 22px rgba(15, 23, 42, 0.07);
                min-height: 172px;
                box-sizing: border-box;
            }}
            .badge {{
                display: inline-block;
                padding: 6px 11px;
                border-radius: 999px;
                background: {s["badge"]};
                color: {s["accent"]};
                font-size: 0.78rem;
                font-weight: 800;
                margin-bottom: 0.75rem;
            }}
            .title {{
                font-size: 1.8rem;
                line-height: 1.08;
                font-weight: 800;
                color: #0f172a;
                margin-bottom: 0.5rem;
            }}
            .subtitle {{
                font-size: 0.9rem;
                line-height: 1.5;
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
    components.html(html, height=190)


inject_ui_css()

render_page_title(
    "Módulo 9 - Panel de decisión",
    "Integra riesgo, volatilidad, señales técnicas y benchmark para producir una postura de acción más clara.",
)


# ==============================
# Sidebar
# ==============================
with st.sidebar:
    st.header("Parámetros")

    horizonte = st.selectbox(
        "Horizonte de análisis",
        ["1 mes", "Trimestre", "Semestre", "1 año", "2 años", "3 años", "5 años", "Personalizado"],
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
        start_date = st.date_input("Fecha inicial", value=DEFAULT_START_DATE, key="decision_start")
        end_date = st.date_input("Fecha final", value=DEFAULT_END_DATE, key="decision_end")

    st.divider()
    st.subheader("Opciones de análisis")
    mostrar_detalle = st.checkbox("Mostrar detalle técnico", value=False)
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


def _build_signal_summary(ohlcv_by_ticker: dict[str, pd.DataFrame]) -> dict:
    favorables = 0
    neutrales = 0
    desfavorables = 0

    for asset_name, meta in ASSETS.items():
        ticker = meta["ticker"]
        try:
            df = ohlcv_by_ticker.get(ticker, pd.DataFrame())
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


def extract_garch_persistence(garch_results: dict) -> float:
    diagnostics_df = garch_results.get("diagnostics", pd.DataFrame())
    if isinstance(diagnostics_df, pd.DataFrame) and not diagnostics_df.empty:
        if {"metrica", "valor"}.issubset(diagnostics_df.columns):
            persist_row = diagnostics_df.loc[
                diagnostics_df["metrica"] == "persistencia_alpha_mas_beta"
            ]
            if not persist_row.empty:
                persist_value = pd.to_numeric(persist_row["valor"], errors="coerce").dropna()
                if not persist_value.empty:
                    return float(persist_value.iloc[0])

    comparison_df = garch_results.get("comparison", pd.DataFrame())
    if not isinstance(comparison_df, pd.DataFrame) or comparison_df.empty:
        return np.nan

    working = comparison_df.copy()

    def _sorted_if_aic(df: pd.DataFrame) -> pd.DataFrame:
        if "AIC" not in df.columns:
            return df
        df = df.copy()
        df["AIC"] = pd.to_numeric(df["AIC"], errors="coerce")
        valid_aic = df.dropna(subset=["AIC"])
        if not valid_aic.empty:
            return valid_aic.sort_values("AIC", ascending=True)
        return df

    if "persistencia" in working.columns:
        working["persistencia"] = pd.to_numeric(working["persistencia"], errors="coerce")
        sorted_df = _sorted_if_aic(working)
        valid_persist = sorted_df.dropna(subset=["persistencia"])
        if not valid_persist.empty:
            return float(valid_persist.iloc[0]["persistencia"])

        valid_persist_any = working.dropna(subset=["persistencia"])
        if not valid_persist_any.empty:
            return float(valid_persist_any.iloc[0]["persistencia"])

    alpha_candidates = ["alpha_1", "alpha[1]", "alpha1", "alpha"]
    beta_candidates = ["beta_1", "beta[1]", "beta1", "beta"]
    alpha_col = next((col for col in alpha_candidates if col in working.columns), None)
    beta_col = next((col for col in beta_candidates if col in working.columns), None)

    if alpha_col and beta_col:
        alpha_series = pd.to_numeric(working[alpha_col], errors="coerce")
        beta_series = pd.to_numeric(working[beta_col], errors="coerce")
        rebuilt = working.copy()
        rebuilt["persistencia_reconstruida"] = alpha_series + beta_series

        sorted_df = _sorted_if_aic(rebuilt)
        valid_rebuilt = sorted_df.dropna(subset=["persistencia_reconstruida"])
        if not valid_rebuilt.empty:
            return float(valid_rebuilt.iloc[0]["persistencia_reconstruida"])

        valid_rebuilt_any = rebuilt.dropna(subset=["persistencia_reconstruida"])
        if not valid_rebuilt_any.empty:
            return float(valid_rebuilt_any.iloc[0]["persistencia_reconstruida"])

    return np.nan


# ==============================
# Datos base
# ==============================
asset_tickers = [meta["ticker"] for meta in ASSETS.values()]
benchmark_ticker = GLOBAL_BENCHMARK
tickers = asset_tickers + [benchmark_ticker]
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
        "Sin datos para estos tickers en el rango seleccionado; se excluyen del panel: "
        + ", ".join(missing_tickers)
    )

risk_analyzer = RiskAnalyzer()
decision_engine = DecisionEngine()
returns_all = bundle["returns"].replace([np.inf, -np.inf], np.nan)
valid_asset_tickers = [ticker for ticker in asset_tickers if ticker in returns_all.columns]
returns = risk_analyzer.clean_returns(returns_all[valid_asset_tickers])

if returns.empty or len(returns) < 30:
    st.error(data_error_message("No hay suficientes datos para construir el panel de decisión."))
    st.stop()

portfolio_returns, weights = risk_analyzer.portfolio_returns(returns)
rf_annual = _get_rf_annual()

# Riesgo extremo
risk_table = risk_analyzer.compute_var_tables(
    portfolio_returns=portfolio_returns,
    asset_returns_df=returns,
    weights=weights,
    confidence_levels=[alpha],
    n_sim=n_sim,
).get(alpha, pd.DataFrame())

var_hist = np.nan
cvar_hist = np.nan
if not risk_table.empty and "Histórico" in risk_table["método"].values:
    row_hist = risk_table.loc[risk_table["método"] == "Histórico"].iloc[0]
    var_hist = float(row_hist["VaR_diario"])
    cvar_hist = float(row_hist["CVaR_diario"])

# GARCH
garch_validation = validar_serie_para_garch(portfolio_returns, min_obs=120, max_null_ratio=0.05)

if garch_validation["ok"]:
    serie_garch = garch_validation["serie_limpia"] * 100.0
    garch_results = fit_garch_models(serie_garch)
else:
    garch_results = {"comparison": pd.DataFrame(), "diagnostics": pd.DataFrame(), "summary_text": ""}

persistencia = extract_garch_persistence(garch_results)

# Benchmark
summary_df = pd.DataFrame()
extras_df = pd.DataFrame()
max_dd = np.nan

if benchmark_ticker in returns_all.columns:
    benchmark_returns = pd.to_numeric(returns_all[benchmark_ticker], errors="coerce").dropna()
    if not benchmark_returns.empty:
        summary_df, extras_df, _, _ = benchmark_summary(
            portfolio_returns=portfolio_returns,
            benchmark_returns=benchmark_returns,
            rf_annual=rf_annual,
        )

if not summary_df.empty:
    try:
        max_dd = float(summary_df.loc[summary_df["serie"] == "Portafolio", "max_drawdown"].iloc[0])
    except Exception:
        max_dd = np.nan

# Señales
signal_summary = decision_engine.build_signal_summary(bundle.get("ohlcv", {}))

# Clasificaciones
risk_view = decision_engine.classify_risk(var_hist, persistencia, max_dd)
bench_view = decision_engine.classify_benchmark(summary_df, extras_df)
decision = decision_engine.final_decision(risk_view["score"], signal_summary["score"], bench_view["score"])


# ==============================
# UI principal
# ==============================
st.markdown("### Resumen del módulo")
st.caption(
    "Panel integrador opcional para resumir riesgo, técnica y benchmark en una postura de acción del portafolio."
)

st.caption(f"Periodo analizado: {start_date} a {end_date}")

hero_text = decision["mensaje_general"]
hero_decision(decision["titulo"], hero_text, level=decision["ui"])

st.markdown("### Pilares de la decisión")
render_section(
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
        caption="Nivel agregado de riesgo del portafolio.",
    )

with c2:
    kpi_card(
        "Técnica",
        signal_summary["lectura"],
        delta="Pilar 2",
        delta_type=signal_summary["ui"] == "positive" and "pos" or signal_summary["ui"] == "danger" and "neg" or "neu",
        caption="Balance de señales por activo.",
    )

with c3:
    kpi_card(
        "Benchmark",
        bench_view["nivel"],
        delta="Pilar 3",
        delta_type="pos" if bench_view["nivel"] == "Superior" else "neg" if bench_view["nivel"] == "Inferior" else "neu",
        caption="Comparación frente al benchmark global.",
    )

render_explanation_expander(
    "Cómo interpretar los pilares",
    [
        "Riesgo: resume VaR histórico, persistencia GARCH y drawdown.",
        "Técnica: resume señales favorables, neutrales y desfavorables.",
        "Benchmark: resume desempeño relativo y Alpha de Jensen.",
        "La decisión final combina esos tres componentes.",
    ],
)

st.markdown("### Métricas mínimas de soporte")
render_section(
    "Indicadores clave",
    "Estas métricas respaldan la decisión sin repetir todo el detalle de módulos anteriores.",
)

k1, k2, k3, k4 = st.columns(4)

with k1:
    kpi_card(
        "VaR histórico",
        f"{var_hist:.2%}" if pd.notna(var_hist) else "N/D",
        caption="Pérdida umbral estimada.",
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
        caption="Memoria de la volatilidad.",
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
        caption="Desempeño ajustado por riesgo.",
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
        caption="Favorables menos desfavorables.",
    )

render_explanation_expander(
    "Cómo interpretar las métricas de soporte",
    [
        "VaR histórico: aproxima la pérdida umbral del portafolio en la ventana analizada.",
        "Persistencia GARCH: refleja cuánta memoria conserva la volatilidad.",
        "Alpha de Jensen: mide desempeño relativo ajustado por riesgo frente al benchmark.",
        "Balance de señales: resume la diferencia entre señales favorables y desfavorables.",
    ]
    + (
        ["Si la persistencia aparece como N/D, significa que el modelo GARCH no entregó un valor alpha + beta válido para la ventana seleccionada."]
        if pd.isna(persistencia)
        else []
    ),
)

st.markdown("### Cómo interpretar el resultado")
st.success(
    f"Decisión sugerida: **{decision['titulo']}**. {decision['mensaje_general']}"
)

render_explanation_expander(
    "Cómo interpretar la decisión integrada",
    [
        "La postura final surge de combinar riesgo agregado, lectura técnica y desempeño relativo frente al benchmark.",
        "Riesgo agregado: se aproxima mediante VaR histórico, persistencia GARCH y drawdown.",
        "Lectura técnica: se resume a partir del balance entre señales favorables y desfavorables de los activos.",
        "Benchmark: se evalúa en términos de retorno relativo y Alpha de Jensen.",
        f"Conclusión formal: {decision['mensaje_formal']}",
        f"Riesgo de implementación de la postura: {decision['mensaje_riesgo']}",
    ],
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
        render_table(detalle_df, hide_index=True, width="stretch")
