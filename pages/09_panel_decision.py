from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from src.config import ASSETS, DEFAULT_START_DATE, DEFAULT_END_DATE, GLOBAL_BENCHMARK, ensure_project_dirs
from src.api.backend_client import BackendAPIError, backend_post, friendly_error_message
from src.download import data_error_message
from src.indicators import compute_all_indicators
from src.risk_metrics import validar_serie_para_garch
from src.garch_models import fit_garch_models
from src.signals import evaluate_signals
from src.benchmark import benchmark_summary
from src.services.decision_engine import DecisionEngine
from src.services.market_data_client import MarketDataClient
from src.services.risk_analyzer import RiskAnalyzer
from src.ui_components import kpi_card, render_explanation_expander, render_section, render_table
from src.ui_layout import configured_assets, configured_period, module_params, render_app_shell, render_portfolio_summary_card
from src.ui_style import apply_global_typography

try:
    from src.api.macro import macro_snapshot
except Exception:
    macro_snapshot = None

ensure_project_dirs()
apply_global_typography()

METHOD_COL = "metodo"
METRIC_COL = "metrica"
VALUE_COL = "valor"


# ==============================
# UI helpers
# ==============================
def sanitize_text(text):
    if text is None:
        return ""
    return str(text).replace("<", "").replace(">", "")


def normalize_risk_table_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    normalized = df.copy()
    method_col = next(
        (col for col in [METHOD_COL, "método", "Metodo", "Método", "method"] if col in normalized.columns),
        None,
    )
    if method_col is not None and method_col != METHOD_COL:
        normalized = normalized.rename(columns={method_col: METHOD_COL})

    if METHOD_COL in normalized.columns:
        normalized.loc[:, METHOD_COL] = normalized.loc[:, METHOD_COL].replace(
            {
                "Paramétrico": "Parametrico",
                "Histórico": "Historico",
            }
        )

    return normalized


def normalize_metric_table_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    normalized = df.copy()
    metric_col = next(
        (col for col in [METRIC_COL, "métrica", "Metrica", "Métrica", "metric", "indicador", "Indicador"] if col in normalized.columns),
        None,
    )
    value_col = next(
        (col for col in [VALUE_COL, "Valor", "value", "Value"] if col in normalized.columns),
        None,
    )
    if metric_col is not None and metric_col != METRIC_COL:
        normalized = normalized.rename(columns={metric_col: METRIC_COL})
    if value_col is not None and value_col != VALUE_COL:
        normalized = normalized.rename(columns={value_col: VALUE_COL})
    return normalized


def metric_value(df: pd.DataFrame, metric_name: str, default=np.nan) -> float:
    normalized = normalize_metric_table_columns(df)
    if normalized.empty or not {METRIC_COL, VALUE_COL}.issubset(normalized.columns):
        return default

    matches = normalized.loc[normalized.loc[:, METRIC_COL] == metric_name, VALUE_COL]
    if matches.empty:
        return default

    value = pd.to_numeric(matches.iloc[0], errors="coerce")
    return float(value) if pd.notna(value) else default


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
            <div class="badge">Decision integrada del portafolio</div>
            <div class="title">{sanitize_text(title)}</div>
            <div class="subtitle">{sanitize_text(subtitle)}</div>
        </div>
    </body>
    </html>
    """
    components.html(html, height=190)


render_app_shell(
    "Módulo 9 - Panel de decisión",
    "Integra riesgo, volatilidad, señales técnicas y benchmark para producir una postura de acción más clara.",
)
ASSETS = configured_assets(ASSETS)
horizonte, start_date, end_date = configured_period(DEFAULT_START_DATE, DEFAULT_END_DATE)
render_portfolio_summary_card(ASSETS)


# ==============================
# Parámetros del módulo
# ==============================
with module_params():
    st.header("Parámetros")
    st.subheader("Opciones de análisis")
    mostrar_detalle = st.checkbox("Mostrar detalle t-cnico", value=False)
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
# Helpers logicos
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
            "mensaje": "El portafolio presenta perdidas extremas potenciales y/o persistencia de volatilidad suficientemente elevadas como para justificar cautela.",
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
            "mensaje": "No fue posible construir una comparacion robusta frente al benchmark.",
        }

    try:
        port_ret = float(summary_df.loc[summary_df["serie"] == "Portafolio", "ret_anualizado"].iloc[0])
        bench_ret = float(summary_df.loc[summary_df["serie"] == "Benchmark", "ret_anualizado"].iloc[0])

        alpha = metric_value(extras_df, "Alpha de Jensen")

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
            "titulo": "Compra tactica",
            "ui": "positive",
            "mensaje_general": "La lectura integrada favorece una postura compradora o de incremento tactico de exposicion.",
            "mensaje_riesgo": "El principal riesgo es que un cambio brusco de mercado revierta la señal técnica y deteriore el perfil de volatilidad.",
            "mensaje_formal": "La combinación de riesgo contenido, sesgo técnico favorable y comparación relativa no adversa respalda una postura de compra táctica dentro de la ventana analizada.",
            "score_total": total,
        }

    if total >= 0:
        return {
            "titulo": "Mantener / compra selectiva",
            "ui": "warning",
            "mensaje_general": "La lectura integrada permite mantener exposicion y considerar compras selectivas, pero no justifica una expansion agresiva de riesgo.",
            "mensaje_riesgo": "El principal riesgo es entrar con confirmacion incompleta y enfrentar un deterioro posterior en benchmark o volatilidad.",
            "mensaje_formal": "La evidencia agregada no es suficientemente fuerte para una postura agresiva, pero tampoco justifica deshacer exposicion. La decision razonable es mantener y, en todo caso, comprar de forma selectiva.",
            "score_total": total,
        }

    if total == -1:
        return {
            "titulo": "Reducir exposicion",
            "ui": "warning",
            "mensaje_general": "La lectura integrada sugiere reducir parcialmente exposicion o evitar nuevas compras hasta que mejoren las condiciones.",
            "mensaje_riesgo": "El riesgo central es mantener una posicion relativamente alta en un contexto donde la evidencia agregada se ha debilitado.",
            "mensaje_formal": "La combinación de señales no favorece una ampliación de posición. La decisión más consistente es reducir exposición marginalmente y esperar mejor confirmación estadística y técnica.",
            "score_total": total,
        }

    return {
        "titulo": "Venta / postura defensiva",
        "ui": "danger",
        "mensaje_general": "La lectura integrada favorece una postura defensiva: reducir exposicion de forma relevante o priorizar salida.",
        "mensaje_riesgo": "El principal riesgo es permanecer sobreexpuesto en un entorno donde coinciden riesgo elevado, deterioro técnico y/o rezago relativo.",
        "mensaje_formal": "La evidencia integrada es adversa. Desde una perspectiva de control de riesgo, la postura mas defendible es de venta o reduccion sustancial de exposicion.",
        "score_total": total,
    }


def extract_garch_persistence(garch_results: dict) -> float:
    diagnostics_df = garch_results.get("diagnostics", pd.DataFrame())
    if isinstance(diagnostics_df, pd.DataFrame) and not diagnostics_df.empty:
        if {METRIC_COL, VALUE_COL}.issubset(diagnostics_df.columns):
            persist_row = diagnostics_df.loc[
                diagnostics_df.loc[:, METRIC_COL] == "persistencia_alpha_mas_beta"
            ]
            if not persist_row.empty:
                persist_value = pd.to_numeric(persist_row.loc[:, VALUE_COL], errors="coerce").dropna()
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


def _finite_float(value, fallback: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return fallback
    return number if np.isfinite(number) else fallback


def _last_numeric(df: pd.DataFrame, column: str, fallback: float) -> float:
    if df is None or df.empty or column not in df.columns:
        return fallback
    values = pd.to_numeric(df[column], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if values.empty:
        return fallback
    return _finite_float(values.iloc[-1], fallback)


def build_ml_risk_payload(
    returns_series: pd.Series,
    ohlcv_by_ticker: dict[str, pd.DataFrame],
    primary_ticker: str | None,
) -> dict:
    clean_returns = (
        pd.to_numeric(returns_series, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )

    def cumulative_return(window: int) -> float:
        tail = clean_returns.tail(window)
        if tail.empty:
            return 0.0
        return _finite_float((1.0 + tail).prod() - 1.0, 0.0)

    def volatility(window: int) -> float:
        tail = clean_returns.tail(window)
        if len(tail) < 2:
            return 0.0
        return _finite_float(tail.std(ddof=1), 0.0)

    def drawdown_20d() -> float:
        tail = clean_returns.tail(20)
        if tail.empty:
            return 0.0
        cumulative = (1.0 + tail).cumprod()
        drawdown = cumulative / cumulative.cummax().replace(0, np.nan) - 1.0
        return _finite_float(drawdown.min(), 0.0)

    # Fallbacks neutrales: el endpoint requiere floats finitos aun si faltan indicadores técnicos.
    rsi = 50.0
    macd_hist = 0.0
    bb_position = 0.5
    close_over_sma20 = 1.0

    if primary_ticker and primary_ticker in ohlcv_by_ticker:
        try:
            indicators = compute_all_indicators(ohlcv_by_ticker[primary_ticker])
            close = _last_numeric(indicators, "Close", np.nan)
            bb_upper = _last_numeric(indicators, "BB_upper", np.nan)
            bb_lower = _last_numeric(indicators, "BB_lower", np.nan)
            sma20 = _last_numeric(indicators, "SMA_20", np.nan)

            rsi = _last_numeric(indicators, "RSI", rsi)
            macd_hist = _last_numeric(indicators, "MACD_hist", macd_hist)
            if np.isfinite(close) and np.isfinite(bb_upper) and np.isfinite(bb_lower) and bb_upper != bb_lower:
                bb_position = _finite_float((close - bb_lower) / (bb_upper - bb_lower), bb_position)
            if np.isfinite(close) and np.isfinite(sma20) and sma20 != 0:
                close_over_sma20 = _finite_float(close / sma20, close_over_sma20)
        except Exception:
            pass

    return {
        "ret_1d": _finite_float(clean_returns.iloc[-1] if not clean_returns.empty else 0.0, 0.0),
        "ret_5d": cumulative_return(5),
        "ret_20d": cumulative_return(20),
        "vol_5d": volatility(5),
        "vol_20d": volatility(20),
        "rsi": rsi,
        "macd_hist": macd_hist,
        "bb_position": bb_position,
        "close_over_sma20": close_over_sma20,
        "drawdown_20d": drawdown_20d(),
    }


def fetch_ml_risk_score(payload: dict) -> dict | None:
    try:
        result = backend_post("/ml/risk-score", payload)
        st.session_state.pop("decision_ml_risk_error", None)
        return result
    except BackendAPIError as exc:
        st.session_state["decision_ml_risk_error"] = exc
        return None
    except Exception as exc:
        st.session_state["decision_ml_risk_error"] = exc
        return None


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
    st.error(data_error_message("No hay suficientes datos para construir el panel de decision."))
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
risk_table = normalize_risk_table_columns(risk_table)

var_hist = np.nan
cvar_hist = np.nan
required_risk_columns = {METHOD_COL, "VaR_diario", "CVaR_diario"}
if risk_table.empty:
    st.warning("No fue posible calcular la tabla de riesgo extremo para el panel de decisión.")
elif not required_risk_columns.issubset(risk_table.columns):
    st.warning("La tabla de riesgo extremo no incluye todas las columnas necesarias para el panel de decisión.")
elif "Historico" in risk_table.loc[:, METHOD_COL].values:
    row_hist = risk_table.loc[risk_table.loc[:, METHOD_COL] == "Historico"].iloc[0]
    var_hist = float(row_hist["VaR_diario"])
    cvar_hist = float(row_hist["CVaR_diario"])
else:
    st.info("No hay VaR histórico disponible para este periodo; el panel continuará con las demás señales.")

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
        extras_df = normalize_metric_table_columns(extras_df)

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
primary_ticker = valid_asset_tickers[0] if valid_asset_tickers else None
ml_risk_payload = build_ml_risk_payload(portfolio_returns, bundle.get("ohlcv", {}), primary_ticker)
ml_risk_score = fetch_ml_risk_score(ml_risk_payload)


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

st.markdown("### Pilares de la decision")
render_section(
    "Que esta empujando la decision",
    "La postura final se construye combinando riesgo agregado, lectura técnica, comparación frente al benchmark y una señal auxiliar ML.",
)

c1, c2, c3, c4 = st.columns(4)

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
        "Tecnica",
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
        caption="Comparacion frente al benchmark global.",
    )

with c4:
    if ml_risk_score:
        ml_risk_level = str(ml_risk_score.get("risk_level", "")).lower()
        ml_delta_type = "neg" if ml_risk_level == "alto" else "neu" if ml_risk_level == "moderado" else "pos"
        ml_score_value = _finite_float(ml_risk_score.get("risk_score"), np.nan)
        ml_horizon = ml_risk_score.get("horizon_days", "N/D")
        ml_model = ml_risk_score.get("model_version", "N/D")
        kpi_card(
            "Riesgo ML",
            ml_risk_level.capitalize() if ml_risk_level else "N/D",
            delta="Pilar 4",
            delta_type=ml_delta_type,
            caption=(
                "Score auxiliar ML a 5 dias. "
                f"Score ML: {ml_score_value:.3f} | Horizonte: {ml_horizon} dias | Modelo: {ml_model}"
                if pd.notna(ml_score_value)
                else f"Score auxiliar ML a 5 dias. Horizonte: {ml_horizon} dias | Modelo: {ml_model}"
            ),
        )
    else:
        kpi_card(
            "Riesgo ML",
            "N/D",
            delta="Pilar 4",
            delta_type="neu",
            caption="No disponible temporalmente.",
        )

if mostrar_detalle and not ml_risk_score and "decision_ml_risk_error" in st.session_state:
    ml_error = st.session_state["decision_ml_risk_error"]
    st.info(f"Riesgo ML no disponible: {friendly_error_message(ml_error)}")
    if getattr(ml_error, "technical_detail", None):
        st.caption(ml_error.technical_detail)

render_explanation_expander(
    "Como interpretar los pilares",
    [
        "Riesgo: resume VaR historico, persistencia GARCH y drawdown.",
        "Técnica: resume señales favorables, neutrales y desfavorables.",
        "Benchmark: resume desempeño relativo y Alpha de Jensen.",
        "Riesgo ML: score auxiliar a 5 días estimado con variables recientes de retorno, volatilidad e indicadores técnicos.",
        "La decision final combina los tres componentes principales y usa el score ML como apoyo visual.",
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
        "VaR historico",
        f"{var_hist:.2%}" if pd.notna(var_hist) else "N/D",
        caption="Perdida umbral estimada.",
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

alpha_jensen = metric_value(extras_df, "Alpha de Jensen")

with k3:
    kpi_card(
        "Alpha de Jensen",
        f"{alpha_jensen:.4f}" if pd.notna(alpha_jensen) else "N/D",
        delta="Comparacion relativa",
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
        "VaR historico: aproxima la perdida umbral del portafolio en la ventana analizada.",
        "Persistencia GARCH: refleja cuanta memoria conserva la volatilidad.",
        "Alpha de Jensen: mide desempeño relativo ajustado por riesgo frente al benchmark.",
        "Balance de señales: resume la diferencia entre señales favorables y desfavorables.",
    ]
    + (
        ["Si la persistencia aparece como N/D, significa que el modelo GARCH no entrego un valor alpha + beta valido para la ventana seleccionada."]
        if pd.isna(persistencia)
        else []
    ),
)

st.markdown("### Como interpretar el resultado")
st.success(
    f"Decision sugerida: **{decision['titulo']}**. {decision['mensaje_general']}"
)

render_explanation_expander(
    "Como interpretar la decision integrada",
    [
        "La postura final surge de combinar riesgo agregado, lectura técnica y desempeño relativo frente al benchmark.",
        "Riesgo agregado: se aproxima mediante VaR historico, persistencia GARCH y drawdown.",
        "Lectura técnica: se resume a partir del balance entre señales favorables y desfavorables de los activos.",
        "Benchmark: se evalua en terminos de retorno relativo y Alpha de Jensen.",
        "El score ML se usa como senal auxiliar y no como recomendacion financiera independiente.",
        f"Conclusion formal: {decision['mensaje_formal']}",
        f"Riesgo de implementacion de la postura: {decision['mensaje_riesgo']}",
    ],
)

if mostrar_detalle:
    with st.expander("Ver detalle técnico del score"):
        detalle_df = pd.DataFrame(
            {
                "Componente": ["Riesgo", "Tecnica", "Benchmark", "Score total"],
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

