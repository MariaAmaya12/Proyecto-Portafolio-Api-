import streamlit as st
import pandas as pd
from pydantic import BaseModel, ValidationError

try:
    from pydantic import model_validator

    PYDANTIC_V2 = True
except ImportError:
    from pydantic import root_validator

    PYDANTIC_V2 = False

from src.config import (
    ASSETS,
    DEFAULT_END_DATE,
    GLOBAL_BENCHMARK,
    get_ticker,
    get_local_benchmark,
    ensure_project_dirs,
)
from src.returns_analysis import compute_return_series
from src.api.macro import macro_snapshot
from src.api.backend_client import BackendAPIError, friendly_error_message
from src.plots import plot_scatter_regression
from src.services.capm_analyzer import CAPMAnalyzer
from src.services.market_data_client import MarketDataClient
from src.ui_components import kpi_card, render_explanation_expander, render_section, render_table
from src.ui_navigation import render_sidebar_navigation
from src.ui_style import apply_global_typography, render_page_title

ensure_project_dirs()
apply_global_typography()
render_sidebar_navigation()

PORTFOLIO_OPTION = "Portafolio (equiponderado)"
WEIGHT_TOL = 1e-4


def _validate_weights_dict(pesos: dict[str, float], tol: float = WEIGHT_TOL) -> None:
    if not pesos:
        raise ValueError("Debe ingresar al menos un peso.")
    invalid = {asset: weight for asset, weight in pesos.items() if weight < 0 or weight > 1}
    if invalid:
        raise ValueError("Todos los pesos deben estar entre 0 y 1.")
    total = sum(pesos.values())
    if abs(total - 1.0) > tol:
        raise ValueError(f"La suma de pesos debe ser 1.00 dentro de una tolerancia de {tol}.")


if PYDANTIC_V2:

    class PortfolioWeightsModel(BaseModel):
        pesos: dict[str, float]

        @model_validator(mode="after")
        def validate_weights(self):
            _validate_weights_dict(self.pesos)
            return self

else:

    class PortfolioWeightsModel(BaseModel):
        pesos: dict[str, float]

        @root_validator
        def validate_weights(cls, values):
            _validate_weights_dict(values.get("pesos") or {})
            return values


def soft_note(title: str, body: str):
    st.markdown(
        f"""
        <div class="soft-explain-box">
            <div class="soft-explain-title">{sanitize_text(title)}</div>
            <div class="soft-explain-body">{sanitize_text(body)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def sanitize_text(text):
    if text is None:
        return ""
    return str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def fmt_num(value, digits=3):
    numeric_value = pd.to_numeric(value, errors="coerce")
    return f"{numeric_value:.{digits}f}" if pd.notna(numeric_value) else "N/D"


def fmt_pct(value, digits=2):
    numeric_value = pd.to_numeric(value, errors="coerce")
    return f"{numeric_value:.{digits}%}" if pd.notna(numeric_value) else "N/D"


def fmt_alpha_daily(value):
    numeric_value = pd.to_numeric(value, errors="coerce")
    return f"{numeric_value:.3%}" if pd.notna(numeric_value) else "N/D"


def infer_ticker_type(ticker_value):
    if ticker_value is None or str(ticker_value).strip() == "":
        return "no resuelto"
    if str(ticker_value).strip().startswith("^"):
        return "indice"
    return "activo/accion"


@st.cache_data(show_spinner=False, ttl=3600)
def fetch_capm_bundle_cached(tickers: tuple[str, ...], start: str, end: str) -> dict:
    return MarketDataClient().fetch_bundle(tickers=list(tickers), start=start, end=end)


@st.cache_data(show_spinner=False, ttl=3600)
def load_capm_close_matrix(tickers: tuple[str, ...], start: str, end: str) -> pd.DataFrame:
    bundle = fetch_capm_bundle_cached(tickers=tickers, start=start, end=end)
    close = bundle.get("close", pd.DataFrame())
    if close is None:
        return pd.DataFrame()
    return close.copy()


def close_series_from_matrix(close_matrix: pd.DataFrame, ticker_value: str) -> pd.Series:
    if close_matrix.empty or ticker_value not in close_matrix.columns:
        return pd.Series(dtype=float)
    return pd.to_numeric(close_matrix[ticker_value], errors="coerce").dropna()


def returns_from_close(close_series: pd.Series) -> pd.Series:
    return compute_return_series(close_series)["simple_return"]


@st.cache_data(show_spinner=False, ttl=3600)
def returns_from_close_cached(close_series: pd.Series) -> pd.Series:
    return returns_from_close(close_series)


def fallback_asset_returns(asset_label: str, asset_ticker: str) -> pd.Series:
    st.warning(f"No se encontró {asset_ticker} en la matriz inicial; se consulta nuevamente al backend.")
    try:
        bundle = fetch_capm_bundle_cached(tickers=(asset_ticker,), start=str(start_date), end=str(end_date))
    except BackendAPIError as exc:
        st.error(friendly_error_message(exc, "No se pudieron obtener datos del activo desde el backend."))
        if exc.technical_detail:
            st.caption(exc.technical_detail)
        st.stop()

    returns = bundle.get("returns", pd.DataFrame())
    if returns.empty or asset_ticker not in returns.columns:
        st.error("El backend no devolvió retornos válidos para el activo seleccionado.")
        st.caption(f"Activo seleccionado: {asset_label} | Ticker: {asset_ticker}")
        st.stop()
    return pd.to_numeric(returns[asset_ticker], errors="coerce").dropna()


def stop_if_benchmark_empty(bench_returns: pd.Series, bench_ticker: str, selected_label: str) -> None:
    if not bench_returns.empty:
        return
    st.error("No hay datos del benchmark desde el backend; intenta recargar o cambiar horizonte.")
    st.caption("Diagnóstico backend CAPM")
    st.code(
        "\n".join(
            [
                f"Activo seleccionado: {selected_label}",
                f"Benchmark usado: {bench_ticker}",
                f"Tipo benchmark inferido: {infer_ticker_type(bench_ticker)}",
                f"Rango: {start_date} a {end_date}",
                f"Retornos benchmark disponibles: {len(bench_returns)}",
            ]
        )
    )
    st.stop()


def build_portfolio_returns(
    asset_names: list[str],
    close_matrix: pd.DataFrame,
    raw_weights: dict[str, float] | None = None,
) -> tuple[pd.Series, list[str], list[str], pd.Series]:
    returns = []
    included = []
    missing = []

    for name in asset_names:
        asset_ticker = get_ticker(name)
        close_series = close_series_from_matrix(close_matrix, asset_ticker)
        if close_series.empty:
            missing.append(f"{name} ({asset_ticker})")
            continue
        returns.append(returns_from_close_cached(close_series).rename(name))
        included.append(name)

    if not returns:
        st.error("No se pudieron construir los retornos del portafolio equiponderado.")
        st.caption("Diagnóstico de descarga CAPM")
        st.code(
            "\n".join(
                [
                    f"Activos sin datos en matriz bundle: {', '.join(missing) if missing else 'N/D'}",
                    f"Rango: {start_date} a {end_date}",
                    f"Columnas disponibles: {list(close_matrix.columns)}",
                ]
            )
        )
        st.stop()

    returns_df = pd.concat(returns, axis=1).dropna(how="any")
    if returns_df.empty:
        st.warning("No hay suficientes retornos alineados para construir el portafolio equiponderado.")
        st.stop()

    if raw_weights:
        weights = pd.Series(
            {name: max(float(raw_weights.get(name, 0.0)), 0.0) for name in returns_df.columns},
            index=returns_df.columns,
            dtype=float,
        )
        if weights.sum() <= 0:
            st.error("La suma de pesos manuales debe ser mayor que cero.")
            st.stop()
        if abs(float(weights.sum()) - 1.0) > WEIGHT_TOL:
            st.error("La suma de pesos manuales debe ser 1.00 para calcular el CAPM del portafolio.")
            st.stop()
    else:
        weights = pd.Series(1 / len(returns_df.columns), index=returns_df.columns, dtype=float)

    return returns_df.mul(weights, axis=1).sum(axis=1).rename(PORTFOLIO_OPTION), included, missing, weights


def build_asset_betas_table(
    analyzer: CAPMAnalyzer,
) -> tuple[pd.DataFrame, list[str]]:
    rows = []
    notes = []
    global_returns = analyzer.get_benchmark_returns(GLOBAL_BENCHMARK)
    global_benchmark_failed = global_returns.empty

    for name in ASSETS.keys():
        asset_ticker = get_ticker(name)
        bench_ticker = get_local_benchmark(name)
        asset_returns = analyzer.get_asset_returns(asset_ticker)
        local_returns = analyzer.get_benchmark_returns(bench_ticker)

        row = {
            "Activo": name,
            "Ticker": asset_ticker,
            "Benchmark usado (local)": bench_ticker,
            "Beta (local)": "Sin datos",
            "Retorno esperado (CAPM local)": "Sin datos",
            "Clasificación (local)": "Sin datos",
            "Alpha diaria (local)": "Sin datos",
            "R² (local)": "Sin datos",
            "p-value beta (local)": "Sin datos",
            "Benchmark global": GLOBAL_BENCHMARK,
            "Beta (ACWI)": "N/D",
            "Retorno esperado (CAPM ACWI)": "N/D",
            "Clasificación (ACWI)": "N/D",
            "Alpha diaria (ACWI)": "N/D",
            "R² (ACWI)": "N/D",
            "p-value beta (ACWI)": "N/D",
        }

        if asset_returns.empty:
            notes.append(f"{name}: sin serie de cierre para {asset_ticker} en la matriz del bundle.")
        elif local_returns.empty:
            notes.append(f"{name}: sin datos del benchmark {bench_ticker}.")
        else:
            local_result = analyzer.compute_for_asset(asset_ticker, bench_ticker)
            if local_result:
                row.update(
                    {
                        "Beta (local)": fmt_num(local_result.get("beta")),
                        "Retorno esperado (CAPM local)": fmt_pct(local_result.get("expected_return_capm_annual"), digits=2),
                        "Clasificación (local)": local_result.get("classification", "Sin datos"),
                        "Alpha diaria (local)": fmt_alpha_daily(local_result.get("alpha_diaria")),
                        "R² (local)": fmt_num(local_result.get("r_squared")),
                        "p-value beta (local)": fmt_num(local_result.get("p_value_beta")),
                    }
                )
            else:
                notes.append(f"{name}: no hubo suficientes retornos alineados para CAPM local.")

            if not global_benchmark_failed:
                global_result = analyzer.compute_for_asset(asset_ticker, GLOBAL_BENCHMARK)
                if global_result:
                    row.update(
                        {
                            "Beta (ACWI)": fmt_num(global_result.get("beta")),
                            "Retorno esperado (CAPM ACWI)": fmt_pct(global_result.get("expected_return_capm_annual"), digits=2),
                            "Clasificación (ACWI)": global_result.get("classification", "N/D"),
                            "Alpha diaria (ACWI)": fmt_alpha_daily(global_result.get("alpha_diaria")),
                            "R² (ACWI)": fmt_num(global_result.get("r_squared")),
                            "p-value beta (ACWI)": fmt_num(global_result.get("p_value_beta")),
                        }
                    )
                else:
                    notes.append(f"{name}: no hubo suficientes retornos alineados contra {GLOBAL_BENCHMARK}.")

        rows.append(row)

    if global_benchmark_failed:
        notes.append(f"No se pudieron calcular columnas ACWI porque {GLOBAL_BENCHMARK} no devolvió datos.")

    return pd.DataFrame(rows), notes


def build_asset_summary_table(betas_df: pd.DataFrame) -> pd.DataFrame:
    return betas_df[
        [
            "Activo",
            "Ticker",
            "Benchmark usado (local)",
            "Beta (local)",
            "Retorno esperado (CAPM local)",
            "Clasificación (local)",
        ]
    ].rename(
        columns={
            "Benchmark usado (local)": "Benchmark local",
            "Beta (local)": "Beta local",
            "Retorno esperado (CAPM local)": "Retorno CAPM local",
            "Clasificación (local)": "Clasificación local",
        }
    )


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

    .soft-explain-box {
        background: #eff6ff;
        border: 1px solid rgba(37, 99, 235, 0.16);
        border-radius: 14px;
        padding: 14px 16px;
        margin: 0.65rem 0 0.9rem 0;
    }

    .soft-explain-title {
        color: #0f172a;
        font-size: 0.95rem;
        font-weight: 750;
        margin-bottom: 0.25rem;
    }

    .soft-explain-body {
        color: #334155;
        font-size: 0.88rem;
        line-height: 1.5;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

render_page_title(
    "Módulo 4 - CAPM y Beta",
    "Evalúa sensibilidad al mercado, rendimiento esperado y riesgo sistemático del activo.",
)

# ==============================
# Sidebar
# ==============================
with st.sidebar:
    st.header("Parámetros CAPM")
    asset_name = st.selectbox("Activo", [PORTFOLIO_OPTION] + list(ASSETS.keys()), index=1)
    manual_weights_enabled = False
    manual_weights = {}
    manual_weights_validation_error = None

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

    if asset_name == PORTFOLIO_OPTION:
        st.divider()
        manual_weights_enabled = st.checkbox("Definir pesos manualmente (opcional)", value=False)
        if manual_weights_enabled:
            st.caption("Ingresa pesos entre 0 y 1. La suma debe ser 1.00 para calcular el CAPM.")
            default_weight = 1 / len(ASSETS)
            for name in ASSETS.keys():
                manual_weights[name] = st.number_input(
                    f"{name} ({get_ticker(name)})",
                    min_value=0.0,
                    max_value=1.0,
                    value=float(default_weight),
                    step=0.01,
                    format="%.4f",
                    key=f"capm_weight_{get_ticker(name)}",
                )

            weights_sum = sum(manual_weights.values())
            diff_to_one = 1.0 - weights_sum
            st.caption(f"Suma de pesos: {weights_sum:.4f}")
            st.caption(f"Diferencia a 1.00: {diff_to_one:.4f}")

            try:
                PortfolioWeightsModel(pesos=manual_weights)
                st.success("Pesos OK: la suma es 1.00.")
            except ValidationError as exc:
                manual_weights_validation_error = exc
                if weights_sum > 1.0 + WEIGHT_TOL:
                    st.error("La suma de pesos supera 1.00. Ajusta los pesos para continuar.")
                elif weights_sum < 1.0 - WEIGHT_TOL:
                    st.warning("La suma de pesos es menor que 1.00. Ajusta los pesos para continuar.")
                else:
                    st.error("Los pesos no cumplen la validación requerida.")

# ==============================
# Datos
# ==============================
macro = macro_snapshot()
rf_annual = (
    macro["risk_free_rate_pct"] / 100
    if macro["risk_free_rate_pct"] == macro["risk_free_rate_pct"]
    else 0.03
)

is_portfolio = asset_name == PORTFOLIO_OPTION
portfolio_components = []
portfolio_missing_components = []
portfolio_weights = pd.Series(dtype=float)
asset_tickers = tuple(get_ticker(name) for name in ASSETS.keys())
benchmark_tickers = tuple(
    dict.fromkeys(
        [GLOBAL_BENCHMARK] + [get_local_benchmark(name) for name in ASSETS.keys()]
    )
)
market_tickers = asset_tickers + benchmark_tickers
try:
    close_matrix = load_capm_close_matrix(market_tickers, str(start_date), str(end_date))
except BackendAPIError as exc:
    st.error(friendly_error_message(exc, "No fue posible obtener datos de mercado desde el backend."))
    if exc.technical_detail:
        st.caption(exc.technical_detail)
    st.stop()

missing_backend_tickers = [ticker for ticker in asset_tickers if ticker not in close_matrix.columns]
if missing_backend_tickers:
    st.warning(
        "Sin datos para estos tickers en el rango seleccionado; se excluyen del análisis: "
        + ", ".join(missing_backend_tickers)
    )
capm_analyzer = CAPMAnalyzer(close=close_matrix, rf_annual=rf_annual)

if is_portfolio:
    ticker = "Portafolio equiponderado"
    benchmark_ticker = GLOBAL_BENCHMARK
    if manual_weights_enabled and manual_weights_validation_error is not None:
        st.error("No se calcula el CAPM del portafolio hasta que los pesos manuales sumen 1.00.")
        st.stop()

    asset_ret, portfolio_components, portfolio_missing_components, portfolio_weights = build_portfolio_returns(
        list(ASSETS.keys()),
        close_matrix,
        manual_weights if manual_weights_enabled else None,
    )

    if benchmark_ticker is None or str(benchmark_ticker).strip() == "":
        st.error("No se pudo resolver el benchmark global del portafolio.")
        st.stop()

    bench_ret = capm_analyzer.get_benchmark_returns(benchmark_ticker)
    stop_if_benchmark_empty(bench_ret, benchmark_ticker, asset_name)
    res = capm_analyzer.compute_for_portfolio(asset_ret, benchmark_ticker)
else:
    ticker = get_ticker(asset_name)
    benchmark_ticker = get_local_benchmark(asset_name)

    if ticker is None or str(ticker).strip() == "":
        st.error("No se pudo resolver el ticker del activo seleccionado.")
        st.caption(f"Activo seleccionado: {asset_name}")
        st.stop()

    if benchmark_ticker is None or str(benchmark_ticker).strip() == "":
        st.error("No se pudo resolver el benchmark local del activo seleccionado.")
        st.caption(f"Activo seleccionado: {asset_name}")
        st.caption(f"Ticker activo: {ticker}")
        st.stop()

    asset_ret = capm_analyzer.get_asset_returns(ticker)
    if asset_ret.empty:
        st.error("El backend no devolvió retornos válidos para el activo seleccionado.")
        st.stop()

    bench_ret = capm_analyzer.get_benchmark_returns(benchmark_ticker)
    stop_if_benchmark_empty(bench_ret, benchmark_ticker, asset_name)
    res = capm_analyzer.compute_for_asset(ticker, benchmark_ticker)

if not res:
    st.warning("No hay suficientes datos alineados para CAPM.")
    st.stop()

# ==============================
# Resultados principales
# ==============================
beta = res.get("beta")
alpha_diaria = res.get("alpha_diaria")
r_squared = res.get("r_squared")
expected_return = res.get("expected_return_capm_annual")
classification = res.get("classification")

beta_delta = None
beta_delta_type = "neu"
if beta is not None:
    if beta > 1:
        beta_delta = "Más sensible que el mercado"
        beta_delta_type = "neg"
    elif beta < 1:
        beta_delta = "Más defensivo que el mercado"
        beta_delta_type = "pos"
    else:
        beta_delta = "Sensibilidad similar al mercado"
        beta_delta_type = "neu"

alpha_delta = None
alpha_delta_type = "neu"
if alpha_diaria is not None:
    if alpha_diaria > 0:
        alpha_delta = "Alpha positivo"
        alpha_delta_type = "pos"
    elif alpha_diaria < 0:
        alpha_delta = "Alpha negativo"
        alpha_delta_type = "neg"

r2_delta = None
r2_delta_type = "neu"
if r_squared is not None:
    if r_squared >= 0.60:
        r2_delta = "Buen ajuste"
        r2_delta_type = "pos"
    elif r_squared >= 0.30:
        r2_delta = "Ajuste moderado"
        r2_delta_type = "neu"
    else:
        r2_delta = "Ajuste bajo"
        r2_delta_type = "neg"

ret_delta = None
ret_delta_type = "neu"
if expected_return is not None:
    if expected_return > rf_annual:
        ret_delta = "Sobre tasa libre de riesgo"
        ret_delta_type = "pos"
    elif expected_return < rf_annual:
        ret_delta = "Bajo tasa libre de riesgo"
        ret_delta_type = "neg"

entity_label = "portafolio" if is_portfolio else "activo"
entity_display = "portafolio equiponderado" if is_portfolio else "activo"
classification_title = "Clasificación del portafolio" if is_portfolio else "Clasificación del activo"

if classification == "Agresivo":
    class_msg = (
        f"El {entity_display} se clasifica como **Agresivo**: su beta sugiere una sensibilidad mayor al benchmark. "
        "Puede amplificar movimientos de mercado al alza o a la baja."
    )
elif classification == "Defensivo":
    class_msg = (
        f"El {entity_display} se clasifica como **Defensivo**: su beta sugiere menor sensibilidad frente al benchmark. "
        "Tiende a moverse con menos intensidad que el mercado de referencia."
    )
elif classification == "Neutro":
    class_msg = (
        f"El {entity_display} se clasifica como **Neutro**: su beta se ubica cerca de 1 y su sensibilidad es similar "
        "a la del benchmark."
    )
else:
    class_msg = "Clasificación no disponible."

# ==============================
# Resumen
# ==============================
st.markdown("### Resumen del módulo")
if is_portfolio:
    st.caption(
        f"Beta del portafolio frente a {benchmark_ticker} y CAPM con tasa libre de riesgo tomada del módulo macro."
    )
else:
    st.caption(
        f"Beta de {asset_name} ({ticker}) frente a {benchmark_ticker} y retorno CAPM con tasa libre de riesgo macro."
    )

st.caption(f"Periodo analizado: {start_date} a {end_date}")

if is_portfolio:
    st.markdown("### Portafolio")
    weights_text = ", ".join(f"{name}: {weight:.2%}" for name, weight in portfolio_weights.items())
    weights_title = "Construcción con pesos manuales validados" if manual_weights_enabled else "Construcción equiponderada: pesos iguales"
    weights_body = (
        f"El portafolio usa los pesos efectivos {weights_text}. "
        f"Los retornos diarios se alinean por fecha y se calculan con inner join. "
        f"Para CAPM se compara contra el benchmark global {benchmark_ticker}."
    )
    soft_note(
        weights_title,
        weights_body,
    )
    if portfolio_missing_components:
        st.caption(
            "No se incluyeron por falta de datos en la matriz del bundle: "
            + ", ".join(portfolio_missing_components)
        )

# ==============================
# KPIs
# ==============================
st.markdown("### KPIs CAPM")
render_section(
    "Resumen ejecutivo del modelo",
    "Beta, ajuste estadístico y retorno esperado del elemento seleccionado.",
)

c1, c2, c3, c4 = st.columns(4)

with c1:
    kpi_card(
        "Beta",
        fmt_num(beta),
        delta=beta_delta,
        delta_type=beta_delta_type,
        caption=f"Sensibilidad del {entity_label} frente al benchmark",
    )

with c2:
    kpi_card(
        "Alpha diaria",
        fmt_alpha_daily(alpha_diaria),
        delta=alpha_delta,
        delta_type=alpha_delta_type,
        caption="Componente diario no explicado por el mercado",
    )

with c3:
    kpi_card(
        "R²",
        fmt_num(r_squared),
        delta=r2_delta,
        delta_type=r2_delta_type,
        caption="Proporción explicada por la regresión CAPM",
    )

with c4:
    kpi_card(
        "Retorno esperado anual",
        fmt_pct(expected_return, digits=2),
        delta=ret_delta,
        delta_type=ret_delta_type,
        caption="Rendimiento estimado bajo CAPM",
    )

render_explanation_expander(
    "Cómo interpretar los KPI CAPM",
    [
        "Beta: sensibilidad frente al benchmark y medida del riesgo sistemático.",
        "Alpha diaria: rendimiento no explicado por el benchmark.",
        "R²: proporción explicada por la regresión CAPM.",
        "Retorno esperado anual: estimación CAPM usando la tasa libre de riesgo desde la API macro.",
    ],
)

# ==============================
# Clasificación
# ==============================
st.markdown(f"### {classification_title}")
soft_note("Lectura rápida de la beta", class_msg)

if is_portfolio:
    betas_df, betas_notes = build_asset_betas_table(capm_analyzer)
    betas_summary_df = build_asset_summary_table(betas_df)

    # ==============================
    # Tabla resumen por activo
    # ==============================
    st.markdown("### Tabla resumen por activo")
    render_table(
        betas_summary_df,
        hide_index=True,
        width="stretch",
    )

    with st.expander("Ver tabla técnica por activo", expanded=False):
        render_table(
            betas_df,
            hide_index=True,
            width="stretch",
        )
        if betas_notes:
            st.caption("Notas técnicas por fallos de descarga o falta de datos alineados.")
            for note in betas_notes:
                st.caption(f"- {note}")

# ==============================
# Regresión CAPM
# ==============================
st.markdown("### Regresión CAPM")
render_section(
    "Relación activo-mercado",
    f"Dispersión de rendimientos del {entity_label} frente al benchmark con su línea de regresión.",
)

fig = plot_scatter_regression(
    x=res["scatter_data"]["market_excess"],
    y=res["scatter_data"]["asset_excess"],
    yhat=res["regression_line"]["y"],
    title="Regresión CAPM",
)
st.plotly_chart(fig, width="stretch")

# ==============================
# Cómo leer el gráfico
# ==============================
render_explanation_expander(
    "Cómo interpretar el gráfico CAPM",
    [
        f"Cada punto representa una observación del {entity_label} frente al benchmark.",
        "La línea de regresión resume la relación promedio.",
        "La pendiente de la línea corresponde a la beta.",
        "La dispersión alrededor de la recta se relaciona con riesgo no sistemático o componente idiosincrático.",
    ],
)

# ==============================
# Tabla técnica
# ==============================
summary_df = pd.DataFrame(
    {
        "Métrica": [
            "Beta",
            "Alpha diaria",
            "R²",
            "p-value beta",
            "Retorno esperado anual",
            "Clasificación",
            "Benchmark utilizado",
            "Tasa libre anual usada",
        ],
        "Valor": [
            fmt_num(res["beta"]),
            fmt_alpha_daily(res["alpha_diaria"]),
            fmt_num(res["r_squared"]),
            fmt_num(res["p_value_beta"]),
            fmt_pct(res["expected_return_capm_annual"], digits=2),
            res["classification"],
            benchmark_ticker,
            fmt_pct(rf_annual, digits=2),
        ],
    }
)

with st.expander("Ver tabla técnica completa"):
    render_table(summary_df, hide_index=True, width="stretch")

# ==============================
# Interpretación económica
# ==============================
st.markdown("### Interpretación económica del CAPM")
st.info(
    f"Beta: **{fmt_num(beta)}** | Retorno CAPM: **{fmt_pct(expected_return, digits=2)}** | Tasa libre usada: **{fmt_pct(rf_annual, digits=2)}**."
)

with st.expander("Ver conclusión e interpretación económica", expanded=False):
    st.write(
        f"""
        La beta estimada de **{fmt_num(beta)}** indica la sensibilidad del {entity_label} frente al benchmark
        **{benchmark_ticker}**. {class_msg}

        El retorno esperado anual bajo CAPM es **{fmt_pct(expected_return, digits=2)}** y se calcula con una
        tasa libre de riesgo anual de **{fmt_pct(rf_annual, digits=2)}** obtenida automáticamente desde la API macro.
        """
    )

with st.expander("Riesgo sistemático, no sistemático y diversificación", expanded=False):
    st.write(
        """
        - **Riesgo sistemático:** depende del mercado y no se elimina fácilmente con diversificación.
        - **Riesgo no sistemático:** es propio del activo o empresa y, en teoría, puede reducirse al diversificar.
        - El **CAPM** remunera principalmente la exposición al riesgo sistemático, capturada por la beta.
        """
    )
