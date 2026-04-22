import streamlit as st
import streamlit.components.v1 as components
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
from src.download import data_error_debug, download_single_ticker, load_market_bundle
from src.returns_analysis import compute_return_series
from src.capm import compute_beta_and_capm
from src.api.macro import macro_snapshot
from src.plots import plot_scatter_regression
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


def get_price_series(df: pd.DataFrame) -> pd.Series:
    return df["Adj Close"] if "Adj Close" in df.columns else df["Close"]


def returns_from_ohlcv(df: pd.DataFrame) -> pd.Series:
    return compute_return_series(get_price_series(df))["simple_return"]


@st.cache_data(show_spinner=False, ttl=3600)
def download_benchmark_cached(bench_ticker: str, start: str, end: str) -> pd.DataFrame:
    return download_single_ticker(ticker=bench_ticker, start=start, end=end)


@st.cache_data(show_spinner=False, ttl=3600)
def load_capm_close_matrix(tickers: tuple[str, ...], start: str, end: str) -> pd.DataFrame:
    bundle = load_market_bundle(tickers=list(tickers), start=start, end=end)
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


@st.cache_data(show_spinner=False, ttl=3600)
def benchmark_returns_cached(bench_ticker: str, start: str, end: str) -> pd.Series:
    bench_df = download_benchmark_cached(bench_ticker, start, end)
    if bench_df.empty:
        return pd.Series(dtype=float)
    return returns_from_ohlcv(bench_df)


def fallback_asset_returns(asset_label: str, asset_ticker: str) -> pd.Series:
    st.warning(f"fallback Yahoo: no se encontró {asset_ticker} en la matriz de cierres del bundle.")
    asset_data = download_single_ticker(ticker=asset_ticker, start=str(start_date), end=str(end_date))
    if asset_data.empty:
        st.error("No se pudieron descargar datos del activo.")
        st.caption("Diagnostico de descarga CAPM")
        st.code(
            "\n".join(
                [
                    f"Activo seleccionado: {asset_label}",
                    f"Ticker activo: {asset_ticker}",
                    f"Rango: {start_date} a {end_date}",
                    f"Filas activo descargadas: {len(asset_data)}",
                    f"Columnas activo: {list(asset_data.columns)}",
                    f"Detalle tecnico: {data_error_debug('No hay excepcion tecnica registrada.')}",
                ]
            )
        )
        st.stop()
    return returns_from_ohlcv(asset_data)


def stop_if_benchmark_empty(bench_df: pd.DataFrame, bench_ticker: str, selected_label: str) -> None:
    if not bench_df.empty:
        return
    st.error("No hay datos del benchmark; intenta recargar o cambiar horizonte.")
    st.caption("Diagnostico de descarga CAPM")
    st.code(
        "\n".join(
            [
                f"Activo seleccionado: {selected_label}",
                f"Benchmark usado: {bench_ticker}",
                f"Tipo benchmark inferido: {infer_ticker_type(bench_ticker)}",
                f"Rango: {start_date} a {end_date}",
                f"Filas benchmark descargadas: {len(bench_df)}",
                f"Columnas benchmark: {list(bench_df.columns)}",
                f"Detalle tecnico: {data_error_debug('No hay excepcion tecnica registrada.')}",
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
        st.caption("Diagnostico de descarga CAPM")
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
    rf_annual_value: float,
    close_matrix: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str]]:
    rows = []
    notes = []
    global_returns = benchmark_returns_cached(GLOBAL_BENCHMARK, str(start_date), str(end_date))
    global_benchmark_failed = global_returns.empty

    for name in ASSETS.keys():
        asset_ticker = get_ticker(name)
        bench_ticker = get_local_benchmark(name)
        close_series = close_series_from_matrix(close_matrix, asset_ticker)
        local_returns = benchmark_returns_cached(bench_ticker, str(start_date), str(end_date))

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

        if close_series.empty:
            notes.append(f"{name}: sin serie de cierre para {asset_ticker} en la matriz del bundle.")
        elif local_returns.empty:
            notes.append(f"{name}: sin datos del benchmark {bench_ticker}.")
        else:
            asset_returns = returns_from_close_cached(close_series)
            local_result = compute_beta_and_capm(
                asset_returns,
                local_returns,
                rf_annual=rf_annual_value,
            )
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
                global_result = compute_beta_and_capm(
                    asset_returns,
                    global_returns,
                    rf_annual=rf_annual_value,
                )
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
                background: linear-gradient(180deg, #eff6ff 0%, #dbeafe 100%);
                border: 1px solid rgba(37, 99, 235, 0.16);
                border-radius: 14px;
                padding: 16px 16px 14px 16px;
                box-shadow: 0 4px 14px rgba(37, 99, 235, 0.08);
                min-height: 156px;
                box-sizing: border-box;
                display: flex;
                flex-direction: column;
                justify-content: space-between;
                gap: 0.55rem;
                overflow: visible;
            }}

            .kpi-label {{
                font-size: 0.82rem;
                font-weight: 700;
                color: #334155;
                margin-bottom: 0.35rem;
                letter-spacing: 0;
                line-height: 1.25;
            }}

            .kpi-value {{
                font-size: 1.55rem;
                font-weight: 800;
                color: #0f172a;
                line-height: 1.12;
                margin-bottom: 0.45rem;
                overflow-wrap: anywhere;
                word-break: normal;
                white-space: normal;
            }}

            .kpi-delta {{
                display: inline-block;
                width: fit-content;
                max-width: 100%;
                font-size: 0.76rem;
                font-weight: 700;
                padding: 0.28rem 0.55rem;
                border-radius: 999px;
                margin-top: 0.10rem;
                line-height: 1.2;
                white-space: normal;
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
                font-size: 0.76rem;
                color: #475569;
                margin-top: 0.45rem;
                line-height: 1.38;
                overflow-wrap: normal;
                word-break: normal;
                white-space: normal;
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

    components.html(html, height=178)


inject_kpi_cards_css()

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
close_matrix = load_capm_close_matrix(asset_tickers, str(start_date), str(end_date))

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

    bench_df = download_benchmark_cached(benchmark_ticker, str(start_date), str(end_date))
    stop_if_benchmark_empty(bench_df, benchmark_ticker, asset_name)
    bench_ret = benchmark_returns_cached(benchmark_ticker, str(start_date), str(end_date))
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

    asset_close = close_series_from_matrix(close_matrix, ticker)
    asset_ret = fallback_asset_returns(asset_name, ticker) if asset_close.empty else returns_from_close_cached(asset_close)

    bench_df = download_benchmark_cached(benchmark_ticker, str(start_date), str(end_date))
    stop_if_benchmark_empty(bench_df, benchmark_ticker, asset_name)
    bench_ret = benchmark_returns_cached(benchmark_ticker, str(start_date), str(end_date))

res = compute_beta_and_capm(asset_ret, bench_ret, rf_annual=rf_annual)

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
    st.write(
        f"""
        Este módulo estima la beta del **{asset_name}** frente al benchmark global **{benchmark_ticker}**,
        evalúa la regresión CAPM y calcula el retorno esperado usando una tasa libre de riesgo tomada
        automáticamente del módulo macro.
        """
    )
else:
    st.write(
        f"""
        Este módulo estima la beta de **{asset_name} ({ticker})** frente a su benchmark local
        **{benchmark_ticker}**, evalúa la regresión CAPM y calcula el retorno esperado usando una tasa
        libre de riesgo tomada automáticamente del módulo macro.
        """
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
section_intro(
    "Resumen ejecutivo del modelo",
    "Aquí se condensan la sensibilidad al mercado, el ajuste del modelo y el retorno esperado estimado bajo CAPM.",
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

# ==============================
# Clasificación
# ==============================
st.markdown(f"### {classification_title}")
soft_note("Lectura rápida de la beta", class_msg)

# ==============================
# Regresión CAPM
# ==============================
st.markdown("### Regresión CAPM")
section_intro(
    "Relación activo-mercado",
    f"El diagrama de dispersión muestra cómo responde el {entity_label} a los movimientos del benchmark y permite interpretar visualmente la beta.",
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
st.markdown("### Cómo leer el gráfico")
st.info(
    f"""
    - Cada punto representa una observación del {entity_label} frente al benchmark.
    - La línea resume la relación promedio entre ambos excesos de retorno.
    - La pendiente de esa línea es la beta: una pendiente mayor implica mayor exposición al riesgo sistemático.
    """
)

with st.expander("Ver interpretación técnica del gráfico"):
    st.write(
        f"""
        El diagrama de dispersión muestra la relación entre el exceso de retorno del benchmark
        y el exceso de retorno del {entity_label}. La pendiente de la recta estimada corresponde a la beta,
        mientras que la dispersión alrededor de la recta se relaciona con el componente idiosincrático.
        """
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
    st.dataframe(summary_df, width="stretch")

if is_portfolio:
    st.markdown("### Betas por activo")
    betas_df, betas_notes = build_asset_betas_table(rf_annual, close_matrix)
    betas_summary_df = betas_df[
        [
            "Activo",
            "Ticker",
            "Benchmark usado (local)",
            "Beta (local)",
            "Retorno esperado (CAPM local)",
            "Clasificación (local)",
            "Benchmark global",
            "Beta (ACWI)",
            "Retorno esperado (CAPM ACWI)",
            "Clasificación (ACWI)",
        ]
    ].rename(
        columns={
            "Benchmark usado (local)": "Benchmark local",
            "Beta (local)": "Beta local",
            "Retorno esperado (CAPM local)": "Retorno CAPM local",
            "Clasificación (local)": "Clasificación local",
            "Beta (ACWI)": "Beta ACWI",
            "Retorno esperado (CAPM ACWI)": "Retorno CAPM ACWI",
            "Clasificación (ACWI)": "Clasificación ACWI",
        }
    )
    st.dataframe(
        betas_summary_df,
        use_container_width=True,
        hide_index=True,
        height=260,
    )

    with st.expander("Ver tabla técnica completa (comparación local vs ACWI)"):
        st.dataframe(
            betas_df,
            use_container_width=True,
            hide_index=True,
            height=320,
        )
        if betas_notes:
            st.caption("Algunas filas aparecen como 'Sin datos' o 'N/D' por fallos de descarga o falta de datos alineados.")
            for note in betas_notes:
                st.caption(f"- {note}")

# ==============================
# Interpretación económica
# ==============================
st.markdown("### Interpretación económica del CAPM")

expected_text = (
    f"El retorno esperado anual bajo CAPM es **{fmt_pct(expected_return, digits=2)}**. "
    f"Este cálculo usa una tasa libre de riesgo anual de **{fmt_pct(rf_annual, digits=2)}** "
    "obtenida automáticamente desde el contexto macro."
)

st.info(
    f"""
    La beta estimada de **{fmt_num(beta)}** indica la sensibilidad del {entity_label} frente al benchmark
    **{benchmark_ticker}**. {class_msg} {expected_text}
    """
)

with st.expander("Riesgo sistemático, no sistemático y diversificación"):
    st.write(
        """
        - **Riesgo sistemático:** depende del mercado y no se elimina fácilmente con diversificación.
        - **Riesgo no sistemático:** es propio del activo o empresa y, en teoría, puede reducirse al diversificar.
        - El **CAPM** remunera principalmente la exposición al riesgo sistemático, capturada por la beta.
        """
    )
