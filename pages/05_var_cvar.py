import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import pandas as pd
from pydantic import BaseModel, ValidationError

try:
    from pydantic import model_validator

    PYDANTIC_V2 = True
except ImportError:
    from pydantic import root_validator

    PYDANTIC_V2 = False

from src.config import ASSETS, DEFAULT_END_DATE, ensure_project_dirs
from src.api.backend_client import BackendAPIError, friendly_error_message
from src.download import data_error_message
from src.risk_metrics import kupiec_test
from src.services.market_data_client import MarketDataClient
from src.services.risk_analyzer import RiskAnalyzer
from src.plots import plot_var_distribution
from src.ui_navigation import render_sidebar_navigation
from src.ui_style import apply_global_typography, render_page_title

ensure_project_dirs()
apply_global_typography()
render_sidebar_navigation()

CONFIDENCE_LEVELS = [0.95, 0.99]
WEIGHT_TOL = 1e-6


def _validate_weights_dict(pesos: dict[str, float], tol: float = WEIGHT_TOL) -> None:
    if not pesos:
        raise ValueError("Debe ingresar al menos un peso.")
    if any(weight < 0 or weight > 1 for weight in pesos.values()):
        raise ValueError("Todos los pesos deben estar entre 0 y 1.")
    if abs(sum(pesos.values()) - 1.0) > tol:
        raise ValueError("La suma de pesos debe ser 1.00.")


def _validate_n_sim(n_sim: int) -> None:
    if int(n_sim) < 10_000:
        raise ValueError("Monte Carlo requiere al menos 10,000 simulaciones.")


if PYDANTIC_V2:

    class PortfolioWeightsModel(BaseModel):
        pesos: dict[str, float]

        @model_validator(mode="after")
        def validate_weights(self):
            _validate_weights_dict(self.pesos)
            return self


    class SimulationConfigModel(BaseModel):
        n_sim: int

        @model_validator(mode="after")
        def validate_simulations(self):
            _validate_n_sim(self.n_sim)
            return self

else:

    class PortfolioWeightsModel(BaseModel):
        pesos: dict[str, float]

        @root_validator
        def validate_weights(cls, values):
            _validate_weights_dict(values.get("pesos") or {})
            return values


    class SimulationConfigModel(BaseModel):
        n_sim: int

        @root_validator
        def validate_simulations(cls, values):
            _validate_n_sim(values.get("n_sim"))
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


def style_risk_table(df: pd.DataFrame):
    return (
        df.style.hide(axis="index")
        .set_table_styles(
            [
                {
                    "selector": "th",
                    "props": [
                        ("background-color", "#EAF3FF"),
                        ("color", "#0f172a"),
                        ("font-weight", "700"),
                        ("border", "1px solid rgba(37, 99, 235, 0.16)"),
                    ],
                },
                {
                    "selector": "td",
                    "props": [
                        ("border", "1px solid rgba(15, 23, 42, 0.06)"),
                    ],
                },
            ]
        )
    )


def fmt_pct_value(value):
    numeric_value = pd.to_numeric(value, errors="coerce")
    return f"{numeric_value:.2%}" if pd.notna(numeric_value) else "N/D"


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
                background: linear-gradient(180deg, #f3f8ff 0%, #eaf3ff 100%);
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
                font-size: 1.58rem;
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
    "Módulo 5 - VaR y CVaR",
    "Evalúa el riesgo extremo del portafolio mediante VaR y CVaR bajo distintos enfoques de estimación.",
)

# ==============================
# Sidebar
# ==============================
with st.sidebar:
    st.header("Parámetros de riesgo")

    horizonte = st.selectbox(
        "Horizonte histórico de análisis",
        [
            "1 año",
            "2 años",
            "5 años",
        ],
        index=0,
    )

    fecha_fin_ref = pd.to_datetime(DEFAULT_END_DATE)

    if horizonte == "1 año":
        start_date = (fecha_fin_ref - pd.DateOffset(years=1)).date()
        end_date = fecha_fin_ref.date()
    elif horizonte == "2 años":
        start_date = (fecha_fin_ref - pd.DateOffset(years=2)).date()
        end_date = fecha_fin_ref.date()
    elif horizonte == "5 años":
        start_date = (fecha_fin_ref - pd.DateOffset(years=5)).date()
        end_date = fecha_fin_ref.date()

    alpha = st.radio("Nivel para mostrar KPIs", CONFIDENCE_LEVELS, index=0, horizontal=True)

    n_sim = int(
        st.number_input(
            "Simulaciones Monte Carlo",
            min_value=10_000,
            value=10_000,
            step=1_000,
            format="%d",
        )
    )
    try:
        SimulationConfigModel(n_sim=n_sim)
    except ValidationError:
        st.error("Monte Carlo requiere al menos 10,000 simulaciones.")
        st.stop()

    manual_weights_enabled = st.checkbox("Definir pesos manualmente (opcional)", value=False)
    manual_weights = {}
    if manual_weights_enabled:
        st.caption("Ingresa pesos entre 0 y 1. La suma debe ser 1.00.")
        default_weight = 1 / len(ASSETS)
        for asset_name, meta in ASSETS.items():
            manual_weights[asset_name] = st.number_input(
                f"{asset_name} ({meta['ticker']})",
                min_value=0.0,
                max_value=1.0,
                value=float(default_weight),
                step=0.01,
                format="%.4f",
                key=f"var_weight_{meta['ticker']}",
            )

        weights_sum = sum(manual_weights.values())
        st.caption(f"Suma actual = {weights_sum:.6f}")
        try:
            PortfolioWeightsModel(pesos=manual_weights)
            st.success("Pesos OK: la suma es 1.00.")
        except ValidationError:
            st.error(f"Pesos inválidos: la suma debe ser 1.00 para continuar. Suma actual = {weights_sum:.6f}.")
            st.stop()

# ==============================
# Carga y preparación de datos
# ==============================
tickers = [meta["ticker"] for meta in ASSETS.values()]
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
        "Sin datos para estos tickers en el rango seleccionado; se excluyen del analisis: "
        + ", ".join(missing_tickers)
    )

risk_analyzer = RiskAnalyzer()
returns = risk_analyzer.clean_returns(bundle["returns"])

if not risk_analyzer.validate_sample(returns, min_rows=30):
    st.error(data_error_message("No hay suficientes datos para calcular métricas de riesgo."))
    st.stop()

if manual_weights_enabled:
    ticker_to_asset = {meta["ticker"]: name for name, meta in ASSETS.items()}
    weights = np.array(
        [manual_weights[ticker_to_asset[ticker]] for ticker in returns.columns],
        dtype=float,
    )
    portfolio_returns, weights = risk_analyzer.portfolio_returns(returns, weights)
else:
    portfolio_returns, weights = risk_analyzer.portfolio_returns(returns)

tables_by_alpha = risk_analyzer.compute_var_tables(
    portfolio_returns=portfolio_returns,
    asset_returns_df=returns,
    weights=weights,
    confidence_levels=CONFIDENCE_LEVELS,
    n_sim=n_sim,
)

table = pd.concat(
    [alpha_table for alpha_table in tables_by_alpha.values() if not alpha_table.empty],
    ignore_index=True,
)
selected_table = tables_by_alpha.get(alpha, pd.DataFrame())

if table.empty:
    st.error("No fue posible calcular VaR y CVaR con los datos disponibles.")
    st.stop()

if selected_table.empty:
    st.error("No fue posible calcular VaR y CVaR para el nivel de confianza seleccionado.")
    st.stop()

# ==============================
# Resumen
# ==============================
st.markdown("### Resumen del módulo")
st.write(
    f"""
    Este módulo compara el **Value at Risk (VaR)** y el **Conditional Value at Risk (CVaR)** del portafolio
    equiponderado bajo enfoques **paramétrico**, **histórico** y **Monte Carlo** con **{n_sim:,} simulaciones**,
    usando la convención de pérdidas positivas para un nivel de confianza de **{int(alpha * 100)}%**.
    """
)

st.caption(f"Periodo analizado: {start_date} a {end_date}")

# ==============================
# Portafolio
# ==============================
st.markdown("### Portafolio analizado")
if manual_weights_enabled:
    st.write("Se usa un portafolio con pesos manuales validados.")
else:
    st.write("Se usa un portafolio equiponderado, es decir, todos los activos tienen el mismo peso.")

# ==============================
# Filas por método
# ==============================
var_hist_row = selected_table.loc[selected_table["método"] == "Histórico"]
var_param_row = selected_table.loc[selected_table["método"] == "Paramétrico"]
var_mc_row = selected_table.loc[selected_table["método"] == "Monte Carlo"]

var_h = float(var_hist_row["VaR_diario"].iloc[0]) if not var_hist_row.empty else None
cvar_h = float(var_hist_row["CVaR_diario"].iloc[0]) if not var_hist_row.empty else None
var_p = float(var_param_row["VaR_diario"].iloc[0]) if not var_param_row.empty else None
cvar_p = float(var_param_row["CVaR_diario"].iloc[0]) if not var_param_row.empty else None
var_mc = float(var_mc_row["VaR_diario"].iloc[0]) if not var_mc_row.empty else None
cvar_mc = float(var_mc_row["CVaR_diario"].iloc[0]) if not var_mc_row.empty else None

# ==============================
# KPIs
# ==============================
st.markdown("### KPIs de riesgo")
section_intro(
    "Resumen ejecutivo del riesgo extremo",
    "Estas métricas resumen la pérdida umbral esperada y la severidad promedio de los escenarios más extremos del portafolio.",
)

kpi_row_1 = st.columns(3)
kpi_row_2 = st.columns(3)

with kpi_row_1[0]:
    kpi_card(
        "VaR paramétrico diario",
        f"{var_p:.2%}" if var_p is not None else "N/D",
        delta=f"{int(alpha * 100)}% confianza",
        caption="Pérdida umbral bajo supuesto de normalidad",
    )

with kpi_row_1[1]:
    kpi_card(
        "VaR histórico diario",
        f"{var_h:.2%}" if var_h is not None else "N/D",
        delta=f"{int(alpha * 100)}% confianza",
        caption="Pérdida umbral con distribución empírica",
    )

with kpi_row_1[2]:
    kpi_card(
        "VaR Monte Carlo diario",
        f"{var_mc:.2%}" if var_mc is not None else "N/D",
        delta=f"{n_sim:,} simulaciones",
        caption="Pérdida umbral mediante simulación probabilística",
    )

with kpi_row_2[0]:
    kpi_card(
        "CVaR paramétrico diario",
        f"{cvar_p:.2%}" if cvar_p is not None else "N/D",
        delta="Cola extrema",
        delta_type="neg",
        caption="Pérdida promedio más allá del VaR paramétrico",
    )

with kpi_row_2[1]:
    kpi_card(
        "CVaR histórico diario",
        f"{cvar_h:.2%}" if cvar_h is not None else "N/D",
        delta="Cola extrema",
        delta_type="neg",
        caption="Pérdida promedio en los peores casos observados",
    )

with kpi_row_2[2]:
    kpi_card(
        "CVaR Monte Carlo diario",
        f"{cvar_mc:.2%}" if cvar_mc is not None else "N/D",
        delta="Cola extrema",
        delta_type="neg",
        caption="Pérdida promedio en escenarios simulados extremos",
    )

with st.expander("Interpretación"):
    st.write(
        f"""
        - **VaR paramétrico diario:** asume normalidad y marca el umbral de pérdida diaria que no debería superarse con **{int(alpha * 100)}%** de confianza.
        - **VaR histórico diario:** usa percentiles empíricos de los rendimientos observados para estimar la pérdida diaria umbral al **{int(alpha * 100)}%**.
        - **VaR Monte Carlo diario:** estima el percentil de pérdida diaria a partir de **{n_sim:,} simulaciones**.
        - **CVaR paramétrico diario:** estima la pérdida diaria promedio condicionada a exceder el VaR paramétrico.
        - **CVaR histórico diario:** promedia las pérdidas diarias más extremas observadas en la cola histórica.
        - **CVaR Monte Carlo diario:** promedia las pérdidas diarias más extremas dentro de los escenarios simulados.
        """
    )

# ==============================
# Gráfico
# ==============================
st.markdown("### Distribución y riesgo extremo")
section_intro(
    "Distribución de rendimientos del portafolio",
    "El histograma muestra la forma empírica de los rendimientos y las líneas señalan los umbrales de VaR y CVaR por método.",
)

st.caption(
    "El histograma muestra la distribución de rendimientos. Las líneas punteadas representan VaR y CVaR por método."
)
fig_var = plot_var_distribution(portfolio_returns, selected_table)
fig_var.update_traces(marker_line_width=0.6, marker_line_color="rgba(15, 23, 42, 0.35)", selector=dict(type="histogram"))
line_styles = {
    "Paramétrico": "#2563eb",
    "Histórico": "#16a34a",
    "Monte Carlo": "#f59e0b",
}
for trace in fig_var.data:
    trace_name = str(getattr(trace, "name", ""))
    if trace_name.startswith("VaR") or trace_name.startswith("CVaR"):
        for method_name, color in line_styles.items():
            if method_name in trace_name:
                trace.line.color = color
                trace.line.width = 3
                trace.line.dash = "dash" if trace_name.startswith("VaR") else "dot"
st.plotly_chart(fig_var, width="stretch")

st.info(
    """
    **Cómo leer este gráfico**

    - El histograma resume cómo se distribuyen los rendimientos del portafolio.
    - Las líneas de **VaR** marcan pérdidas umbral bajo distintos métodos.
    - Las líneas de **CVaR** muestran pérdidas promedio más severas en escenarios extremos.
    - Cuando el CVaR es más alto que el VaR, las pérdidas extremas pueden ser considerablemente más intensas.
    """
)

with st.expander("Leyenda de líneas (VaR/CVaR)"):
    st.write(
        """
        - **Rendimientos (histograma):** distribución empírica de rendimientos diarios del portafolio.
        - **VaR Paramétrico:** umbral de pérdida diaria estimado bajo normalidad.
        - **CVaR Paramétrico:** pérdida promedio más allá del VaR paramétrico.
        - **VaR Histórico:** umbral de pérdida calculado con rendimientos observados.
        - **CVaR Histórico:** promedio de pérdidas históricas en la cola extrema.
        - **VaR Monte Carlo:** umbral de pérdida a partir de escenarios simulados.
        - **CVaR Monte Carlo:** promedio de pérdidas extremas dentro de los escenarios simulados.
        """
    )

# ==============================
# Tabla
# ==============================
st.markdown("### Comparación VaR / CVaR")
compact_table = table[["confianza", "método", "VaR_diario", "CVaR_diario"]].copy()
compact_table = compact_table.rename(
    columns={
        "confianza": "Confianza",
        "método": "Método",
        "VaR_diario": "VaR diario",
        "CVaR_diario": "CVaR diario",
    }
)
compact_table["Confianza"] = compact_table["Confianza"].map(lambda x: f"{x:.0%}")
for col in ["VaR diario", "CVaR diario"]:
    compact_table[col] = compact_table[col].map(fmt_pct_value)

st.dataframe(style_risk_table(compact_table), use_container_width=True, height=260)

complete_table = table.copy().rename(
    columns={
        "confianza": "Confianza",
        "método": "Método",
        "VaR_diario": "VaR diario",
        "CVaR_diario": "CVaR diario",
        "VaR_anualizado": "VaR anualizado",
        "CVaR_anualizado": "CVaR anualizado",
    }
)
complete_table["Confianza"] = complete_table["Confianza"].map(lambda x: f"{x:.0%}")
for col in ["VaR diario", "CVaR diario", "VaR anualizado", "CVaR anualizado"]:
    if col in complete_table.columns:
        complete_table[col] = complete_table[col].map(fmt_pct_value)

with st.expander("Ver tabla completa de VaR y CVaR"):
    st.dataframe(style_risk_table(complete_table), use_container_width=True)

st.info(
    f"""
    Para el nivel seleccionado de **{int(alpha * 100)}%**, el VaR histórico diario indica una pérdida umbral de
    **{fmt_pct_value(var_h)}** y el CVaR histórico diario estima una pérdida promedio de cola de
    **{fmt_pct_value(cvar_h)}**. Si el CVaR supera claramente al VaR, los escenarios extremos son más severos
    que el umbral inicial. La comparación entre métodos ayuda a ver cuánto dependen los resultados del supuesto
    normal, de la historia observada o de simulaciones Monte Carlo.
    """
)

with st.expander("Backtesting (opcional) - Test de Kupiec"):
    section_intro(
        "Validación del VaR estimado",
        "Este bloque contrasta si la frecuencia observada de violaciones del VaR es coherente con la esperada bajo el nivel de confianza seleccionado.",
    )
    st.write(
        """
        Una **violación** ocurre cuando la pérdida observada supera el VaR estimado. La tasa esperada de
        violaciones es \(1 - \alpha\): por ejemplo, 5% para 95% de confianza y 1% para 99%.
        El test de Kupiec evalúa si la frecuencia observada de violaciones es compatible con esa tasa esperada.
        Si el **p-value** es mayor que 0.05, no se rechaza la calibración del VaR; si es menor o igual, hay evidencia
        de que el VaR no está bien calibrado.
        """
    )

    backtest_method = st.selectbox(
        "Método para backtesting",
        ["Histórico", "Paramétrico", "Monte Carlo"],
        index=0,
    )
    method_row = selected_table.loc[selected_table["método"] == backtest_method]
    backtest_var = float(method_row["VaR_diario"].iloc[0]) if not method_row.empty else None

    if backtest_var is not None:
        kupiec = kupiec_test(
            returns=portfolio_returns,
            var=backtest_var,
            alpha=alpha,
        )

        if kupiec:
            col1, col2, col3 = st.columns(3)

            with col1:
                kpi_card(
                    "Violaciones",
                    str(kupiec["violations"]),
                    caption="Número de veces que la pérdida superó el VaR",
                )

            with col2:
                kpi_card(
                    "Observadas (%)",
                    f"{kupiec['observed_fail_rate'] * 100:.2f}%",
                    caption="Frecuencia empírica de violaciones",
                )

            with col3:
                kpi_card(
                    "Esperadas (%)",
                    f"{kupiec['expected_fail_rate'] * 100:.2f}%",
                    caption="Frecuencia teórica bajo el modelo",
                )

            st.write(f"**p-value:** {kupiec['p_value']:.4f}")
            st.write(f"**Conclusión:** {kupiec['conclusion']}")

            if kupiec["p_value"] > 0.05:
                st.success(
                    f"El VaR {backtest_method.lower()} es consistente con la frecuencia de pérdidas observadas en la muestra."
                )
            else:
                st.error(
                    f"El VaR {backtest_method.lower()} no es consistente con la frecuencia de pérdidas observadas. "
                    "Esto sugiere que el modelo puede estar subestimando o sobreestimando el riesgo."
                )

            st.info(
                "El test de Kupiec compara la proporción esperada de violaciones del VaR con la proporción observada. "
                "Es una forma de evaluar si el modelo de riesgo está calibrado de manera razonable."
            )
        else:
            st.warning("No se pudo ejecutar el test de Kupiec.")
    else:
        st.warning(f"No hay VaR {backtest_method.lower()} disponible para ejecutar el test de Kupiec.")
