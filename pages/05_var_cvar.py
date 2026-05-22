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
from src.ui_layout import configured_assets, configured_period, module_params, render_app_shell, render_portfolio_summary_card
from src.ui_style import apply_global_typography

ensure_project_dirs()
apply_global_typography()

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


def normalize_risk_table_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    normalized = df.copy()
    method_col = next(
        (col for col in ["metodo", "método", "Metodo", "Método", "method"] if col in normalized.columns),
        None,
    )
    if method_col is not None and method_col != "metodo":
        normalized = normalized.rename(columns={method_col: "metodo"})

    if "metodo" in normalized.columns:
        normalized["metodo"] = normalized["metodo"].replace(
            {
                "Paramétrico": "Parametrico",
                "Histórico": "Historico",
            }
        )

    return normalized


def table_for_var_plot(df: pd.DataFrame) -> pd.DataFrame:
    plot_df = df.copy()
    if "metodo" in plot_df.columns:
        plot_df["método"] = plot_df["metodo"].replace(
            {
                "Parametrico": "Paramétrico",
                "Historico": "Histórico",
            }
        )
    return plot_df


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

render_app_shell(
    "Módulo 5 - VaR y CVaR",
    "Evalua el riesgo extremo del portafolio mediante VaR y CVaR bajo distintos enfoques de estimacion.",
)
ASSETS = configured_assets(ASSETS)
horizonte, start_date, end_date = configured_period(default_end=DEFAULT_END_DATE)
render_portfolio_summary_card(ASSETS)

# ==============================
# Parámetros del módulo
# ==============================
with module_params():
    st.header("Parámetros de riesgo")

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
            st.error(f"Pesos inv-lidos: la suma debe ser 1.00 para continuar. Suma actual = {weights_sum:.6f}.")
            st.stop()

# ==============================
# Carga y preparacion de datos
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
        "Sin datos para estos tickers en el rango seleccionado; se excluyen del análisis: "
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

non_empty_tables = [
    normalize_risk_table_columns(alpha_table)
    for alpha_table in tables_by_alpha.values()
    if alpha_table is not None and not alpha_table.empty
]
table = pd.concat(non_empty_tables, ignore_index=True) if non_empty_tables else pd.DataFrame()
selected_table = normalize_risk_table_columns(tables_by_alpha.get(alpha, pd.DataFrame()))

if table.empty:
    st.error("No fue posible calcular VaR y CVaR con los datos disponibles.")
    st.stop()

if selected_table.empty:
    st.error("No fue posible calcular VaR y CVaR para el nivel de confianza seleccionado.")
    st.stop()

required_risk_columns = {"metodo", "VaR_diario", "CVaR_diario"}
if not required_risk_columns.issubset(selected_table.columns) or not required_risk_columns.issubset(table.columns):
    st.warning("La tabla de riesgo no incluye todas las columnas necesarias para mostrar VaR/CVaR.")
    st.stop()

# ==============================
# Resumen
# ==============================
st.markdown("### Resumen del módulo")
st.write(
    f"""
    Este módulo compara el **Value at Risk (VaR)** y el **Conditional Value at Risk (CVaR)** del portafolio
    equiponderado bajo enfoques **parametrico**, **historico** y **Monte Carlo** con **{n_sim:,} simulaciones**,
    usando la convencion de perdidas positivas para un nivel de confianza de **{int(alpha * 100)}%**.
    """
)

st.caption(f"Periodo analizado: {start_date} a {end_date}")

# ==============================
# Supuesto de pesos
# ==============================
st.markdown("### Supuesto de pesos")
if manual_weights_enabled:
    st.write("Se usa un portafolio con pesos manuales validados.")
else:
    st.write("Se usa un portafolio equiponderado, es decir, todos los activos tienen el mismo peso.")

# ==============================
# Filas por metodo
# ==============================
var_hist_row = selected_table.loc[selected_table["metodo"] == "Historico"]
var_param_row = selected_table.loc[selected_table["metodo"] == "Parametrico"]
var_mc_row = selected_table.loc[selected_table["metodo"] == "Monte Carlo"]

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
        "VaR parametrico diario",
        f"{var_p:.2%}" if var_p is not None else "N/D",
        delta=f"{int(alpha * 100)}% confianza",
        caption="Perdida umbral bajo supuesto de normalidad",
    )

with kpi_row_1[1]:
    kpi_card(
        "VaR historico diario",
        f"{var_h:.2%}" if var_h is not None else "N/D",
        delta=f"{int(alpha * 100)}% confianza",
        caption="Perdida umbral con distribucion empirica",
    )

with kpi_row_1[2]:
    kpi_card(
        "VaR Monte Carlo diario",
        f"{var_mc:.2%}" if var_mc is not None else "N/D",
        delta=f"{n_sim:,} simulaciones",
        caption="Perdida umbral mediante simulacion probabilistica",
    )

with kpi_row_2[0]:
    kpi_card(
        "CVaR parametrico diario",
        f"{cvar_p:.2%}" if cvar_p is not None else "N/D",
        delta="Cola extrema",
        delta_type="neg",
        caption="Perdida promedio mas alla del VaR parametrico",
    )

with kpi_row_2[1]:
    kpi_card(
        "CVaR historico diario",
        f"{cvar_h:.2%}" if cvar_h is not None else "N/D",
        delta="Cola extrema",
        delta_type="neg",
        caption="Perdida promedio en los peores casos observados",
    )

with kpi_row_2[2]:
    kpi_card(
        "CVaR Monte Carlo diario",
        f"{cvar_mc:.2%}" if cvar_mc is not None else "N/D",
        delta="Cola extrema",
        delta_type="neg",
        caption="Perdida promedio en escenarios simulados extremos",
    )

with st.expander("Interpretacion"):
    st.write(
        f"""
        - **VaR parametrico diario:** asume normalidad y marca el umbral de perdida diaria que no deberia superarse con **{int(alpha * 100)}%** de confianza.
        - **VaR historico diario:** usa percentiles empiricos de los rendimientos observados para estimar la perdida diaria umbral al **{int(alpha * 100)}%**.
        - **VaR Monte Carlo diario:** estima el percentil de perdida diaria a partir de **{n_sim:,} simulaciones**.
        - **CVaR parametrico diario:** estima la perdida diaria promedio condicionada a exceder el VaR parametrico.
        - **CVaR historico diario:** promedia las perdidas diarias mas extremas observadas en la cola historica.
        - **CVaR Monte Carlo diario:** promedia las perdidas diarias mas extremas dentro de los escenarios simulados.
        """
    )

# ==============================
# Grafico
# ==============================
st.markdown("### Distribucion y riesgo extremo")
section_intro(
    "Distribucion de rendimientos del portafolio",
    "El histograma muestra la forma empirica de los rendimientos y las lineas senalan los umbrales de VaR y CVaR por metodo.",
)

st.caption(
    "El histograma muestra la distribucion de rendimientos. Las lineas punteadas representan VaR y CVaR por metodo."
)
plot_table = table_for_var_plot(selected_table[selected_table["metodo"] != "Monte Carlo KDE"].copy())
fig_var = plot_var_distribution(portfolio_returns, plot_table)
fig_var.update_traces(marker_line_width=0.6, marker_line_color="rgba(15, 23, 42, 0.35)", selector=dict(type="histogram"))
line_styles = {
    "Parametrico": "#2563eb",
    "Paramétrico": "#2563eb",
    "Historico": "#16a34a",
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
    **Como leer este grafico**

    - El histograma resume como se distribuyen los rendimientos del portafolio.
    - Las lineas de **VaR** marcan perdidas umbral bajo distintos metodos.
    - Las lineas de **CVaR** muestran perdidas promedio mas severas en escenarios extremos.
    - Cuando el CVaR es mas alto que el VaR, las perdidas extremas pueden ser considerablemente mas intensas.
    """
)

with st.expander("Leyenda de lineas (VaR/CVaR)"):
    st.write(
        """
        - **Rendimientos (histograma):** distribucion empirica de rendimientos diarios del portafolio.
        - **VaR Parametrico:** umbral de perdida diaria estimado bajo normalidad.
        - **CVaR Parametrico:** perdida promedio mas alla del VaR parametrico.
        - **VaR Historico:** umbral de perdida calculado con rendimientos observados.
        - **CVaR Historico:** promedio de perdidas historicas en la cola extrema.
        - **VaR Monte Carlo:** umbral de perdida a partir de escenarios simulados.
        - **CVaR Monte Carlo:** promedio de perdidas extremas dentro de los escenarios simulados.
        """
    )

# ==============================
# Tabla
# ==============================
st.markdown("### Comparacion VaR / CVaR")
compact_table = table[["confianza", "metodo", "VaR_diario", "CVaR_diario"]].copy()
compact_table = compact_table.rename(
    columns={
        "confianza": "Confianza",
        "metodo": "Metodo",
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
        "metodo": "Metodo",
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

with st.expander("Interpretacion de la tabla comparativa"):
    st.write(
        """
        - **Monte Carlo KDE:** usa una distribucion empirica suavizada de los rendimientos historicos del portafolio.
        - A diferencia del enfoque Monte Carlo normal, no asume normalidad estricta en los rendimientos del portafolio.
        - Sirve para comparar la sensibilidad del VaR/CVaR frente al supuesto normal.
        - No debe interpretarse como el metodo real o necesariamente superior, sino como una aproximacion alternativa.
        """
    )

st.info(
    f"""
    Para el nivel seleccionado de **{int(alpha * 100)}%**, el VaR historico diario indica una perdida umbral de
    **{fmt_pct_value(var_h)}** y el CVaR historico diario estima una perdida promedio de cola de
    **{fmt_pct_value(cvar_h)}**. Si el CVaR supera claramente al VaR, los escenarios extremos son mas severos
    que el umbral inicial. La comparacion entre metodos ayuda a ver cuanto dependen los resultados del supuesto
    normal, de la historia observada o de simulaciones Monte Carlo.
    """
)

with st.expander("Backtesting (opcional) - Test de Kupiec"):
    section_intro(
        "Validacion del VaR estimado",
        "Este bloque contrasta si la frecuencia observada de violaciones del VaR es coherente con la esperada bajo el nivel de confianza seleccionado.",
    )
    st.write(
        r"""
        Una **violacion** ocurre cuando la perdida observada supera el VaR estimado. La tasa esperada de
        violaciones es \(1 - \alpha\): por ejemplo, 5% para 95% de confianza y 1% para 99%.
        El test de Kupiec evalua si la frecuencia observada de violaciones es compatible con esa tasa esperada.
        Si el **p-value** es mayor que 0.05, no se rechaza la calibracion del VaR; si es menor o igual, hay evidencia
        de que el VaR no esta bien calibrado.
        """
    )

    backtest_method = st.selectbox(
        "Metodo para backtesting",
        ["Historico", "Parametrico", "Monte Carlo"],
        index=0,
    )
    method_row = selected_table.loc[selected_table["metodo"] == backtest_method]
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
                    caption="Numero de veces que la perdida supero el VaR",
                )

            with col2:
                kpi_card(
                    "Observadas (%)",
                    f"{kupiec['observed_fail_rate'] * 100:.2f}%",
                    caption="Frecuencia empirica de violaciones",
                )

            with col3:
                kpi_card(
                    "Esperadas (%)",
                    f"{kupiec['expected_fail_rate'] * 100:.2f}%",
                    caption="Frecuencia teorica bajo el modelo",
                )

            st.write(f"**p-value:** {kupiec['p_value']:.4f}")
            st.write(f"**Conclusion:** {kupiec['conclusion']}")

            if kupiec["p_value"] > 0.05:
                st.success(
                    f"El VaR {backtest_method.lower()} es consistente con la frecuencia de perdidas observadas en la muestra."
                )
            else:
                st.error(
                    f"El VaR {backtest_method.lower()} no es consistente con la frecuencia de perdidas observadas. "
                    "Esto sugiere que el modelo puede estar subestimando o sobreestimando el riesgo."
                )

            st.info(
                "El test de Kupiec compara la proporcion esperada de violaciones del VaR con la proporcion observada. "
                "Es una forma de evaluar si el modelo de riesgo esta calibrado de manera razonable."
            )
        else:
            st.warning("No se pudo ejecutar el test de Kupiec.")
    else:
        st.warning(f"No hay VaR {backtest_method.lower()} disponible para ejecutar el test de Kupiec.")

