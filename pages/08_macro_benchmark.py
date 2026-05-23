import pandas as pd
import streamlit as st

from src.api.backend_client import BackendAPIError, friendly_error_message
from src.api.macro import macro_snapshot
from src.benchmark import benchmark_summary
from src.config import (
    ASSETS,
    DEFAULT_END_DATE,
    DEFAULT_START_DATE,
    GLOBAL_BENCHMARK,
    ensure_project_dirs,
)
from src.download import data_error_message
from src.plots import plot_benchmark_base100
from src.services.market_data_client import MarketDataClient
from src.services.portfolio_optimizer import PortfolioOptimizer
from src.ui_components import (
    kpi_card,
    module_header,
    render_chart_explanation,
    render_explanation_expander,
    render_insight,
    render_kpi_help,
    render_section,
    render_table,
)
from src.ui_layout import configured_assets, configured_period, module_params, render_app_shell, render_portfolio_summary_card
from src.ui_style import apply_global_typography

ensure_project_dirs()
apply_global_typography()


def fmt_pct(value, digits=2):
    numeric_value = pd.to_numeric(value, errors="coerce")
    return f"{numeric_value:.{digits}%}" if pd.notna(numeric_value) else "N/D"


def fmt_num(value, digits=4):
    numeric_value = pd.to_numeric(value, errors="coerce")
    return f"{numeric_value:.{digits}f}" if pd.notna(numeric_value) else "N/D"


def first_valid_macro_value(data: dict, *keys: str):
    for key in keys:
        value = pd.to_numeric(data.get(key), errors="coerce")
        if pd.notna(value):
            return float(value)
    return None


def normalize_metric_table_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    normalized = df.copy()
    metric_col = next(
        (col for col in ["metrica", "métrica", "Metrica", "Métrica", "metric", "indicador", "Indicador"] if col in normalized.columns),
        None,
    )
    value_col = next(
        (col for col in ["valor", "Valor", "value", "Value"] if col in normalized.columns),
        None,
    )
    if metric_col is not None and metric_col != "metrica":
        normalized = normalized.rename(columns={metric_col: "metrica"})
    if value_col is not None and value_col != "valor":
        normalized = normalized.rename(columns={value_col: "valor"})
    return normalized


def metric_value(df: pd.DataFrame, metric_name: str):
    normalized = normalize_metric_table_columns(df)
    if normalized.empty or not {"metrica", "valor"}.issubset(normalized.columns):
        return None

    matches = normalized.loc[normalized["metrica"] == metric_name, "valor"]
    if matches.empty:
        return None

    value = pd.to_numeric(matches.iloc[0], errors="coerce")
    return float(value) if pd.notna(value) else None


def macro_display(value, formatter):
    return formatter(value) if value is not None else "No disponible"


def render_conclusion_box(message: str):
    st.markdown(
        f"""
        <div style="
            background: linear-gradient(180deg, #e0f2fe 0%, #f0f9ff 100%);
            border: 1px solid rgba(14, 116, 144, 0.18);
            border-radius: 18px;
            padding: 16px 18px;
            box-shadow: 0 4px 14px rgba(14, 116, 144, 0.08);
            color: #0f172a;
            line-height: 1.5;
            margin: 0.25rem 0 0.75rem 0;
        ">
            {message}
        </div>
        """,
        unsafe_allow_html=True,
    )


render_app_shell(
    "Módulo 8 - Contexto macro y benchmark",
    "Contexto macro y comparación del portafolio óptimo frente al benchmark global.",
)
ASSETS = configured_assets(ASSETS)
horizonte, start_date, end_date = configured_period(DEFAULT_START_DATE, DEFAULT_END_DATE)
render_portfolio_summary_card(ASSETS)
module_header(
    "Contexto macro y benchmark",
    "Indicadores macroeconómicos del período y comparación del portafolio óptimo de máximo Sharpe frente al benchmark global.",
    badge="Alpha de Jensen · TE · IR",
)

with module_params():
    st.header("Parámetros")
    mostrar_detalle = st.checkbox("Mostrar detalle adicional", value=False)


st.caption("Portafolio óptimo usado: máximo Sharpe.")
st.caption(f"Periodo analizado: {start_date} a {end_date}")

tickers = [meta["ticker"] for meta in ASSETS.values()] + [GLOBAL_BENCHMARK]
market_client = MarketDataClient()
optimizer = PortfolioOptimizer()

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

returns = optimizer.prepare_returns(bundle["returns"])
if returns.empty or GLOBAL_BENCHMARK not in returns.columns:
    st.error(data_error_message("No fue posible construir benchmark global."))
    st.stop()

asset_columns = [meta["ticker"] for meta in ASSETS.values() if meta["ticker"] in returns.columns]
asset_returns = returns[asset_columns].copy()
if asset_returns.empty or asset_returns.shape[1] < 2:
    st.error(data_error_message("No hay suficientes activos con retornos válidos para construir el portafolio óptimo."))
    st.stop()

benchmark_returns = returns[GLOBAL_BENCHMARK]

macro = macro_snapshot()
rf_annual = (
    macro["risk_free_rate_pct"] / 100
    if macro["risk_free_rate_pct"] == macro["risk_free_rate_pct"]
    else 0.03
)

sim_df = optimizer.simulate(asset_returns, rf_annual=rf_annual, n_portfolios=10000)
_, max_sharpe = optimizer.optimal_portfolios(sim_df)

if max_sharpe is None or getattr(max_sharpe, "empty", False):
    st.error("No fue posible identificar el portafolio óptimo de máximo Sharpe.")
    st.stop()

max_sharpe_weights = []
for ticker in asset_returns.columns:
    weight_value = pd.to_numeric(max_sharpe.get(f"w_{ticker}"), errors="coerce")
    max_sharpe_weights.append(0.0 if pd.isna(weight_value) else float(weight_value))

portfolio_returns = asset_returns.mul(max_sharpe_weights, axis=1).sum(axis=1)

summary_df, extras_df, cum_port, cum_bench = benchmark_summary(
    portfolio_returns=portfolio_returns,
    benchmark_returns=benchmark_returns,
    rf_annual=rf_annual,
)
extras_df = normalize_metric_table_columns(extras_df)

if summary_df.empty:
    st.error("No fue posible construir la comparación entre portafolio óptimo y benchmark.")
    st.stop()

st.markdown("### Indicadores macroeconomicos")
render_section(
    "Contexto macro",
    "Rf, inflacion y tasa de cambio obtenidas via API para contextualizar el periodo.",
)

rf_pct = first_valid_macro_value(macro, "risk_free_rate_pct")
inflation_yoy = first_valid_macro_value(macro, "inflation_yoy")
usdcop_market = first_valid_macro_value(macro, "usdcop_market", "cop_per_usd")

macro_cols = st.columns(3)
with macro_cols[0]:
    kpi_card(
        "Tasa libre de riesgo",
        macro_display(rf_pct, lambda value: f"{value:.2f}%"),
        caption="Usada en Sharpe, CAPM y benchmark",
    )
with macro_cols[1]:
    kpi_card(
        "Inflacion interanual",
        macro_display(inflation_yoy, fmt_pct),
        caption="Referencia macro del periodo",
    )
with macro_cols[2]:
    kpi_card(
        "Tasa de cambio USD/COP",
        macro_display(usdcop_market, lambda value: f"{value:.2f}"),
        caption="Dato de mercado mas reciente disponible",
    )
missing_macro_fields = []
if rf_pct is None:
    missing_macro_fields.append("tasa libre de riesgo")
if inflation_yoy is None:
    missing_macro_fields.append("inflación interanual")
if usdcop_market is None:
    missing_macro_fields.append("USD/COP")
if missing_macro_fields:
    st.info(
        "No hay dato macro disponible para "
        + ", ".join(missing_macro_fields)
        + ". La lectura continúa con la información disponible."
    )
render_explanation_expander(
    "Como interpretar los indicadores macroeconomicos",
    [
        "Muestra la tasa libre de riesgo, la inflacion interanual y la tasa USD/COP usadas para contextualizar el periodo analizado.",
        f"En el resultado actual, la referencia libre de riesgo es {macro_display(rf_pct, lambda value: f'{value:.2f}%')}, la inflacion observada es {macro_display(inflation_yoy, fmt_pct)} y el USD/COP reportado es {macro_display(usdcop_market, lambda value: f'{value:.2f}')}.",
        "Financieramente, una Rf mas alta eleva el retorno exigido, la inflacion condiciona el retorno real y el tipo de cambio ayuda a leer presiones externas o sensibilidad cambiaria.",
    ],
)

st.markdown("### Portafolio óptimo vs benchmark")
render_section(
    "Comparacion principal",
    "Desempeño acumulado en base 100 del portafolio óptimo de máximo Sharpe contra el benchmark.",
)
benchmark_fig = plot_benchmark_base100(cum_port, cum_bench)
if benchmark_fig.data:
    benchmark_fig.data[0].name = "Portafolio óptimo"
    if hasattr(benchmark_fig.data[0], "hovertemplate") and benchmark_fig.data[0].hovertemplate:
        benchmark_fig.data[0].hovertemplate = benchmark_fig.data[0].hovertemplate.replace(
            "Portafolio", "Portafolio óptimo"
        )
if len(benchmark_fig.data) > 1:
    benchmark_fig.data[1].name = "Benchmark"
    if hasattr(benchmark_fig.data[1], "hovertemplate") and benchmark_fig.data[1].hovertemplate:
        benchmark_fig.data[1].hovertemplate = benchmark_fig.data[1].hovertemplate.replace(
            "Benchmark global", "Benchmark"
        ).replace("Benchmark", "Benchmark")
st.plotly_chart(benchmark_fig, width="stretch")

try:
    ret_port = float(summary_df.loc[summary_df["serie"] == "Portafolio", "ret_acumulado"].iloc[0])
except Exception:
    ret_port = None

try:
    ret_bench = float(summary_df.loc[summary_df["serie"] == "Benchmark", "ret_acumulado"].iloc[0])
except Exception:
    ret_bench = None

chart_ret_comparison_text = "no hay un dato suficiente para comparar el cierre relativo"
if ret_port is not None and ret_bench is not None:
    if ret_port > ret_bench:
        chart_ret_comparison_text = (
            f"el portafolio termina por encima del benchmark, con {fmt_pct(ret_port)} frente a {fmt_pct(ret_bench)}"
        )
    elif ret_port < ret_bench:
        chart_ret_comparison_text = (
            f"el portafolio termina por debajo del benchmark, con {fmt_pct(ret_port)} frente a {fmt_pct(ret_bench)}"
        )
    else:
        chart_ret_comparison_text = (
            f"portafolio y benchmark terminan con el mismo retorno acumulado, {fmt_pct(ret_port)}"
        )

render_chart_explanation(
    "Como leer el grafico base 100",
    "Compara portafolio óptimo y benchmark en una base común de 100 para seguir su trayectoria acumulada durante el periodo.",
    [
        f"En el resultado actual, {chart_ret_comparison_text}.",
        "Cuando la curva del portafolio se mantiene por encima, hay liderazgo relativo; cuando converge o cae por debajo, la ventaja se reduce o se revierte.",
        "Financieramente, el grafico muestra la trayectoria del retorno relativo, pero la lectura debe confirmarse con alpha, tracking error, information ratio y la tabla comparativa.",
    ],
)

alpha_jensen = metric_value(extras_df, "Alpha de Jensen")
tracking_error = metric_value(extras_df, "Tracking Error")
information_ratio = metric_value(extras_df, "Information Ratio")
missing_benchmark_metrics = [
    name
    for name, value in [
        ("Alpha de Jensen", alpha_jensen),
        ("Tracking Error", tracking_error),
        ("Information Ratio", information_ratio),
    ]
    if value is None
]
if missing_benchmark_metrics:
    st.info(
        "No hay dato suficiente para "
        + ", ".join(missing_benchmark_metrics)
        + ". Revisa que el portafolio y el benchmark tengan retornos alineados en el periodo."
    )

ret_delta = None
ret_delta_type = "neu"
if ret_port is not None and ret_bench is not None:
    diff_ret = ret_port - ret_bench
    ret_delta = f"Diferencia: {diff_ret:.2%}"
    ret_delta_type = "pos" if diff_ret > 0 else "neg" if diff_ret < 0 else "neu"

alpha_delta = None
alpha_delta_type = "neu"
if alpha_jensen is not None:
    if alpha_jensen > 0:
        alpha_delta = "Alpha positivo"
        alpha_delta_type = "pos"
    elif alpha_jensen < 0:
        alpha_delta = "Alpha negativo"
        alpha_delta_type = "neg"

te_delta = None
te_delta_type = "neu"
if tracking_error is not None:
    te_delta = "Riesgo activo"
    te_delta_type = "neu"

ir_delta = None
ir_delta_type = "neu"
if information_ratio is not None:
    if information_ratio > 0:
        ir_delta = "Activo positivo"
        ir_delta_type = "pos"
    elif information_ratio < 0:
        ir_delta = "Activo negativo"
        ir_delta_type = "neg"

st.markdown("### KPIs de benchmark")
render_section(
    "Desempeño relativo",
    "Métricas principales del portafolio óptimo frente al benchmark global.",
)

metric_cols = st.columns(4)
with metric_cols[0]:
    kpi_card(
        "Retorno acumulado portafolio",
        fmt_pct(ret_port),
        delta=ret_delta,
        delta_type=ret_delta_type,
        caption="Portafolio óptimo de máximo Sharpe",
    )
with metric_cols[1]:
    kpi_card(
        "Alpha de Jensen",
        fmt_num(alpha_jensen),
        delta=alpha_delta,
        delta_type=alpha_delta_type,
        caption="Exceso de desempeño ajustado por riesgo",
    )
with metric_cols[2]:
    kpi_card(
        "Tracking Error",
        fmt_num(tracking_error),
        delta=te_delta,
        delta_type=te_delta_type,
        caption="Desviación frente al benchmark",
    )
with metric_cols[3]:
    kpi_card(
        "Information Ratio",
        fmt_num(information_ratio),
        delta=ir_delta,
        delta_type=ir_delta_type,
        caption="Retorno activo por unidad de riesgo activo",
    )
render_kpi_help(
    "Como interpretar los KPIs de benchmark",
    [
        "Muestra el retorno acumulado del portafolio y tres métricas de lectura relativa frente al benchmark.",
        f"En el resultado actual, el portafolio registra {fmt_pct(ret_port)}, el alpha de Jensen es {fmt_num(alpha_jensen)}, el tracking error es {fmt_num(tracking_error)} y el information ratio es {fmt_num(information_ratio)}.",
        "Financieramente, alpha evalua si hubo valor agregado ajustado por beta, tracking error mide cuanto se aparta la estrategia del benchmark e information ratio juzga si ese riesgo activo fue bien compensado.",
    ],
)

st.markdown("### Tabla de desempeño")
performance_table = summary_df.rename(
    columns={
        "serie": "Serie",
        "ret_acumulado": "Retorno acumulado",
        "ret_anualizado": "Retorno anualizado",
        "vol_anualizada": "Volatilidad",
        "sharpe": "Sharpe",
        "max_drawdown": "Max. drawdown",
    }
).copy()

for col in ["Retorno acumulado", "Retorno anualizado", "Volatilidad", "Max. drawdown"]:
    if col in performance_table.columns:
        performance_table[col] = performance_table[col].map(
            lambda value: f"{value:.2%}" if pd.notna(value) else "N/D"
        )
if "Sharpe" in performance_table.columns:
    performance_table["Sharpe"] = performance_table["Sharpe"].map(
        lambda value: f"{value:.3f}" if pd.notna(value) else "N/D"
    )

port_row = summary_df.loc[summary_df["serie"] == "Portafolio"]
bench_row = summary_df.loc[summary_df["serie"] == "Benchmark"]

port_vol = float(port_row["vol_anualizada"].iloc[0]) if not port_row.empty else None
bench_vol = float(bench_row["vol_anualizada"].iloc[0]) if not bench_row.empty else None
port_sharpe = float(port_row["sharpe"].iloc[0]) if not port_row.empty else None
bench_sharpe = float(bench_row["sharpe"].iloc[0]) if not bench_row.empty else None
port_drawdown = float(port_row["max_drawdown"].iloc[0]) if not port_row.empty else None
bench_drawdown = float(bench_row["max_drawdown"].iloc[0]) if not bench_row.empty else None

table_rows = performance_table.to_dict("records")
port_table_row = next((row for row in table_rows if row["Serie"] == "Portafolio"), None)
bench_table_row = next((row for row in table_rows if row["Serie"] == "Benchmark"), None)

ret_comparison_text = "no hay un dato suficiente para comparar el cierre relativo"
if ret_port is not None and ret_bench is not None:
    if ret_port > ret_bench:
        ret_comparison_text = (
            f"el portafolio termina por encima del benchmark, con {fmt_pct(ret_port)} frente a {fmt_pct(ret_bench)}"
        )
    elif ret_port < ret_bench:
        ret_comparison_text = (
            f"el portafolio termina por debajo del benchmark, con {fmt_pct(ret_port)} frente a {fmt_pct(ret_bench)}"
        )
    else:
        ret_comparison_text = f"portafolio y benchmark terminan con el mismo retorno acumulado, {fmt_pct(ret_port)}"

alpha_text = f"Alpha de Jensen en {fmt_num(alpha_jensen)}: no hay dato suficiente para una lectura ajustada por riesgo."
if alpha_jensen is not None:
    if alpha_jensen > 0:
        alpha_text = (
            f"Alpha de Jensen en {fmt_num(alpha_jensen)}: indica desempeño superior ajustado por riesgo frente al benchmark."
        )
    elif alpha_jensen < 0:
        alpha_text = (
            f"Alpha de Jensen en {fmt_num(alpha_jensen)}: indica desempeño inferior ajustado por riesgo frente al benchmark."
        )
    else:
        alpha_text = "Alpha de Jensen en 0.0000: no hay evidencia de creacion de valor ajustada por riesgo."

tracking_error_text = f"Tracking Error en {fmt_num(tracking_error)}: no hay dato suficiente para medir separacion frente al benchmark."
if tracking_error is not None:
    tracking_error_text = (
        f"Tracking Error en {fmt_num(tracking_error)}: refleja el grado de separacion frente al benchmark y no es bueno o malo por si solo."
    )

information_ratio_text = (
    f"Information Ratio en {fmt_num(information_ratio)}: no hay dato suficiente para evaluar la eficiencia del retorno activo."
)
if information_ratio is not None:
    if information_ratio > 0:
        information_ratio_text = (
            f"Information Ratio en {fmt_num(information_ratio)}: hubo retorno activo positivo por unidad de riesgo activo."
        )
    elif information_ratio < 0:
        information_ratio_text = (
            f"Information Ratio en {fmt_num(information_ratio)}: hubo retorno activo negativo frente al riesgo activo asumido."
        )
    else:
        information_ratio_text = "Information Ratio en 0.0000: la gestion activa no agrego retorno por unidad de riesgo activo."

vol_text = "no hay lectura concluyente de volatilidad relativa."
if port_vol is not None and bench_vol is not None:
    if port_vol > bench_vol:
        vol_text = f"el portafolio asume mayor volatilidad que el benchmark ({fmt_pct(port_vol)} vs {fmt_pct(bench_vol)})"
    elif port_vol < bench_vol:
        vol_text = f"el portafolio asume menor volatilidad que el benchmark ({fmt_pct(port_vol)} vs {fmt_pct(bench_vol)})"
    else:
        vol_text = f"portafolio y benchmark muestran la misma volatilidad ({fmt_pct(port_vol)})"

sharpe_text = "Sharpe: no hay dato suficiente para comparar eficiencia riesgo-retorno."
if port_sharpe is not None and bench_sharpe is not None:
    if port_sharpe > bench_sharpe:
        sharpe_text = (
            f"Sharpe mide retorno por unidad de volatilidad; la tabla favorece al portafolio ({fmt_num(port_sharpe, 3)} vs {fmt_num(bench_sharpe, 3)})."
        )
    elif port_sharpe < bench_sharpe:
        sharpe_text = (
            f"Sharpe mide retorno por unidad de volatilidad; la tabla favorece al benchmark ({fmt_num(bench_sharpe, 3)} vs {fmt_num(port_sharpe, 3)})."
        )
    else:
        sharpe_text = f"Sharpe mide retorno por unidad de volatilidad; ambas series muestran la misma eficiencia ({fmt_num(port_sharpe, 3)})."

drawdown_text = "Maximo drawdown: no hay dato suficiente para comparar severidad de caidas."
if port_drawdown is not None and bench_drawdown is not None:
    if port_drawdown > bench_drawdown:
        drawdown_text = (
            f"El drawdown del portafolio fue menos profundo que el del benchmark ({fmt_pct(port_drawdown)} vs {fmt_pct(bench_drawdown)})."
        )
    elif port_drawdown < bench_drawdown:
        drawdown_text = (
            f"El drawdown del portafolio fue mas profundo que el del benchmark ({fmt_pct(port_drawdown)} vs {fmt_pct(bench_drawdown)})."
        )
    else:
        drawdown_text = f"Portafolio y benchmark registran el mismo drawdown maximo ({fmt_pct(port_drawdown)})."

risk_compensation_text = "No hay evidencia suficiente para juzgar si el retorno compenso el riesgo observado."
if ret_port is not None and ret_bench is not None and port_sharpe is not None and bench_sharpe is not None:
    if ret_port > ret_bench and port_sharpe >= bench_sharpe:
        risk_compensation_text = "La ventaja en retorno tambien se sostiene en Sharpe, por lo que el riesgo observado parece haber sido compensado."
    elif ret_port > ret_bench:
        risk_compensation_text = "El portafolio gana en retorno acumulado, pero la eficiencia ajustada por volatilidad no mejora con la misma claridad."
    elif ret_port <= ret_bench and port_sharpe > bench_sharpe:
        risk_compensation_text = "Aunque el retorno acumulado no supera al benchmark, el Sharpe sugiere una administracion del riesgo relativamente mas eficiente."
    else:
        risk_compensation_text = "El riesgo observado no se traduce en una mejora clara de retorno ajustado por riesgo frente al benchmark."

table_header_text = "La tabla compara portafolio y benchmark."
if port_table_row and bench_table_row:
    table_header_text = (
        f"La tabla compara portafolio y benchmark; muestra {port_table_row['Retorno acumulado']} vs {bench_table_row['Retorno acumulado']} "
        f"en retorno acumulado y {port_table_row['Volatilidad']} vs {bench_table_row['Volatilidad']} en volatilidad."
    )

render_table(performance_table, hide_index=True, width="stretch")
render_explanation_expander(
    "Cómo interpretar la tabla de desempeño",
    [
        "Muestra la comparacion consolidada entre portafolio y benchmark en retorno, riesgo, eficiencia y drawdown.",
        f"En el resultado actual, {table_header_text.lower()}",
        f"Financieramente, la lectura conjunta parte de {ret_comparison_text}, se matiza con {vol_text}, y se valida con eficiencia y perdidas extremas: {sharpe_text} {drawdown_text} {risk_compensation_text}",
    ],
)

st.markdown("### Conclusion")
port_beats_benchmark = ret_port is not None and ret_bench is not None and ret_port > ret_bench
alpha_positive = alpha_jensen is not None and pd.notna(alpha_jensen) and alpha_jensen > 0

if port_beats_benchmark and alpha_positive:
    render_conclusion_box(
        "El portafolio óptimo supera al benchmark y también muestra alpha positivo; la ventaja se observa tanto en retorno acumulado como en desempeño ajustado por riesgo.",
    )
elif port_beats_benchmark:
    render_conclusion_box(
        "El portafolio supera al benchmark en retorno acumulado, pero el alpha no confirma una ventaja ajustada por riesgo.",
    )
elif alpha_positive:
    render_conclusion_box(
        "Aunque no supera al benchmark en retorno acumulado, el alpha positivo sugiere una mejora ajustada por riesgo.",
    )
else:
    render_conclusion_box(
        "El portafolio no supera al benchmark y el alpha no evidencia creacion de valor ajustada por riesgo.",
    )
render_explanation_expander(
    "Como interpretar la conclusion",
    [
        "Muestra una conclusion ejecutiva sobre si el portafolio gana o pierde frente al benchmark en retorno y calidad del resultado.",
        f"En el resultado actual, el mensaje se apoya en retorno del portafolio {fmt_pct(ret_port)}, retorno del benchmark {fmt_pct(ret_bench)} y alpha de Jensen {fmt_num(alpha_jensen)}.",
        "Financieramente, el insight sintetiza la evidencia principal, pero su solidez depende de que alpha, tracking error, information ratio y la tabla respalden la misma lectura.",
    ],
)

if mostrar_detalle:
    weights_df = pd.DataFrame(
        {
            "Activo": asset_returns.columns,
            "Peso": max_sharpe_weights,
        }
    )
    weights_df["Participacion"] = weights_df["Peso"].map(lambda x: f"{x:.2%}")
    weights_df = weights_df.sort_values("Peso", ascending=False).reset_index(drop=True)
    with st.expander("Detalle adicional", expanded=False):
        st.markdown(
            "- Muestra la composición del portafolio óptimo y métricas complementarias del contraste con el benchmark.\n"
            "- Usa esta sección para revisar concentración por activo y ampliar el diagnóstico sin recargar la vista principal."
        )
        st.markdown("**Pesos del portafolio óptimo**")
        render_table(weights_df[["Activo", "Participacion"]], hide_index=True, width="stretch")

        st.markdown("**Métricas adicionales**")
        extras_display_df = extras_df.rename(columns={"metrica": "Métrica", "valor": "Valor"})
        render_table(extras_display_df, hide_index=True, width="stretch")

