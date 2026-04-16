import streamlit as st

from src.config import (
    ASSETS,
    GLOBAL_BENCHMARK,
    DEFAULT_START_DATE,
    DEFAULT_END_DATE,
    ensure_project_dirs,
)
from src.download import load_market_bundle
from src.preprocess import equal_weight_portfolio
from src.api.macro import macro_snapshot
from src.benchmark import benchmark_summary
from src.plots import plot_benchmark_base100


ensure_project_dirs()

st.title("Módulo 8 - Contexto macro y benchmark")
st.caption(
    "Compara el desempeño del portafolio frente a un benchmark global y contextualiza los resultados con variables macroeconómicas."
)

with st.sidebar:
    st.header("Parámetros")
    start_date = st.date_input("Fecha inicial", value=DEFAULT_START_DATE, key="bm_start")
    end_date = st.date_input("Fecha final", value=DEFAULT_END_DATE, key="bm_end")

    st.divider()
    modo = st.radio(
        "Modo de visualización",
        ["General", "Estadístico"],
        index=0,
    )

    mostrar_tablas = st.checkbox("Mostrar tablas completas", value=False)

    mostrar_interpretacion_tecnica = False
    if modo == "Estadístico":
        mostrar_interpretacion_tecnica = st.checkbox(
            "Mostrar interpretación técnica",
            value=False,
        )

st.markdown("### Resumen del módulo")
if modo == "General":
    st.write(
        """
        Este módulo permite comparar si el portafolio tuvo un comportamiento mejor, similar o peor
        que su índice de referencia. Además, muestra algunas variables macroeconómicas que ayudan
        a entender el entorno financiero en el periodo analizado.
        """
    )
else:
    st.write(
        """
        Este módulo evalúa el desempeño relativo del portafolio frente a un benchmark mediante métricas
        como Alpha de Jensen, Tracking Error, Information Ratio y máximo drawdown, incorporando además
        variables macroeconómicas relevantes para contextualizar el análisis.
        """
    )

tickers = [meta["ticker"] for meta in ASSETS.values()] + [GLOBAL_BENCHMARK]
bundle = load_market_bundle(tickers=tickers, start=str(start_date), end=str(end_date))
returns = bundle["returns"].dropna()

if returns.empty or GLOBAL_BENCHMARK not in returns.columns:
    st.error("No fue posible construir benchmark global.")
    st.stop()

portfolio_returns = equal_weight_portfolio(
    returns[[c for c in returns.columns if c != GLOBAL_BENCHMARK]]
)
benchmark_returns = returns[GLOBAL_BENCHMARK]

macro = macro_snapshot()
rf_annual = (
    macro["risk_free_rate_pct"] / 100
    if macro["risk_free_rate_pct"] == macro["risk_free_rate_pct"]
    else 0.03
)

if "source" in macro and macro["source"]:
    st.caption(f"Fuente macro: {macro['source']}")

if "last_updated" in macro and macro["last_updated"]:
    st.caption(f"Última actualización: {macro['last_updated']}")

with st.expander("Ver estado de carga de variables macro"):
    if macro["inflation_yoy"] != macro["inflation_yoy"]:
        st.warning("No se pudo obtener inflación desde API. Usando fallback o valor no disponible.")

    if macro["usdcop_market"] != macro["usdcop_market"]:
        st.warning("No se pudo obtener USD/COP spot desde API. Usando fallback o valor no disponible.")

    if macro["cop_per_usd"] != macro["cop_per_usd"]:
        st.warning("No se pudo obtener USD/COP promedio anual desde API. Usando fallback o valor no disponible.")

st.markdown("### Indicadores macroeconómicos")
col1, col2, col3, col4 = st.columns(4)
col1.metric(
    "Tasa libre de riesgo (%)",
    f"{macro['risk_free_rate_pct']:.2f}"
    if macro["risk_free_rate_pct"] == macro["risk_free_rate_pct"]
    else "N/D",
)
col2.metric(
    "Inflación interanual",
    f"{macro['inflation_yoy']:.2%}"
    if macro["inflation_yoy"] == macro["inflation_yoy"]
    else "N/D",
)
col3.metric(
    "USD/COP (spot)",
    f"{macro['usdcop_market']:.2f}"
    if macro["usdcop_market"] == macro["usdcop_market"]
    else "N/D",
)
col4.metric(
    "USD/COP (promedio anual)",
    f"{macro['cop_per_usd']:.2f}"
    if macro["cop_per_usd"] == macro["cop_per_usd"]
    else "N/D",
)

summary_df, extras_df, cum_port, cum_bench = benchmark_summary(
    portfolio_returns=portfolio_returns,
    benchmark_returns=benchmark_returns,
    rf_annual=rf_annual,
)

st.markdown("### Comparación visual")
st.plotly_chart(plot_benchmark_base100(cum_port, cum_bench), width="stretch")

if modo == "General":
    st.info(
        """
        **Cómo leer este gráfico**

        - Si la línea del portafolio termina por encima del benchmark, hubo mejor desempeño acumulado.
        - Si ambas líneas se mueven muy parecido, el portafolio siguió de cerca al índice de referencia.
        - Caídas pronunciadas indican periodos de pérdida acumulada y mayor presión de riesgo.
        """
    )
else:
    with st.expander("Ver interpretación técnica del gráfico"):
        st.write(
            """
            El gráfico base 100 permite comparar trayectorias acumuladas normalizadas del portafolio
            y del benchmark. La separación entre ambas curvas refleja desempeño relativo, mientras que
            la amplitud de las caídas ayuda a identificar episodios de drawdown y sensibilidad a choques
            de mercado.
            """
        )

st.markdown("### KPIs de desempeño relativo")

try:
    ret_port = float(summary_df.loc[summary_df["serie"] == "Portafolio", "ret_acumulado"].iloc[0])
except Exception:
    ret_port = None

try:
    ret_bench = float(summary_df.loc[summary_df["serie"] == "Benchmark", "ret_acumulado"].iloc[0])
except Exception:
    ret_bench = None

try:
    alpha_jensen = float(extras_df.loc[extras_df["métrica"] == "Alpha de Jensen", "valor"].iloc[0])
except Exception:
    alpha_jensen = None

try:
    tracking_error = float(extras_df.loc[extras_df["métrica"] == "Tracking Error", "valor"].iloc[0])
except Exception:
    tracking_error = None

c1, c2, c3, c4 = st.columns(4)
c1.metric("Retorno acumulado portafolio", f"{ret_port:.2%}" if ret_port is not None else "N/D")
c2.metric("Retorno acumulado benchmark", f"{ret_bench:.2%}" if ret_bench is not None else "N/D")
c3.metric("Alpha de Jensen", f"{alpha_jensen:.4f}" if alpha_jensen is not None else "N/D")
c4.metric("Tracking Error", f"{tracking_error:.4f}" if tracking_error is not None else "N/D")

st.markdown("### Tablas de resultados")
if mostrar_tablas:
    st.subheader("Desempeño: portafolio vs benchmark")
    st.dataframe(summary_df, width="stretch")

    st.subheader("Métricas adicionales")
    st.dataframe(extras_df, width="stretch")

if modo == "General":
    st.markdown("### Interpretación")
    st.success(
        """
        **Lectura sencilla de resultados**

        - El benchmark sirve como punto de comparación para saber si el portafolio realmente agregó valor.
        - Un alpha positivo sugiere mejor desempeño que el esperado según su riesgo de mercado.
        - Un tracking error alto indica que el portafolio se aleja bastante del benchmark.
        - Un drawdown alto indica que en algún momento hubo una caída acumulada fuerte.
        """
    )
else:
    st.markdown("### Interpretación técnica")
    if mostrar_interpretacion_tecnica:
        st.info(
            """
            **Interpretación del benchmark y del contexto macro**

            - El gráfico base 100 permite comparar visualmente la trayectoria acumulada del portafolio frente al benchmark.
            - Un **Alpha de Jensen** positivo sugiere que el portafolio obtuvo un desempeño superior al explicado por su nivel de riesgo sistemático.
            - Un **Tracking Error** alto indica mayor desviación frente al benchmark, mientras que uno bajo sugiere un comportamiento más cercano al índice de referencia.
            - El **Information Ratio** resume cuánto retorno activo genera el portafolio por unidad de riesgo activo.
            - El **máximo drawdown** muestra la peor caída acumulada desde un máximo previo, y es clave para evaluar pérdidas severas.
            - El contexto macroeconómico, en particular la tasa libre de riesgo, la inflación y la tasa de cambio, ayuda a interpretar el entorno financiero en el que se evalúa el portafolio.
            """
        )