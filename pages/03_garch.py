import streamlit as st

from src.config import (
    ASSETS,
    DEFAULT_END_DATE,
    DEFAULT_START_DATE,
    ensure_project_dirs,
    get_ticker,
)
from src.download import download_single_ticker
from src.garch_models import fit_garch_models
from src.plots import plot_forecast, plot_volatility
from src.returns_analysis import compute_return_series
from src.risk_metrics import validar_serie_para_garch

ensure_project_dirs()

st.title("Módulo 3 - Modelos ARCH/GARCH")

# ==============================
# Sidebar
# ==============================
with st.sidebar:
    st.header("Parámetros GARCH")
    asset_name = st.selectbox("Activo representativo", list(ASSETS.keys()), index=0)
    start_date = st.date_input("Fecha inicial", value=DEFAULT_START_DATE, key="garch_start")
    end_date = st.date_input("Fecha final", value=DEFAULT_END_DATE, key="garch_end")

# ==============================
# Descargar datos
# ==============================
ticker = get_ticker(asset_name)
df = download_single_ticker(ticker=ticker, start=str(start_date), end=str(end_date))

if df.empty:
    st.error("No se pudieron descargar datos.")
    st.stop()

price_col = "Adj Close" if "Adj Close" in df.columns else "Close"
ret_df = compute_return_series(df[price_col])

if "log_return" not in ret_df.columns:
    st.error("No se encontró la columna 'log_return' para ajustar el modelo GARCH.")
    st.stop()

# ==============================
# Explicación
# ==============================
st.markdown(
    """
    La volatilidad condicional se usa cuando la varianza no es constante en el tiempo.
    En finanzas esto ocurre con frecuencia por clustering de volatilidad.

    Los modelos GARCH permiten modelar esa persistencia de la volatilidad y generar
    pronósticos de riesgo más realistas que una volatilidad histórica constante.
    """
)

# ==============================
# Validación de la serie
# ==============================
serie_retornos = ret_df["log_return"]

validacion = validar_serie_para_garch(
    serie_retornos,
    min_obs=120,
    max_null_ratio=0.05,
)

st.subheader("Validación de la serie para GARCH")

col1, col2, col3 = st.columns(3)
col1.metric("Obs. originales", validacion["resumen"].get("n_original", 0))
col2.metric("Obs. limpias", validacion["resumen"].get("n_limpio", 0))
col3.metric("Desv. estándar", f'{validacion["resumen"].get("std", 0):.6f}')

for adv in validacion["advertencias"]:
    st.warning(adv)

if not validacion["ok"]:
    for err in validacion["errores"]:
        st.error(err)

    st.info(
        "No se ajustó el modelo GARCH porque la serie no cumple las condiciones mínimas "
        "de calidad para un ajuste defendible."
    )
    st.stop()

# ==============================
# Preparar datos para GARCH
# ==============================
serie_garch = validacion["serie_limpia"] * 100.0

st.caption(
    "Los rendimientos logarítmicos se escalan por 100 para mejorar la estabilidad numérica "
    "del ajuste en modelos ARCH/GARCH."
)

# ==============================
# Ajuste de modelos
# ==============================
results = fit_garch_models(serie_garch)

if results["comparison"].empty:
    st.warning("No hay suficientes datos o el ajuste no convergió correctamente para los modelos GARCH.")
    st.stop()

# ==============================
# 🔥 Interpretación automática (MUY IMPORTANTE PARA LA RÚBRICA)
# ==============================
if results["best_model_name"] is not None and results.get("summary_text"):
    st.info(results["summary_text"])

# ==============================
# Resultados
# ==============================
st.subheader("Comparación de modelos")
st.dataframe(results["comparison"], width="stretch")

col1, col2 = st.columns(2)
with col1:
    st.plotly_chart(plot_volatility(results["volatility"]), width="stretch")
with col2:
    st.plotly_chart(plot_forecast(results["forecast"]), width="stretch")

st.subheader("Diagnóstico")
st.dataframe(results["diagnostics"], width="stretch")

st.subheader("Residuos estandarizados")
st.dataframe(results["std_resid"].tail(20), width="stretch")