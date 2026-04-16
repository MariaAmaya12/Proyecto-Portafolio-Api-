import streamlit as st
import pandas as pd

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
st.caption("Analiza volatilidad condicional y pronósticos de riesgo a partir de rendimientos del activo.")

# ==============================
# Sidebar
# ==============================
with st.sidebar:
    st.header("Parámetros GARCH")
    asset_name = st.selectbox("Activo representativo", list(ASSETS.keys()), index=0)

    horizonte = st.selectbox(
        "Horizonte de análisis",
        [
            "1 mes",
            "Trimestre",
            "Semestre",
            "1 año",
            "3 años",
            "5 años",
            "Personalizado",
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
    elif horizonte == "3 años":
        start_date = (fecha_fin_ref - pd.DateOffset(years=3)).date()
        end_date = fecha_fin_ref.date()
    elif horizonte == "5 años":
        start_date = (fecha_fin_ref - pd.DateOffset(years=5)).date()
        end_date = fecha_fin_ref.date()
    else:
        start_date = st.date_input("Fecha inicial", value=DEFAULT_START_DATE, key="garch_start")
        end_date = st.date_input("Fecha final", value=DEFAULT_END_DATE, key="garch_end")

    st.divider()
    st.subheader("Modo de visualización")
    modo = st.radio(
        "Selecciona el nivel de detalle",
        ["General", "Estadístico"],
        index=0,
    )

    st.divider()
    st.subheader("Opciones de visualización")
    mostrar_tablas = st.checkbox("Mostrar tablas completas", value=False)

    mostrar_diagnostico = False
    mostrar_residuos = False
    if modo == "Estadístico":
        mostrar_diagnostico = st.checkbox("Mostrar diagnóstico del modelo", value=True)
        mostrar_residuos = st.checkbox("Mostrar residuos estandarizados", value=False)

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

serie_retornos = ret_df["log_return"]

# ==============================
# Resumen
# ==============================
st.markdown("### Resumen del módulo")
if modo == "General":
    st.write(
        f"""
        Este módulo evalúa si la volatilidad de **{asset_name} ({ticker})** cambia a lo largo del tiempo.
        En mercados financieros esto es importante porque los periodos tranquilos y turbulentos no suelen
        alternarse de manera uniforme.
        """
    )
else:
    st.write(
        f"""
        Este módulo ajusta modelos ARCH/GARCH sobre los rendimientos logarítmicos de **{asset_name} ({ticker})**
        para modelar heterocedasticidad condicional, comparar especificaciones y generar pronósticos de volatilidad.
        """
    )

st.caption(f"Periodo analizado: {start_date} a {end_date}")

# ==============================
# Explicación base
# ==============================
if modo == "General":
    st.info(
        """
        **Idea central**
        - La volatilidad no siempre es constante.
        - Los modelos GARCH ayudan a capturar periodos de calma y periodos de alta incertidumbre.
        - Esto permite medir mejor el riesgo que usar una sola volatilidad fija.
        """
    )
else:
    with st.expander("Ver fundamento teórico"):
        st.write(
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
validacion = validar_serie_para_garch(
    serie_retornos,
    min_obs=120,
    max_null_ratio=0.05,
)

st.markdown("### Validación de la serie para GARCH")

n_original = validacion["resumen"].get("n_original", 0)
n_limpio = validacion["resumen"].get("n_limpio", 0)
std_val = validacion["resumen"].get("std", 0)

col1, col2, col3 = st.columns(3)
col1.metric("Obs. originales", n_original)
col2.metric("Obs. limpias", n_limpio)
col3.metric("Desv. estándar", f"{std_val:.6f}")

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

if modo == "Estadístico":
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
# KPIs
# ==============================
st.markdown("### KPIs del ajuste")

best_model = results.get("best_model_name", None)
n_models = len(results["comparison"]) if "comparison" in results else 0

forecast_last = None
try:
    forecast_last = float(results["forecast"]["volatilidad_pronosticada"].iloc[-1])
except Exception:
    forecast_last = None

vol_last = None
try:
    vol_cols = list(results["volatility"].columns)
    if vol_cols:
        vol_last = float(results["volatility"][vol_cols[0]].dropna().iloc[-1])
except Exception:
    vol_last = None

c1, c2, c3, c4 = st.columns(4)
c1.metric("Activo", asset_name)
c2.metric("Modelos comparados", n_models)
c3.metric("Mejor modelo", str(best_model) if best_model is not None else "N/D")
c4.metric("Forecast final", f"{forecast_last:.4f}" if forecast_last is not None else "N/D")

# ==============================
# Interpretación automática
# ==============================
st.markdown("### Interpretación")

if results["best_model_name"] is not None and results.get("summary_text"):
    if modo == "General":
        st.success(results["summary_text"])
    else:
        st.info(results["summary_text"])
else:
    st.warning("No se generó un resumen automático del mejor modelo.")

# ==============================
# Comparación de modelos
# ==============================
st.markdown("### Comparación de modelos")

if mostrar_tablas:
    st.dataframe(results["comparison"], width="stretch")
else:
    with st.expander("Ver comparación completa de modelos"):
        st.dataframe(results["comparison"], width="stretch")

# ==============================
# Gráficos
# ==============================
st.markdown("### Volatilidad y pronóstico")

col1, col2 = st.columns(2)
with col1:
    st.plotly_chart(plot_volatility(results["volatility"]), width="stretch")
with col2:
    st.plotly_chart(plot_forecast(results["forecast"]), width="stretch")

if modo == "General":
    st.info(
        """
        **Cómo leer estos gráficos**

        - El primer gráfico muestra cómo cambia la volatilidad estimada a lo largo del tiempo.
        - El segundo resume el pronóstico de volatilidad a futuro.
        - Valores altos indican mayor incertidumbre y, por tanto, mayor riesgo.
        """
    )
else:
    with st.expander("Ver interpretación técnica de volatilidad y forecast"):
        st.write(
            """
            La serie de volatilidad condicional captura persistencia y clustering de la varianza,
            mientras que el forecast resume la trayectoria esperada de volatilidad bajo el modelo
            seleccionado. Esto permite comparar riesgo reciente y riesgo prospectivo.
            """
        )

# ==============================
# Diagnóstico
# ==============================
if modo == "Estadístico" and mostrar_diagnostico:
    st.markdown("### Diagnóstico")
    st.dataframe(results["diagnostics"], width="stretch")

# ==============================
# Residuos estandarizados
# ==============================
if modo == "Estadístico" and mostrar_residuos:
    st.markdown("### Residuos estandarizados")
    st.dataframe(results["std_resid"].tail(20), width="stretch")