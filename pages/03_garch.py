import streamlit as st
import pandas as pd

from src.config import (
    ASSETS,
    DEFAULT_END_DATE,
    ensure_project_dirs,
    get_ticker,
)
from src.download import data_error_message, download_single_ticker
from src.garch_models import fit_garch_models
from src.plots import plot_forecast, plot_standardized_residuals, plot_volatility
from src.returns_analysis import compute_return_series
from src.risk_metrics import validar_serie_para_garch
from src.ui_components import kpi_card, render_explanation_expander, render_section, render_table
from src.ui_navigation import render_sidebar_navigation
from src.ui_style import apply_global_typography, render_page_title

ensure_project_dirs()
apply_global_typography()
render_sidebar_navigation()


# ==============================
# Estilos UI
# ==============================
def inject_module_css():
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


def fmt_num(value):
    numeric_value = pd.to_numeric(value, errors="coerce")
    return f"{numeric_value:.3f}" if pd.notna(numeric_value) else "N/D"


def fmt_pvalue(value):
    numeric_value = pd.to_numeric(value, errors="coerce")
    if pd.isna(numeric_value):
        return "N/D"
    if numeric_value < 0.001:
        return "< 0.001"
    return f"{numeric_value:.3f}"


inject_module_css()

render_page_title(
    "Módulo 3: ARCH/GARCH",
    "Modela volatilidad condicional y pronósticos de riesgo sobre rendimientos del activo.",
)

# ==============================
# Sidebar
# ==============================
with st.sidebar:
    st.header("Filtros")
    asset_name = st.selectbox("Activo", list(ASSETS.keys()), index=0)

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

# ==============================
# Descargar datos
# ==============================
ticker = get_ticker(asset_name)
df = download_single_ticker(ticker=ticker, start=str(start_date), end=str(end_date))

if df.empty:
    st.error(data_error_message("No se pudieron descargar datos."))
    st.stop()

price_col = "Adj Close" if "Adj Close" in df.columns else "Close"
ret_df = compute_return_series(df[price_col])

if "log_return" not in ret_df.columns:
    st.error("No se encontró la columna 'log_return' para ajustar el modelo GARCH.")
    st.stop()

serie_retornos = ret_df["log_return"]

# ==============================
# Validación de la serie
# ==============================
validacion = validar_serie_para_garch(
    serie_retornos,
    min_obs=120,
    max_null_ratio=0.05,
)

if not validacion["ok"]:
    for err in validacion["errores"]:
        st.error(err)

    st.info(
        "No se ajustó el modelo GARCH porque la serie no cumple las condiciones mínimas "
        "de calidad para un ajuste defendible."
    )
    st.stop()

for adv in validacion["advertencias"]:
    st.warning(adv)

serie_garch = validacion["serie_limpia"] * 100.0
volatilidad_movil_21d = serie_garch.rolling(window=21).std()

# ==============================
# Encabezado del módulo
# ==============================
render_section(
    "Volatilidad condicional",
    f"Evalúa modelos ARCH/GARCH para comparar especificaciones y pronosticar la volatilidad de {asset_name} ({ticker}).",
)
st.caption(f"Periodo analizado: {start_date} a {end_date}")
render_explanation_expander(
    "Fundamento del módulo",
    [
        "La volatilidad condicional cambia en el tiempo y suele agruparse en periodos de calma o turbulencia.",
        "Los modelos ARCH/GARCH permiten estimar esa dinámica sin asumir una volatilidad histórica constante.",
        "La serie validada se escala por 100 antes del ajuste para mejorar la estabilidad numérica del modelo.",
    ],
)

# ==============================
# Ajuste de modelos
# ==============================
results = fit_garch_models(serie_garch)

if results["comparison"].empty:
    st.warning("No hay suficientes datos o el ajuste no convergió correctamente para los modelos GARCH.")
    st.stop()

# ==============================
# Resultados principales
# ==============================
comparison_df = results["comparison"].copy()
if "AIC" in comparison_df.columns:
    comparison_df = comparison_df.sort_values("AIC", ascending=True).reset_index(drop=True)

best_model = results.get("best_model_name", None)
best_row = pd.DataFrame()
if best_model is not None and "modelo" in comparison_df.columns:
    best_row = comparison_df.loc[comparison_df["modelo"] == best_model]

if best_row.empty and not comparison_df.empty:
    best_row = comparison_df.head(1)
    best_model = best_row.iloc[0].get("modelo", best_model)

best_aic = None
best_bic = None
best_loglik = None
best_converged = "N/D"
persistence = None

if not best_row.empty:
    row = best_row.iloc[0]
    best_loglik = pd.to_numeric(row.get("loglik"), errors="coerce")
    best_aic = pd.to_numeric(row.get("AIC"), errors="coerce")
    best_bic = pd.to_numeric(row.get("BIC"), errors="coerce")
    best_converged = next((row[col] for col in row.index if str(col).startswith("convergi")), "N/D")
    persistence = pd.to_numeric(row.get("persistencia"), errors="coerce")

forecast_last = None
try:
    forecast_last = float(results["forecast"]["volatilidad_pronosticada"].iloc[-1])
except Exception:
    forecast_last = None

diagnostics_df = results.get("diagnostics", pd.DataFrame()).copy()
diagnostic_map = {}
if not diagnostics_df.empty and {"metrica", "valor"}.issubset(diagnostics_df.columns):
    diagnostic_map = dict(zip(diagnostics_df["metrica"], diagnostics_df["valor"]))

jb_stat = pd.to_numeric(diagnostic_map.get("jb_residuos_stat"), errors="coerce")
jb_pvalue = pd.to_numeric(diagnostic_map.get("jb_residuos_pvalue"), errors="coerce")

if pd.notna(jb_pvalue):
    normality_rejected = jb_pvalue < 0.05
    normality_decision = (
        f"Se rechaza normalidad en residuos estandarizados (p-value {fmt_pvalue(jb_pvalue)})."
        if normality_rejected
        else f"No se rechaza normalidad en residuos estandarizados (p-value {fmt_pvalue(jb_pvalue)})."
    )
else:
    normality_rejected = None
    normality_decision = "No fue posible evaluar normalidad de residuos estandarizados."

if pd.notna(persistence):
    if persistence >= 0.90:
        persistence_label = "Alta persistencia"
        persistence_delta = "pos"
    elif persistence >= 0.75:
        persistence_label = "Persistencia media"
        persistence_delta = "neu"
    else:
        persistence_label = "Persistencia baja"
        persistence_delta = "neg"
else:
    persistence_label = None
    persistence_delta = "neu"

# ==============================
# KPIs del mejor modelo
# ==============================
st.markdown("### KPIs del mejor modelo")
render_section(
    "Modelo seleccionado",
    "Indicadores principales de la especificación ganadora bajo el criterio de selección disponible.",
)

k1, k2, k3 = st.columns(3)
with k1:
    kpi_card(
        "Mejor modelo",
        str(best_model) if best_model is not None else "N/D",
        caption="Referencia seleccionada por menor AIC",
    )
with k2:
    kpi_card(
        "AIC",
        fmt_num(best_aic),
        caption="Menor valor favorece el ajuste relativo",
    )
with k3:
    kpi_card(
        "BIC",
        fmt_num(best_bic),
        caption="Criterio con penalización por complejidad",
    )

k4, k5, k6 = st.columns(3)
with k4:
    kpi_card(
        "Persistencia",
        fmt_num(persistence),
        delta=persistence_label,
        delta_type=persistence_delta,
        caption="Memoria de la volatilidad estimada",
    )
with k5:
    kpi_card(
        "Volatilidad pronosticada",
        fmt_num(forecast_last),
        caption="Último valor del pronóstico a 10 pasos",
    )
with k6:
    kpi_card(
        "Log-Likelihood",
        fmt_num(best_loglik),
        caption="Log-verosimilitud del modelo ganador",
    )

if best_model is None:
    st.warning("No se generó una lectura automática del mejor modelo.")

render_explanation_expander(
    "Cómo interpretar los KPI del modelo GARCH",
    [
        "El AIC/BIC sirven para comparar modelos; menor valor indica mejor ajuste relativo.",
        "La persistencia cercana a 1 indica alta memoria de la volatilidad.",
        "La volatilidad estimada/pronosticada mide la magnitud esperada de fluctuación del activo.",
        "El mejor modelo no significa predicción perfecta, sino mejor ajuste relativo dentro de los modelos evaluados.",
    ],
)
# ==============================
# Comparación de modelos
# ==============================
st.markdown("### 2. Comparación entre modelos")
render_section(
    "Tabla comparativa",
    "Se comparan las especificaciones candidatas con criterios de ajuste y métricas de volatilidad.",
)

preferred_columns = [
    "modelo",
    "AIC",
    "BIC",
    "loglik",
    "persistencia",
]
visible_columns = [col for col in preferred_columns if col in comparison_df.columns]
comparison_display_df = comparison_df[visible_columns].copy()
if "AIC" in comparison_display_df.columns:
    comparison_display_df = comparison_display_df.sort_values("AIC", ascending=True).reset_index(drop=True)
if "modelo" in comparison_display_df.columns:
    comparison_display_df["Selección"] = comparison_display_df["modelo"].apply(
        lambda model_name: "Mejor modelo" if model_name == best_model else ""
    )
comparison_display_df = comparison_display_df.rename(
    columns={
        "modelo": "Modelo",
        "loglik": "Log-Likelihood",
        "persistencia": "Persistencia",
    }
)
for column in comparison_display_df.columns:
    if column not in {"Modelo", "Selección"}:
        comparison_display_df[column] = pd.to_numeric(comparison_display_df[column], errors="coerce").apply(fmt_num)

render_table(comparison_display_df, width="stretch", hide_index=True)

render_explanation_expander(
    "Cómo leer la comparación de modelos",
    [
        "Se comparan modelos por ajuste relativo.",
        "AIC/BIC penalizan complejidad.",
        "El modelo con menor AIC se toma como referencia si esa es la regla usada.",
        "La tabla ayuda a justificar por qué se seleccionó el mejor modelo.",
    ],
)

# ==============================
# Diagnóstico
# ==============================
st.markdown("### 3. Diagnóstico del modelo seleccionado")
render_section(
    "Diagnóstico",
    "Se resume si el ajuste converge y si los residuos estandarizados mantienen señales relevantes.",
)

d1, d2, d3, d4 = st.columns(4)

with d1:
    kpi_card(
        "Convergencia",
        str(best_converged),
        caption="Estado de convergencia del ajuste",
    )

with d2:
    kpi_card(
        "JB residuos est.",
        fmt_num(jb_stat),
        caption="Jarque-Bera sobre residuos estandarizados",
    )

with d3:
    kpi_card(
        "p-value JB",
        fmt_pvalue(jb_pvalue),
        delta="Rechaza normalidad" if normality_rejected else "No rechaza" if normality_rejected is False else None,
        delta_type="neg" if normality_rejected else "pos" if normality_rejected is False else "neu",
        caption="Prueba sobre residuos estandarizados",
    )

with d4:
    kpi_card(
        "Persistencia",
        fmt_num(persistence),
        delta=persistence_label,
        delta_type=persistence_delta,
        caption="Memoria de la volatilidad",
    )

st.caption(normality_decision)

render_explanation_expander(
    "Cómo interpretar el diagnóstico",
    [
        "La convergencia indica si el modelo logró estimarse correctamente.",
        "Jarque-Bera se aplica a residuos estandarizados para revisar normalidad residual.",
        "Un p-value muy pequeño sugiere que pueden persistir colas o eventos extremos no explicados por completo.",
        "Una persistencia alta indica que los choques de volatilidad tardan más en disiparse.",
    ],
)

if not diagnostics_df.empty:
    with st.expander("Ver detalle técnico del diagnóstico"):
        diagnostics_display_df = diagnostics_df.copy()
        if "valor" in diagnostics_display_df.columns:
            diagnostics_display_df["valor"] = diagnostics_display_df["valor"].apply(
                lambda value: fmt_num(value) if pd.notna(pd.to_numeric(value, errors="coerce")) else value
            )
        render_table(diagnostics_display_df, width="stretch", hide_index=True)
else:
    st.info("No se generaron diagnósticos adicionales para el modelo seleccionado.")

# ==============================
# Residuos estandarizados
# ==============================
st.markdown("### 4. Residuos estandarizados")
if "std_resid" in results and results["std_resid"] is not None:
    st.plotly_chart(plot_standardized_residuals(results["std_resid"]), width="stretch")
    render_explanation_expander(
        "Cómo interpretar los residuos estandarizados",
        [
            "Residuos alrededor de cero sugieren que el modelo captura buena parte de la estructura media.",
            "Picos extremos persistentes indican episodios que el modelo no absorbe completamente.",
            "Esta revisión complementa el diagnóstico formal y ayuda a evaluar riesgo extremo.",
        ],
    )

    with st.expander("Ver residuos estandarizados"):
        render_table(results["std_resid"].tail(20), width="stretch", hide_index=False)
else:
    st.info("No se generaron residuos estandarizados para el modelo seleccionado.")

# ==============================
# Volatilidad condicional estimada
# ==============================
st.markdown("### 5. Volatilidad condicional estimada")
st.caption(
    "La gráfica compara cómo cada especificación modela la evolución de la volatilidad condicional a lo largo del tiempo."
)
volatility_plot_df = results["volatility"].copy()
volatility_plot_df.insert(
    0,
    "Volatilidad móvil 21 días",
    volatilidad_movil_21d.reindex(volatility_plot_df.index),
)
volatility_fig = plot_volatility(volatility_plot_df)
volatility_fig.update_traces(
    selector=dict(name="Volatilidad móvil 21 días"),
    line=dict(color="#64748B", width=2.6, dash="dot"),
)
st.plotly_chart(volatility_fig, width="stretch")

render_explanation_expander(
    "Cómo interpretar la volatilidad estimada",
    [
        "La Volatilidad móvil 21 días es una referencia empírica aproximada calculada a partir de los rendimientos recientes.",
        "Los modelos ARCH, GARCH y EGARCH son estimaciones suavizadas de la volatilidad condicional.",
        "Un buen modelo no tiene que replicar exactamente todos los movimientos de la Volatilidad móvil 21 días.",
        "Lo importante es que capture los principales cambios de régimen o episodios de mayor riesgo.",
        "Si las líneas de los modelos aumentan en los mismos periodos que la Volatilidad móvil 21 días, están capturando adecuadamente episodios de mayor incertidumbre.",
        "La volatilidad condicional permite identificar tramos donde la incertidumbre se concentra.",
        "Una persistencia elevada refuerza la lectura de episodios de riesgo agrupado.",
        "ARCH, GARCH y EGARCH modelan la volatilidad con distintos supuestos sobre choques y memoria.",
    ],
)

render_explanation_expander(
    "Qué significan ARCH, GARCH y EGARCH",
    [
        "ARCH se enfoca en cómo los choques recientes afectan la volatilidad actual.",
        "GARCH combina choques recientes con persistencia de volatilidad pasada.",
        "EGARCH permite capturar respuestas asimétricas ante choques positivos y negativos.",
    ],
)

# ==============================
# Pronóstico de volatilidad
# ==============================
horizon_steps = None
try:
    horizon_steps = int(results["forecast"]["horizonte"].max())
except Exception:
    horizon_steps = None

forecast_title = (
    f"### 6. Pronóstico de volatilidad ({horizon_steps} pasos)"
    if horizon_steps is not None
    else "### 6. Pronóstico de volatilidad (10 pasos)"
)
st.markdown(forecast_title)
st.caption(
    "El gráfico muestra un pronóstico de volatilidad a varios pasos hacia adelante."
)
st.plotly_chart(
    plot_forecast(results["forecast"]),
    width="stretch",
)
render_explanation_expander(
    "Cómo interpretar el pronóstico de volatilidad",
    [
        "El pronóstico no predice dirección del precio.",
        "Pronostica intensidad esperada de la volatilidad.",
        "Mayor volatilidad implica mayor incertidumbre/riesgo.",
        "Debe complementarse con VaR/CVaR, CAPM y benchmark.",
    ],
)

# ==============================
# Conclusión
# ==============================
st.markdown("### Conclusión")
if best_model is not None:
    conclusion_parts = [
        f"El modelo ganador fue **{best_model}** por criterio AIC.",
    ]

    if pd.notna(persistence):
        conclusion_parts.append(
            f"La persistencia de **{fmt_num(persistence)}** indica el grado de memoria de los choques de volatilidad."
        )

    conclusion_parts.append(normality_decision)

    if forecast_last is not None:
        conclusion_parts.append(
            f"El pronóstico final de volatilidad de **{fmt_num(forecast_last)}** resume la volatilidad esperada al cierre del horizonte."
        )

    conclusion_parts.append(
        "Para la gestión de riesgo y VaR, estos resultados ayudan a distinguir entre riesgo reciente, persistencia y riesgo prospectivo."
    )

    st.success(" ".join(conclusion_parts))
else:
    st.info("No fue posible construir una conclusión porque no se seleccionó un modelo final.")
