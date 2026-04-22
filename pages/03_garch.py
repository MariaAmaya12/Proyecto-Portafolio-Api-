import streamlit as st
import streamlit.components.v1 as components
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
from src.ui_navigation import render_sidebar_navigation
from src.ui_style import apply_global_typography, render_page_title

ensure_project_dirs()
apply_global_typography()
render_sidebar_navigation()


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


def kpi_card(title, value, delta=None, delta_type="neu", caption=""):
    title = sanitize_text(title)
    value = sanitize_text(value)
    delta = sanitize_text(delta) if delta is not None else ""
    caption = sanitize_text(caption)

    delta_html = ""
    if delta:
        delta_html = f'<div class="kpi-delta {delta_type}">{delta}</div>'

    caption_html = f'<div class="kpi-caption">{caption}</div>' if caption else ""

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
                align-items: center;
                gap: 0.55rem;
                overflow: visible;
                text-align: center;
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
                font-size: 1.82rem;
                font-weight: 800;
                color: #0f172a;
                line-height: 1.08;
                margin-bottom: 0.45rem;
                overflow-wrap: anywhere;
                word-break: normal;
                white-space: normal;
                width: 100%;
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
            {caption_html}
        </div>
    </body>
    </html>
    """

    components.html(html, height=194)


inject_kpi_cards_css()

render_page_title(
    "Módulo 3 - Modelos ARCH/GARCH",
    "Analiza volatilidad condicional y pronósticos de riesgo a partir de rendimientos del activo.",
)

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

# ==============================
# Resumen
# ==============================
st.markdown("### Resumen del módulo")
st.write(
    f"""
    Este módulo ajusta modelos ARCH/GARCH sobre los rendimientos logarítmicos de **{asset_name} ({ticker})**
    para modelar heterocedasticidad condicional, comparar especificaciones y generar pronósticos de volatilidad.
    La lectura integra validación de datos, selección del mejor modelo, persistencia, volatilidad de largo plazo
    y diagnóstico del ajuste.
    """
)

st.caption(f"Periodo analizado: {start_date} a {end_date}")

# ==============================
# Fundamento teórico
# ==============================
with st.expander("Ver fundamento teórico"):
    st.caption(
        "Este bloque justifica por qué se usan modelos de volatilidad condicional antes de revisar resultados."
    )
    st.write(
        """
        La volatilidad condicional se usa cuando la varianza no es constante en el tiempo.
        En finanzas esto ocurre con frecuencia por clustering de volatilidad: periodos de calma
        suelen alternarse con periodos de turbulencia.

        Los modelos GARCH permiten modelar la persistencia de la volatilidad y generar
        pronósticos de riesgo más realistas que una volatilidad histórica constante.

        Para este ajuste, la serie cumple condiciones mínimas de longitud, limpieza y variabilidad.
        Además, los rendimientos logarítmicos se escalan por 100 para mejorar la estabilidad numérica
        del ajuste en modelos ARCH/GARCH.
        """
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
best_converged = "N/D"
long_run_vol = None
persistence = None

if not best_row.empty:
    row = best_row.iloc[0]
    best_aic = pd.to_numeric(row.get("AIC"), errors="coerce")
    best_bic = pd.to_numeric(row.get("BIC"), errors="coerce")
    best_converged = row.get("convergió", row.get("convergiÃ³", "N/D"))
    persistence = pd.to_numeric(row.get("persistencia"), errors="coerce")

    omega = pd.to_numeric(row.get("omega"), errors="coerce")
    if pd.notna(omega) and pd.notna(persistence) and persistence < 1:
        long_run_var = omega / (1 - persistence)
        if long_run_var > 0:
            long_run_vol = long_run_var ** 0.5

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
section_intro(
    "Resumen analítico del ajuste",
    "Estos indicadores resumen la selección, persistencia y riesgo prospectivo del modelo ganador.",
)

k1, k2, k3 = st.columns(3)

with k1:
    kpi_card(
        "Mejor modelo",
        str(best_model) if best_model is not None else "N/D",
        caption="Modelo con menor AIC",
    )

with k2:
    kpi_card(
        "AIC",
        fmt_num(best_aic),
        caption="Criterio de información del modelo ganador",
    )

with k3:
    kpi_card(
        "Persistencia",
        fmt_num(persistence),
        delta=persistence_label,
        delta_type=persistence_delta,
        caption="Suma alpha + beta cuando aplica",
    )

k4, k5 = st.columns(2)

with k4:
    kpi_card(
        "Forecast final",
        fmt_num(forecast_last),
        caption="Último valor pronosticado de volatilidad",
    )

with k5:
    kpi_card(
        "Vol. largo plazo",
        fmt_num(long_run_vol),
        caption="Nivel de reversión si el modelo es estacionario",
    )

with st.expander("¿Qué significa cada KPI?"):
    st.write(
        """
        **Mejor modelo:** especificación que tuvo mejor desempeño comparativo entre los modelos estimados.

        **AIC:** criterio de información; un valor más bajo suele indicar mejor equilibrio entre ajuste y complejidad.

        **Persistencia:** mide qué tanto duran los choques de volatilidad. Si es alta, los episodios de incertidumbre pueden prolongarse.

        **Forecast final:** volatilidad esperada al final del horizonte pronosticado.

        **Volatilidad de largo plazo:** nivel al que tendería la volatilidad si el modelo es estacionario.
        """
    )

# ==============================
# Lectura del modelo seleccionado
# ==============================
st.markdown("### Lectura del modelo seleccionado")

if best_model is not None:
    lectura = [
        f"El modelo **{best_model}** fue seleccionado por presentar el menor AIC dentro de las especificaciones estimadas."
    ]

    if pd.notna(persistence):
        lectura.append(
            f"La persistencia estimada es **{fmt_num(persistence)}**, lo que indica qué tan rápido se disipan los choques de volatilidad."
        )
        if persistence >= 0.90:
            lectura.append(
                "Este nivel es consistente con clustering de volatilidad: los episodios de alta incertidumbre tienden a mantenerse por varios periodos."
            )
        else:
            lectura.append(
                "La persistencia no es extrema, por lo que los choques de volatilidad tienden a disiparse con mayor rapidez."
            )

    st.info(" ".join(lectura))
else:
    st.warning("No se generó una lectura automática del mejor modelo.")

# ==============================
# Comparación de modelos
# ==============================
with st.expander("Ver comparación completa entre modelos"):
    section_intro(
        "Selección de especificación",
        "Se comparan modelos candidatos mediante log-verosimilitud, AIC, BIC, convergencia y parámetros estimados.",
    )

    preferred_columns = [
        "modelo",
        "loglik",
        "AIC",
        "BIC",
        "convergió",
        "convergiÃ³",
        "mu",
        "omega",
        "alpha_1",
        "beta_1",
        "persistencia",
    ]
    visible_columns = [col for col in preferred_columns if col in comparison_df.columns]
    st.dataframe(comparison_df[visible_columns], width="stretch")

    if best_model is not None:
        criterio = f"AIC = {fmt_num(best_aic)}" if pd.notna(best_aic) else "menor AIC disponible"
        st.caption(f"El modelo con mejor criterio de información es **{best_model}** ({criterio}).")

# ==============================
# Diagnóstico
# ==============================
st.markdown("### Diagnóstico del modelo")

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

with st.expander("¿Cómo interpretar este diagnóstico?"):
    st.write(
        """
        **Convergencia:** indica si el modelo logró estimarse correctamente.

        **JB residuos est.:** prueba de Jarque-Bera aplicada a los residuos estandarizados del modelo.

        **p-value JB:** ayuda a decidir si se rechaza normalidad. Si es muy pequeño, como **< 0.001**, se rechaza normalidad en residuos.

        **Persistencia:** indica si los choques de volatilidad duran poco o mucho.

        Si el p-value es muy pequeño, el modelo puede estar capturando la volatilidad, pero todavía pueden quedar colas o comportamientos extremos no explicados por completo.
        """
    )

if normality_rejected:
    soft_note(
        "Lectura del diagnóstico",
        "Se rechaza normalidad en residuos estandarizados; el modelo captura la dinámica de volatilidad, "
        "pero no elimina completamente rasgos no normales. Esto sugiere que el riesgo extremo puede no estar "
        "totalmente capturado por una aproximación normal.",
    )
elif normality_rejected is False:
    soft_note(
        "Lectura del diagnóstico",
        "No se rechaza normalidad en residuos estandarizados; bajo esta prueba, los residuos no muestran "
        "evidencia estadística fuerte contra normalidad.",
    )
else:
    soft_note("Lectura del diagnóstico", normality_decision)

if not diagnostics_df.empty:
    with st.expander("Ver detalle técnico del diagnóstico"):
        st.dataframe(diagnostics_df, width="stretch")
else:
    st.info("No se generaron diagnósticos adicionales para el modelo seleccionado.")

if "std_resid" in results and results["std_resid"] is not None:
    st.plotly_chart(plot_standardized_residuals(results["std_resid"]), width="stretch")
    soft_note(
        "Lectura de los residuos estandarizados",
        "Si los residuos oscilan alrededor de cero, el modelo está capturando buena parte de la estructura media. "
        "Si persisten picos extremos, todavía hay episodios que el modelo no absorbe completamente; esto es importante "
        "para evaluar riesgo extremo.",
    )

    with st.expander("Ver residuos estandarizados"):
        st.dataframe(results["std_resid"].tail(20), width="stretch")

# ==============================
# Volatilidad condicional estimada
# ==============================
st.markdown("### Volatilidad condicional estimada")
st.caption(
    "La gráfica compara cómo cada especificación modela la evolución de la volatilidad condicional a lo largo del tiempo."
)
st.plotly_chart(plot_volatility(results["volatility"]), width="stretch")

with st.expander("¿Qué significan ARCH, GARCH y EGARCH?"):
    st.write(
        """
        **ARCH** se enfoca en cómo los choques recientes afectan la volatilidad actual.

        **GARCH** combina el efecto de choques recientes con la persistencia de la volatilidad pasada.
        Por eso suele capturar mejor periodos donde la incertidumbre se mantiene durante varios días.

        **EGARCH** permite capturar respuestas asimétricas: por ejemplo, cuando malas noticias aumentan
        la volatilidad más que buenas noticias de tamaño similar.
        """
    )

soft_note(
    "Lectura de la volatilidad estimada",
    "La volatilidad condicional permite identificar clustering: tramos donde la incertidumbre se concentra "
    "y tarda en disiparse. Una persistencia elevada refuerza la lectura de episodios de riesgo agrupado.",
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
    f"### Pronóstico de volatilidad ({horizon_steps} pasos)"
    if horizon_steps is not None
    else "### Pronóstico de volatilidad (N pasos)"
)
st.markdown(forecast_title)
st.caption(
    "El gráfico muestra un pronóstico de volatilidad a varios pasos hacia adelante. "
    "La línea de volatilidad de largo plazo se incluye cuando el modelo seleccionado es estacionario."
)
st.plotly_chart(
    plot_forecast(
        results["forecast"],
        long_run_vol=long_run_vol,
    ),
    width="stretch",
)
soft_note(
    "Lectura del pronóstico",
    "Este forecast resume la volatilidad esperada para un horizonte futuro de varios pasos. "
    "Si el modelo es estacionario, la trayectoria pronosticada tiende a acercarse a la volatilidad de largo plazo.",
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
            f"El forecast final de **{fmt_num(forecast_last)}** resume la volatilidad esperada al cierre del horizonte."
        )

    conclusion_parts.append(
        "Para la gestión de riesgo y VaR, estos resultados ayudan a distinguir entre riesgo reciente, persistencia y riesgo prospectivo."
    )

    st.success(" ".join(conclusion_parts))
else:
    st.info("No fue posible construir una conclusión porque no se seleccionó un modelo final.")
