import streamlit as st
import streamlit.components.v1 as components
import pandas as pd

from src.config import ASSETS, DEFAULT_START_DATE, DEFAULT_END_DATE, get_ticker, ensure_project_dirs
from src.download import data_error_message, download_single_ticker
from src.returns_analysis import (
    compute_return_series,
    descriptive_stats,
    normality_tests,
    qq_plot_data,
)
from src.plots import plot_histogram_with_normal, plot_qq, plot_box
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
                border: 1px solid rgba(37, 99, 235, 0.20);
                border-radius: 18px;
                padding: 20px 18px 16px 18px;
                box-shadow: 0 6px 18px rgba(37, 99, 235, 0.10);
                min-height: 160px;
                box-sizing: border-box;
                display: flex;
                flex-direction: column;
                justify-content: space-between;
                overflow-wrap: anywhere;
            }}

            .kpi-label {{
                font-size: 0.88rem;
                font-weight: 700;
                color: #1e3a8a;
                margin-bottom: 0.35rem;
                letter-spacing: 0;
            }}

            .kpi-value {{
                font-size: 1.78rem;
                font-weight: 800;
                color: #0f172a;
                line-height: 1.14;
                margin-bottom: 0.45rem;
                overflow-wrap: anywhere;
            }}

            .kpi-delta {{
                display: inline-block;
                width: fit-content;
                font-size: 0.80rem;
                font-weight: 700;
                padding: 0.28rem 0.55rem;
                border-radius: 999px;
                margin-top: 0.10rem;
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
                font-size: 0.80rem;
                color: #334155;
                margin-top: 0.65rem;
                line-height: 1.42;
                overflow-wrap: anywhere;
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

    components.html(html, height=190)


def format_p_value(p_value):
    if pd.isna(p_value):
        return "Sin datos"
    if p_value < 0.0001:
        return "< 0.0001"
    return f"{p_value:.4f}"


def render_statistical_interpretation(
    mean_ret,
    vol_ret,
    skew_value,
    kurt_value,
    jb_p_value,
    shapiro_p_value,
):
    mean_text = (
        "cercana a cero, lo que sugiere ausencia de sesgo fuerte en el retorno promedio"
        if abs(mean_ret) < 0.0005
        else "positiva, lo que indica un sesgo promedio favorable en el periodo"
        if mean_ret > 0
        else "negativa, lo que indica un sesgo promedio desfavorable en el periodo"
    )
    skew_text = (
        "sin dato suficiente para evaluar asimetría"
        if skew_value is None
        else "negativa, asociada a cola izquierda y mayor atención a pérdidas extremas"
        if skew_value < -0.5
        else "positiva, asociada a mayor peso de movimientos favorables extremos"
        if skew_value > 0.5
        else "moderada, sin un sesgo direccional fuerte"
    )
    kurt_text = (
        "sin dato suficiente para evaluar curtosis"
        if kurt_value is None
        else "alta, consistente con colas pesadas y mayor presencia de eventos extremos"
        if kurt_value > 3
        else "sin señal fuerte de colas pesadas bajo el umbral usado"
    )
    jb_text = (
        "sin decisión formal por falta de datos"
        if jb_p_value is None
        else f"rechaza normalidad (p-value {format_p_value(jb_p_value)})"
        if jb_p_value < 0.05
        else f"no rechaza normalidad (p-value {format_p_value(jb_p_value)})"
    )
    shapiro_text = (
        "sin dato disponible"
        if shapiro_p_value is None
        else f"p-value {format_p_value(shapiro_p_value)}"
    )
    risk_text = (
        "conviene contrastar el VaR paramétrico normal con métodos históricos o enfoques menos dependientes de normalidad"
        if (jb_p_value is not None and jb_p_value < 0.05) or (kurt_value is not None and kurt_value > 3)
        else "el VaR paramétrico puede servir como referencia, pero debe validarse frente a métodos históricos"
    )
    skew_display = "sin datos" if skew_value is None else f"{skew_value:.2f}"
    kurt_display = "sin datos" if kurt_value is None else f"{kurt_value:.2f}"
    st.info(
        f"""
        **Lectura estadística integrada**

        - **Media:** {mean_ret:.4%}, {mean_text}.
        - **Volatilidad:** {vol_ret:.4%}, mide la dispersión diaria observada.
        - **Asimetría:** {skew_display}; lectura {skew_text}.
        - **Curtosis:** {kurt_display}; lectura {kurt_text}.
        - **Normalidad:** Jarque-Bera {jb_text}; Shapiro-Wilk reporta {shapiro_text}.
        - **Implicación para riesgo:** {risk_text}.
        """
    )


inject_kpi_cards_css()

render_page_title(
    "Módulo 2 - Rendimientos y propiedades empíricas",
    "Analiza la distribución de rendimientos del activo y sus principales propiedades estadísticas.",
)

# ==============================
# Sidebar
# ==============================
with st.sidebar:
    st.header("Parámetros")
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
    elif horizonte == "2 años":
        start_date = (fecha_fin_ref - pd.DateOffset(years=2)).date()
        end_date = fecha_fin_ref.date()
    elif horizonte == "3 años":
        start_date = (fecha_fin_ref - pd.DateOffset(years=3)).date()
        end_date = fecha_fin_ref.date()
    elif horizonte == "5 años":
        start_date = (fecha_fin_ref - pd.DateOffset(years=5)).date()
        end_date = fecha_fin_ref.date()
    else:
        start_date = st.date_input("Fecha inicial", value=DEFAULT_START_DATE, key="ret_start")
        end_date = st.date_input("Fecha final", value=DEFAULT_END_DATE, key="ret_end")

return_type = "log_return"
return_type_label = "Rendimiento logarítmico"

# ==============================
# Datos
# ==============================
ticker = get_ticker(asset_name)
df = download_single_ticker(ticker=ticker, start=str(start_date), end=str(end_date))

if df.empty:
    st.error(data_error_message("No se pudieron descargar datos."))
    st.stop()

price_series = df["Adj Close"] if "Adj Close" in df.columns else df["Close"]
ret_df = compute_return_series(price_series)

if ret_df.empty or return_type not in ret_df.columns:
    st.error("No fue posible calcular la serie de rendimientos.")
    st.stop()

series = ret_df[return_type].dropna()

if series.empty:
    st.error("La serie de rendimientos está vacía.")
    st.stop()

desc_df = descriptive_stats(series)
norm_df = normality_tests(series)
norm_df["interpretacion"] = norm_df["p_value"].apply(
    lambda p: "Sin datos suficientes"
    if pd.isna(p)
    else "Se rechaza normalidad"
    if p < 0.05
    else "No se rechaza normalidad"
)
qq_df = qq_plot_data(series)

comparison_df = pd.DataFrame(
    [
        {
            "tipo_rendimiento": "Simple",
            "media": ret_df["simple_return"].mean(),
            "volatilidad": ret_df["simple_return"].std(ddof=1),
            "minimo": ret_df["simple_return"].min(),
            "maximo": ret_df["simple_return"].max(),
        },
        {
            "tipo_rendimiento": "Logarítmico",
            "media": ret_df["log_return"].mean(),
            "volatilidad": ret_df["log_return"].std(ddof=1),
            "minimo": ret_df["log_return"].min(),
            "maximo": ret_df["log_return"].max(),
        },
    ]
)

skew_value = float(desc_df.loc["asimetria", "valor"]) if "asimetria" in desc_df.index else None
kurt_value = float(desc_df.loc["curtosis", "valor"]) if "curtosis" in desc_df.index else None
jb_rows = norm_df[norm_df["test"] == "Jarque-Bera"]
jb_p_value = float(jb_rows["p_value"].iloc[0]) if not jb_rows.empty and pd.notna(jb_rows["p_value"].iloc[0]) else None
shapiro_rows = norm_df[norm_df["test"] == "Shapiro-Wilk"]
shapiro_p_value = (
    float(shapiro_rows["p_value"].iloc[0])
    if not shapiro_rows.empty and pd.notna(shapiro_rows["p_value"].iloc[0])
    else None
)

norm_display_df = norm_df.copy()
norm_display_df["estadistico"] = norm_display_df["estadistico"].apply(
    lambda value: "Sin datos" if pd.isna(value) else f"{value:.4f}"
)
norm_display_df["p_value"] = norm_display_df["p_value"].apply(format_p_value)

if jb_p_value is None:
    jb_decision = "Sin datos"
    jb_delta = "p-value no disponible"
    jb_delta_type = "neu"
elif jb_p_value < 0.05:
    jb_decision = "Rechazada"
    jb_delta = f"p-value {format_p_value(jb_p_value)}"
    jb_delta_type = "neg"
else:
    jb_decision = "No rechazada"
    jb_delta = f"p-value {format_p_value(jb_p_value)}"
    jb_delta_type = "pos"

comparison_display_df = comparison_df.copy()
for column in ["media", "volatilidad", "minimo", "maximo"]:
    comparison_display_df[column] = comparison_display_df[column].apply(lambda value: f"{value:.4%}")

# ==============================
# Resumen
# ==============================
st.markdown("### Resumen del módulo")
st.write(
    f"""
    Este módulo caracteriza la distribución de rendimientos de **{asset_name} ({ticker})** usando
    rendimientos logarítmicos, estadísticos descriptivos, histograma, boxplot, Q-Q plot
    y pruebas de normalidad. El objetivo es identificar dispersión, asimetría, colas pesadas y
    posibles implicaciones para riesgo y VaR.
    """
)

st.caption(f"Periodo analizado: {start_date} a {end_date}")
st.caption("El análisis principal se mantiene sobre rendimientos logarítmicos y se compara explícitamente con rendimientos simples.")

# ==============================
# KPIs
# ==============================
st.markdown("### KPIs de rendimientos")
section_intro(
    "Resumende la distribución",
    "Aquí se sintetizan la rentabilidad media, la volatilidad y los extremos observados en la serie de rendimientos seleccionada.",
)

mean_ret = series.mean()
vol_ret = series.std(ddof=1)
min_ret = series.min()
max_ret = series.max()
q1_ret = series.quantile(0.25)
q3_ret = series.quantile(0.75)
iqr_ret = q3_ret - q1_ret
lower_fence = q1_ret - 1.5 * iqr_ret
upper_fence = q3_ret + 1.5 * iqr_ret
outlier_count = int(((series < lower_fence) | (series > upper_fence)).sum())

prom_delta = None
prom_delta_type = "neu"
if abs(mean_ret) < 0.0005:
    prom_delta = "Sesgo débil"
    prom_delta_type = "neu"
elif mean_ret > 0:
    prom_delta = "Sesgo positivo"
    prom_delta_type = "pos"
else:
    prom_delta = "Sesgo negativo"
    prom_delta_type = "neg"

vol_delta = None
vol_delta_type = "neu"
if vol_ret < 0.01:
    vol_delta = "Baja volatilidad"
    vol_delta_type = "pos"
elif vol_ret < 0.02:
    vol_delta = "Volatilidad moderada"
    vol_delta_type = "neu"
else:
    vol_delta = "Alta volatilidad"
    vol_delta_type = "neg"

col1, col2, col3 = st.columns(3)

with col1:
    kpi_card(
        "Promedio",
        f"{mean_ret:.3%}",
        delta=prom_delta,
        delta_type=prom_delta_type,
        caption=f"Media de la serie de {return_type_label.lower()}",
    )

with col2:
    kpi_card(
        "Volatilidad",
        f"{vol_ret:.3%}",
        delta=vol_delta,
        delta_type=vol_delta_type,
        caption="Desviación estándar muestral de rendimientos",
    )

with col3:
    kpi_card(
        "Normalidad (JB)",
        jb_decision,
        delta=jb_delta,
        delta_type=jb_delta_type,
        caption="Decisión basada en Jarque-Bera al 5%",
    )

col4, col5 = st.columns(2)

with col4:
    kpi_card(
        "Mínimo",
        f"{min_ret:.3%}",
        caption="Peor rendimiento observado en el periodo",
    )

with col5:
    kpi_card(
        "Máximo",
        f"{max_ret:.3%}",
        caption="Mejor rendimiento observado en el periodo",
    )

with st.expander("Interpretación de KPIs"):
    st.markdown(
        f"""
        - **Promedio:** resume la dirección media de los rendimientos diarios. En **{asset_name}** es **{mean_ret:.4%}**;
          si está cerca de cero, no hay un sesgo fuerte de rentabilidad diaria promedio.
        - **Volatilidad:** mide cuán dispersos son los rendimientos alrededor de su media. El valor actual es
          **{vol_ret:.4%}**, clasificado como **{vol_delta.lower()}**, y funciona como primera señal de incertidumbre.
        - **Normalidad (JB):** indica si la distribución se parece a una normal simple. Una normalidad rechazada
          sugiere que una aproximación normal puede no capturar bien colas, asimetrías o riesgo extremo.
        - **Mínimo:** muestra el peor retorno observado del periodo (**{min_ret:.4%}**), útil para dimensionar pérdidas extremas históricas.
        - **Máximo:** muestra el mejor retorno observado del periodo (**{max_ret:.4%}**), útil para comparar la amplitud de movimientos positivos.
        """
    )

# ==============================
# Gráficos principales
# ==============================
st.markdown("### Distribución de rendimientos")
section_intro(
    "Forma de la distribución",
    "Los gráficos permiten visualizar la dispersión, la asimetría y la presencia de valores extremos en los rendimientos.",
)

col3, col4 = st.columns(2)
with col3:
    st.plotly_chart(plot_histogram_with_normal(series), width="stretch")
with col4:
    st.plotly_chart(plot_box(series), width="stretch")

skew_distribution_text = (
    "sesgo hacia pérdidas extremas"
    if skew_value is not None and skew_value < -0.5
    else "sesgo hacia movimientos positivos extremos"
    if skew_value is not None and skew_value > 0.5
    else "asimetría moderada, sin sesgo direccional fuerte"
)
dispersion_text = (
    "concentrados alrededor del centro"
    if vol_ret < 0.01
    else "con dispersión moderada alrededor del centro"
    if vol_ret < 0.02
    else "ampliamente dispersos, con variaciones diarias intensas"
)
outlier_text = (
    "no se observan outliers bajo el criterio IQR"
    if outlier_count == 0
    else f"aparecen {outlier_count} observaciones atípicas bajo el criterio IQR"
)

st.caption(
    f"Se observa una distribución {dispersion_text}, con {skew_distribution_text} y donde {outlier_text}."
)

with st.expander("Interpretación del histograma y boxplot"):
    st.markdown(
        f"""
        - **Dispersión:** la volatilidad diaria de **{vol_ret:.4%}** sugiere rendimientos **{dispersion_text}**.
        - **Asimetría:** el valor de **{"Sin datos" if skew_value is None else f"{skew_value:.2f}"}** apunta a **{skew_distribution_text}**.
        - **Extremos:** el boxplot indica que **{outlier_text}**, coherente con un mínimo de **{min_ret:.4%}** y un máximo de **{max_ret:.4%}**.
        - **Lectura de riesgo:** una distribución con más dispersión o con extremos visibles puede subestimar riesgo si se resume solo con media y desviación estándar.
        """
    )

# ==============================
# Gráfico Q-Q
# ==============================
st.markdown("### Gráfico Q-Q")
section_intro(
    "Contraste visual con normalidad",
    "El gráfico Q-Q compara cuantiles estandarizados para verificar si la serie sigue la forma esperada bajo normalidad.",
)

qq_fig = plot_qq(qq_df)
qq_fig.update_yaxes(scaleanchor="x", scaleratio=1)
qq_fig.update_layout(height=540)
st.plotly_chart(qq_fig, width="stretch")

st.caption("El Q-Q plot permite contrastar visualmente la normalidad, especialmente en las colas.")

with st.expander("Interpretación del gráfico Q-Q"):
    st.markdown(
        f"""
        - Si los puntos siguen la diagonal, la serie se aproxima a una normal; desviaciones marcadas en las colas sugieren no normalidad.
        - Este contraste visual debe leerse junto con **Jarque-Bera = {jb_decision.lower()}** y **Shapiro-Wilk = {format_p_value(shapiro_p_value)}**.
        - Cuando las colas se apartan de la diagonal, la evidencia visual refuerza la presencia de eventos extremos y limita una aproximación normal simple para medir riesgo.
        """
    )

# ==============================
# Tablas principales
# ==============================
st.markdown("### Resumen estadístico")
section_intro(
    "Estadísticos y normalidad",
    "Estas tablas profundizan en las medidas descriptivas y en las pruebas usadas para evaluar normalidad.",
)

col1, col2 = st.columns(2)
with col1:
    st.markdown("#### Estadísticos descriptivos")
    st.dataframe(desc_df, width="stretch")
with col2:
    st.markdown("#### Pruebas de normalidad")
    st.dataframe(norm_display_df, width="stretch")

with st.expander("Interpretación de estadísticos y pruebas"):
    render_statistical_interpretation(
        mean_ret,
        vol_ret,
        skew_value,
        kurt_value,
        jb_p_value,
        shapiro_p_value,
    )

st.markdown("### Comparación de tipos de rendimiento")
section_intro(
    "Rendimiento simple vs. logarítmico",
    "La comparación es descriptiva y reutiliza directamente las columnas calculadas en la serie de rendimientos.",
)
st.dataframe(comparison_display_df, width="stretch", hide_index=True)

with st.expander("Interpretación de la comparación de rendimientos"):
    simple_mean = comparison_df.loc[comparison_df["tipo_rendimiento"] == "Simple", "media"].iloc[0]
    log_mean = comparison_df.loc[comparison_df["tipo_rendimiento"] == "Logarítmico", "media"].iloc[0]
    simple_vol = comparison_df.loc[comparison_df["tipo_rendimiento"] == "Simple", "volatilidad"].iloc[0]
    log_vol = comparison_df.loc[comparison_df["tipo_rendimiento"] == "Logarítmico", "volatilidad"].iloc[0]
    st.markdown(
        f"""
        - La comparación usa las columnas ya calculadas en `ret_df`: `simple_return` y `log_return`.
        - En esta muestra, la media simple es **{simple_mean:.4%}** y la media logarítmica es **{log_mean:.4%}**.
        - La volatilidad simple es **{simple_vol:.4%}** y la volatilidad logarítmica es **{log_vol:.4%}**.
        - Para el resto del módulo se mantiene el enfoque principal sobre **{return_type_label.lower()}**, que es la base de las pruebas y gráficos principales.
        """
    )

# ==============================
# Hechos estilizados
# ==============================
st.markdown("### Hechos estilizados")
section_intro(
    "Lectura empírica de la serie",
    "Se resume si la muestra exhibe rasgos frecuentes en retornos financieros, usando las estadísticas y extremos observados.",
)

rolling_vol = series.rolling(20).std()
high_vol_threshold = rolling_vol.quantile(0.75)
high_vol_days = rolling_vol > high_vol_threshold if pd.notna(high_vol_threshold) else pd.Series(False, index=rolling_vol.index)
high_vol_clusters = int((high_vol_days & high_vol_days.shift(1).fillna(False)).sum())
negative_extremes = int((series < lower_fence).sum())
positive_extremes = int((series > upper_fence).sum())
skew_metric_display = "Sin datos" if skew_value is None else f"{skew_value:.2f}"
heavy_tails_text = (
    f"La curtosis de {kurt_value:.2f} y Jarque-Bera con p-value {format_p_value(jb_p_value)} sugieren colas pesadas; hay señal cuando la curtosis supera 3 o cuando Jarque-Bera rechaza normalidad."
    if kurt_value is not None and jb_p_value is not None and (kurt_value > 3 or jb_p_value < 0.05)
    else "La muestra no muestra una señal fuerte de colas pesadas con las métricas disponibles."
)
volatility_clustering_text = (
    f"La volatilidad móvil de 20 periodos muestra {high_vol_clusters} coincidencias consecutivas por encima del percentil 75, lo que aporta evidencia descriptiva de agrupamiento de volatilidad."
    if high_vol_clusters > 0
    else "La volatilidad móvil de 20 periodos no muestra suficientes coincidencias consecutivas por encima del percentil 75 como para sugerir agrupamiento claro en esta muestra."
)
leverage_text = (
    f"La asimetría de {skew_metric_display}, el mínimo de {min_ret:.4%} y {negative_extremes} outliers negativos frente a {positive_extremes} positivos sugieren una lectura exploratoria compatible con efecto apalancamiento."
    if skew_value is not None and skew_value < 0 and abs(min_ret) > abs(max_ret)
    else f"La asimetría de {skew_metric_display}, el mínimo de {min_ret:.4%} y el máximo de {max_ret:.4%} no bastan para afirmar un efecto apalancamiento; la lectura es exploratoria."
)

heavy_tails_detected = bool(
    kurt_value is not None
    and jb_p_value is not None
    and (kurt_value > 3 or jb_p_value < 0.05)
)
leverage_signal = bool(
    skew_value is not None and skew_value < 0 and abs(min_ret) > abs(max_ret)
)

stylized_cards = [
    {
        "title": "Colas pesadas",
        "status": "Señal detectada" if heavy_tails_detected else "Sin señal fuerte",
        "summary": (
            "Curtosis elevada o rechazo de normalidad sugieren eventos extremos más frecuentes."
            if heavy_tails_detected
            else "La muestra no muestra una desviación fuerte frente a una normal en esta lectura."
        ),
    },
    {
        "title": "Agrupamiento de volatilidad",
        "status": "Señal descriptiva" if high_vol_clusters > 0 else "Sin señal clara",
        "summary": (
            "La volatilidad móvil alta aparece en días consecutivos y apunta a persistencia temporal."
            if high_vol_clusters > 0
            else "La volatilidad móvil no muestra persistencia alta suficientemente clara en la muestra."
        ),
    },
    {
        "title": "Efecto apalancamiento",
        "status": "Señal exploratoria" if leverage_signal else "No concluyente",
        "summary": (
            "Los extremos negativos dominan la lectura y son compatibles con mayor sensibilidad al riesgo bajista."
            if leverage_signal
            else "La asimetría y los extremos observados no alcanzan para una lectura concluyente."
        ),
    },
]

card_columns = st.columns(3)
for column, card in zip(card_columns, stylized_cards):
    with column:
        st.markdown(
            f"""
            <div style="
                background: linear-gradient(180deg, #eff6ff 0%, #dbeafe 100%);
                border: 1px solid rgba(37, 99, 235, 0.20);
                border-radius: 18px;
                padding: 18px 16px;
                box-shadow: 0 6px 18px rgba(37, 99, 235, 0.10);
                min-height: 170px;
                margin-bottom: 0.5rem;
            ">
                <div style="font-size: 0.86rem; font-weight: 700; color: #1e3a8a; margin-bottom: 0.45rem;">
                    {card["title"]}
                </div>
                <div style="
                    display: inline-block;
                    font-size: 0.78rem;
                    font-weight: 700;
                    color: #1e40af;
                    background: rgba(255, 255, 255, 0.7);
                    border-radius: 999px;
                    padding: 0.28rem 0.55rem;
                    margin-bottom: 0.75rem;
                ">
                    {card["status"]}
                </div>
                <div style="font-size: 0.82rem; line-height: 1.45; color: #334155;">
                    {card["summary"]}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

with st.expander("Interpretación de hechos estilizados"):
    st.markdown(
        f"""
        - **Colas pesadas:** {heavy_tails_text}
        - **Agrupamiento de volatilidad:** {volatility_clustering_text} Esta lectura usa volatilidad móvil como evidencia descriptiva y no constituye una prueba formal ARCH/GARCH.
        - **Efecto apalancamiento:** {leverage_text} Como en este módulo no se estima una relación formal entre choques negativos y volatilidad futura, debe leerse como una señal exploratoria y no como prueba concluyente.
        """
    )

# ==============================
# Datos recientes
# ==============================
st.markdown("### Últimos rendimientos")
st.dataframe(ret_df.tail(15), width="stretch")

with st.expander("Interpretación de la tabla de rendimientos recientes"):
    st.markdown(
        """
        - La tabla permite contrastar observaciones recientes de `simple_return` y `log_return` sin recalcularlas manualmente.
        - Es útil para identificar si los episodios extremos del periodo también se concentran en las fechas más recientes.
        """
    )

# ==============================
# Conclusión
# ==============================
st.markdown("### Conclusión")

normality_conclusion = (
    "no hay suficientes datos para una decisión formal de normalidad"
    if jb_p_value is None
    else "se rechaza normalidad"
    if jb_p_value < 0.05
    else "no se rechaza normalidad"
)
tails_conclusion = (
    "no hay suficientes datos para evaluar colas"
    if kurt_value is None
    else "hay señal de colas pesadas"
    if kurt_value > 3
    else "no aparece una señal fuerte de colas pesadas"
)
skew_conclusion = (
    "no hay suficientes datos para evaluar asimetría"
    if skew_value is None
    else "predomina riesgo de cola izquierda"
    if skew_value < -0.5
    else "predomina sesgo positivo"
    if skew_value > 0.5
    else "la asimetría no muestra sesgo direccional fuerte"
)
var_conclusion = (
    "conviene contrastar el VaR paramétrico normal con métodos históricos o no normales."
    if (jb_p_value is not None and jb_p_value < 0.05) or (kurt_value is not None and kurt_value > 3)
    else "el VaR paramétrico puede usarse como referencia, pero debe validarse contra métodos históricos."
)
kurt_display = "sin datos" if kurt_value is None else f"{kurt_value:.2f}"
skew_display = "sin datos" if skew_value is None else f"{skew_value:.2f}"
st.info(
    f"""
    - **Normalidad:** {normality_conclusion} con Jarque-Bera (p-value {format_p_value(jb_p_value)}).
    - **Colas:** {tails_conclusion} (curtosis {kurt_display}).
    - **Asimetría:** {skew_conclusion} (asimetría {skew_display}).
    - **Puente a VaR:** {var_conclusion}
    """
)
