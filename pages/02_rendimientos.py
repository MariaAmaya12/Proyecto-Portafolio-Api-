import streamlit as st
import pandas as pd

from src.config import ASSETS, DEFAULT_START_DATE, DEFAULT_END_DATE, get_ticker, ensure_project_dirs
from src.download import data_error_message
from src.returns_analysis import (
    compute_return_series,
    descriptive_stats,
    normality_tests,
    qq_plot_data,
)
from src.services.market_data_client import MarketDataClient
from src.plots import plot_histogram_with_normal, plot_qq, plot_box
from src.ui_components import kpi_card, section_intro
from src.ui_layout import configured_assets, configured_period, module_params, render_app_shell, render_selected_asset_card
from src.ui_style import apply_global_typography

ensure_project_dirs()
apply_global_typography()



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
        "sin dato suficiente para evaluar asimetria"
        if skew_value is None
        else "negativa, asociada a cola izquierda y mayor atencion a perdidas extremas"
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
        else "sin senal fuerte de colas pesadas bajo el umbral usado"
    )
    jb_text = (
        "sin decision formal por falta de datos"
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
        "conviene contrastar el VaR parametrico normal con metodos historicos o enfoques menos dependientes de normalidad"
        if (jb_p_value is not None and jb_p_value < 0.05) or (kurt_value is not None and kurt_value > 3)
        else "el VaR parametrico puede servir como referencia, pero debe validarse frente a metodos historicos"
    )
    skew_display = "sin datos" if skew_value is None else f"{skew_value:.2f}"
    kurt_display = "sin datos" if kurt_value is None else f"{kurt_value:.2f}"
    st.info(
        f"""
        **Lectura estadistica integrada**

        - **Media:** {mean_ret:.4%}, {mean_text}.
        - **Volatilidad:** {vol_ret:.4%}, mide la dispersion diaria observada.
        - **Asimetria:** {skew_display}; lectura {skew_text}.
        - **Curtosis:** {kurt_display}; lectura {kurt_text}.
        - **Normalidad:** Jarque-Bera {jb_text}; Shapiro-Wilk reporta {shapiro_text}.
        - **Implicacion para riesgo:** {risk_text}.
        """
    )


render_app_shell(
    "Módulo 2 - Rendimientos y propiedades empíricas",
    "Analiza la distribucion de rendimientos del activo y sus principales propiedades estadisticas.",
)
ASSETS = configured_assets(ASSETS)
horizonte, start_date, end_date = configured_period(DEFAULT_START_DATE, DEFAULT_END_DATE)
asset_name, ticker = render_selected_asset_card(ASSETS, key="m2_asset_selector")

# ==============================
# Parámetros del módulo
# ==============================
with module_params():
    st.caption("Este módulo usa el activo y horizonte definidos en la vista principal.")

return_type = "log_return"
return_type_label = "Rendimiento logaritmico"

# ==============================
# Datos
# ==============================
market_client = MarketDataClient()

bundle = market_client.fetch_bundle(
    tickers=[ticker],
    start=str(start_date),
    end=str(end_date),
)

ohlcv_map = bundle.get("ohlcv", {})
df = ohlcv_map.get(ticker)

if df is None:
    df = pd.DataFrame()

if df.empty:
    missing_tickers = market_client.missing_tickers(bundle)

    if ticker in missing_tickers:
        st.warning(
            f"No hay datos disponibles para {ticker} en el rango seleccionado. "
            "Prueba con un horizonte más amplio, por ejemplo 2 años."
        )
    else:
        st.error(
            data_error_message(
                "No se pudieron obtener datos desde el backend para el activo seleccionado."
            )
        )

    st.stop()

price_series = df["Adj Close"] if "Adj Close" in df.columns else df["Close"]
ret_df = compute_return_series(price_series)

if ret_df.empty or return_type not in ret_df.columns:
    st.error("No fue posible calcular la serie de rendimientos.")
    st.stop()

series = ret_df[return_type].dropna()

if series.empty:
    st.error("La serie de rendimientos esta vacia.")
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
            "tipo_rendimiento": "Logaritmico",
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
    rendimientos logaritmicos, estadisticos descriptivos, histograma, boxplot, Q-Q plot
    y pruebas de normalidad. El objetivo es identificar dispersion, asimetria, colas pesadas y
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
    "Resumen de la distribucion",
    "Aqui se sintetizan la rentabilidad media, la volatilidad y los extremos observados en la serie de rendimientos seleccionada.",
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
    prom_delta = "Sesgo debil"
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
        caption="Decision basada en Jarque-Bera al 5%",
    )

col4, col5 = st.columns(2)

with col4:
    kpi_card(
        "Minimo",
        f"{min_ret:.3%}",
        caption="Peor rendimiento observado en el periodo",
    )

with col5:
    kpi_card(
        "Maximo",
        f"{max_ret:.3%}",
        caption="Mejor rendimiento observado en el periodo",
    )

with st.expander("Interpretacion de KPIs"):
    st.markdown(
        f"""
        - **Promedio:** resume la direccion media de los rendimientos diarios. En **{asset_name}** es **{mean_ret:.4%}**;
          si esta cerca de cero, no hay un sesgo fuerte de rentabilidad diaria promedio.
        - **Volatilidad:** mide cuan dispersos son los rendimientos alrededor de su media. El valor actual es
          **{vol_ret:.4%}**, clasificado como **{vol_delta.lower()}**, y funciona como primera senal de incertidumbre.
        - **Normalidad (JB):** indica si la distribucion se parece a una normal simple. Una normalidad rechazada
          sugiere que una aproximacion normal puede no capturar bien colas, asimetrias o riesgo extremo.
        - **Minimo:** muestra el peor retorno observado del periodo (**{min_ret:.4%}**), util para dimensionar perdidas extremas historicas.
        - **Maximo:** muestra el mejor retorno observado del periodo (**{max_ret:.4%}**), util para comparar la amplitud de movimientos positivos.
        """
    )

# ==============================
# Graficos de distribución en tabs
# ==============================
st.markdown("### Distribución de rendimientos")
section_intro(
    "Forma, dispersión y contraste con normalidad",
    "Histograma, boxplot y Q-Q plot permiten visualizar la forma de la distribución, detectar extremos y evaluar la hipótesis de normalidad.",
)

skew_distribution_text = (
    "sesgo hacia perdidas extremas"
    if skew_value is not None and skew_value < -0.5
    else "sesgo hacia movimientos positivos extremos"
    if skew_value is not None and skew_value > 0.5
    else "asimetria moderada, sin sesgo direccional fuerte"
)
dispersion_text = (
    "concentrados alrededor del centro"
    if vol_ret < 0.01
    else "con dispersion moderada alrededor del centro"
    if vol_ret < 0.02
    else "ampliamente dispersos, con variaciones diarias intensas"
)
outlier_text = (
    "no se observan outliers bajo el criterio IQR"
    if outlier_count == 0
    else f"aparecen {outlier_count} observaciones atipicas bajo el criterio IQR"
)

tab_hist, tab_box, tab_qq = st.tabs(["Histograma", "Boxplot", "Q-Q Plot"])

with tab_hist:
    st.plotly_chart(plot_histogram_with_normal(series), width="stretch")
    st.caption(
        f"Se observa una distribucion {dispersion_text}, con {skew_distribution_text} y donde {outlier_text}."
    )
    with st.expander("Interpretacion del histograma"):
        st.markdown(
            f"""
        - **Dispersion:** la volatilidad diaria de **{vol_ret:.4%}** sugiere rendimientos **{dispersion_text}**.
        - **Asimetria:** el valor de **{"Sin datos" if skew_value is None else f"{skew_value:.2f}"}** apunta a **{skew_distribution_text}**.
        - **Lectura de riesgo:** una distribución con más dispersión puede subestimar riesgo si se resume solo con media y desviación estándar.
            """
        )

with tab_box:
    st.plotly_chart(plot_box(series), width="stretch")
    st.caption(f"El boxplot muestra que {outlier_text}, con mínimo de {min_ret:.4%} y máximo de {max_ret:.4%}.")
    with st.expander("Interpretacion del boxplot"):
        st.markdown(
            f"""
        - **Extremos:** el boxplot indica que **{outlier_text}**, coherente con un minimo de **{min_ret:.4%}** y un maximo de **{max_ret:.4%}**.
        - **Asimetria:** el valor de **{"Sin datos" if skew_value is None else f"{skew_value:.2f}"}** apunta a **{skew_distribution_text}**.
        - **Lectura de riesgo:** observaciones atípicas visibles en el boxplot señalan eventos extremos que el VaR paramétrico normal puede subestimar.
            """
        )

with tab_qq:
    qq_fig = plot_qq(qq_df)
    qq_fig.update_yaxes(scaleanchor="x", scaleratio=1)
    qq_fig.update_layout(height=480, margin=dict(l=48, r=28, t=58, b=46))
    st.plotly_chart(qq_fig, width="stretch")
    st.caption("El Q-Q plot permite contrastar visualmente la normalidad, especialmente en las colas.")
    with st.expander("Interpretacion del grafico Q-Q"):
        st.markdown(
            f"""
        - Si los puntos siguen la diagonal, la serie se aproxima a una normal; diferencias marcadas en las colas sugieren no normalidad.
        - Este contraste visual debe leerse junto con **Jarque-Bera = {jb_decision.lower()}** y **Shapiro-Wilk = {format_p_value(shapiro_p_value)}**.
        - Cuando las colas se apartan de la diagonal, la evidencia visual refuerza la presencia de eventos extremos y limita una aproximacion normal simple para medir riesgo.
            """
        )

# ==============================
# Tablas principales
# ==============================
st.markdown("### Resumen estadistico")
section_intro(
    "Estadisticos y normalidad",
    "Estas tablas profundizan en las medidas descriptivas y en las pruebas usadas para evaluar normalidad.",
)

col1, col2 = st.columns(2)
with col1:
    st.markdown("#### Estadisticos descriptivos")
    st.dataframe(desc_df, width="stretch")
with col2:
    st.markdown("#### Pruebas de normalidad")
    st.dataframe(norm_display_df, width="stretch")

with st.expander("Interpretacion de estadisticos y pruebas"):
    render_statistical_interpretation(
        mean_ret,
        vol_ret,
        skew_value,
        kurt_value,
        jb_p_value,
        shapiro_p_value,
    )

st.markdown("### Comparacion de tipos de rendimiento")
section_intro(
    "Rendimiento simple vs. logaritmico",
    "La comparacion es descriptiva y reutiliza directamente las columnas calculadas en la serie de rendimientos.",
)
st.dataframe(comparison_display_df, width="stretch", hide_index=True)

with st.expander("Interpretacion de la comparacion de rendimientos"):
    simple_mean = comparison_df.loc[comparison_df["tipo_rendimiento"] == "Simple", "media"].iloc[0]
    log_mean = comparison_df.loc[comparison_df["tipo_rendimiento"] == "Logaritmico", "media"].iloc[0]
    simple_vol = comparison_df.loc[comparison_df["tipo_rendimiento"] == "Simple", "volatilidad"].iloc[0]
    log_vol = comparison_df.loc[comparison_df["tipo_rendimiento"] == "Logaritmico", "volatilidad"].iloc[0]
    st.markdown(
        f"""
        - La comparacion usa las columnas ya calculadas en `ret_df`: `simple_return` y `log_return`.
        - En esta muestra, la media simple es **{simple_mean:.4%}** y la media logaritmica es **{log_mean:.4%}**.
        - La volatilidad simple es **{simple_vol:.4%}** y la volatilidad logaritmica es **{log_vol:.4%}**.
        - Para el resto del módulo se mantiene el enfoque principal sobre **{return_type_label.lower()}**, que es la base de las pruebas y gráficos principales.
        """
    )

# ==============================
# Hechos estilizados
# ==============================
st.markdown("### Hechos estilizados")
section_intro(
    "Lectura empirica de la serie",
    "Se resume si la muestra exhibe rasgos frecuentes en retornos financieros, usando las estadisticas y extremos observados.",
)

rolling_vol = series.rolling(20).std()
high_vol_threshold = rolling_vol.quantile(0.75)
high_vol_days = rolling_vol > high_vol_threshold if pd.notna(high_vol_threshold) else pd.Series(False, index=rolling_vol.index)
high_vol_clusters = int((high_vol_days & high_vol_days.shift(1).fillna(False)).sum())
negative_extremes = int((series < lower_fence).sum())
positive_extremes = int((series > upper_fence).sum())
skew_metric_display = "Sin datos" if skew_value is None else f"{skew_value:.2f}"
heavy_tails_text = (
    f"La curtosis de {kurt_value:.2f} y Jarque-Bera con p-value {format_p_value(jb_p_value)} sugieren colas pesadas; hay senal cuando la curtosis supera 3 o cuando Jarque-Bera rechaza normalidad."
    if kurt_value is not None and jb_p_value is not None and (kurt_value > 3 or jb_p_value < 0.05)
    else "La muestra no muestra una señal fuerte de colas pesadas con las métricas disponibles."
)
volatility_clustering_text = (
    f"La volatilidad movil de 20 periodos muestra {high_vol_clusters} coincidencias consecutivas por encima del percentil 75, lo que aporta un indicio descriptivo de agrupamiento de volatilidad."
    if high_vol_clusters > 0
    else "La volatilidad movil de 20 periodos no muestra suficientes coincidencias consecutivas por encima del percentil 75 como para sugerir un indicio claro de agrupamiento en esta muestra."
)
leverage_text = (
    f"La asimetria de {skew_metric_display}, el minimo de {min_ret:.4%} y {negative_extremes} outliers negativos frente a {positive_extremes} positivos son compatibles con riesgo bajista. Esta lectura es exploratoria y no permite afirmar efecto apalancamiento de forma concluyente."
    if skew_value is not None and skew_value < 0 and abs(min_ret) > abs(max_ret)
    else f"La asimetria de {skew_metric_display}, el minimo de {min_ret:.4%}, el maximo de {max_ret:.4%} y los extremos observados no bastan para afirmar efecto apalancamiento; la lectura se limita a una evaluacion descriptiva de riesgo bajista."
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
        "status": "Senal detectada" if heavy_tails_detected else "Sin senal fuerte",
        "summary": (
            "Curtosis elevada o rechazo de normalidad sugieren eventos extremos mas frecuentes."
            if heavy_tails_detected
            else "La muestra no muestra una desviación fuerte frente a una normal en esta lectura."
        ),
    },
    {
        "title": "Indicio de agrupamiento de volatilidad",
        "status": "Senal descriptiva" if high_vol_clusters > 0 else "Sin senal clara",
        "summary": (
            "La volatilidad movil alta aparece en dias consecutivos y apunta a persistencia temporal descriptiva."
            if high_vol_clusters > 0
            else "La volatilidad movil no muestra persistencia alta suficientemente clara en la muestra."
        ),
    },
    {
        "title": "Riesgo bajista / posible efecto apalancamiento",
        "status": "Senal exploratoria" if leverage_signal else "No concluyente",
        "summary": (
            "Los extremos negativos dominan la lectura y son compatibles con riesgo bajista, sin constituir prueba formal."
            if leverage_signal
            else "La asimetria y los extremos observados no alcanzan para una lectura concluyente."
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

with st.expander("Interpretacion de hechos estilizados"):
    st.markdown(
        f"""
        - **Colas pesadas:** {heavy_tails_text}
        - **Indicio de agrupamiento de volatilidad:** {volatility_clustering_text} Esta lectura usa volatilidad movil como evidencia descriptiva y no constituye una prueba formal ARCH/GARCH.
        - **Riesgo bajista / posible efecto apalancamiento:** {leverage_text} Una prueba formal requeriria modelos como **EGARCH** o **GJR-GARCH**, o estimar explicitamente la relacion entre choques negativos y volatilidad futura.
        """
    )

# ==============================
# Datos recientes
# ==============================
st.markdown("### Últimos rendimientos")
st.dataframe(ret_df.tail(15), width="stretch")

with st.expander("Interpretacion de la tabla de rendimientos recientes"):
    st.markdown(
        """
        - La tabla permite contrastar observaciones recientes de `simple_return` y `log_return` sin recalcularlas manualmente.
        - Es util para identificar si los episodios extremos del periodo tambien se concentran en las fechas mas recientes.
        """
    )

# ==============================
# Conclusión del módulo
# ==============================
st.markdown("### Conclusión del módulo")

interpretacion_media = (
    "cercana a cero"
    if abs(mean_ret) < 0.0005
    else "positiva"
    if mean_ret > 0
    else "negativa"
)
interpretacion_volatilidad = (
    "baja"
    if vol_ret < 0.01
    else "moderada"
    if vol_ret < 0.02
    else "alta"
)
resultado_normalidad = (
    "no ofrecen suficientes datos para una decision formal de normalidad"
    if jb_p_value is None
    else f"rechazan normalidad bajo Jarque-Bera al 5% (p-value {format_p_value(jb_p_value)})"
    if jb_p_value < 0.05
    else f"no rechazan normalidad bajo Jarque-Bera al 5% (p-value {format_p_value(jb_p_value)})"
)
interpretacion_normalidad = (
    "se aparta de una normal simple y puede concentrar riesgos en colas o extremos"
    if jb_p_value is not None and jb_p_value < 0.05
    else "no muestra evidencia estadistica fuerte contra normalidad en esta prueba, aunque sigue siendo necesario revisar colas y extremos"
    if jb_p_value is not None
    else "no puede contrastarse formalmente con los datos disponibles"
)
asimetria_conclusion = (
    "asimetria no disponible"
    if skew_value is None
    else f"asimetria negativa ({skew_value:.2f}), compatible con mayor peso relativo de perdidas extremas"
    if skew_value < -0.5
    else f"asimetria positiva ({skew_value:.2f}), compatible con mayor peso relativo de movimientos favorables extremos"
    if skew_value > 0.5
    else f"asimetria moderada ({skew_value:.2f}), sin sesgo direccional fuerte"
)
curtosis_conclusion = (
    "curtosis no disponible"
    if kurt_value is None
    else f"curtosis elevada ({kurt_value:.2f}), consistente con colas mas pesadas"
    if kurt_value > 3
    else f"curtosis de {kurt_value:.2f}, sin senal fuerte de colas pesadas bajo el umbral usado"
)
extremos_conclusion = (
    f"minimo observado de {min_ret:.4%}, maximo de {max_ret:.4%} y {outlier_count} observaciones atipicas bajo criterio IQR"
)
st.info(
    f"""
    En el periodo analizado, los rendimientos logaritmicos de **{asset_name} ({ticker})** presentan una media diaria **{interpretacion_media}** ({mean_ret:.4%}) y una volatilidad **{interpretacion_volatilidad}** ({vol_ret:.4%}). Las pruebas de normalidad **{resultado_normalidad}**, lo que indica que la distribucion **{interpretacion_normalidad}**.

    La **{asimetria_conclusion}**, la **{curtosis_conclusion}** y los valores extremos observados ({extremos_conclusion}) sugieren que el riesgo no debe resumirse únicamente con media y desviación estándar. Por esta razón, para la medición de riesgo en módulos posteriores, especialmente **VaR/CVaR** y **GARCH**, es recomendable complementar el análisis con enfoques empíricos y modelos de volatilidad.
    """
)

