import streamlit as st
import streamlit.components.v1 as components
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
                background: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%);
                border: 1px solid rgba(15, 23, 42, 0.08);
                border-radius: 18px;
                padding: 18px 18px 14px 18px;
                box-shadow: 0 4px 14px rgba(15, 23, 42, 0.06);
                min-height: 124px;
                box-sizing: border-box;
                display: flex;
                flex-direction: column;
                justify-content: space-between;
            }}

            .kpi-label {{
                font-size: 0.88rem;
                font-weight: 600;
                color: #475569;
                margin-bottom: 0.35rem;
                letter-spacing: 0.2px;
            }}

            .kpi-value {{
                font-size: 1.85rem;
                font-weight: 800;
                color: #0f172a;
                line-height: 1.1;
                margin-bottom: 0.45rem;
                word-break: break-word;
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
                font-size: 0.78rem;
                color: #64748b;
                margin-top: 0.65rem;
                line-height: 1.35;
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

    components.html(html, height=145)


inject_kpi_cards_css()

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

    mostrar_long_run = st.checkbox("Mostrar volatilidad de largo plazo", value=True)
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
section_intro(
    "Calidad de la serie de entrada",
    "Antes de ajustar el modelo, verificamos observaciones útiles, limpieza de datos y variabilidad mínima.",
)

n_original = validacion["resumen"].get("n_original", 0)
n_limpio = validacion["resumen"].get("n_limpio", 0)
std_val = validacion["resumen"].get("std", 0)

col1, col2, col3 = st.columns(3)

with col1:
    kpi_card(
        "Obs. originales",
        f"{n_original:,}".replace(",", "."),
        caption="Cantidad total de datos descargados",
    )

with col2:
    ratio_util = (n_limpio / n_original) if n_original else None
    kpi_card(
        "Obs. limpias",
        f"{n_limpio:,}".replace(",", "."),
        delta=f"{ratio_util:.1%} útiles" if ratio_util is not None else None,
        delta_type="pos" if ratio_util is not None and ratio_util >= 0.95 else "neu",
        caption="Datos válidos para estimar el modelo",
    )

with col3:
    kpi_card(
        "Desv. estándar",
        f"{std_val:.6f}",
        caption="Dispersión de los rendimientos logarítmicos",
    )

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
# Volatilidad de largo plazo
# ==============================
long_run_vol = None
persistence = None

try:
    best_model_name = results.get("best_model_name")
    comparison_df = results.get("comparison")

    if best_model_name is not None and comparison_df is not None and not comparison_df.empty:
        best_row = comparison_df.loc[comparison_df["modelo"] == best_model_name]

        if not best_row.empty:
            omega = pd.to_numeric(best_row["omega"], errors="coerce").iloc[0]
            alpha_1 = pd.to_numeric(best_row["alpha_1"], errors="coerce").iloc[0]

            beta_1 = None
            if "beta_1" in best_row.columns:
                beta_1 = pd.to_numeric(best_row["beta_1"], errors="coerce").iloc[0]

            if pd.notna(omega) and pd.notna(alpha_1):
                persistence = alpha_1 + beta_1 if pd.notna(beta_1) else alpha_1

                if persistence < 1:
                    long_run_var = omega / (1 - persistence)
                    if long_run_var > 0:
                        long_run_vol = long_run_var ** 0.5
except Exception:
    long_run_vol = None
    persistence = None

# ==============================
# KPIs
# ==============================
st.markdown("### KPIs del ajuste")
section_intro(
    "Resumen ejecutivo del modelo",
    "Aquí se condensan los resultados principales del ajuste para lectura rápida y presentación.",
)

best_model = results.get("best_model_name", None)
n_models = len(results["comparison"]) if "comparison" in results else 0

forecast_last = None
try:
    forecast_last = float(results["forecast"]["volatilidad_pronosticada"].iloc[-1])
except Exception:
    forecast_last = None

c1, c2, c3, c4 = st.columns(4)

with c1:
    kpi_card(
        "Activo",
        asset_name,
        caption=f"Ticker de referencia: {ticker}",
    )

with c2:
    kpi_card(
        "Modelos comparados",
        str(n_models),
        caption="Especificaciones evaluadas en el ajuste",
    )

with c3:
    kpi_card(
        "Mejor modelo",
        str(best_model) if best_model is not None else "N/D",
        caption="Selección según criterios de comparación",
    )

with c4:
    kpi_card(
        "Forecast final",
        f"{forecast_last:.4f}" if forecast_last is not None else "N/D",
        caption="Último valor pronosticado de volatilidad",
    )

if modo == "Estadístico":
    extra1, extra2 = st.columns(2)

    with extra1:
        if persistence is not None:
            if persistence >= 0.90:
                delta_text = "Alta persistencia"
                delta_type = "pos"
            elif persistence >= 0.75:
                delta_text = "Persistencia media"
                delta_type = "neu"
            else:
                delta_text = "Persistencia baja"
                delta_type = "neg"
        else:
            delta_text = None
            delta_type = "neu"

        kpi_card(
            "Persistencia",
            f"{persistence:.4f}" if persistence is not None else "N/D",
            delta=delta_text,
            delta_type=delta_type,
            caption="Memoria de los choques de volatilidad",
        )

    with extra2:
        kpi_card(
            "Volatilidad largo plazo",
            f"{long_run_vol:.4f}" if long_run_vol is not None else "N/D",
            caption="Nivel al que tendería la volatilidad si el modelo es estacionario",
        )

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
st.caption(
    "El forecast muestra la evolución esperada de la volatilidad para distintos horizontes. "
    "Opcionalmente se incluye la volatilidad de largo plazo del modelo."
)

st.markdown("### Volatilidad y pronóstico")

col1, col2 = st.columns(2)
with col1:
    st.plotly_chart(plot_volatility(results["volatility"]), width="stretch")
with col2:
    st.plotly_chart(
        plot_forecast(
            results["forecast"],
            long_run_vol=long_run_vol if mostrar_long_run else None,
        ),
        width="stretch",
    )

if modo == "General":
    st.info(
        """
        **Cómo leer la volatilidad condicional estimada**

        - El primer gráfico muestra cómo cambia la volatilidad estimada a lo largo del tiempo.
        - Valores altos indican mayor incertidumbre y, por tanto, mayor riesgo.

        **Cómo leer el pronóstico de volatilidad**

        - El horizonte representa el número de días hacia el futuro.
        - En el corto plazo, la volatilidad refleja choques recientes.
        - A medida que aumenta el horizonte, la volatilidad tiende a estabilizarse.
        - La línea punteada muestra la **volatilidad de largo plazo** del modelo.
        - Esto refleja el comportamiento típico de modelos GARCH (mean reversion).
        """
    )
else:
    with st.expander("Ver interpretación técnica de volatilidad y forecast"):
        st.write(
            """
            La serie de volatilidad condicional captura persistencia y clustering de la varianza,
            mientras que el forecast resume la trayectoria esperada de volatilidad bajo el modelo
            seleccionado. Esto permite comparar riesgo reciente y riesgo prospectivo.

            El horizonte indica el número de días hacia el futuro para los cuales se proyecta la volatilidad.
            Si el modelo es estacionario, el forecast tiende a converger hacia una volatilidad de largo plazo,
            reflejando reversión a la media en la dinámica GARCH.
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