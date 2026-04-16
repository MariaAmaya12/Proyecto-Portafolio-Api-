import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import pandas as pd

from src.config import ASSETS, DEFAULT_START_DATE, DEFAULT_END_DATE, ensure_project_dirs
from src.download import load_market_bundle
from src.markowitz import (
    simulate_portfolios,
    efficient_frontier,
    minimum_variance_portfolio,
    maximum_sharpe_portfolio,
    weights_table,
)
from src.plots import plot_correlation_heatmap, plot_frontier
from src.api.macro import macro_snapshot
from src.portfolio_optimization import optimize_target_return

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
            <div class="kpi-caption">{caption}</div>
        </div>
    </body>
    </html>
    """

    components.html(html, height=145)


def ensure_dataframe(obj):
    if isinstance(obj, pd.Series):
        return obj.to_frame().T
    if isinstance(obj, pd.DataFrame):
        return obj
    return pd.DataFrame([obj])


def safe_get_first(obj, key):
    try:
        if isinstance(obj, pd.Series):
            return obj.get(key, None)
        if isinstance(obj, pd.DataFrame):
            if key in obj.columns and not obj.empty:
                return obj[key].iloc[0]
        if isinstance(obj, dict):
            return obj.get(key, None)
    except Exception:
        return None
    return None


def format_weights_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    cols = {c.lower(): c for c in out.columns}

    activo_col = cols.get("activo", list(out.columns)[0])
    peso_col = cols.get("peso", list(out.columns)[1])

    out = out[[activo_col, peso_col]].copy()
    out.columns = ["Activo", "Peso"]
    out["Peso"] = pd.to_numeric(out["Peso"], errors="coerce")
    out["Participación"] = out["Peso"].map(lambda x: f"{x:.2%}" if pd.notna(x) else "N/D")
    out["Peso"] = out["Peso"].round(4)
    out = out.sort_values("Peso", ascending=False).reset_index(drop=True)
    return out


inject_kpi_cards_css()

st.title("Módulo 6 - Optimización de portafolio (Markowitz)")
st.caption("Explora portafolios eficientes, diversificación, relación riesgo-retorno y soluciones óptimas bajo Markowitz.")

# ==============================
# Sidebar
# ==============================
with st.sidebar:
    st.header("Parámetros de optimización")

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
        start_date = st.date_input("Fecha inicial", value=DEFAULT_START_DATE, key="mk_start")
        end_date = st.date_input("Fecha final", value=DEFAULT_END_DATE, key="mk_end")

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

    mostrar_detalle_correlacion = False
    mostrar_interpretacion_tecnica = False

    with st.expander("Filtros secundarios"):
        n_portfolios = st.slider(
            "Número de portafolios",
            min_value=5000,
            max_value=50000,
            value=10000,
            step=5000,
        )

        target_return = st.slider(
            "Retorno objetivo (%)",
            min_value=0.0,
            max_value=0.30,
            value=0.10,
            step=0.01,
        )

        if modo == "Estadístico":
            mostrar_detalle_correlacion = st.checkbox("Mostrar matriz de correlación tabular", value=False)
            mostrar_interpretacion_tecnica = st.checkbox("Mostrar interpretación técnica", value=True)

# ==============================
# Carga de datos
# ==============================
tickers = [meta["ticker"] for meta in ASSETS.values()]
bundle = load_market_bundle(tickers=tickers, start=str(start_date), end=str(end_date))

returns = (
    bundle["returns"]
    .replace([np.inf, -np.inf], np.nan)
    .dropna(how="any")
)

if returns.empty or returns.shape[0] < 2 or returns.shape[1] < 2:
    st.error("No hay suficientes datos de retornos alineados para ejecutar Markowitz.")
    st.write({
        "shape_returns": bundle["returns"].shape,
        "na_por_activo": bundle["returns"].isna().sum().to_dict(),
    })
    st.stop()

macro = macro_snapshot()
rf_annual = (
    macro["risk_free_rate_pct"] / 100
    if macro["risk_free_rate_pct"] == macro["risk_free_rate_pct"]
    else 0.03
)

# ==============================
# Simulación y soluciones óptimas
# ==============================
sim_df = simulate_portfolios(returns, rf_annual=rf_annual, n_portfolios=n_portfolios)

if sim_df.empty:
    st.error("La simulación de portafolios no generó resultados válidos.")
    st.write({
        "shape_returns_filtrado": returns.shape,
        "rf_annual": rf_annual,
        "n_portfolios": n_portfolios,
    })
    st.stop()

frontier_df = efficient_frontier(sim_df)
min_var = minimum_variance_portfolio(sim_df)
max_sharpe = maximum_sharpe_portfolio(sim_df)

if min_var is None or max_sharpe is None:
    st.error("No fue posible identificar los portafolios óptimos.")
    st.stop()

min_var_df = ensure_dataframe(min_var)
max_sharpe_df = ensure_dataframe(max_sharpe)

if min_var_df.empty or max_sharpe_df.empty:
    st.error("No fue posible identificar los portafolios óptimos.")
    st.stop()

min_var_weights_df = format_weights_df(weights_table(min_var))
max_sharpe_weights_df = format_weights_df(weights_table(max_sharpe))

# ==============================
# Resumen
# ==============================
st.markdown("### Resumen del módulo")
if modo == "General":
    st.write(
        """
        Este módulo construye múltiples combinaciones de portafolios para identificar aquellas que ofrecen
        una mejor relación entre **retorno esperado** y **riesgo**. Se resaltan el portafolio de
        **mínima varianza**, el de **máximo Sharpe** y una solución con **retorno objetivo**.
        """
    )
else:
    st.write(
        """
        Este módulo implementa el enfoque media-varianza de **Markowitz**, simulando portafolios factibles
        para aproximar la frontera eficiente, identificar el portafolio de **mínima varianza**, el de
        **máximo Sharpe** y resolver una optimización condicionada a un **retorno objetivo**.
        """
    )

st.caption(f"Periodo analizado: {start_date} a {end_date}")

# ==============================
# KPIs principales
# ==============================
st.markdown("### KPIs del módulo")
section_intro(
    "Resumen ejecutivo de optimización",
    "Aquí se resume el universo analizado, el tamaño de la simulación y las características centrales de las soluciones óptimas.",
)

n_assets = returns.shape[1]
n_obs = returns.shape[0]

min_var_return = safe_get_first(min_var, "return")
min_var_vol = safe_get_first(min_var, "volatility")
max_sharpe_return = safe_get_first(max_sharpe, "return")
max_sharpe_vol = safe_get_first(max_sharpe, "volatility")
max_sharpe_ratio = safe_get_first(max_sharpe, "sharpe")

c1, c2, c3, c4 = st.columns(4)

with c1:
    kpi_card(
        "Activos analizados",
        str(n_assets),
        caption="Número de activos incluidos en el universo",
    )

with c2:
    kpi_card(
        "Observaciones",
        str(n_obs),
        caption="Cantidad de retornos alineados utilizados",
    )

with c3:
    kpi_card(
        "Portafolios simulados",
        f"{n_portfolios:,}".replace(",", "."),
        caption="Combinaciones generadas para aproximar la frontera",
    )

with c4:
    kpi_card(
        "Tasa libre de riesgo",
        f"{rf_annual:.2%}",
        caption="Usada para el cálculo del ratio Sharpe",
    )

# ==============================
# Portafolios óptimos
# ==============================
st.markdown("### Portafolios destacados")
section_intro(
    "Soluciones óptimas del modelo",
    "Se comparan los portafolios más relevantes de la optimización: el más defensivo y el más eficiente.",
)

c5, c6, c7, c8 = st.columns(4)

with c5:
    kpi_card(
        "Retorno mín. varianza",
        f"{float(min_var_return):.2%}" if min_var_return is not None else "N/D",
        caption="Retorno esperado del portafolio más estable",
    )

with c6:
    kpi_card(
        "Volatilidad mín. varianza",
        f"{float(min_var_vol):.2%}" if min_var_vol is not None else "N/D",
        delta="Menor riesgo disponible",
        delta_type="pos",
        caption="Portafolio con la menor volatilidad estimada",
    )

with c7:
    kpi_card(
        "Retorno máx. Sharpe",
        f"{float(max_sharpe_return):.2%}" if max_sharpe_return is not None else "N/D",
        caption="Retorno esperado del portafolio más eficiente",
    )

with c8:
    kpi_card(
        "Sharpe máximo",
        f"{float(max_sharpe_ratio):.3f}" if max_sharpe_ratio is not None else "N/D",
        delta="Mejor eficiencia riesgo-retorno",
        delta_type="pos",
        caption="Portafolio con mayor ratio Sharpe",
    )

# ==============================
# Correlación
# ==============================
corr = returns.corr()

st.markdown("### Matriz de correlación")
section_intro(
    "Relación entre activos",
    "La matriz de correlación ayuda a entender el potencial de diversificación entre los activos del portafolio.",
)

st.plotly_chart(plot_correlation_heatmap(corr), width="stretch")

if modo == "General":
    st.info(
        """
        **Cómo leer esta matriz**

        - Correlaciones altas indican que dos activos tienden a moverse de forma parecida.
        - Correlaciones más bajas o moderadas ayudan a diversificar.
        - En Markowitz, la diversificación es clave para reducir volatilidad agregada.
        """
    )
else:
    if mostrar_detalle_correlacion:
        with st.expander("Ver matriz de correlación en tabla"):
            st.dataframe(corr.round(4), width="stretch")

        st.info(
            """
            En términos de media-varianza, la matriz de correlación es fundamental porque determina cuánto riesgo
            conjunto puede reducirse mediante diversificación. Correlaciones menores tienden a ampliar el espacio
            de portafolios eficientes.
            """
        )

# ==============================
# Frontera eficiente
# ==============================
st.markdown("### Frontera eficiente")
section_intro(
    "Relación riesgo-retorno",
    "El gráfico resume el espacio de portafolios posibles y destaca la frontera eficiente junto con las soluciones óptimas.",
)

st.plotly_chart(plot_frontier(sim_df, frontier_df, min_var, max_sharpe), width="stretch")

if modo == "General":
    st.info(
        """
        **Cómo leer este gráfico**

        - Cada punto representa un portafolio posible.
        - La frontera eficiente reúne las mejores combinaciones para cada nivel de riesgo.
        - El portafolio de mínima varianza minimiza volatilidad.
        - El portafolio de máximo Sharpe maximiza eficiencia entre retorno esperado y riesgo.
        """
    )
else:
    with st.expander("Ver interpretación técnica de la frontera eficiente"):
        st.write(
            """
            La nube de portafolios representa combinaciones factibles generadas por simulación. La frontera eficiente
            aproxima el conjunto de soluciones dominantes en el espacio media-varianza. El portafolio de mínima varianza
            resuelve el problema de minimización del riesgo, mientras que el de máximo Sharpe maximiza la pendiente de
            la línea de asignación de capital dada la tasa libre de riesgo.
            """
        )

# ==============================
# Pesos de portafolios óptimos
# ==============================
st.markdown("### Composición de portafolios óptimos")
section_intro(
    "Pesos recomendados",
    "Estas tablas muestran cómo se distribuye la participación de cada activo en las soluciones óptimas principales.",
)

col1, col2 = st.columns(2)

with col1:
    st.subheader("Portafolio de mínima varianza")
    st.dataframe(
        min_var_weights_df,
        width="stretch",
        hide_index=True,
    )

with col2:
    st.subheader("Portafolio de máximo Sharpe")
    st.dataframe(
        max_sharpe_weights_df,
        width="stretch",
        hide_index=True,
    )

if mostrar_tablas:
    with st.expander("Ver tablas completas de pesos y resultados"):
        st.markdown("#### Portafolio de mínima varianza")
        st.dataframe(min_var_df, width="stretch", hide_index=True)
        st.markdown("#### Portafolio de máximo Sharpe")
        st.dataframe(max_sharpe_df, width="stretch", hide_index=True)

# ==============================
# Optimización con retorno objetivo
# ==============================
st.markdown("### Optimización con retorno objetivo")
section_intro(
    "Solución condicionada",
    "Aquí se busca un portafolio que cumpla un retorno objetivo específico sujeto a las restricciones del modelo.",
)

result = optimize_target_return(returns, target_return)

if result is not None:
    target_delta = None
    target_delta_type = "neu"

    if result["return"] >= target_return:
        target_delta = "Objetivo alcanzado"
        target_delta_type = "pos"
    else:
        target_delta = "Cercano al objetivo"
        target_delta_type = "neu"

    col3, col4 = st.columns([1, 1.2])

    with col3:
        kpi_card(
            "Retorno esperado",
            f"{result['return']:.2%}",
            delta=target_delta,
            delta_type=target_delta_type,
            caption=f"Objetivo solicitado: {target_return:.2%}",
        )

        kpi_card(
            "Volatilidad",
            f"{result['volatility']:.2%}",
            caption="Riesgo estimado de la solución encontrada",
        )

    with col4:
        st.markdown("#### Pesos del portafolio objetivo")
        target_weights_df = pd.DataFrame(
            {
                "Activo": returns.columns,
                "Peso": np.round(result["weights"], 4),
            }
        )
        target_weights_df["Participación"] = target_weights_df["Peso"].map(lambda x: f"{x:.2%}")
        target_weights_df = target_weights_df.sort_values("Peso", ascending=False).reset_index(drop=True)

        st.dataframe(
            target_weights_df,
            width="stretch",
            hide_index=True,
        )

else:
    st.warning("No se pudo encontrar solución para ese nivel de retorno.")

# ==============================
# Interpretación
# ==============================
st.markdown("### Interpretación")

if modo == "General":
    st.success(
        """
        **Lectura sencilla**

        - Este módulo muestra que no existe una única mejor cartera: todo depende del equilibrio entre retorno y riesgo.
        - La frontera eficiente resume las combinaciones más convenientes.
        - El portafolio de mínima varianza prioriza estabilidad.
        - El portafolio de máximo Sharpe prioriza eficiencia.
        - El portafolio con retorno objetivo adapta la solución a una meta concreta.
        """
    )
else:
    if mostrar_interpretacion_tecnica:
        st.info(
            """
            **Interpretación técnica**

            - El enfoque de Markowitz modela el problema de asignación óptima en términos de media y varianza.
            - La frontera eficiente representa el conjunto de portafolios no dominados.
            - El portafolio de mínima varianza minimiza el riesgo total sujeto a las restricciones del problema.
            - El portafolio de máximo Sharpe maximiza el exceso de retorno por unidad de riesgo.
            - La solución con retorno objetivo impone una restricción adicional sobre la rentabilidad esperada, lo que puede incrementar la volatilidad necesaria para alcanzar dicha meta.
            """
        )