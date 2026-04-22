import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator

from src.config import ASSETS, DEFAULT_START_DATE, DEFAULT_END_DATE, TRADING_DAYS, ensure_project_dirs
from src.download import data_error_message, load_market_bundle
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
                border: 1px solid rgba(37, 99, 235, 0.18);
                border-radius: 18px;
                box-shadow: 0 4px 14px rgba(30, 64, 175, 0.10);
                min-height: 195px;
                height: 195px;
                box-sizing: border-box;
                overflow: visible;
            }}

            .kpi-card-inner {{
                min-height: 195px;
                height: 195px;
                box-sizing: border-box;
                padding: 18px 18px 16px 18px;
                display: flex;
                flex-direction: column;
                justify-content: flex-start;
            }}

            .kpi-title,
            .kpi-label {{
                font-size: 0.88rem;
                font-weight: 600;
                color: #475569;
                margin-bottom: 0.35rem;
                letter-spacing: 0.2px;
                line-height: 1.22;
                min-height: 44px;
                overflow-wrap: anywhere;
                white-space: normal;
            }}

            .kpi-value {{
                font-size: 1.85rem;
                font-weight: 800;
                color: #0f172a;
                line-height: 1.1;
                margin-bottom: 0.45rem;
                overflow-wrap: anywhere;
                white-space: normal;
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

            .kpi-badge-slot {{
                min-height: 34px;
                display: flex;
                align-items: center;
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
                margin-top: auto;
                line-height: 1.35;
                overflow-wrap: anywhere;
                white-space: normal;
            }}
        </style>
    </head>
    <body>
        <div class="kpi-card">
            <div class="kpi-card-inner">
                <div class="kpi-title kpi-label">{title}</div>
                <div class="kpi-value">{value}</div>
                <div class="kpi-badge-slot">{delta_html}</div>
                <div class="kpi-caption">{caption}</div>
            </div>
        </div>
    </body>
    </html>
    """

    components.html(html, height=215)


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


class ModuleParams(BaseModel):
    n_portfolios: int = Field(ge=10000)
    target_return: float = Field(ge=0.03, le=0.20)
    evaluate_manual: bool = False


class ManualPortfolioInput(BaseModel):
    weights: list[float]

    @field_validator("weights")
    @classmethod
    def validate_weight_values(cls, weights):
        if not weights:
            raise ValueError("debe incluir al menos un peso")

        invalid = [weight for weight in weights if weight < 0 or weight > 1]
        if invalid:
            raise ValueError("todos los pesos deben estar entre 0 y 1")

        return weights

    @model_validator(mode="after")
    def validate_weight_sum(self):
        if abs(sum(self.weights) - 1.0) > 1e-6:
            raise ValueError("los pesos deben sumar 1")
        return self


def show_validation_error(title: str, exc: ValidationError):
    details = "; ".join(error["msg"] for error in exc.errors())
    st.error(f"{title}: {details}.")


def calculate_manual_portfolio(returns_df: pd.DataFrame, weights: np.ndarray, rf_annual: float) -> dict:
    mean_returns = returns_df.mean().values * TRADING_DAYS
    cov_matrix = returns_df.cov().values * TRADING_DAYS

    port_return = float(weights @ mean_returns)
    port_vol = float(np.sqrt(weights.T @ cov_matrix @ weights))
    sharpe = np.nan if port_vol <= 0 else float((port_return - rf_annual) / port_vol)

    return {
        "return": port_return,
        "volatility": port_vol,
        "sharpe": sharpe,
        "weights": weights,
    }


def prepare_frontier_figure(sim_df, frontier_df, min_var, max_sharpe, manual_portfolio=None):
    fig = plot_frontier(sim_df, frontier_df, min_var, max_sharpe)

    if fig.data:
        first_trace = fig.data[0]
        if hasattr(first_trace, "marker") and first_trace.marker:
            first_trace.marker.colorbar.update(
                x=1.08,
                y=0.54,
                len=0.62,
                thickness=12,
                title=dict(text="Sharpe", font=dict(size=11)),
                tickfont=dict(size=10),
            )

    if manual_portfolio is not None:
        fig.add_scatter(
            x=[manual_portfolio["volatility"]],
            y=[manual_portfolio["return"]],
            mode="markers",
            marker=dict(
                size=13,
                symbol="circle-open",
                color="#facc15",
                line=dict(color="#854d0e", width=2),
            ),
            name="Portafolio manual",
            hovertemplate=(
                "Portafolio manual<br>"
                "Volatilidad: %{x:.2%}<br>"
                "Retorno: %{y:.2%}<extra></extra>"
            ),
        )

    fig.update_layout(
        margin=dict(l=40, r=150, t=60, b=105),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.22,
            xanchor="center",
            x=0.5,
            bgcolor="rgba(0,0,0,0)",
            font=dict(size=10),
        ),
    )
    return fig


inject_kpi_cards_css()

render_page_title(
    "Módulo 6 - Optimización de portafolio (Markowitz)",
    "Explora portafolios eficientes, diversificación, relación riesgo-retorno y soluciones óptimas bajo Markowitz.",
)

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
        start_date = st.date_input("Fecha inicial", value=DEFAULT_START_DATE, key="mk_start")
        end_date = st.date_input("Fecha final", value=DEFAULT_END_DATE, key="mk_end")

    st.divider()
    n_portfolios = st.slider(
        "Número de portafolios",
        min_value=10000,
        max_value=50000,
        value=10000,
        step=1000,
    )

    target_return = st.slider(
        "Retorno objetivo (%)",
        min_value=0.03,
        max_value=0.20,
        value=0.10,
        step=0.01,
    )

    evaluar_manual = st.toggle("Evaluar portafolio manual", value=False)

if n_portfolios < 10000:
    st.warning("El número de portafolios no puede ser menor a 10.000. Se usará 10.000.")
    n_portfolios = 10000

try:
    params = ModuleParams(
        n_portfolios=int(n_portfolios),
        target_return=float(target_return),
        evaluate_manual=bool(evaluar_manual),
    )
except ValidationError as exc:
    show_validation_error("Parámetros inválidos del módulo", exc)
    st.stop()

n_portfolios = params.n_portfolios
target_return = params.target_return
evaluar_manual = params.evaluate_manual

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
    st.error(data_error_message("No hay suficientes datos de retornos alineados para ejecutar Markowitz."))
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

manual_portfolio = None
manual_weights_df = None

if evaluar_manual:
    with st.expander("Portafolio manual (opcional)", expanded=True):
        st.caption("Ingresa pesos en formato decimal. Ejemplo: 0.25 equivale a 25%. La participación se deriva automáticamente.")

        default_weight = 1 / len(returns.columns)
        manual_weights = []
        weight_cols = st.columns(min(3, len(returns.columns)))

        for idx, asset in enumerate(returns.columns):
            with weight_cols[idx % len(weight_cols)]:
                manual_weights.append(
                    st.number_input(
                        f"Peso {asset}",
                        min_value=0.0,
                        max_value=1.0,
                        value=float(default_weight),
                        step=0.01,
                        format="%.6f",
                        key=f"manual_weight_{asset}",
                    )
                )

        manual_weights = np.array(manual_weights, dtype=float)
        manual_weight_sum = float(manual_weights.sum())

        try:
            manual_input = ManualPortfolioInput(weights=manual_weights.tolist())
        except ValidationError:
            st.error(f"Suma de pesos: {manual_weight_sum:.6f} - Error. Los pesos deben estar entre 0 y 1 y sumar 1.")
            st.stop()

        st.success(f"Suma de pesos: {manual_weight_sum:.6f} - OK")
        manual_weights = np.array(manual_input.weights, dtype=float)
        manual_portfolio = calculate_manual_portfolio(returns, manual_weights, rf_annual)
        manual_weights_df = pd.DataFrame(
            {
                "Activo": returns.columns,
                "Peso": np.round(manual_weights, 6),
            }
        )
        manual_weights_df["Participación"] = manual_weights_df["Peso"].map(lambda x: f"{x:.2%}")

        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric("Retorno esperado", f"{manual_portfolio['return']:.2%}")
        with m2:
            st.metric("Volatilidad", f"{manual_portfolio['volatility']:.2%}")
        with m3:
            st.metric(
                "Sharpe",
                f"{manual_portfolio['sharpe']:.3f}" if np.isfinite(manual_portfolio["sharpe"]) else "N/D",
            )

        st.dataframe(manual_weights_df, width="stretch", hide_index=True)

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
st.write(
    """
    Este módulo construye múltiples combinaciones de portafolios para identificar alternativas eficientes
    entre **retorno esperado** y **riesgo**. Se resaltan el portafolio de **mínima varianza**, el de
    **máximo Sharpe** y una solución condicionada por un **retorno objetivo**.
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

with st.expander("Interpretación (KPIs del módulo)"):
    st.write(
        f"""
        - **Activos analizados:** se optimiza sobre {n_assets} activos; este universo define cuántas piezas tiene disponible el modelo para diversificar.
        - **Observaciones:** se usan {n_obs} retornos alineados para estimar retornos esperados, volatilidades y correlaciones.
        - **Portafolios simulados:** se evalúan {n_portfolios:,} combinaciones aleatorias; el control impide bajar de 10.000 simulaciones.
        - **Tasa libre de riesgo:** la referencia anual es {rf_annual:.2%}; se usa para calcular el exceso de retorno en el ratio Sharpe.
        """
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

with st.expander("Interpretación (portafolios destacados)", expanded=False):
    st.write(
        f"""
        - **Mínima varianza:** ofrece un retorno esperado de {float(min_var_return):.2%} con volatilidad de {float(min_var_vol):.2%}; es la opción más defensiva dentro de la simulación.
        - **Máximo Sharpe:** ofrece un retorno esperado de {float(max_sharpe_return):.2%} con Sharpe de {float(max_sharpe_ratio):.3f}; es la mejor eficiencia riesgo-retorno bajo la tasa libre usada.
        - **Lectura conjunta:** mínima varianza prioriza estabilidad, mientras que máximo Sharpe prioriza compensación por unidad de riesgo.
        """
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

with st.expander("Leyenda de la matriz de correlación"):
    st.write(
        """
        - Una correlación alta y positiva indica que dos activos tienden a moverse en la misma dirección.
        - Una correlación negativa indica que suelen moverse en direcciones opuestas, lo que puede reducir el riesgo conjunto.
        - Una correlación cercana a 0 sugiere poca relación lineal entre sus movimientos.
        - En Markowitz, combinar activos con correlaciones bajas o negativas ayuda a diversificar y puede disminuir la volatilidad del portafolio.
        - La escala de colores permite identificar rápidamente relaciones fuertes, débiles o inversas entre pares de activos.
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

frontier_fig = prepare_frontier_figure(sim_df, frontier_df, min_var, max_sharpe, manual_portfolio)
st.plotly_chart(frontier_fig, width="stretch")

with st.expander("Interpretación: frontera eficiente"):
    st.write(
        """
        - Portafolios: la nube de puntos representa combinaciones simuladas de pesos entre los activos.
        - Frontera eficiente: la línea reúne portafolios dominantes, es decir, los que ofrecen mayor retorno para un nivel de riesgo comparable.
        - Mínima varianza: el marcador identifica el portafolio con menor volatilidad estimada.
        - Máximo Sharpe: el marcador identifica el portafolio con mejor exceso de retorno por unidad de riesgo.
        - Portafolio manual: si ingresas pesos válidos, aparece como un punto adicional para compararlo contra las soluciones del modelo.
        - Escala de color: representa el ratio Sharpe de los portafolios simulados; tonos más intensos indican mayor eficiencia riesgo-retorno.
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

min_var_top = min_var_weights_df.head(2)
max_sharpe_top = max_sharpe_weights_df.head(2)
min_var_top_text = ", ".join(f"{row['Activo']} ({row['Participación']})" for _, row in min_var_top.iterrows())
max_sharpe_top_text = ", ".join(f"{row['Activo']} ({row['Participación']})" for _, row in max_sharpe_top.iterrows())

with st.expander("Interpretación: composición de portafolios"):
    st.write(
        f"""
        - **Mínima varianza:** los mayores pesos están en {min_var_top_text}. Si uno o dos activos concentran gran parte del peso, la cartera gana estabilidad por esos activos pero reduce diversificación.
        - **Máximo Sharpe:** los mayores pesos están en {max_sharpe_top_text}. Esta asignación prioriza eficiencia riesgo-retorno, por lo que puede concentrarse más en activos con mejor compensación histórica.
        - Una composición más distribuida reduce dependencia de activos específicos; una más concentrada puede mejorar una métrica objetivo, pero aumenta sensibilidad a esos activos.
        """
    )

with st.expander("Ver detalles técnicos (tablas completas)"):
    st.markdown("#### Portafolio de mínima varianza")
    st.dataframe(min_var_df, width="stretch", hide_index=True)
    st.markdown("#### Portafolio de máximo Sharpe")
    st.dataframe(max_sharpe_df, width="stretch", hide_index=True)
    st.markdown("#### Matriz de correlación")
    st.dataframe(corr.round(4), width="stretch")

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

    target_top = target_weights_df.head(2)
    target_top_text = ", ".join(f"{row['Activo']} ({row['Participación']})" for _, row in target_top.iterrows())

    with st.expander("Interpretación del portafolio con retorno objetivo"):
        st.write(
            f"""
            - **Retorno esperado:** la solución alcanza {result['return']:.2%} frente al objetivo seleccionado de {target_return:.2%}.
            - **Volatilidad:** el riesgo anualizado de esta cartera es {result['volatility']:.2%}; ese es el costo de riesgo asociado a la meta elegida.
            - **Pesos:** los mayores pesos del portafolio objetivo están en {target_top_text}.
            - Esta solución sirve para analizar una meta específica de rentabilidad; no reemplaza al portafolio de mínima varianza ni al de máximo Sharpe.
            """
        )

else:
    st.warning("No se pudo encontrar solución para ese nivel de retorno.")

# ==============================
# Interpretación
# ==============================
st.markdown("### Interpretación")

st.success(
    """
    **Lectura sencilla**

    - La matriz de correlación muestra qué tan parecidos son los movimientos entre activos; relaciones bajas o negativas favorecen la diversificación.
    - La frontera eficiente resume las combinaciones que logran una mejor relación entre retorno esperado y volatilidad.
    - El portafolio de mínima varianza es la alternativa más defensiva porque prioriza reducir el riesgo estimado.
    - El portafolio de máximo Sharpe busca la mejor compensación entre retorno adicional y riesgo asumido.
    - El retorno objetivo agrega una meta concreta de rentabilidad y puede exigir aceptar más volatilidad para alcanzarla.
    - La decisión final depende de si se prefiere estabilidad, eficiencia riesgo-retorno o cumplir una meta específica.
    """
)

with st.expander("Ver interpretación técnica"):
    st.write(
        """
        - Markowitz estima portafolios en el espacio media-varianza usando retornos esperados, volatilidades y covarianzas.
        - La correlación entre activos determina cuánto riesgo conjunto puede reducirse mediante diversificación.
        - La frontera eficiente aproxima el conjunto de portafolios no dominados frente a las combinaciones simuladas.
        - El portafolio de mínima varianza minimiza la volatilidad total del portafolio bajo las restricciones disponibles.
        - El portafolio de máximo Sharpe maximiza el exceso de retorno por unidad de riesgo usando la tasa libre de riesgo.
        - La optimización con retorno objetivo impone una restricción adicional de rentabilidad; si la meta es exigente, el modelo puede requerir una asignación más riesgosa.
        """
    )
