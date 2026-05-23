import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator

from src.config import ASSETS, DEFAULT_START_DATE, DEFAULT_END_DATE, ensure_project_dirs
from src.api.backend_client import BackendAPIError, friendly_error_message
from src.download import data_error_message
from src.plots import plot_correlation_heatmap, plot_frontier
from src.api.macro import macro_snapshot
from src.portfolio_optimization import optimize_target_return
from src.services.market_data_client import MarketDataClient
from src.services.portfolio_optimizer import PortfolioOptimizer
from src.ui_components import conclusion_box, kpi_card, module_header, section_intro, sanitize_text
from src.ui_layout import configured_assets, configured_period, module_params, render_app_shell, render_portfolio_summary_card
from src.ui_style import apply_global_typography

ensure_project_dirs()
apply_global_typography()


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


# ==============================
# Encabezado y configuración
# ==============================
render_app_shell(
    "Módulo 6 - Optimización de portafolio (Markowitz)",
    "Explora portafolios eficientes, diversificación, relación riesgo-retorno y soluciones óptimas bajo Markowitz.",
)
ASSETS = configured_assets(ASSETS)
horizonte, start_date, end_date = configured_period(DEFAULT_START_DATE, DEFAULT_END_DATE)
render_portfolio_summary_card(ASSETS)
module_header(
    "Módulo 6 – Optimización de portafolio Markowitz",
    "Construye portafolios eficientes, compara soluciones óptimas y evalúa la relación riesgo-retorno de los activos seleccionados.",
    badge="Frontera eficiente · Diversificación · Riesgo-retorno",
)

# ==============================
# Parámetros del módulo (sidebar)
# ==============================
with module_params():
    st.caption("Configura la simulación y el análisis de optimización.")

    n_portfolios = st.slider(
        "Número de portafolios simulados",
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
        format="%.2f",
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
market_client = MarketDataClient()
try:
    bundle = market_client.fetch_bundle(tickers=tickers, start=str(start_date), end=str(end_date))
except BackendAPIError as exc:
    st.error(friendly_error_message(exc, "No fue posible obtener datos de mercado desde el backend."))
    if exc.technical_detail:
        st.caption(exc.technical_detail)
    st.stop()

missing_tickers = market_client.missing_tickers(bundle)
if missing_tickers:
    st.warning(
        "Sin datos para estos tickers en el rango seleccionado; se excluyen de la optimización: "
        + ", ".join(missing_tickers)
    )

optimizer = PortfolioOptimizer()
returns = optimizer.prepare_returns(bundle["returns"])

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

# ==============================
# Portafolio manual (opcional)
# ==============================
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
        manual_portfolio = optimizer.manual_portfolio(returns, manual_weights, rf_annual)
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
sim_df = optimizer.simulate(returns, rf_annual=rf_annual, n_portfolios=n_portfolios)

if sim_df.empty:
    st.error("La simulación de portafolios no generó resultados válidos.")
    st.write({
        "shape_returns_filtrado": returns.shape,
        "rf_annual": rf_annual,
        "n_portfolios": n_portfolios,
    })
    st.stop()

frontier_df = optimizer.efficient_frontier(sim_df)
min_var, max_sharpe = optimizer.optimal_portfolios(sim_df)

if min_var is None or max_sharpe is None:
    st.error("No fue posible identificar los portafolios óptimos.")
    st.stop()

min_var_df = ensure_dataframe(min_var)
max_sharpe_df = ensure_dataframe(max_sharpe)

if min_var_df.empty or max_sharpe_df.empty:
    st.error("No fue posible identificar los portafolios óptimos.")
    st.stop()

min_var_weights_df = optimizer.weights_frame(min_var).rename(columns={"Participacion": "Participación"})
max_sharpe_weights_df = optimizer.weights_frame(max_sharpe).rename(columns={"Participacion": "Participación"})

# Ordenar por peso descendente para mejor lectura
min_var_weights_df = min_var_weights_df.sort_values("Peso", ascending=False).reset_index(drop=True)
max_sharpe_weights_df = max_sharpe_weights_df.sort_values("Peso", ascending=False).reset_index(drop=True)

# Asegurar columna Participación en formato porcentaje
if "Participación" not in min_var_weights_df.columns or min_var_weights_df["Participación"].dtype != object:
    min_var_weights_df["Participación"] = min_var_weights_df["Peso"].map(lambda x: f"{x:.2%}")
if "Participación" not in max_sharpe_weights_df.columns or max_sharpe_weights_df["Participación"].dtype != object:
    max_sharpe_weights_df["Participación"] = max_sharpe_weights_df["Peso"].map(lambda x: f"{x:.2%}")

n_assets = returns.shape[1]
n_obs = returns.shape[0]

min_var_return = safe_get_first(min_var, "return")
min_var_vol = safe_get_first(min_var, "volatility")
max_sharpe_return = safe_get_first(max_sharpe, "return")
max_sharpe_vol = safe_get_first(max_sharpe, "volatility")
max_sharpe_ratio = safe_get_first(max_sharpe, "sharpe")

# Portafolio con retorno objetivo (precomputado)
result = optimize_target_return(returns, target_return)

# Correlación (precomputada para Tab 4)
corr = returns.corr()

# Métricas de diversificación
_corr_vals = corr.values
_mask = ~np.eye(_corr_vals.shape[0], dtype=bool)
_off_diag = _corr_vals[_mask]
_corr_promedio = float(np.mean(_off_diag)) if len(_off_diag) > 0 else None
_corr_max = float(np.max(_off_diag)) if len(_off_diag) > 0 else None
_corr_min = float(np.min(_off_diag)) if len(_off_diag) > 0 else None

# Figura frontera eficiente
frontier_fig = prepare_frontier_figure(sim_df, frontier_df, min_var, max_sharpe, manual_portfolio)

# Textos para composición de portafolios
min_var_top = min_var_weights_df.head(2)
max_sharpe_top = max_sharpe_weights_df.head(2)
min_var_top_text = ", ".join(f"{row['Activo']} ({row['Participación']})" for _, row in min_var_top.iterrows())
max_sharpe_top_text = ", ".join(f"{row['Activo']} ({row['Participación']})" for _, row in max_sharpe_top.iterrows())

# ==============================
# PESTAÑAS PRINCIPALES
# ==============================
tab_resumen, tab_frontera, tab_composicion, tab_correlacion, tab_objetivo = st.tabs([
    "Resumen ejecutivo",
    "Frontera eficiente",
    "Composición de portafolios",
    "Diversificación y correlación",
    "Retorno objetivo",
])

# ==============================
# TAB 1 – Resumen ejecutivo
# ==============================
with tab_resumen:
    st.caption(f"Período analizado: {start_date} a {end_date} · {n_assets} activos · {n_portfolios:,} portafolios simulados")

    section_intro(
        "Optimización de portafolio bajo Markowitz",
        "Identifica combinaciones eficientes de activos, compara el portafolio de mínima varianza con el de máximo Sharpe y evalúa una solución condicionada a un retorno objetivo.",
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        kpi_card(
            "Retorno mín. varianza",
            f"{float(min_var_return):.2%}" if min_var_return is not None else "N/D",
            delta="Portafolio defensivo",
            delta_type="neu",
            caption="Retorno esperado del portafolio más estable.",
        )

    with c2:
        kpi_card(
            "Volatilidad mín. varianza",
            f"{float(min_var_vol):.2%}" if min_var_vol is not None else "N/D",
            delta="Menor riesgo disponible",
            delta_type="pos",
            caption="Riesgo estimado de la solución más estable.",
        )

    with c3:
        kpi_card(
            "Retorno máx. Sharpe",
            f"{float(max_sharpe_return):.2%}" if max_sharpe_return is not None else "N/D",
            delta="Mayor eficiencia",
            delta_type="pos",
            caption="Retorno esperado del portafolio más eficiente.",
        )

    with c4:
        kpi_card(
            "Sharpe máximo",
            f"{float(max_sharpe_ratio):.3f}" if max_sharpe_ratio is not None else "N/D",
            delta="Mejor riesgo-retorno",
            delta_type="pos",
            caption="Compensación por unidad de riesgo.",
        )

    conclusion_box(
        "El portafolio de mínima varianza prioriza estabilidad y reducción de riesgo, mientras que el portafolio de máximo Sharpe busca mayor eficiencia riesgo-retorno. "
        "La frontera eficiente permite visualizar las combinaciones posibles y la matriz de correlación ayuda a explicar el potencial de diversificación entre activos.",
        kind="success",
        label="Lectura rápida del módulo",
    )

    with st.expander("Parámetros de la simulación", expanded=False):
        section_intro(
            "Universo y configuración",
            "Tamaño de la simulación, activos incluidos y tasa libre de riesgo usada para el ratio Sharpe.",
        )
        _pc1, _pc2, _pc3, _pc4 = st.columns(4)
        with _pc1:
            kpi_card(
                "Activos analizados",
                str(n_assets),
                caption="Número de activos incluidos en el universo.",
            )
        with _pc2:
            kpi_card(
                "Observaciones",
                str(n_obs),
                caption="Cantidad de retornos alineados utilizados.",
            )
        with _pc3:
            kpi_card(
                "Portafolios simulados",
                f"{n_portfolios:,}".replace(",", "."),
                caption="Combinaciones generadas para aproximar la frontera.",
            )
        with _pc4:
            kpi_card(
                "Tasa libre de riesgo",
                f"{rf_annual:.2%}",
                caption="Usada para el cálculo del ratio Sharpe.",
            )

# ==============================
# TAB 2 – Frontera eficiente
# ==============================
with tab_frontera:
    section_intro(
        "Relación riesgo-retorno",
        "El gráfico resume el universo de portafolios simulados y destaca las soluciones óptimas: mínima varianza y máximo Sharpe.",
    )

    st.plotly_chart(frontier_fig, use_container_width=True)

    with st.expander("Cómo interpretar la frontera eficiente", expanded=False):
        st.write(
            """
            - **Cada punto** representa una combinación posible de pesos entre los activos seleccionados.
            - **La frontera eficiente** agrupa los portafolios con mejor retorno para cada nivel de riesgo.
            - **Mínima varianza:** reduce la volatilidad al mínimo posible dado el universo de activos.
            - **Máximo Sharpe:** maximiza la compensación por unidad de riesgo, usando la tasa libre de referencia.
            - **Escala de color:** representa el ratio Sharpe de los portafolios simulados; tonos más intensos indican mayor eficiencia.
            """
        )

# ==============================
# TAB 3 – Composición de portafolios
# ==============================
with tab_composicion:
    section_intro(
        "Composición de portafolios óptimos",
        "Distribución de pesos en los dos portafolios clave: el más defensivo y el más eficiente.",
    )

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Portafolio de mínima varianza")
        _display_mv = min_var_weights_df[["Activo", "Participación"]].copy()
        st.dataframe(_display_mv, use_container_width=True, hide_index=True)

        _mv_bar = go.Figure(go.Bar(
            x=min_var_weights_df["Activo"].tolist(),
            y=min_var_weights_df["Peso"].tolist(),
            marker_color="#2563eb",
            text=min_var_weights_df["Participación"].tolist(),
            textposition="outside",
        ))
        _mv_bar.update_layout(
            title="Mínima varianza — composición",
            yaxis_title="Peso",
            height=280,
            margin=dict(l=20, r=20, t=40, b=50),
            yaxis=dict(tickformat=".0%"),
        )
        st.plotly_chart(_mv_bar, use_container_width=True)

    with col2:
        st.subheader("Portafolio de máximo Sharpe")
        _display_ms = max_sharpe_weights_df[["Activo", "Participación"]].copy()
        st.dataframe(_display_ms, use_container_width=True, hide_index=True)

        _ms_bar = go.Figure(go.Bar(
            x=max_sharpe_weights_df["Activo"].tolist(),
            y=max_sharpe_weights_df["Peso"].tolist(),
            marker_color="#16a34a",
            text=max_sharpe_weights_df["Participación"].tolist(),
            textposition="outside",
        ))
        _ms_bar.update_layout(
            title="Máximo Sharpe — composición",
            yaxis_title="Peso",
            height=280,
            margin=dict(l=20, r=20, t=40, b=50),
            yaxis=dict(tickformat=".0%"),
        )
        st.plotly_chart(_ms_bar, use_container_width=True)

    with st.expander("Interpretación: composición de portafolios", expanded=False):
        st.write(
            f"""
            - **Mínima varianza:** los mayores pesos están en {min_var_top_text}. Este portafolio tiende a distribuir más el peso para reducir riesgo conjunto.
            - **Máximo Sharpe:** los mayores pesos están en {max_sharpe_top_text}. Puede concentrarse en activos con mejor relación retorno-riesgo histórica.
            - Una composición distribuida reduce la dependencia de activos específicos; una más concentrada puede mejorar una métrica objetivo, pero aumenta la sensibilidad a esos activos.
            """
        )

    with st.expander("Detalle completo de tablas", expanded=False):
        st.markdown("#### Portafolio de mínima varianza")
        st.dataframe(min_var_df, use_container_width=True, hide_index=True)
        st.markdown("#### Portafolio de máximo Sharpe")
        st.dataframe(max_sharpe_df, use_container_width=True, hide_index=True)

# ==============================
# TAB 4 – Diversificación y correlación
# ==============================
with tab_correlacion:
    section_intro(
        "Relación entre activos",
        "La matriz de correlación permite evaluar si los activos se mueven de forma similar o si aportan diversificación al portafolio.",
    )

    if _corr_promedio is not None:
        _dc1, _dc2, _dc3 = st.columns(3)
        with _dc1:
            _cp_delta = "pos" if _corr_promedio < 0.3 else ("neu" if _corr_promedio < 0.6 else "neg")
            kpi_card(
                "Correlación promedio",
                f"{_corr_promedio:.2f}",
                delta="Baja" if _corr_promedio < 0.3 else ("Moderada" if _corr_promedio < 0.6 else "Alta"),
                delta_type=_cp_delta,
                caption="Promedio de correlaciones entre todos los pares de activos.",
            )
        with _dc2:
            kpi_card(
                "Mayor correlación",
                f"{_corr_max:.2f}",
                delta="Par más correlacionado",
                delta_type="neg" if _corr_max > 0.7 else "neu",
                caption="Correlación más alta entre un par de activos.",
            )
        with _dc3:
            kpi_card(
                "Menor correlación",
                f"{_corr_min:.2f}",
                delta="Mayor potencial de diversificación" if _corr_min < 0 else "Baja correlación positiva",
                delta_type="pos" if _corr_min < 0.3 else "neu",
                caption="Correlación más baja (o negativa) entre un par de activos.",
            )

    st.plotly_chart(plot_correlation_heatmap(corr), use_container_width=True)

    with st.expander("Leyenda de la matriz de correlación", expanded=False):
        st.write(
            """
            - **Correlación cercana a 1:** los activos se mueven en la misma dirección. Poca diversificación.
            - **Correlación cercana a 0:** relación débil entre los activos. Aporte moderado de diversificación.
            - **Correlación negativa:** los activos tienden a moverse en direcciones opuestas. Posible beneficio de diversificación.
            - En Markowitz, combinar activos con correlaciones bajas o negativas puede reducir la volatilidad total del portafolio.
            - La escala de colores permite identificar rápidamente relaciones fuertes, débiles o inversas entre pares de activos.
            """
        )

    with st.expander("Tabla de correlaciones completa", expanded=False):
        st.dataframe(corr.round(2), use_container_width=True)

# ==============================
# TAB 5 – Retorno objetivo
# ==============================
with tab_objetivo:
    section_intro(
        "Solución condicionada",
        "Evalúa un portafolio que intenta alcanzar un retorno objetivo específico sujeto a las restricciones del modelo.",
    )

    if result is not None:
        target_delta = "Objetivo alcanzado" if result["return"] >= target_return else "Cercano al objetivo"
        target_delta_type = "pos" if result["return"] >= target_return else "neu"

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
                caption="Riesgo estimado de la solución encontrada.",
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
                target_weights_df[["Activo", "Participación"]],
                use_container_width=True,
                hide_index=True,
            )

        target_top = target_weights_df.head(2)
        target_top_text = ", ".join(f"{row['Activo']} ({row['Participación']})" for _, row in target_top.iterrows())

        with st.expander("Interpretación del portafolio con retorno objetivo", expanded=False):
            st.write(
                f"""
                - **Retorno esperado:** la solución alcanza {result['return']:.2%} frente al objetivo seleccionado de {target_return:.2%}.
                - **Volatilidad:** el riesgo anualizado de esta cartera es {result['volatility']:.2%}; ese es el costo de riesgo asociado a la meta elegida.
                - **Pesos:** los mayores pesos del portafolio objetivo están en {target_top_text}.
                - Permite evaluar una meta concreta de rentabilidad. Si el retorno objetivo es alto, puede implicar mayor volatilidad.
                - La solución depende de restricciones, retornos estimados y la matriz de covarianzas.
                - Esta solución no reemplaza al portafolio de mínima varianza ni al de máximo Sharpe; es un análisis complementario.
                """
            )

    else:
        st.warning("No se pudo encontrar solución para ese nivel de retorno objetivo.")
        st.info("Intenta reducir el retorno objetivo en el panel lateral o ampliar el período de análisis.")

    conclusion_box(
        "La frontera eficiente resume las combinaciones con mejor relación riesgo-retorno posibles para el universo de activos. "
        "El portafolio de mínima varianza prioriza estabilidad reduciendo el riesgo estimado; el de máximo Sharpe maximiza la compensación por unidad de riesgo. "
        "El retorno objetivo condiciona la optimización hacia una meta concreta de rentabilidad, que puede implicar mayor volatilidad.",
        kind="success",
        label="Conclusión del módulo",
    )
