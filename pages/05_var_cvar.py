import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import pandas as pd
from pydantic import BaseModel, ValidationError

try:
    from pydantic import model_validator

    PYDANTIC_V2 = True
except ImportError:
    from pydantic import root_validator

    PYDANTIC_V2 = False

from src.config import ASSETS, DEFAULT_END_DATE, ensure_project_dirs
from src.api.backend_client import BackendAPIError, friendly_error_message
from src.download import data_error_message
from src.risk_metrics import kupiec_test
from src.services.market_data_client import MarketDataClient
from src.services.risk_analyzer import RiskAnalyzer
from src.plots import plot_var_distribution
from src.ui_components import conclusion_box, kpi_card, module_header, section_intro, sanitize_text
from src.ui_layout import configured_assets, configured_period, module_params, render_app_shell, render_portfolio_summary_card
from src.ui_style import apply_global_typography

ensure_project_dirs()
apply_global_typography()

CONFIDENCE_LEVELS = [0.95, 0.99]
WEIGHT_TOL = 1e-6


def _validate_weights_dict(pesos: dict[str, float], tol: float = WEIGHT_TOL) -> None:
    if not pesos:
        raise ValueError("Debe ingresar al menos un peso.")
    if any(weight < 0 or weight > 1 for weight in pesos.values()):
        raise ValueError("Todos los pesos deben estar entre 0 y 1.")
    if abs(sum(pesos.values()) - 1.0) > tol:
        raise ValueError("La suma de pesos debe ser 1.00.")


def _validate_n_sim(n_sim: int) -> None:
    if int(n_sim) < 10_000:
        raise ValueError("Monte Carlo requiere al menos 10,000 simulaciones.")


if PYDANTIC_V2:

    class PortfolioWeightsModel(BaseModel):
        pesos: dict[str, float]

        @model_validator(mode="after")
        def validate_weights(self):
            _validate_weights_dict(self.pesos)
            return self

    class SimulationConfigModel(BaseModel):
        n_sim: int

        @model_validator(mode="after")
        def validate_simulations(self):
            _validate_n_sim(self.n_sim)
            return self

else:

    class PortfolioWeightsModel(BaseModel):
        pesos: dict[str, float]

        @root_validator
        def validate_weights(cls, values):
            _validate_weights_dict(values.get("pesos") or {})
            return values

    class SimulationConfigModel(BaseModel):
        n_sim: int

        @root_validator
        def validate_simulations(cls, values):
            _validate_n_sim(values.get("n_sim"))
            return values


def style_risk_table(df: pd.DataFrame):
    return (
        df.style.hide(axis="index")
        .set_table_styles(
            [
                {
                    "selector": "th",
                    "props": [
                        ("background-color", "#EAF3FF"),
                        ("color", "#0f172a"),
                        ("font-weight", "700"),
                        ("border", "1px solid rgba(37, 99, 235, 0.16)"),
                    ],
                },
                {
                    "selector": "td",
                    "props": [
                        ("border", "1px solid rgba(15, 23, 42, 0.06)"),
                    ],
                },
            ]
        )
    )


def fmt_pct_value(value):
    numeric_value = pd.to_numeric(value, errors="coerce")
    return f"{numeric_value:.2%}" if pd.notna(numeric_value) else "N/D"


def normalize_risk_table_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    normalized = df.copy()
    method_col = next(
        (col for col in ["metodo", "método", "Metodo", "Método", "method"] if col in normalized.columns),
        None,
    )
    if method_col is not None and method_col != "metodo":
        normalized = normalized.rename(columns={method_col: "metodo"})

    if "metodo" in normalized.columns:
        normalized["metodo"] = normalized["metodo"].replace(
            {
                "Paramétrico": "Parametrico",
                "Histórico": "Historico",
            }
        )

    return normalized


def table_for_var_plot(df: pd.DataFrame) -> pd.DataFrame:
    plot_df = df.copy()
    if "metodo" in plot_df.columns:
        plot_df["método"] = plot_df["metodo"].replace(
            {
                "Parametrico": "Paramétrico",
                "Historico": "Histórico",
            }
        )
    return plot_df


render_app_shell(
    "Módulo 5 - VaR y CVaR",
    "Evalua el riesgo extremo del portafolio mediante VaR y CVaR bajo distintos enfoques de estimacion.",
)
ASSETS = configured_assets(ASSETS)
horizonte, start_date, end_date = configured_period(default_end=DEFAULT_END_DATE)
render_portfolio_summary_card(ASSETS)
module_header(
    "Value at Risk y CVaR del portafolio",
    "Comparación de pérdida umbral y pérdida extrema bajo enfoques paramétrico, histórico, Monte Carlo y KDE.",
    badge="VaR · CVaR · Monte Carlo · Kupiec",
)

# ==============================
# Parámetros del módulo
# ==============================
with module_params():
    st.header("Parámetros de riesgo")

    alpha = st.radio("Nivel de confianza", CONFIDENCE_LEVELS, index=0, horizontal=True)

    n_sim = int(
        st.number_input(
            "Simulaciones Monte Carlo",
            min_value=10_000,
            value=10_000,
            step=1_000,
            format="%d",
        )
    )
    try:
        SimulationConfigModel(n_sim=n_sim)
    except ValidationError:
        st.error("Monte Carlo requiere al menos 10,000 simulaciones.")
        st.stop()

    manual_weights_enabled = st.checkbox("Definir pesos manualmente (opcional)", value=False)
    manual_weights = {}
    if manual_weights_enabled:
        st.caption("Ingresa pesos entre 0 y 1. La suma debe ser 1.00.")
        default_weight = 1 / len(ASSETS)
        for asset_name, meta in ASSETS.items():
            manual_weights[asset_name] = st.number_input(
                f"{asset_name} ({meta['ticker']})",
                min_value=0.0,
                max_value=1.0,
                value=float(default_weight),
                step=0.01,
                format="%.4f",
                key=f"var_weight_{meta['ticker']}",
            )

        weights_sum = sum(manual_weights.values())
        st.caption(f"Suma actual = {weights_sum:.6f}")
        try:
            PortfolioWeightsModel(pesos=manual_weights)
            st.success("Pesos OK: la suma es 1.00.")
        except ValidationError:
            st.error(f"Pesos inválidos: la suma debe ser 1.00. Suma actual = {weights_sum:.6f}.")
            st.stop()

# ==============================
# Carga y preparacion de datos
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
        "Sin datos para estos tickers en el rango seleccionado; se excluyen del análisis: "
        + ", ".join(missing_tickers)
    )

risk_analyzer = RiskAnalyzer()
returns = risk_analyzer.clean_returns(bundle["returns"])

if not risk_analyzer.validate_sample(returns, min_rows=30):
    st.error(data_error_message("No hay suficientes datos para calcular métricas de riesgo."))
    st.stop()

if manual_weights_enabled:
    ticker_to_asset = {meta["ticker"]: name for name, meta in ASSETS.items()}
    weights = np.array(
        [manual_weights[ticker_to_asset[ticker]] for ticker in returns.columns],
        dtype=float,
    )
    portfolio_returns, weights = risk_analyzer.portfolio_returns(returns, weights)
else:
    portfolio_returns, weights = risk_analyzer.portfolio_returns(returns)

tables_by_alpha = risk_analyzer.compute_var_tables(
    portfolio_returns=portfolio_returns,
    asset_returns_df=returns,
    weights=weights,
    confidence_levels=CONFIDENCE_LEVELS,
    n_sim=n_sim,
)

non_empty_tables = [
    normalize_risk_table_columns(alpha_table)
    for alpha_table in tables_by_alpha.values()
    if alpha_table is not None and not alpha_table.empty
]
table = pd.concat(non_empty_tables, ignore_index=True) if non_empty_tables else pd.DataFrame()
selected_table = normalize_risk_table_columns(tables_by_alpha.get(alpha, pd.DataFrame()))

if table.empty:
    st.error("No fue posible calcular VaR y CVaR con los datos disponibles.")
    st.stop()

if selected_table.empty:
    st.error("No fue posible calcular VaR y CVaR para el nivel de confianza seleccionado.")
    st.stop()

required_risk_columns = {"metodo", "VaR_diario", "CVaR_diario"}
if not required_risk_columns.issubset(selected_table.columns) or not required_risk_columns.issubset(table.columns):
    st.warning("La tabla de riesgo no incluye todas las columnas necesarias para mostrar VaR/CVaR.")
    st.stop()

# ==============================
# Variables por método (precompute before tabs)
# ==============================
var_hist_row = selected_table.loc[selected_table["metodo"] == "Historico"]
var_param_row = selected_table.loc[selected_table["metodo"] == "Parametrico"]
var_mc_row = selected_table.loc[selected_table["metodo"] == "Monte Carlo"]
var_kde_row = selected_table.loc[selected_table["metodo"] == "Monte Carlo KDE"]

var_h = float(var_hist_row["VaR_diario"].iloc[0]) if not var_hist_row.empty else None
cvar_h = float(var_hist_row["CVaR_diario"].iloc[0]) if not var_hist_row.empty else None
var_p = float(var_param_row["VaR_diario"].iloc[0]) if not var_param_row.empty else None
cvar_p = float(var_param_row["CVaR_diario"].iloc[0]) if not var_param_row.empty else None
var_mc = float(var_mc_row["VaR_diario"].iloc[0]) if not var_mc_row.empty else None
cvar_mc = float(var_mc_row["CVaR_diario"].iloc[0]) if not var_mc_row.empty else None
var_kde = float(var_kde_row["VaR_diario"].iloc[0]) if not var_kde_row.empty else None
cvar_kde = float(var_kde_row["CVaR_diario"].iloc[0]) if not var_kde_row.empty else None

# Kupiec sobre VaR histórico (precompute for resumen)
_kupiec_hist = kupiec_test(returns=portfolio_returns, var=var_h, alpha=alpha) if var_h is not None else None
_kupiec_calibrated = (
    _kupiec_hist is not None
    and _kupiec_hist.get("p_value") is not None
    and _kupiec_hist["p_value"] > 0.05
) if _kupiec_hist else None

# Bar chart data
import plotly.graph_objects as _go_var

_bar_methods: list[str] = []
_bar_var_vals: list[float] = []
_bar_cvar_vals: list[float] = []
for _bm, _bv, _bcv in [
    ("Paramétrico", var_p, cvar_p),
    ("Histórico", var_h, cvar_h),
    ("Monte Carlo", var_mc, cvar_mc),
    ("MC KDE", var_kde, cvar_kde),
]:
    if _bv is not None:
        _bar_methods.append(_bm)
        _bar_var_vals.append(abs(_bv) * 100)
        _bar_cvar_vals.append(abs(_bcv) * 100 if _bcv is not None else 0)

# Summary table
compact_table = table[["confianza", "metodo", "VaR_diario", "CVaR_diario"]].copy()
_var_num = pd.to_numeric(compact_table["VaR_diario"], errors="coerce").abs()
_cvar_num = pd.to_numeric(compact_table["CVaR_diario"], errors="coerce").abs()
compact_table["Diferencia CVaR-VaR"] = (_cvar_num - _var_num).map(fmt_pct_value)
compact_table = compact_table.rename(columns={
    "confianza": "Confianza",
    "metodo": "Método",
    "VaR_diario": "VaR diario",
    "CVaR_diario": "CVaR diario",
})
compact_table["Confianza"] = compact_table["Confianza"].map(lambda x: f"{x:.0%}")
for col in ["VaR diario", "CVaR diario"]:
    compact_table[col] = compact_table[col].map(fmt_pct_value)

complete_table = table.copy().rename(columns={
    "confianza": "Confianza",
    "metodo": "Método",
    "VaR_diario": "VaR diario",
    "CVaR_diario": "CVaR diario",
    "VaR_anualizado": "VaR anualizado",
    "CVaR_anualizado": "CVaR anualizado",
})
complete_table["Confianza"] = complete_table["Confianza"].map(lambda x: f"{x:.0%}")
for col in ["VaR diario", "CVaR diario", "VaR anualizado", "CVaR anualizado"]:
    if col in complete_table.columns:
        complete_table[col] = complete_table[col].map(fmt_pct_value)

# ==============================
# Pestañas internas
# ==============================
tab_resumen, tab_metodos, tab_distribucion, tab_backtesting, tab_detalle = st.tabs([
    "Resumen ejecutivo",
    "Métodos VaR/CVaR",
    "Distribución y colas",
    "Backtesting Kupiec",
    "Detalle técnico",
])

# =========================================
# Tab 1: Resumen ejecutivo
# =========================================
with tab_resumen:
    section_intro(
        "Value at Risk y Conditional Value at Risk del portafolio",
        f"Comparación de pérdida umbral y pérdida extrema bajo enfoques paramétrico, histórico, Monte Carlo y KDE · {int(alpha * 100)}% confianza · {start_date} a {end_date}",
    )

    _r1, _r2, _r3, _r4 = st.columns(4)
    with _r1:
        kpi_card("Activos", str(len(ASSETS)), caption="Activos en el portafolio.")
    with _r2:
        kpi_card("Confianza", f"{int(alpha * 100)}%", caption="Nivel de confianza activo.")
    with _r3:
        kpi_card("Simulaciones MC", f"{n_sim:,}", caption="Escenarios Monte Carlo.")
    with _r4:
        kpi_card("Pesos", "Manuales" if manual_weights_enabled else "Equiponderados", caption="Distribución del portafolio.")

    st.markdown("---")
    section_intro("KPIs de riesgo extremo", "Pérdida umbral y pérdida extrema diaria por los métodos más representativos.")

    _k1, _k2, _k3, _k4 = st.columns(4)
    with _k1:
        kpi_card(
            "VaR Histórico diario",
            f"{var_h:.2%}" if var_h is not None else "N/D",
            delta=f"{int(alpha * 100)}% confianza",
            caption="Pérdida umbral empírica.",
        )
    with _k2:
        kpi_card(
            "CVaR Histórico diario",
            f"{cvar_h:.2%}" if cvar_h is not None else "N/D",
            delta="Cola extrema",
            delta_type="neg",
            caption="Pérdida promedio en la cola.",
        )
    with _k3:
        kpi_card(
            "VaR Monte Carlo diario",
            f"{var_mc:.2%}" if var_mc is not None else "N/D",
            delta=f"{n_sim:,} simulaciones",
            caption="Pérdida umbral simulada.",
        )
    with _k4:
        kpi_card(
            "CVaR Monte Carlo diario",
            f"{cvar_mc:.2%}" if cvar_mc is not None else "N/D",
            delta="Cola extrema",
            delta_type="neg",
            caption="Pérdida promedio simulada extrema.",
        )

    if var_kde is not None:
        st.caption(
            f"Monte Carlo KDE — VaR: {fmt_pct_value(var_kde)}  ·  CVaR: {fmt_pct_value(cvar_kde)}  "
            "· Distribución empírica suavizada sin supuesto de normalidad estricta."
        )

    _calib_text = (
        "El test de Kupiec indica que el modelo **está bien calibrado**."
        if _kupiec_calibrated is True
        else "El test de Kupiec sugiere **revisar la calibración** del modelo."
        if _kupiec_calibrated is False
        else "No fue posible ejecutar el test de Kupiec para el rango seleccionado."
    )
    _diff_text = ""
    if var_h is not None and cvar_h is not None:
        _diff = abs(cvar_h) - abs(var_h)
        _diff_text = (
            f" La diferencia entre CVaR y VaR histórico es {_diff:.2%}, lo que indica "
            + (
                "pérdidas extremas considerablemente más severas que el umbral."
                if _diff > 0.003
                else "cola extrema relativamente compacta."
            )
        )
    conclusion_box(
        f"Para el {int(alpha * 100)}% de confianza, el VaR histórico diario es {fmt_pct_value(var_h)} "
        f"y el CVaR histórico es {fmt_pct_value(cvar_h)}.{_diff_text} {_calib_text}",
        kind="success" if _kupiec_calibrated else "warn",
        label="Conclusión del riesgo extremo",
    )

# =========================================
# Tab 2: Métodos VaR/CVaR
# =========================================
with tab_metodos:
    section_intro(
        "Comparación VaR/CVaR por método",
        "Cada método utiliza supuestos distintos para estimar la pérdida umbral y la pérdida extrema promedio del portafolio.",
    )

    if _bar_methods:
        _bar_fig = _go_var.Figure()
        _bar_fig.add_trace(_go_var.Bar(
            name="VaR diario",
            x=_bar_methods,
            y=_bar_var_vals,
            marker_color="#2563eb",
            text=[f"{v:.2f}%" for v in _bar_var_vals],
            textposition="outside",
        ))
        _bar_fig.add_trace(_go_var.Bar(
            name="CVaR diario",
            x=_bar_methods,
            y=_bar_cvar_vals,
            marker_color="#dc2626",
            text=[f"{v:.2f}%" for v in _bar_cvar_vals],
            textposition="outside",
        ))
        _bar_fig.update_layout(
            barmode="group",
            title=f"Comparación VaR y CVaR por método ({int(alpha * 100)}% confianza)",
            yaxis_title="Pérdida estimada (%)",
            height=380,
            margin=dict(l=40, r=20, t=50, b=50),
            legend=dict(orientation="h", y=-0.18),
        )
        st.plotly_chart(_bar_fig, width="stretch")
    else:
        st.warning("No hay datos suficientes para construir la comparación por método.")

    st.markdown("#### Tabla resumida")
    _tbl_alpha = f"{int(alpha * 100)}%"
    _tbl_display = compact_table[compact_table["Confianza"] == _tbl_alpha].copy() if "Confianza" in compact_table.columns else compact_table.copy()
    st.dataframe(style_risk_table(_tbl_display), use_container_width=True, height=220)

    with st.expander("Ver tabla completa de VaR y CVaR"):
        st.dataframe(style_risk_table(complete_table), use_container_width=True)

    with st.expander("Cómo interpretar la comparación entre métodos"):
        st.write(
            """
            - **VaR** mide la pérdida umbral: el nivel que no debería superarse con la confianza seleccionada.
            - **CVaR** mide la pérdida promedio **más allá** del VaR; siempre será mayor o igual al VaR.
            - Si CVaR es mucho mayor que VaR, la cola extrema de pérdidas es más severa.
            - Las diferencias entre métodos reflejan distintos supuestos sobre la distribución de rendimientos.
            - **Monte Carlo KDE** modela la distribución empírica suavizada sin asumir normalidad estricta; útil para capturar mejor la forma real de la distribución.
            """
        )

# =========================================
# Tab 3: Distribución y colas
# =========================================
with tab_distribucion:
    section_intro(
        "Distribución de rendimientos y colas de pérdida",
        "El histograma muestra la forma empírica de los rendimientos. Las líneas indican los umbrales de VaR y CVaR por método seleccionado.",
    )

    _metodos_disponibles = [
        m for m, v in [("Paramétrico", var_p), ("Histórico", var_h), ("Monte Carlo", var_mc)]
        if v is not None
    ]
    _metodos_visibles = st.multiselect(
        "Métodos a mostrar en el histograma",
        _metodos_disponibles,
        default=_metodos_disponibles[:2] if len(_metodos_disponibles) >= 2 else _metodos_disponibles,
        key="m5_dist_methods",
    )

    if _metodos_visibles:
        _method_map_back = {
            "Paramétrico": "Parametrico",
            "Histórico": "Historico",
            "Monte Carlo": "Monte Carlo",
        }
        _filtered_table = selected_table[
            selected_table["metodo"].isin([_method_map_back.get(m, m) for m in _metodos_visibles])
        ].copy()
        _plot_table = table_for_var_plot(_filtered_table)
        _fig_var = plot_var_distribution(portfolio_returns, _plot_table)
        _fig_var.update_traces(
            marker_line_width=0.6,
            marker_line_color="rgba(15, 23, 42, 0.35)",
            selector=dict(type="histogram"),
        )
        _line_styles = {
            "Parametrico": "#2563eb", "Paramétrico": "#2563eb",
            "Historico": "#16a34a", "Histórico": "#16a34a",
            "Monte Carlo": "#f59e0b",
        }
        for _trace in _fig_var.data:
            _tname = str(getattr(_trace, "name", ""))
            if _tname.startswith("VaR") or _tname.startswith("CVaR"):
                for _mn, _mc_color in _line_styles.items():
                    if _mn in _tname:
                        _trace.line.color = _mc_color
                        _trace.line.width = 3
                        _trace.line.dash = "dash" if _tname.startswith("VaR") else "dot"
        st.plotly_chart(_fig_var, width="stretch")
    else:
        st.info("Selecciona al menos un método para visualizar el histograma.")

    with st.expander("Cómo leer este gráfico"):
        st.write(
            """
            - El histograma resume la distribución empírica de rendimientos diarios del portafolio.
            - Las líneas **VaR** (trazo discontinuo) marcan la pérdida umbral bajo cada método.
            - Las líneas **CVaR** (trazo punteado) muestran la pérdida promedio más severa en la cola.
            - La cola izquierda concentra los escenarios de pérdida extrema.
            - Cuando CVaR supera visualmente al VaR, las pérdidas extremas son más intensas que el umbral.
            """
        )

# =========================================
# Tab 4: Backtesting Kupiec
# =========================================
with tab_backtesting:
    section_intro(
        "Validación del VaR mediante Test de Kupiec",
        "Evalúa si la frecuencia observada de pérdidas que superan el VaR es coherente con la frecuencia esperada bajo el nivel de confianza seleccionado.",
    )

    _bt_method = st.selectbox(
        "Método para backtesting",
        ["Histórico", "Paramétrico", "Monte Carlo"],
        index=0,
        key="m5_bt_method",
    )
    _bt_method_key = {
        "Histórico": "Historico",
        "Paramétrico": "Parametrico",
        "Monte Carlo": "Monte Carlo",
    }.get(_bt_method, _bt_method)
    _bt_row = selected_table.loc[selected_table["metodo"] == _bt_method_key]
    _bt_var = float(_bt_row["VaR_diario"].iloc[0]) if not _bt_row.empty else None

    if _bt_var is not None:
        _kupiec = kupiec_test(returns=portfolio_returns, var=_bt_var, alpha=alpha)
        if _kupiec:
            _pv = _kupiec.get("p_value")
            _obs_rate = _kupiec["observed_fail_rate"]
            _exp_rate = _kupiec["expected_fail_rate"]
            _is_ok = _pv is not None and _pv > 0.05

            _bk1, _bk2, _bk3, _bk4 = st.columns(4)
            with _bk1:
                kpi_card(
                    "Violaciones observadas",
                    str(_kupiec["violations"]),
                    caption="Pérdidas que superaron el VaR estimado.",
                )
            with _bk2:
                kpi_card(
                    "Tasa observada",
                    f"{_obs_rate:.2%}",
                    delta=f"Esperada: {_exp_rate:.2%}",
                    delta_type="pos" if abs(_obs_rate - _exp_rate) < 0.02 else "neg",
                    caption=f"Confianza {int(alpha * 100)}%.",
                )
            with _bk3:
                kpi_card(
                    "p-value Kupiec",
                    f"{_pv:.4f}" if _pv is not None else "N/D",
                    delta="Bien calibrado" if _is_ok else "Revisar calibración",
                    delta_type="pos" if _is_ok else "neg",
                    caption="H0: tasa observada = tasa esperada.",
                )
            with _bk4:
                kpi_card(
                    "Estado del modelo",
                    "Bien calibrado" if _is_ok else "Revisar",
                    delta_type="pos" if _is_ok else "neg",
                    caption="Verde: p-value > 0.05 · Rojo: p-value ≤ 0.05.",
                )

            if _is_ok:
                conclusion_box(
                    f"El VaR {_bt_method} al {int(alpha * 100)}% está bien calibrado: "
                    f"no se rechaza que la tasa observada de violaciones ({_obs_rate:.2%}) "
                    f"sea compatible con la tasa esperada ({_exp_rate:.2%}). {_kupiec.get('conclusion', '')}",
                    kind="success",
                    label="Modelo bien calibrado",
                )
            else:
                conclusion_box(
                    f"El VaR {_bt_method} muestra posible desajuste: la tasa observada ({_obs_rate:.2%}) "
                    f"difiere de la esperada ({_exp_rate:.2%}). {_kupiec.get('conclusion', '')}",
                    kind="warn",
                    label="Revisar calibración del modelo",
                )
        else:
            st.warning("No se pudo ejecutar el test de Kupiec para este método.")
    else:
        st.warning(f"No hay VaR {_bt_method} disponible para ejecutar el test de Kupiec.")

    with st.expander("Qué evalúa el Test de Kupiec"):
        st.write(
            r"""
            Una **violación** ocurre cuando la pérdida observada supera el VaR estimado.
            La tasa esperada de violaciones es \(1 - \alpha\): 5% para 95% de confianza y 1% para 99%.

            El test de Kupiec evalúa si la frecuencia observada de violaciones es compatible con esa tasa esperada.
            - **p-value > 0.05**: no se rechaza la calibración del VaR → modelo bien calibrado.
            - **p-value ≤ 0.05**: hay evidencia de que el VaR no está bien calibrado.

            El test requiere una muestra suficientemente larga. Resultados con pocos datos deben interpretarse con cautela.
            """
        )

# =========================================
# Tab 5: Detalle técnico
# =========================================
with tab_detalle:
    section_intro(
        "Detalle técnico del módulo",
        "Supuestos, pesos, parámetros de simulación y metodología de los cálculos.",
    )

    with st.expander("Supuesto de pesos del portafolio"):
        if manual_weights_enabled:
            st.write("Se usan pesos manuales validados (suma = 1.00).")
            for _an, _am in ASSETS.items():
                st.caption(f"{_an} ({_am['ticker']}): {manual_weights.get(_an, 0):.4f}")
        else:
            st.write("Se usa un portafolio equiponderado. Todos los activos tienen el mismo peso.")
            _eq_weight = 1 / len(ASSETS) if ASSETS else 0
            for _an, _am in ASSETS.items():
                st.caption(f"{_an} ({_am['ticker']}): {_eq_weight:.4f}")

    with st.expander("Parámetros del módulo"):
        st.write(
            f"""
            - **Nivel de confianza activo:** {int(alpha * 100)}%
            - **Simulaciones Monte Carlo:** {n_sim:,}
            - **Periodo:** {start_date} a {end_date}
            - **Activos:** {len(ASSETS)}
            - **Pesos:** {"Manuales" if manual_weights_enabled else "Equiponderados"}
            """
        )

    with st.expander("Tabla completa de VaR y CVaR (todos los niveles)"):
        st.dataframe(style_risk_table(complete_table), use_container_width=True)

    with st.expander("Notas metodológicas"):
        st.write(
            f"""
            - **VaR paramétrico:** asume normalidad en los rendimientos; se calcula como cuantil de la distribución normal.
            - **VaR histórico:** usa el percentil empírico de los rendimientos observados; no asume forma distribucional.
            - **VaR Monte Carlo:** simula {n_sim:,} trayectorias y toma el cuantil de la distribución simulada.
            - **Monte Carlo KDE:** usa distribución empírica suavizada (Kernel Density Estimation) sin asumir normalidad estricta.
            - **CVaR (Expected Shortfall):** en todos los métodos, es el promedio de pérdidas que exceden el VaR estimado.
            - **Test de Kupiec:** contrasta estadísticamente si la frecuencia observada de violaciones es compatible con la esperada.
            """
        )
