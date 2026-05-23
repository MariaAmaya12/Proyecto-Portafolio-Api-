import streamlit as st
import pandas as pd

from src.config import (
    ASSETS,
    DEFAULT_END_DATE,
    ensure_project_dirs,
    get_ticker,
)
from src.download import data_error_message
from src.garch_models import fit_garch_models
from src.volatility import ewma_volatility
from src.plots import plot_forecast, plot_standardized_residuals, plot_volatility
from src.returns_analysis import compute_return_series
from src.risk_metrics import validar_serie_para_garch
from src.services.market_data_client import MarketDataClient
from src.ui_components import conclusion_box, kpi_card, module_header, render_explanation_expander, render_section, render_table
from src.ui_layout import configured_assets, configured_period, module_params, render_app_shell, render_selected_asset_card
from src.ui_style import apply_global_typography

ensure_project_dirs()
apply_global_typography()


def fmt_num(value):
    numeric_value = pd.to_numeric(value, errors="coerce")
    return f"{numeric_value:.3f}" if pd.notna(numeric_value) else "N/D"


def fmt_pvalue(value):
    numeric_value = pd.to_numeric(value, errors="coerce")
    if pd.isna(numeric_value):
        return "N/D"
    if numeric_value < 0.0001:
        return "< 0.0001"
    return f"{numeric_value:.4f}"


render_app_shell(
    "Módulo 3: ARCH/GARCH",
    "Modela volatilidad condicional y pronosticos de riesgo sobre rendimientos del activo.",
)
ASSETS = configured_assets(ASSETS)
horizonte, start_date, end_date = configured_period(default_end=DEFAULT_END_DATE)
asset_name, ticker = render_selected_asset_card(ASSETS, key="m3_asset_selector")

# ==============================
# Parámetros del módulo
# ==============================
with module_params():
    st.caption("Este módulo usa el activo y horizonte definidos en la vista principal.")
    _lambda = st.slider(
        "Lambda EWMA",
        min_value=0.80,
        max_value=0.99,
        value=0.94,
        step=0.01,
        format="%.2f",
        key="m3_lambda",
        help="Mayor lambda = mayor peso al pasado. RiskMetrics estándar usa 0.94.",
    )

# ==============================
# Descargar datos
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
            f"No hay datos suficientes para {ticker} en el rango seleccionado. "
            "Para ajustar modelos GARCH se recomienda usar al menos 1 año completo, "
            "idealmente 2 años o más."
        )
    else:
        st.error(
            data_error_message(
                "No se pudieron obtener datos desde el backend para el activo seleccionado."
            )
        )

    st.stop()

price_col = "Adj Close" if "Adj Close" in df.columns else "Close"
ret_df = compute_return_series(df[price_col])

if "log_return" not in ret_df.columns:
    st.error("No se encontro la columna 'log_return' para ajustar el modelo GARCH.")
    st.stop()

serie_retornos = ret_df["log_return"]

# ==============================
# Validacion de la serie
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
        "No se ajusto el modelo GARCH porque la serie no cumple las condiciones minimas "
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
module_header(
    "Módulo 3: Volatilidad condicional",
    "Estimación de volatilidad con EWMA y modelos ARCH/GARCH para evaluar persistencia, riesgo reciente y pronóstico de volatilidad.",
    badge="EWMA · ARCH · GARCH · EGARCH",
)
st.caption(f"Periodo analizado: {start_date} a {end_date}  ·  Activo: {asset_name} ({ticker})")

# ==============================
# Ajuste de modelos GARCH
# ==============================
results = fit_garch_models(serie_garch)

if results["comparison"].empty:
    st.warning("No hay suficientes datos o el ajuste no convergio correctamente para los modelos GARCH.")
    st.stop()

# ==============================
# Variables de resultado
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
    if persistence >= 1.0:
        persistence_label = "Revisar estacionariedad"
        persistence_delta = "neg"
    elif persistence >= 0.90:
        persistence_label = "Alta persistencia"
        persistence_delta = "neg"
    elif persistence >= 0.75:
        persistence_label = "Persistencia media"
        persistence_delta = "neu"
    else:
        persistence_label = "Persistencia baja"
        persistence_delta = "neg"
else:
    persistence_label = None
    persistence_delta = "neu"

# EWMA computations (configurable lambda)
try:
    _ewma_daily = ewma_volatility(serie_retornos, lambda_=_lambda, annualize=False)
    _ewma_annual = ewma_volatility(serie_retornos, lambda_=_lambda, annualize=True, periods_per_year=252)
except Exception:
    _ewma_daily = None
    _ewma_annual = None

if _ewma_annual is not None and pd.notna(_ewma_annual):
    if _ewma_annual < 0.15:
        _ewma_signal = "Baja"
        _ewma_signal_type = "pos"
    elif _ewma_annual <= 0.30:
        _ewma_signal = "Moderada"
        _ewma_signal_type = "neu"
    else:
        _ewma_signal = "Alta"
        _ewma_signal_type = "neg"
else:
    _ewma_signal = "N/D"
    _ewma_signal_type = "neu"

horizon_steps = None
try:
    horizon_steps = int(results["forecast"]["horizonte"].max())
except Exception:
    horizon_steps = 10

_vol_current = None
try:
    _vol_df_raw = results.get("volatility")
    if _vol_df_raw is not None and not _vol_df_raw.empty:
        _best_col = None
        if best_model:
            for _c in _vol_df_raw.columns:
                if str(best_model).upper() in str(_c).upper():
                    _best_col = _c
                    break
        if _best_col is None:
            _best_col = _vol_df_raw.columns[0]
        _vals = _vol_df_raw[_best_col].dropna()
        if not _vals.empty:
            _vol_current = float(_vals.iloc[-1])
except Exception:
    _vol_current = None

import plotly.graph_objects as _go

# ==============================
# Pestañas internas
# ==============================
_tabs = st.tabs(["EWMA", "Comparación", "Modelos GARCH", "Diagnóstico", "Pronóstico", "Detalle técnico"])

# =========================================
# Tab 1: EWMA
# =========================================
with _tabs[0]:
    render_section(
        "EWMA: volatilidad con ponderación exponencial",
        "El modelo EWMA asigna mayor peso a los rendimientos recientes. Es útil para capturar cambios rápidos en el riesgo sin estimar un modelo paramétrico completo.",
    )
    st.caption(f"Lambda configurado: **{_lambda:.2f}**  ·  Modifícalo en el panel lateral → *Parámetros del módulo*.")

    _e1, _e2, _e3, _e4 = st.columns(4)
    with _e1:
        kpi_card(
            "Vol. EWMA diaria",
            f"{_ewma_daily * 100:.2f}%" if _ewma_daily is not None and pd.notna(_ewma_daily) else "N/D",
            caption="Volatilidad diaria en escala porcentual.",
        )
    with _e2:
        kpi_card(
            "Vol. EWMA anualizada",
            f"{_ewma_annual * 100:.2f}%" if _ewma_annual is not None and pd.notna(_ewma_annual) else "N/D",
            caption=f"Escalada × √252 con λ={_lambda:.2f}.",
        )
    with _e3:
        kpi_card(
            "Lambda",
            f"{_lambda:.2f}",
            caption="Mayor lambda = mayor memoria del pasado.",
        )
    with _e4:
        kpi_card(
            "Señal de riesgo EWMA",
            _ewma_signal,
            delta_type=_ewma_signal_type,
            caption="Baja <15% · Moderada 15–30% · Alta >30% (anualizada).",
        )

    try:
        _ewma_chart = serie_retornos.ewm(alpha=(1 - _lambda), adjust=False).std() * 100
        _rolling_21 = serie_retornos.rolling(window=21).std() * 100
        _ewma_fig = _go.Figure()
        _ewma_fig.add_trace(_go.Scatter(
            x=_ewma_chart.index,
            y=_ewma_chart.values,
            mode="lines",
            name=f"EWMA (λ={_lambda:.2f})",
            line=dict(color="#0EA5E9", width=2),
        ))
        _ewma_fig.add_trace(_go.Scatter(
            x=_rolling_21.index,
            y=_rolling_21.values,
            mode="lines",
            name="Rolling 21 días",
            line=dict(color="#64748B", width=1.8, dash="dot"),
        ))
        _ewma_fig.update_layout(
            title=f"Evolución de la volatilidad EWMA (λ={_lambda:.2f})",
            xaxis_title="Fecha",
            yaxis_title="Volatilidad (%)",
            height=340,
            margin=dict(l=40, r=20, t=50, b=50),
            legend=dict(orientation="h", y=-0.2),
        )
        st.plotly_chart(_ewma_fig, width="stretch")
    except Exception:
        st.info("No fue posible generar el gráfico EWMA.")

    conclusion_box(
        "EWMA permite observar cómo cambia el riesgo reciente del activo. "
        "Si la volatilidad EWMA sube, los rendimientos recientes muestran mayor inestabilidad. "
        "A diferencia de GARCH, EWMA no selecciona parámetros por AIC/BIC: su comportamiento depende directamente del valor de lambda.",
        kind="success",
        label="Lectura para exposición",
    )

# =========================================
# Tab 2: Comparación EWMA vs GARCH
# =========================================
with _tabs[1]:
    render_section(
        "Dos enfoques complementarios",
        "EWMA reacciona rápidamente a choques recientes. GARCH estima una estructura condicional de volatilidad y permite evaluar persistencia, ajuste estadístico y pronóstico.",
    )

    _ewma_daily_pct = (_ewma_daily * 100) if _ewma_daily is not None and pd.notna(_ewma_daily) else None
    if _ewma_daily_pct is not None and forecast_last is not None:
        _diff_daily = _ewma_daily_pct - forecast_last
        if abs(_diff_daily) < 0.3:
            _lectura_comp = "Niveles similares"
            _comp_type = "neu"
        elif _diff_daily > 0:
            _lectura_comp = "EWMA > GARCH"
            _comp_type = "neg"
        else:
            _lectura_comp = "EWMA < GARCH"
            _comp_type = "pos"
    else:
        _lectura_comp = "No disponible"
        _comp_type = "neu"

    _c1, _c2, _c3, _c4 = st.columns(4)
    with _c1:
        kpi_card(
            "Vol. EWMA diaria",
            f"{_ewma_daily_pct:.2f}%" if _ewma_daily_pct is not None else "N/D",
            caption=f"λ={_lambda:.2f}, escala porcentual.",
        )
    with _c2:
        kpi_card(
            "Vol. pronosticada GARCH",
            fmt_num(forecast_last),
            caption=f"Diaria, modelo {best_model or 'N/D'} al último paso.",
        )
    with _c3:
        kpi_card(
            "Mejor modelo GARCH",
            str(best_model) if best_model else "N/D",
            caption="Seleccionado por menor AIC.",
        )
    with _c4:
        kpi_card(
            "Lectura comparativa",
            _lectura_comp,
            delta_type=_comp_type,
            caption="EWMA diaria vs pronóstico GARCH (escala ×100).",
        )

    st.markdown("#### Comparación conceptual: EWMA vs GARCH")
    _comp_conceptual = pd.DataFrame([
        {"Aspecto": "Tipo de modelo", "EWMA": "Ponderación exponencial", "GARCH/EGARCH": "Paramétrico condicional"},
        {"Aspecto": "Parámetros", "EWMA": f"Solo lambda (λ={_lambda:.2f})", "GARCH/EGARCH": "Estimados por máxima verosimilitud"},
        {"Aspecto": "Reacción a choques", "EWMA": "Alta (muy reactivo)", "GARCH/EGARCH": "Moderada (estructura persistente)"},
        {"Aspecto": "Persistencia", "EWMA": "Implícita en lambda", "GARCH/EGARCH": f"Explícita ({fmt_num(persistence)})"},
        {"Aspecto": "Selección estadística", "EWMA": "No usa AIC/BIC", "GARCH/EGARCH": "Selección por AIC/BIC"},
        {"Aspecto": "Uso en gestión de riesgo", "EWMA": "Referencia reactiva de corto plazo", "GARCH/EGARCH": "Persistencia y pronóstico condicional"},
    ])
    render_table(_comp_conceptual, hide_index=True, width="stretch")

    render_explanation_expander(
        "EWMA vs GARCH: diferencias clave",
        [
            f"EWMA asigna mayor peso a retornos recientes con λ={_lambda:.2f} (RiskMetrics usa 0.94).",
            "GARCH estima parámetros por máxima verosimilitud e incluye una constante de largo plazo.",
            "EWMA es más reactivo; GARCH captura mejor la estructura de persistencia.",
            "Cuando ambos coinciden en el nivel de riesgo, la señal es más robusta.",
        ],
    )

    conclusion_box(
        "EWMA se usa como referencia reactiva de corto plazo, mientras que GARCH/EGARCH permite modelar "
        "persistencia condicional. Si ambos modelos señalan niveles similares de riesgo, "
        "la lectura de volatilidad es más robusta.",
        kind="success",
        label="Conclusión comparativa",
    )

# =========================================
# Tab 3: Modelos GARCH
# =========================================
with _tabs[2]:
    render_section(
        "Selección de modelos ARCH/GARCH",
        "Se comparan ARCH(1), GARCH(1,1) y EGARCH(1,1) usando criterios de información. El menor AIC/BIC favorece el mejor ajuste relativo penalizando complejidad.",
    )

    if pd.notna(persistence) and persistence >= 1.0:
        st.warning(
            f"Persistencia elevada ({fmt_num(persistence)}): revisar estacionariedad o interpretación "
            "según la especificación del modelo. Para EGARCH, la persistencia puede calcularse de forma distinta a alpha+beta."
        )

    _g1, _g2, _g3, _g4, _g5 = st.columns(5)
    with _g1:
        kpi_card(
            "Mejor modelo",
            str(best_model) if best_model else "N/D",
            caption="Seleccionado por menor AIC.",
        )
    with _g2:
        kpi_card(
            "AIC",
            fmt_num(best_aic),
            caption="Menor valor = mejor ajuste relativo.",
        )
    with _g3:
        kpi_card(
            "BIC",
            fmt_num(best_bic),
            caption="Penaliza complejidad del modelo.",
        )
    with _g4:
        kpi_card(
            "Log-Likelihood",
            fmt_num(best_loglik),
            caption="Log-verosimilitud del modelo ganador.",
        )
    with _g5:
        _persist_warn = pd.notna(persistence) and persistence >= 1.0
        kpi_card(
            "Persistencia",
            fmt_num(persistence),
            delta="⚠ Revisar" if _persist_warn else persistence_label,
            delta_type="neg" if _persist_warn else persistence_delta,
            caption="Memoria de la volatilidad estimada.",
        )

    if best_model is None:
        st.warning("No se generó una lectura automática del mejor modelo.")

    st.markdown("#### Tabla comparativa — modelos GARCH")
    _garch_cols = ["modelo", "AIC", "BIC", "loglik", "persistencia"]
    _vis_cols = [c for c in _garch_cols if c in comparison_df.columns]
    _garch_display = comparison_df[_vis_cols].copy()
    if "AIC" in _garch_display.columns:
        _garch_display = _garch_display.sort_values("AIC", ascending=True).reset_index(drop=True)
    if "modelo" in _garch_display.columns:
        _garch_display["Selección"] = _garch_display["modelo"].apply(
            lambda m: "Mejor" if m == best_model else ""
        )
    _garch_display = _garch_display.rename(columns={
        "modelo": "Modelo",
        "loglik": "Log-Likelihood",
        "persistencia": "Persistencia",
    })
    for _col in _garch_display.columns:
        if _col not in {"Modelo", "Selección"}:
            _garch_display[_col] = pd.to_numeric(_garch_display[_col], errors="coerce").apply(fmt_num)
    render_table(_garch_display, hide_index=True, width="stretch")

    render_explanation_expander(
        "Cómo leer AIC/BIC",
        [
            "AIC y BIC son criterios de selección que equilibran ajuste y complejidad.",
            "Un AIC menor indica mejor ajuste relativo; BIC penaliza más los parámetros adicionales.",
            "El modelo con menor AIC se selecciona como referencia cuando converge correctamente.",
            "Para EGARCH, la persistencia no equivale directamente a alpha+beta del GARCH estándar.",
        ],
    )

# =========================================
# Tab 4: Diagnóstico
# =========================================
with _tabs[3]:
    render_section(
        "Diagnóstico del modelo seleccionado",
        "Se evalúa si el modelo converge y si los residuos estandarizados conservan patrones relevantes después del ajuste.",
    )

    _d1, _d2, _d3, _d4 = st.columns(4)
    with _d1:
        kpi_card(
            "Convergencia",
            str(best_converged),
            caption="Estado de convergencia del ajuste.",
        )
    with _d2:
        kpi_card(
            "JB residuos est.",
            fmt_num(jb_stat),
            caption="Jarque-Bera sobre residuos estandarizados.",
        )
    with _d3:
        _pv_fmt = (
            f"{jb_pvalue:.4f}" if pd.notna(jb_pvalue) and jb_pvalue >= 0.0001
            else "< 0.0001" if pd.notna(jb_pvalue)
            else "N/D"
        )
        kpi_card(
            "p-value JB",
            _pv_fmt,
            delta="Rechaza normalidad" if normality_rejected else "No rechaza" if normality_rejected is False else None,
            delta_type="neg" if normality_rejected else "pos" if normality_rejected is False else "neu",
            caption="H0: residuos con distribución normal.",
        )
    with _d4:
        _norm_label = (
            "Rechaza normalidad" if normality_rejected
            else "No rechaza normalidad" if normality_rejected is False
            else "N/D"
        )
        _norm_type = "neg" if normality_rejected else "pos" if normality_rejected is False else "neu"
        kpi_card(
            "Normalidad residuos",
            _norm_label,
            delta_type=_norm_type,
            caption=normality_decision,
        )

    st.caption("ARCH-LM: no disponible en la implementación actual. Ver pestaña Detalle técnico para contexto metodológico.")

    if "std_resid" in results and results["std_resid"] is not None:
        st.markdown("#### Residuos estandarizados")
        st.plotly_chart(plot_standardized_residuals(results["std_resid"]), width="stretch")
        with st.expander("Ver tabla de residuos estandarizados"):
            render_table(results["std_resid"].tail(30), width="stretch", hide_index=False)
    else:
        st.info("No se generaron residuos estandarizados para el modelo seleccionado.")

    conclusion_box(
        "Si los residuos estandarizados rechazan normalidad, esto sugiere colas pesadas o choques extremos "
        "no capturados completamente por el supuesto de distribución. "
        "Esto no invalida automáticamente el modelo, pero debe mencionarse como limitación metodológica.",
        kind="warn",
        label="Diagnóstico del modelo",
    )

# =========================================
# Tab 5: Pronóstico
# =========================================
with _tabs[4]:
    render_section(
        "Pronóstico de volatilidad",
        "El modelo seleccionado proyecta la volatilidad esperada para los próximos pasos, útil para anticipar riesgo prospectivo.",
    )

    _p1, _p2, _p3 = st.columns(3)
    with _p1:
        kpi_card(
            "Volatilidad actual",
            fmt_num(_vol_current),
            caption=f"Último valor estimado por {best_model or 'N/D'}.",
        )
    with _p2:
        kpi_card(
            "Volatilidad pronosticada",
            fmt_num(forecast_last),
            caption=f"Último paso del pronóstico a {horizon_steps} pasos.",
        )
    with _p3:
        kpi_card(
            "Horizonte",
            f"{horizon_steps} pasos" if horizon_steps else "N/D",
            caption="Pasos hacia adelante proyectados por el modelo.",
        )

    st.plotly_chart(plot_forecast(results["forecast"]), width="stretch")

    _conclusion_parts = [
        "EWMA permite capturar cambios recientes en la volatilidad del activo, "
        "mientras que el modelo GARCH/EGARCH seleccionado permite evaluar persistencia y pronosticar riesgo condicional. "
        "En conjunto, ambos enfoques fortalecen la lectura del riesgo: combinan sensibilidad de corto plazo "
        "con modelación estadística de la volatilidad."
    ]
    if best_model and pd.notna(persistence):
        _conclusion_parts.append(
            f"El modelo {best_model} presenta una persistencia de {fmt_num(persistence)}, lo que indica "
            + (
                "alta memoria en los choques de volatilidad."
                if persistence >= 0.90
                else "memoria moderada en los choques de volatilidad."
            )
        )
    _conclusion_parts.append(normality_decision)

    conclusion_box(
        " ".join(_conclusion_parts),
        kind="success",
        label="Conclusión del análisis de volatilidad",
    )

# =========================================
# Tab 6: Detalle técnico
# =========================================
with _tabs[5]:
    render_section(
        "Detalle técnico del módulo",
        "Fundamento metodológico, parámetros del modelo y tablas técnicas de respaldo.",
    )

    with st.expander("Fundamento del módulo"):
        st.write(
            """
            - La volatilidad condicional cambia en el tiempo y suele agruparse en periodos de calma o turbulencia.
            - Los modelos ARCH/GARCH permiten estimar esa dinámica sin asumir una volatilidad histórica constante.
            - La serie validada se escala por 100 antes del ajuste para mejorar la estabilidad numérica del modelo.
            - EWMA es la referencia más inmediata: sin estimación paramétrica, con reactividad directa al lambda elegido.
            """
        )

    with st.expander("Qué significan ARCH, GARCH y EGARCH"):
        st.write(
            """
            - **ARCH** modela la volatilidad actual en función de los choques cuadráticos pasados.
            - **GARCH** combina choques recientes con la volatilidad condicional pasada (memoria).
            - **EGARCH** captura respuestas asimétricas ante choques positivos y negativos.
            - **EWMA** asigna mayor peso a retornos recientes con el parámetro lambda (sin estimación por MV).
            """
        )

    with st.expander("Volatilidad condicional estimada (todos los modelos)"):
        st.caption("Comparación de cómo cada especificación modela la evolución de la volatilidad condicional a lo largo del tiempo.")
        _vol_plot_df = results["volatility"].copy()
        _vol_plot_df.insert(0, "Volatilidad movil 21 dias", volatilidad_movil_21d.reindex(_vol_plot_df.index))
        _vol_fig = plot_volatility(_vol_plot_df)
        _vol_fig.update_traces(
            selector=dict(name="Volatilidad movil 21 dias"),
            line=dict(color="#64748B", width=2.6, dash="dot"),
        )
        st.plotly_chart(_vol_fig, width="stretch")

    if not diagnostics_df.empty:
        with st.expander("Diagnóstico técnico completo"):
            _diag_display = diagnostics_df.copy()
            if "valor" in _diag_display.columns:
                _diag_display["valor"] = _diag_display["valor"].apply(
                    lambda v: fmt_num(v) if pd.notna(pd.to_numeric(v, errors="coerce")) else v
                )
            render_table(_diag_display, width="stretch", hide_index=True)

    with st.expander("Notas metodológicas"):
        st.write(
            """
            - ARCH-LM no está disponible en la implementación actual. Para una validación completa, se recomienda complementar con esta prueba.
            - La persistencia de EGARCH puede no equivaler directamente a alpha+beta del GARCH estándar.
            - El módulo escala los retornos por 100 antes del ajuste GARCH. Los resultados están en esa escala (unidades ×100).
            - EWMA se calcula sobre los retornos en escala decimal. La volatilidad EWMA se multiplica por 100 para comparación visual.
            - Los valores de volatilidad EWMA se muestran en porcentaje (e.g., 1.79%). Los valores GARCH también están escalados ×100.
            """
        )
