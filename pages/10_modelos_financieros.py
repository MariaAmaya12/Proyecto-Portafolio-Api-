from __future__ import annotations

import math

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.api.backend_client import BackendAPIError, backend_post, friendly_error_message
from src.ui_components import conclusion_box, kpi_card, module_header, render_explanation_expander, render_section, render_table
from src.ui_layout import module_params, render_app_shell
from src.ui_style import apply_global_typography


DEFAULT_RETURNS_TEXT = "0.01, -0.005, 0.003, -0.002, 0.004, -0.006, 0.002"
DEFAULT_STRESS_RETURNS_TEXT = "0.01,-0.005,0.003,-0.002"
DEFAULT_STRESS_WEIGHTS_TEXT = "0.25,0.25,0.25,0.25"
DEFAULT_MATRIX_RETURNS_TEXT = "0.01,-0.005,0.003,-0.002\n0.006,-0.002,0.001,-0.004\n-0.008,0.004,-0.003,0.002"


apply_global_typography()


# ==============================
# Helpers de parseo y validación
# ==============================
def parse_float_list(raw_values: str, label: str = "valores", min_length: int = 1) -> list[float]:
    if not raw_values or not raw_values.strip():
        raise ValueError(f"Ingresa {label} separados por coma.")
    values: list[float] = []
    invalid_tokens: list[str] = []

    for token in raw_values.split(","):
        cleaned = token.strip()
        if not cleaned:
            continue
        try:
            value = float(cleaned)
        except ValueError:
            invalid_tokens.append(cleaned)
            continue
        if not math.isfinite(value):
            invalid_tokens.append(cleaned)
            continue
        values.append(value)

    if invalid_tokens:
        invalid_text = ", ".join(invalid_tokens[:5])
        raise ValueError(f"Estos valores no son numéricos válidos: {invalid_text}.")

    if len(values) < min_length:
        raise ValueError(f"Ingresa al menos {min_length} valores válidos para {label}.")

    return values


def parse_returns(raw_returns: str) -> list[float]:
    return parse_float_list(raw_returns, label="retornos diarios", min_length=2)


def parse_float_matrix(raw_values: str, label: str = "retornos") -> list[list[float]]:
    if not raw_values or not raw_values.strip():
        raise ValueError(f"Ingresa {label} en filas separadas por salto de línea o punto y coma.")

    rows = [row.strip() for row in raw_values.replace(";", "\n").splitlines() if row.strip()]
    matrix = [parse_float_list(row, label=f"{label} de la fila {index + 1}", min_length=1) for index, row in enumerate(rows)]

    expected_columns = len(matrix[0]) if matrix else 0
    if expected_columns == 0:
        raise ValueError(f"Ingresa al menos una columna de {label}.")
    if any(len(row) != expected_columns for row in matrix):
        raise ValueError("Todas las filas de retornos deben tener la misma cantidad de activos.")
    return matrix


def validate_equal_length(list_a: list[float], list_b: list[float], label_a: str, label_b: str) -> None:
    if len(list_a) != len(list_b):
        raise ValueError(f"{label_a} y {label_b} deben tener la misma cantidad de elementos.")


def validate_weights(weights: list[float]) -> None:
    if not weights:
        raise ValueError("Ingresa al menos un peso.")
    if any(weight < 0 for weight in weights):
        raise ValueError("Los pesos no pueden ser negativos.")
    if not math.isclose(sum(weights), 1.0, rel_tol=0.0, abs_tol=1e-6):
        raise ValueError("La suma de pesos debe ser aproximadamente 1.00.")


# ==============================
# Helpers de formato
# ==============================
def format_pct(value: object) -> str:
    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return "N/D"
    if not math.isfinite(numeric_value):
        return "N/D"
    return f"{numeric_value:.2%}"


def format_currency(value: object) -> str:
    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return "N/D"
    if not math.isfinite(numeric_value):
        return "N/D"
    return f"{numeric_value:,.2f}"


def format_number(value: object) -> str:
    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return "N/D"
    if not math.isfinite(numeric_value):
        return "N/D"
    return f"{numeric_value:.8f}"


def format_dec3(value: object) -> str:
    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return "N/D"
    if not math.isfinite(numeric_value):
        return "N/D"
    return f"{numeric_value:.3f}"


def render_backend_error(exc: BackendAPIError, default: str) -> None:
    st.error(friendly_error_message(exc, default))


def build_key_value_table(result: dict) -> pd.DataFrame:
    rows = []
    for key, value in result.items():
        if value is None:
            continue
        if isinstance(value, list):
            display_value = ", ".join(format_number(item) for item in value)
        elif isinstance(value, dict):
            display_value = str(value)
        else:
            display_value = format_number(value)
        rows.append({"Métrica": str(key), "Valor": display_value})
    return pd.DataFrame(rows)


# ==============================
# Pestaña: Resumen
# ==============================
def render_summary_tab() -> None:
    render_section(
        "¿Para qué sirve este módulo?",
        "Integra modelos financieros que no reemplazan los módulos de riesgo anteriores, sino que los complementan con análisis de sensibilidad, valoración y escenarios adversos.",
    )

    _s1, _s2, _s3 = st.columns(3)

    with _s1:
        st.markdown("#### Renta fija")
        st.info(
            "**Métrica clave:** Duración modificada\n\n"
            "Mide la sensibilidad del precio del bono ante cambios en la tasa de interés. "
            "La convexidad ajusta esta relación para movimientos más grandes.",
        )

    with _s2:
        st.markdown("#### Opciones")
        st.info(
            "**Métrica clave:** Precio call / put\n\n"
            "Valora derivados bajo el modelo Black-Scholes y permite revisar la sensibilidad "
            "mediante las griegas: Delta, Gamma, Vega, Theta y Rho.",
        )

    with _s3:
        st.markdown("#### Stress testing")
        st.warning(
            "**Métrica clave:** Pérdida extrema / CVaR\n\n"
            "Evalúa el impacto de shocks adversos sobre el portafolio. "
            "Complementa la medición de VaR/CVaR del Módulo 5.",
        )

    conclusion_box(
        "Orden sugerido para la exposición: "
        "(1) presentar el módulo como complementario al análisis de riesgo; "
        "(2) renta fija — sensibilidad a tasas mediante duración y convexidad; "
        "(3) opciones — valoración de derivados con Black-Scholes; "
        "(4) stress testing — cómo responde el portafolio ante shocks adversos; "
        "(5) escenario combinado — integración de múltiples shocks simultáneos.",
        kind="success",
        label="Orden sugerido para la exposición",
    )

    with st.expander("Conexión con los demás módulos", expanded=False):
        st.write(
            """
            - **Stress testing** complementa el VaR/CVaR del **Módulo 5**.
            - **Renta fija** extiende el análisis de tasas relevante para el contexto macroeconómico.
            - **Opciones** añade valoración de derivados que no está cubierta en los módulos anteriores.
            - **Escenario combinado** integra shocks multifactoriales para una evaluación más completa de resiliencia.
            - La volatilidad EWMA se analiza y compara con GARCH en el **Módulo 3 — ARCH/GARCH**.
            """
        )


# ==============================
# Pestaña: Renta fija
# ==============================
def render_bond_metrics_block() -> None:
    render_section(
        "Parámetros del bono",
        "Calcula precio, duración y convexidad a partir de las características del instrumento de renta fija.",
    )

    with st.form("bond_metrics_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            face_value = st.number_input("Valor nominal", min_value=0.01, value=1000.0, step=100.0, format="%.2f")
            coupon_rate = st.number_input("Cupón anual", min_value=0.0, value=0.05, step=0.005, format="%.4f")
        with c2:
            market_rate = st.number_input("Tasa de mercado", min_value=0.0, value=0.045, step=0.005, format="%.4f")
            maturity_years = st.number_input("Años al vencimiento", min_value=0.25, value=5.0, step=0.25, format="%.2f")
        with c3:
            frequency = st.selectbox("Frecuencia de pago", [1, 2, 4, 12], index=1)

        submitted = st.form_submit_button("Calcular métricas de bono", type="primary")

    if not submitted:
        return

    payload = {
        "face_value": float(face_value),
        "coupon_rate": float(coupon_rate),
        "market_rate": float(market_rate),
        "maturity_years": float(maturity_years),
        "frequency": int(frequency),
    }

    try:
        result = backend_post("/fixed-income/bond-metrics", payload)
    except BackendAPIError as exc:
        render_backend_error(exc, "No fue posible calcular las métricas del bono.")
        return

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        kpi_card(
            "Precio del bono",
            format_currency(result.get("price")),
            caption="Valor presente de flujos futuros.",
        )
    with c2:
        kpi_card(
            "Duración Macaulay",
            format_dec3(result.get("macaulay_duration")),
            caption="Plazo promedio ponderado de cobro.",
        )
    with c3:
        kpi_card(
            "Duración modificada",
            format_dec3(result.get("modified_duration")),
            delta="Sensibilidad a tasas",
            delta_type="neu",
            caption="Sensibilidad aproximada del precio ante cambios en tasa.",
        )
    with c4:
        kpi_card(
            "Convexidad",
            format_dec3(result.get("convexity")),
            caption="Ajuste de segundo orden para cambios mayores en tasa.",
        )

    conclusion_box(
        "La duración modificada aproxima cuánto cambia el precio del bono ante una variación en la tasa de interés. "
        "Una mayor duración implica mayor sensibilidad al riesgo de tasa. "
        "La convexidad ajusta esta relación cuando los cambios de tasa son más grandes.",
        kind="success",
        label="Interpretación de renta fija",
    )

    with st.expander("Cómo interpretar las métricas de bono", expanded=False):
        st.write(
            """
            - **Precio del bono:** valor presente de los flujos futuros descontados a la tasa de mercado.
            - **Duración Macaulay:** plazo promedio ponderado de cobro, expresado en años.
            - **Duración modificada:** sensibilidad porcentual del precio ante un cambio de 1% en la tasa. Si la duración modificada es 4.5, un aumento de tasa del 1% reduce el precio ~4.5%.
            - **Convexidad:** corrige la aproximación lineal de la duración para movimientos mayores de tasa.
            """
        )


def render_nelson_siegel_block() -> None:
    render_section(
        "Curva Nelson-Siegel",
        "Estima la curva de rendimientos a partir de parámetros de nivel, pendiente y curvatura.",
    )

    with st.form("nelson_siegel_form"):
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            beta0 = st.number_input("beta0 (nivel)", value=0.04, step=0.005, format="%.4f")
        with c2:
            beta1 = st.number_input("beta1 (pendiente)", value=-0.02, step=0.005, format="%.4f")
        with c3:
            beta2 = st.number_input("beta2 (curvatura)", value=0.03, step=0.005, format="%.4f")
        with c4:
            tau = st.number_input("tau (escala)", min_value=0.0001, value=1.5, step=0.1, format="%.4f")

        maturities_text = st.text_input("Vencimientos separados por coma (años)", value="0.5,1,2,5,10")
        submitted = st.form_submit_button("Calcular curva Nelson-Siegel", type="primary")

    if not submitted:
        return

    try:
        maturities = parse_float_list(maturities_text, label="vencimientos", min_length=1)
        if any(maturity <= 0 for maturity in maturities):
            raise ValueError("Los vencimientos deben ser positivos.")
    except ValueError as exc:
        st.warning(str(exc))
        return

    payload = {
        "beta0": float(beta0),
        "beta1": float(beta1),
        "beta2": float(beta2),
        "tau": float(tau),
        "maturities": maturities,
    }

    try:
        result = backend_post("/fixed-income/nelson-siegel", payload)
    except BackendAPIError as exc:
        render_backend_error(exc, "No fue posible calcular la curva Nelson-Siegel.")
        return

    raw_maturities: list[float] = result.get("maturities", [])
    raw_yields: list[float] = result.get("yields", [])

    if not raw_maturities or not raw_yields:
        st.warning("El backend no devolvió puntos para la curva.")
        return

    yields_pct = [y * 100 for y in raw_yields]
    labels = [f"{y:.2f}%" for y in yields_pct]

    st.caption(
        "La curva muestra la tasa estimada para distintos vencimientos, permitiendo observar "
        "la pendiente y forma temporal de la estructura de tasas."
    )

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=raw_maturities,
            y=yields_pct,
            mode="lines+markers+text",
            line=dict(color="#2563eb", width=2.5),
            marker=dict(size=9, color="#2563eb", line=dict(color="white", width=1.5)),
            text=labels,
            textposition="top center",
            textfont=dict(size=11, color="#1e3a8a"),
            hovertemplate="<b>Vencimiento:</b> %{x} años<br><b>Tasa estimada:</b> %{y:.2f}%<extra></extra>",
            name="Nelson-Siegel",
        )
    )
    fig.update_layout(
        title=dict(
            text="Curva estimada de rendimientos Nelson-Siegel",
            font=dict(size=15, color="#0f172a"),
            x=0.01,
        ),
        xaxis=dict(
            title="Vencimiento (años)",
            gridcolor="#f1f5f9",
            showgrid=True,
            zeroline=False,
        ),
        yaxis=dict(
            title="Tasa estimada (%)",
            tickformat=".2f",
            ticksuffix="%",
            gridcolor="#f1f5f9",
            showgrid=True,
            zeroline=False,
        ),
        height=440,
        margin=dict(l=55, r=40, t=65, b=55),
        paper_bgcolor="white",
        plot_bgcolor="white",
        legend=dict(orientation="h", y=-0.18, x=0.5, xanchor="center"),
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True)

    # Interpretación dinámica de la pendiente
    if len(yields_pct) >= 2:
        _diff = yields_pct[-1] - yields_pct[0]
        if _diff > 0.10:
            _slope = "pendiente positiva, lo que indica tasas mayores para vencimientos largos frente a vencimientos cortos"
            _kind = "success"
        elif _diff < -0.10:
            _slope = "pendiente negativa (curva invertida), lo que indica tasas menores para vencimientos largos"
            _kind = "warn"
        else:
            _slope = "relativamente plana, con diferencias pequeñas entre vencimientos cortos y largos"
            _kind = "success"
        conclusion_box(
            f"Lectura de la curva: la estructura estimada presenta {_slope}.",
            kind=_kind,
            label="Interpretación de la curva",
        )

    curve_display_df = pd.DataFrame(
        {
            "Vencimiento (años)": raw_maturities,
            "Tasa estimada": labels,
        }
    )
    with st.expander("Ver tabla de vencimientos y tasas", expanded=False):
        render_table(curve_display_df, hide_index=True, width="stretch")


def render_fixed_income_tab() -> None:
    render_bond_metrics_block()
    st.divider()
    render_nelson_siegel_block()


# ==============================
# Pestaña: Opciones
# ==============================
def option_payload_from_inputs(spot: float, strike: float, rate: float, volatility: float, time_to_maturity: float) -> dict:
    return {
        "spot": float(spot),
        "strike": float(strike),
        "rate": float(rate),
        "volatility": float(volatility),
        "time_to_maturity": float(time_to_maturity),
    }


def render_options_tab() -> None:
    render_section(
        "Parámetros de valoración Black-Scholes",
        "Valora opciones call y put, y calcula las griegas de sensibilidad del precio de la opción.",
    )

    with st.form("options_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            spot = st.number_input("Spot (precio actual)", min_value=0.01, value=100.0, step=1.0, format="%.2f")
            strike = st.number_input("Strike (precio de ejercicio)", min_value=0.01, value=100.0, step=1.0, format="%.2f")
        with c2:
            rate = st.number_input("Tasa libre de riesgo", value=0.05, step=0.005, format="%.4f")
            volatility = st.number_input("Volatilidad", min_value=0.0001, value=0.20, step=0.01, format="%.4f")
        with c3:
            time_to_maturity = st.number_input("Tiempo al vencimiento (años)", min_value=0.0001, value=1.0, step=0.25, format="%.4f")

        c_btn1, c_btn2 = st.columns(2)
        with c_btn1:
            price_submitted = st.form_submit_button("Calcular Black-Scholes", type="primary")
        with c_btn2:
            greeks_submitted = st.form_submit_button("Calcular griegas")

    if not (price_submitted or greeks_submitted):
        st.info("Configura los parámetros y calcula precios Black-Scholes o griegas desde el backend.")
        return

    payload = option_payload_from_inputs(spot, strike, rate, volatility, time_to_maturity)

    if price_submitted:
        try:
            result = backend_post("/options/black-scholes", payload)
        except BackendAPIError as exc:
            render_backend_error(exc, "No fue posible calcular precios Black-Scholes.")
            return

        _oc1, _oc2 = st.columns(2)
        with _oc1:
            kpi_card(
                "Precio call",
                format_currency(result.get("call_price")),
                delta="Opción de compra",
                delta_type="pos",
                caption="Valor teórico de comprar el activo al precio de ejercicio.",
            )
        with _oc2:
            kpi_card(
                "Precio put",
                format_currency(result.get("put_price")),
                delta="Opción de venta",
                delta_type="neg",
                caption="Valor teórico de vender el activo al precio de ejercicio.",
            )

        conclusion_box(
            "El precio call es mayor cuando el subyacente supera el strike; el put cuando ocurre lo contrario. "
            "Ambos dependen de la volatilidad implícita y el tiempo al vencimiento.",
            kind="success",
            label="Interpretación Black-Scholes",
        )

        with st.expander("Supuestos del modelo Black-Scholes", expanded=False):
            st.write(
                """
                - El precio del subyacente sigue un movimiento browniano geométrico.
                - La volatilidad es constante durante la vida de la opción.
                - No hay dividendos, costos de transacción ni oportunidades de arbitraje.
                - El resultado es teórico y puede diferir de precios de mercado donde la volatilidad implícita varía.
                """
            )

    if greeks_submitted:
        try:
            result = backend_post("/options/greeks", payload)
        except BackendAPIError as exc:
            render_backend_error(exc, "No fue posible calcular las griegas.")
            return

        st.markdown("##### Griegas de la opción")

        _greek_map = [
            ("delta", "Delta", "Sensibilidad al precio del subyacente."),
            ("gamma", "Gamma", "Sensibilidad del delta al precio."),
            ("vega", "Vega", "Sensibilidad a la volatilidad."),
            ("theta", "Theta", "Sensibilidad al paso del tiempo."),
            ("rho", "Rho", "Sensibilidad a la tasa de interés."),
        ]

        _available = [item for item in _greek_map if result.get(item[0]) is not None]
        if _available:
            _gcols = st.columns(len(_available))
            for col, (key, label, caption) in zip(_gcols, _available):
                with col:
                    kpi_card(label, format_dec3(result.get(key)), caption=caption)
        else:
            st.warning("El backend no devolvió griegas para los parámetros ingresados.")

        with st.expander("Interpretación de las griegas", expanded=False):
            st.write(
                """
                - **Delta:** cambio en el precio de la opción ante un cambio unitario en el subyacente. Call positivo, put negativo.
                - **Gamma:** velocidad de cambio del delta. Alta gamma implica sensibilidad no lineal al precio.
                - **Vega:** sensibilidad a la volatilidad. Un vega alto indica que el precio cambia mucho ante variaciones de volatilidad.
                - **Theta:** pérdida de valor por el paso del tiempo (decaimiento temporal). Generalmente negativo.
                - **Rho:** sensibilidad a cambios en la tasa libre de riesgo.
                """
            )

        with st.expander("Tabla técnica de griegas", expanded=False):
            render_table(build_key_value_table(result), hide_index=True, width="stretch")


# ==============================
# Pestaña: Stress testing
# ==============================
def render_portfolio_stress_block() -> None:
    render_section(
        "Escenario de estrés del portafolio",
        "Evalúa pérdidas bajo shocks adversos aplicando un choque uniforme a los retornos de los activos.",
    )

    with st.expander("Editar datos del escenario", expanded=True):
        with st.form("portfolio_stress_form"):
            st.caption("Cada fila representa una observación y cada columna un activo.")
            returns_text = st.text_area("Retornos por activo (filas = observaciones, columnas = activos)", value=DEFAULT_MATRIX_RETURNS_TEXT, height=110)
            weights_text = st.text_input("Pesos separados por coma (deben sumar 1.0)", value=DEFAULT_STRESS_WEIGHTS_TEXT)
            c1, c2 = st.columns(2)
            with c1:
                price_shock = st.number_input("Shock por activo", value=-0.05, step=0.01, format="%.4f",
                                               help="Choque aplicado uniformemente a cada activo. Ejemplo: -0.05 = −5%.")
            with c2:
                confidence_level = st.selectbox("Nivel de confianza", [0.90, 0.95, 0.99], index=1)

            submitted = st.form_submit_button("Calcular estrés de portafolio", type="primary")

    if not submitted:
        return

    try:
        returns_matrix = parse_float_matrix(returns_text, label="retornos")
        weights = parse_float_list(weights_text, label="pesos", min_length=1)
        validate_equal_length(returns_matrix[0], weights, "retornos por activo", "pesos")
        validate_weights(weights)
    except ValueError as exc:
        st.warning(str(exc))
        return

    shocks = {f"asset_{index}": float(price_shock) for index in range(len(weights))}
    payload = {
        "returns": returns_matrix,
        "weights": weights,
        "shocks": shocks,
        "confidence_level": float(confidence_level),
    }

    try:
        result = backend_post("/stress/portfolio", payload)
    except BackendAPIError as exc:
        render_backend_error(exc, "No fue posible calcular el estrés del portafolio.")
        return

    st.markdown("##### Resultados del escenario de estrés")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        kpi_card(
            "Pérdida extrema / CVaR",
            format_pct(result.get("cvar")),
            delta="Cola extrema",
            delta_type="neg",
            caption="Pérdida promedio esperada en el peor percentil.",
        )
    with c2:
        kpi_card(
            "VaR",
            format_pct(result.get("var")),
            delta=f"{confidence_level:.0%} confianza",
            delta_type="neg",
            caption="Pérdida umbral del escenario.",
        )
    with c3:
        kpi_card(
            "Máximo drawdown",
            format_pct(result.get("max_drawdown")),
            caption="Caída acumulada máxima estimada.",
        )
    with c4:
        kpi_card(
            "Retorno estresado medio",
            format_pct(result.get("mean_return")),
            caption="Media de retornos bajo el shock aplicado.",
        )

    conclusion_box(
        f"Bajo el shock aplicado ({price_shock:.1%} por activo, nivel {confidence_level:.0%}), "
        f"la pérdida extrema (CVaR) representa la pérdida promedio esperada en el percentil más adverso. "
        "Este análisis complementa el VaR/CVaR histórico del Módulo 5 con un escenario de estrés específico.",
        kind="warn",
        label="Lectura del escenario de estrés",
    )

    with st.expander("Detalle completo del escenario de estrés", expanded=False):
        render_table(build_key_value_table(result), hide_index=True, width="stretch")


def render_stress_testing_tab() -> None:
    render_portfolio_stress_block()

    with st.expander("Cómo interpretar el stress testing", expanded=False):
        st.write(
            """
            - El stress testing **no predice el futuro**; evalúa la sensibilidad del portafolio bajo supuestos adversos.
            - El shock se aplica uniformemente a cada activo. En escenarios reales, los shocks pueden ser heterogéneos.
            - Un CVaR elevado indica alta vulnerabilidad del portafolio ante condiciones extremas.
            - Compara el CVaR del escenario de estrés con el VaR histórico del Módulo 5 para dimensionar el impacto.
            """
        )


# ==============================
# Pestaña: Escenario combinado
# ==============================
def render_combined_stress_tab() -> None:
    render_section(
        "Escenario combinado",
        "Integra shocks simultáneos de precio, retornos, volatilidad y tasa para evaluar resiliencia ante crisis multifactoriales.",
    )

    with st.form("combined_stress_form"):
        st.markdown("**Parámetros de mercado**")
        c1, c2, c3 = st.columns(3)
        with c1:
            prices_text = st.text_input("Precios separados por coma", value="100,102,98,95")
        with c2:
            returns_text = st.text_input("Retornos separados por coma", value=DEFAULT_STRESS_RETURNS_TEXT)
        with c3:
            weights_val = st.number_input("Peso de la serie", value=1.0, step=0.1, format="%.4f")

        st.markdown("**Shocks del escenario**")
        c4, c5, c6 = st.columns(3)
        with c4:
            price_shock = st.number_input("Shock de precio", value=-0.08, step=0.01, format="%.4f")
        with c5:
            volatility_multiplier = st.number_input("Multiplicador de volatilidad", min_value=0.0001, value=1.5, step=0.1, format="%.4f")
        with c6:
            rate_shock = st.number_input("Shock de tasa", value=0.01, step=0.005, format="%.4f")

        st.markdown("**Parámetros del bono**")
        c7, c8, c9 = st.columns(3)
        with c7:
            bond_price = st.number_input("Precio del bono", min_value=0.01, value=1000.0, step=50.0, format="%.2f")
        with c8:
            modified_duration = st.number_input("Duración modificada", min_value=0.0, value=4.5, step=0.25, format="%.4f")
        with c9:
            convexity = st.number_input("Convexidad", min_value=0.0, value=25.0, step=1.0, format="%.4f")

        submitted = st.form_submit_button("Calcular escenario combinado", type="primary")

    if not submitted:
        return

    try:
        prices = parse_float_list(prices_text, label="precios", min_length=1)
        returns = parse_float_list(returns_text, label="retornos", min_length=1)
    except ValueError as exc:
        st.warning(str(exc))
        return

    payload = {
        "prices": prices,
        "returns": returns,
        "weights": float(weights_val),
        "price_shock": float(price_shock),
        "volatility_multiplier": float(volatility_multiplier),
        "bond_price": float(bond_price),
        "modified_duration": float(modified_duration),
        "convexity": float(convexity),
        "rate_shock": float(rate_shock),
    }

    try:
        result = backend_post("/stress/combined-scenario", payload)
    except BackendAPIError as exc:
        render_backend_error(exc, "No fue posible calcular el escenario combinado.")
        return

    summary = result.get("scenario_summary")

    if isinstance(summary, dict) and summary:
        st.markdown("##### Métricas del escenario combinado")
        _label_map = {
            "mean_return": ("Retorno medio", "neu"),
            "min_return": ("Retorno mínimo", "neg"),
            "max_drawdown": ("Máximo drawdown", "neg"),
            "var": ("VaR", "neg"),
            "cvar": ("CVaR", "neg"),
        }
        _available_keys = [k for k in _label_map if k in summary]
        if _available_keys:
            _cols = st.columns(min(len(_available_keys), 5))
            for col, key in zip(_cols, _available_keys):
                label, delta_type = _label_map[key]
                with col:
                    kpi_card(
                        label,
                        format_pct(summary.get(key)),
                        delta_type=delta_type,
                        caption="Escenario combinado.",
                    )

    conclusion_box(
        "El escenario combinado integra shocks simultáneos de precio, tasa y volatilidad. "
        "El VaR y CVaR permiten cuantificar la pérdida umbral y la pérdida promedio en condiciones extremas. "
        "Es el análisis más completo para evaluar resiliencia ante crisis multifactoriales.",
        kind="warn",
        label="Lectura del escenario combinado",
    )

    with st.expander("Ver detalle técnico del escenario combinado", expanded=False):
        render_table(build_key_value_table(result), hide_index=True, width="stretch")

    if isinstance(summary, dict) and summary:
        with st.expander("Tabla resumen del escenario", expanded=False):
            render_table(build_key_value_table(summary), hide_index=True, width="stretch")


# ==============================
# Configuración del módulo
# ==============================
render_app_shell(
    "Módulo 10 – Modelos financieros avanzados",
    "Modelos complementarios para analizar sensibilidad, valoración y estrés financiero del portafolio.",
)
module_header(
    "Módulo 10 – Modelos financieros avanzados",
    "Modelos complementarios para analizar sensibilidad de bonos ante tasas, valoración de opciones con Black-Scholes y escenarios de estrés para pérdidas extremas.",
    badge="Renta fija · Opciones · Stress testing",
)

with module_params():
    st.caption("Los parámetros de cada modelo se configuran dentro de su pestaña correspondiente.")

# ==============================
# PESTAÑAS PRINCIPALES
# ==============================
tabs = st.tabs(["Resumen", "Renta fija", "Opciones", "Stress testing", "Escenario combinado"])

with tabs[0]:
    render_summary_tab()

with tabs[1]:
    render_fixed_income_tab()

with tabs[2]:
    render_options_tab()

with tabs[3]:
    render_stress_testing_tab()

with tabs[4]:
    render_combined_stress_tab()
