from __future__ import annotations

import math

import pandas as pd
import streamlit as st

from src.api.backend_client import BackendAPIError, backend_post, friendly_error_message
from src.ui_components import kpi_card, render_explanation_expander, render_section, render_table
from src.ui_layout import module_params, render_app_shell
from src.ui_style import apply_global_typography


DEFAULT_RETURNS_TEXT = "0.01, -0.005, 0.003, -0.002, 0.004, -0.006, 0.002"
DEFAULT_STRESS_RETURNS_TEXT = "0.01,-0.005,0.003,-0.002"
DEFAULT_STRESS_WEIGHTS_TEXT = "0.25,0.25,0.25,0.25"
DEFAULT_MATRIX_RETURNS_TEXT = "0.01,-0.005,0.003,-0.002\n0.006,-0.002,0.001,-0.004\n-0.008,0.004,-0.003,0.002"


apply_global_typography()


def inject_page_css() -> None:
    st.markdown(
        """
        <style>
        .section-intro-box {
            background: #ffffff;
            border: 1px solid rgba(15, 23, 42, 0.08);
            border-radius: 18px;
            padding: 16px 18px;
            box-shadow: 0 4px 14px rgba(15, 23, 42, 0.06);
            margin-bottom: 0.8rem;
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
        raise ValueError(f"Estos valores no son numericos validos: {invalid_text}.")

    if len(values) < min_length:
        raise ValueError(f"Ingresa al menos {min_length} valores validos para {label}.")

    return values


def parse_returns(raw_returns: str) -> list[float]:
    """Parse comma-separated daily returns entered by the user."""
    return parse_float_list(raw_returns, label="retornos diarios", min_length=2)


def parse_float_matrix(raw_values: str, label: str = "retornos") -> list[list[float]]:
    if not raw_values or not raw_values.strip():
        raise ValueError(f"Ingresa {label} en filas separadas por salto de linea o punto y coma.")

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
        rows.append({"Metrica": str(key), "Valor": display_value})
    return pd.DataFrame(rows)


def render_ewma_results(result: dict) -> None:
    st.markdown("### Resultado")

    c1, c2, c3 = st.columns(3)
    with c1:
        kpi_card(
            "Volatilidad EWMA",
            format_pct(result.get("ewma_volatility")),
            delta="Anualizada" if result.get("annualize") else "Diaria",
            caption="Estimacion devuelta por el backend.",
        )
    with c2:
        kpi_card(
            "Varianza EWMA",
            format_number(result.get("ewma_variance")),
            delta="Base diaria",
            caption="Varianza exponencialmente ponderada.",
        )
    with c3:
        kpi_card(
            "Observaciones",
            result.get("observations", "N/D"),
            delta=f"lambda = {result.get('lambda_', 'N/D')}",
            caption=f"Periodos por año: {result.get('periods_per_year', 'N/D')}.",
        )

    result_df = pd.DataFrame(
        [
            {
                "Metrica": "Volatilidad EWMA",
                "Valor": format_pct(result.get("ewma_volatility")),
            },
            {
                "Metrica": "Varianza EWMA",
                "Valor": format_number(result.get("ewma_variance")),
            },
            {
                "Metrica": "Lambda",
                "Valor": result.get("lambda_", "N/D"),
            },
            {
                "Metrica": "Anualizada",
                "Valor": "Si" if result.get("annualize") else "No",
            },
            {
                "Metrica": "Observaciones",
                "Valor": result.get("observations", "N/D"),
            },
        ]
    )
    render_table(result_df, hide_index=True, width="stretch")

    render_explanation_expander(
        "Como interpretar EWMA",
        [
            "EWMA asigna mayor peso a los retornos mas recientes y menor peso a observaciones mas antiguas.",
            "Un lambda mas alto suaviza mas la serie y hace que la volatilidad reaccione con mayor lentitud.",
            "Si la opción anualizada está activa, el backend escala la volatilidad usando los periodos por año seleccionados.",
        ],
    )


def render_ewma_tab() -> None:
    render_section(
        "Volatilidad EWMA",
        "Calcula volatilidad con ponderacion exponencial sin replicar formulas en Streamlit; la estimacion viene del backend.",
    )

    with st.form("ewma_volatility_form"):
        returns_text = st.text_area(
            "Retornos diarios separados por coma",
            value=DEFAULT_RETURNS_TEXT,
            height=120,
            help="Usa retornos en formato decimal. Por ejemplo, 0.01 representa 1%.",
        )

        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            lambda_value = st.number_input(
                "Lambda",
                min_value=0.0001,
                max_value=0.9999,
                value=0.94,
                step=0.01,
                format="%.4f",
            )
        with col2:
            annualize = st.checkbox("Annualize", value=True)
        with col3:
            periods_per_year = st.number_input(
                "Periodos por año",
                min_value=1,
                value=252,
                step=1,
                format="%d",
            )

        submitted = st.form_submit_button("Calcular volatilidad EWMA", type="primary")

    if not submitted:
        st.info("Ingresa una serie de retornos y ejecuta el calculo para consultar el endpoint `/volatility/ewma`.")
        return

    try:
        returns = parse_returns(returns_text)
    except ValueError as exc:
        st.warning(str(exc))
        return

    payload = {
        "returns": returns,
        "lambda_": float(lambda_value),
        "annualize": bool(annualize),
        "periods_per_year": int(periods_per_year),
    }

    try:
        result = backend_post("/volatility/ewma", payload)
    except BackendAPIError as exc:
        render_backend_error(exc, "No fue posible calcular la volatilidad EWMA.")
        return

    render_ewma_results(result)


def render_bond_metrics_block() -> None:
    st.markdown("### Metricas de bono")
    with st.form("bond_metrics_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            face_value = st.number_input("Valor nominal", min_value=0.01, value=1000.0, step=100.0, format="%.2f")
            coupon_rate = st.number_input("Cupon anual", min_value=0.0, value=0.05, step=0.005, format="%.4f")
        with c2:
            market_rate = st.number_input("Tasa de mercado", min_value=0.0, value=0.045, step=0.005, format="%.4f")
            maturity_years = st.number_input("Anos al vencimiento", min_value=0.25, value=5.0, step=0.25, format="%.2f")
        with c3:
            frequency = st.selectbox("Frecuencia de pago", [1, 2, 4, 12], index=1)

        submitted = st.form_submit_button("Calcular metricas de bono", type="primary")

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
        render_backend_error(exc, "No fue posible calcular las metricas del bono.")
        return

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        kpi_card("Precio del bono", format_currency(result.get("price")), caption="Valor presente de flujos futuros.")
    with c2:
        kpi_card("Duracion Macaulay", format_number(result.get("macaulay_duration")), caption="Plazo promedio ponderado de cobro.")
    with c3:
        kpi_card("Duracion modificada", format_number(result.get("modified_duration")), caption="Sensibilidad aproximada ante cambios en tasa.")
    with c4:
        kpi_card("Convexidad", format_number(result.get("convexity")), caption="Ajuste de segundo orden para cambios mayores.")

    render_explanation_expander(
        "Como interpretar las metricas de bono",
        [
            "Precio del bono: valor presente de los flujos futuros.",
            "Duracion modificada: sensibilidad aproximada del precio ante cambios en tasa.",
            "Convexidad: ajuste de segundo orden para cambios mas grandes en tasa.",
        ],
    )


def render_nelson_siegel_block() -> None:
    st.markdown("### Curva Nelson-Siegel")
    with st.form("nelson_siegel_form"):
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            beta0 = st.number_input("beta0", value=0.04, step=0.005, format="%.4f")
        with c2:
            beta1 = st.number_input("beta1", value=-0.02, step=0.005, format="%.4f")
        with c3:
            beta2 = st.number_input("beta2", value=0.03, step=0.005, format="%.4f")
        with c4:
            tau = st.number_input("tau", min_value=0.0001, value=1.5, step=0.1, format="%.4f")

        maturities_text = st.text_input("Vencimientos separados por coma", value="0.5,1,2,5,10")
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

    curve_df = pd.DataFrame(
        {
            "maturity": result.get("maturities", []),
            "yield": result.get("yields", []),
        }
    )
    if curve_df.empty:
        st.warning("El backend no devolvio puntos para la curva.")
        return

    render_table(curve_df, hide_index=True, width="stretch")
    st.line_chart(curve_df.set_index("maturity")["yield"])


def render_fixed_income_tab() -> None:
    render_section("Renta fija", "Metricas de bonos y curvas Nelson-Siegel calculadas por el backend.")
    render_bond_metrics_block()
    st.divider()
    render_nelson_siegel_block()


def option_payload_from_inputs(spot: float, strike: float, rate: float, volatility: float, time_to_maturity: float) -> dict:
    return {
        "spot": float(spot),
        "strike": float(strike),
        "rate": float(rate),
        "volatility": float(volatility),
        "time_to_maturity": float(time_to_maturity),
    }


def render_options_tab() -> None:
    render_section("Opciones", "Valoracion Black-Scholes y griegas calculadas por endpoints del backend.")

    with st.form("options_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            spot = st.number_input("Spot", min_value=0.01, value=100.0, step=1.0, format="%.2f")
            strike = st.number_input("Strike", min_value=0.01, value=100.0, step=1.0, format="%.2f")
        with c2:
            rate = st.number_input("Tasa libre de riesgo", value=0.05, step=0.005, format="%.4f")
            volatility = st.number_input("Volatilidad", min_value=0.0001, value=0.20, step=0.01, format="%.4f")
        with c3:
            time_to_maturity = st.number_input("Tiempo al vencimiento", min_value=0.0001, value=1.0, step=0.25, format="%.4f")

        c_btn1, c_btn2 = st.columns(2)
        with c_btn1:
            price_submitted = st.form_submit_button("Calcular Black-Scholes", type="primary")
        with c_btn2:
            greeks_submitted = st.form_submit_button("Calcular griegas")

    if not (price_submitted or greeks_submitted):
        st.info("Usa los controles para consultar precios Black-Scholes o griegas desde el backend.")
        return

    payload = option_payload_from_inputs(spot, strike, rate, volatility, time_to_maturity)

    if price_submitted:
        try:
            result = backend_post("/options/black-scholes", payload)
        except BackendAPIError as exc:
            render_backend_error(exc, "No fue posible calcular precios Black-Scholes.")
            return

        c1, c2 = st.columns(2)
        with c1:
            kpi_card("Precio call", format_currency(result.get("call_price")), caption="Valor teorico de comprar el activo al strike.")
        with c2:
            kpi_card("Precio put", format_currency(result.get("put_price")), caption="Valor teorico de vender el activo al strike.")

        render_explanation_expander(
            "Como interpretar Black-Scholes",
            [
                "El call representa el valor teorico de comprar el activo a un precio de ejercicio.",
                "El put representa el valor teorico de vender el activo a un precio de ejercicio.",
                "El resultado es teorico y depende de supuestos como volatilidad constante y ausencia de arbitraje.",
            ],
        )

    if greeks_submitted:
        try:
            result = backend_post("/options/greeks", payload)
        except BackendAPIError as exc:
            render_backend_error(exc, "No fue posible calcular las griegas.")
            return

        greeks_df = build_key_value_table(result)
        render_table(greeks_df, hide_index=True, width="stretch")
        render_explanation_expander(
            "Como interpretar las griegas",
            [
                "Delta mide sensibilidad al precio del subyacente.",
                "Gamma mide sensibilidad del delta.",
                "Vega mide sensibilidad a la volatilidad.",
                "Theta mide sensibilidad al paso del tiempo.",
                "Rho mide sensibilidad a la tasa de interes.",
            ],
        )


def render_portfolio_stress_block() -> None:
    st.markdown("### Estres de portafolio")
    st.caption("Cada fila representa una observacion y cada columna un activo. Tambien puedes ingresar una sola fila.")

    with st.form("portfolio_stress_form"):
        returns_text = st.text_area("Retornos por activo", value=DEFAULT_MATRIX_RETURNS_TEXT, height=110)
        weights_text = st.text_input("Pesos separados por coma", value=DEFAULT_STRESS_WEIGHTS_TEXT)
        c1, c2 = st.columns(2)
        with c1:
            price_shock = st.number_input("Shock por activo", value=-0.05, step=0.01, format="%.4f")
        with c2:
            confidence_level = st.selectbox("Nivel de confianza", [0.90, 0.95, 0.99], index=1)

        submitted = st.form_submit_button("Calcular estres de portafolio", type="primary")

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
        render_backend_error(exc, "No fue posible calcular el stress del portafolio.")
        return

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        kpi_card("Retorno estresado medio", format_pct(result.get("mean_return")), caption="Media de retornos bajo shocks.")
    with c2:
        kpi_card("Perdida extrema", format_pct(result.get("cvar")), delta="CVaR", delta_type="neg", caption="Perdida promedio de cola.")
    with c3:
        kpi_card("VaR", format_pct(result.get("var")), delta=f"{confidence_level:.0%}", caption="Perdida umbral del escenario.")
    with c4:
        kpi_card("Maximo drawdown", format_pct(result.get("max_drawdown")), caption="Caida acumulada maxima estimada.")

    render_table(build_key_value_table(result), hide_index=True, width="stretch")


def render_combined_stress_block() -> None:
    st.markdown("### Escenario combinado")

    with st.form("combined_stress_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            prices_text = st.text_input("Precios separados por coma", value="100,102,98,95")
            price_shock = st.number_input("Shock de precio", value=-0.08, step=0.01, format="%.4f")
        with c2:
            returns_text = st.text_input("Retornos separados por coma", value=DEFAULT_STRESS_RETURNS_TEXT)
            volatility_multiplier = st.number_input("Multiplicador de volatilidad", min_value=0.0001, value=1.5, step=0.1, format="%.4f")
        with c3:
            weights = st.number_input("Peso de la serie", value=1.0, step=0.1, format="%.4f")
            rate_shock = st.number_input("Shock de tasa", value=0.01, step=0.005, format="%.4f")

        c4, c5, c6 = st.columns(3)
        with c4:
            bond_price = st.number_input("Precio del bono", min_value=0.01, value=1000.0, step=50.0, format="%.2f")
        with c5:
            modified_duration = st.number_input("Duracion modificada", min_value=0.0, value=4.5, step=0.25, format="%.4f")
        with c6:
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
        "weights": float(weights),
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

    render_table(build_key_value_table(result), hide_index=True, width="stretch")
    summary = result.get("scenario_summary")
    if isinstance(summary, dict) and summary:
        st.markdown("#### Resumen del escenario")
        render_table(build_key_value_table(summary), hide_index=True, width="stretch")


def render_stress_testing_tab() -> None:
    render_section("Stress testing", "Escenarios de estres para portafolios y escenarios combinados calculados por backend.")
    render_portfolio_stress_block()
    st.divider()
    render_combined_stress_block()
    render_explanation_expander(
        "Como interpretar stress testing",
        [
            "El stress testing no predice el futuro.",
            "Sirve para evaluar sensibilidad del portafolio bajo escenarios adversos.",
            "El escenario combinado permite observar efectos simultaneos de precio, tasa y volatilidad.",
        ],
    )


inject_page_css()

render_app_shell(
    "Modelos financieros avanzados",
    "Aplica modelos complementarios de riesgo financiero desde el backend, manteniendo el frontend como capa de interaccion y visualizacion.",
)

with module_params():
    st.caption("Los insumos de cada modelo se editan dentro de su pestana correspondiente.")

tabs = st.tabs(["Volatilidad EWMA", "Renta fija", "Opciones", "Stress testing"])

with tabs[0]:
    render_ewma_tab()

with tabs[1]:
    render_fixed_income_tab()

with tabs[2]:
    render_options_tab()

with tabs[3]:
    render_stress_testing_tab()

