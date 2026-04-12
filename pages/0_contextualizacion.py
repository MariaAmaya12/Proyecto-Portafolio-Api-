import streamlit as st

from src.config import ASSETS, GLOBAL_BENCHMARK, ensure_project_dirs

ensure_project_dirs()

st.set_page_config(page_title="Contextualización de activos", layout="wide")

st.title("Contextualización de activos")
st.caption(
    "Resumen cualitativo y lectura de riesgo de cada activo del portafolio, "
    "coherente con los módulos de CAPM, VaR/CVaR, Markowitz y benchmark."
)

# -----------------------------------------------------------------------------
# Contexto cualitativo enriquecido por activo
# -----------------------------------------------------------------------------
ASSET_CONTEXT = {
    "Seven & i Holdings": {
        "sector": "Retail de conveniencia",
        "tipo_exposicion": "Consumo defensivo en Asia",
        "resumen": (
            "Holding japonés centrado en el negocio de tiendas de conveniencia. "
            "Dentro del portafolio representa exposición a consumo recurrente y "
            "a un formato minorista que suele comportarse de forma más defensiva "
            "que sectores cíclicos como energía."
        ),
        "tesis": [
            "Aporta exposición a consumo básico y gasto recurrente en Japón.",
            "Permite incorporar diversificación geográfica hacia Asia desarrollada.",
            "Es útil para contrastar un retail defensivo frente a activos más cíclicos del portafolio.",
        ],
        "riesgos": [
            "Presión sobre márgenes por costos operativos y competencia.",
            "Desaceleración del consumo interno en Japón.",
            "Riesgo de ejecución en el negocio principal de conveniencia.",
        ],
        "catalizadores": [
            "Mejoras operativas en tiendas y eficiencia comercial.",
            "Resultados trimestrales sólidos en el negocio de conveniencia.",
            "Fortalecimiento del posicionamiento del formato de proximidad.",
        ],
        "drivers_riesgo": [
            "Consumo doméstico japonés.",
            "Márgenes del negocio minorista.",
            "Tipo de cambio y percepción del mercado japonés.",
        ],
        "lectura_riesgo": (
            "En términos de riesgo, este activo debería comportarse como una pieza "
            "relativamente defensiva dentro del portafolio. En CAPM podría mostrar "
            "una beta más moderada que BP, mientras que en Markowitz puede ayudar a "
            "diversificar frente a energía y frente a Latinoamérica."
        ),
        "rol_portafolio": (
            "Activo defensivo asiático orientado a consumo recurrente y retail de conveniencia."
        ),
    },
    "Alimentation Couche-Tard": {
        "sector": "Convenience & mobility",
        "tipo_exposicion": "Consumo de proximidad y movilidad en Norteamérica y Europa",
        "resumen": (
            "Compañía canadiense líder en conveniencia y movilidad, con presencia internacional. "
            "En el portafolio representa una combinación entre consumo de proximidad, tráfico vial "
            "y negocio asociado a estaciones de servicio y conveniencia."
        ),
        "tesis": [
            "Aporta exposición a un operador global con huella internacional.",
            "Combina ventas minoristas con exposición al negocio de movilidad.",
            "Diversifica frente a Asia, Latinoamérica y energía pura.",
        ],
        "riesgos": [
            "Dependencia del tráfico, movilidad y entorno de consumo.",
            "Presión de costos operativos y laborales.",
            "Sensibilidad de parte del negocio a combustibles y márgenes de movilidad.",
        ],
        "catalizadores": [
            "Expansión internacional y crecimiento inorgánico.",
            "Mejoras en eficiencia y ventas comparables.",
            "Fortaleza operativa de la red Circle K y del negocio de conveniencia.",
        ],
        "drivers_riesgo": [
            "Tráfico de clientes y movilidad.",
            "Márgenes de combustible y tienda.",
            "Consumo en Norteamérica y Europa.",
        ],
        "lectura_riesgo": (
            "Este activo suele ocupar una posición intermedia: no es tan defensivo como "
            "retail alimentario puro, pero tampoco tan cíclico como energía. En VaR/CVaR "
            "puede amplificar caídas cuando se deteriora el consumo o la movilidad, aunque "
            "en Markowitz puede aportar diversificación por modelo de negocio."
        ),
        "rol_portafolio": (
            "Activo de conveniencia y movilidad con exposición internacional y perfil mixto entre defensivo y cíclico."
        ),
    },
    "FEMSA": {
        "sector": "Retail de proximidad, bebidas y negocios relacionados",
        "tipo_exposicion": "Consumo y diversificación empresarial en Latinoamérica",
        "resumen": (
            "Empresa mexicana con exposición a retail de proximidad, bebidas y negocios "
            "relacionados. Dentro del portafolio cumple el papel de activo latinoamericano "
            "diversificado, con sensibilidad tanto al consumo regional como a variables "
            "macroeconómicas y cambiarias."
        ),
        "tesis": [
            "Representa el componente latinoamericano del portafolio.",
            "Combina negocios relativamente defensivos con exposición regional.",
            "Aporta diversificación frente a Asia, Europa y energía.",
        ],
        "riesgos": [
            "Riesgo cambiario y sensibilidad al entorno macro regional.",
            "Presión en márgenes del retail y costos operativos.",
            "Dependencia parcial del comportamiento del consumo en América Latina.",
        ],
        "catalizadores": [
            "Mejora del entorno macro regional.",
            "Crecimiento operativo en formatos de proximidad.",
            "Fortaleza de negocios vinculados al consumo recurrente.",
        ],
        "drivers_riesgo": [
            "Tipo de cambio y tasas en la región.",
            "Consumo masivo en Latinoamérica.",
            "Ejecución operativa de sus unidades de negocio.",
        ],
        "lectura_riesgo": (
            "En CAPM este activo puede reflejar tanto riesgo sistemático del mercado mexicano "
            "como riesgo adicional por exposición regional. En VaR/CVaR puede ser sensible a "
            "episodios de depreciación cambiaria o deterioro macro. En Markowitz es clave porque "
            "introduce una fuente de riesgo distinta a la europea y asiática."
        ),
        "rol_portafolio": (
            "Activo latinoamericano diversificado que agrega exposición regional, consumo y riesgo cambiario."
        ),
    },
    "BP": {
        "sector": "Energía",
        "tipo_exposicion": "Ciclo energético global",
        "resumen": (
            "Compañía energética integrada con exposición a distintas etapas de la cadena de valor "
            "del sector. Dentro del portafolio introduce sensibilidad al ciclo energético global y "
            "una fuente de riesgo claramente distinta a los activos de retail y consumo."
        ),
        "tesis": [
            "Incorpora exposición directa al sector energético.",
            "Diversifica sectorialmente frente al bloque de retail y consumo.",
            "Permite capturar movimientos del ciclo global de petróleo, gas y energía.",
        ],
        "riesgos": [
            "Alta sensibilidad a precios internacionales de energía.",
            "Riesgo geopolítico y regulatorio.",
            "Volatilidad elevada por cambios de estrategia, inversión y transición energética.",
        ],
        "catalizadores": [
            "Recuperación de precios energéticos.",
            "Resultados sólidos en upstream/downstream.",
            "Mejoras en eficiencia y claridad estratégica.",
        ],
        "drivers_riesgo": [
            "Precio del petróleo y gas.",
            "Entorno geopolítico global.",
            "Percepción del mercado sobre transición energética.",
        ],
        "lectura_riesgo": (
            "Es el activo más claramente cíclico del portafolio. En CAPM debería ser de los más "
            "sensibles al mercado; en VaR/CVaR puede aumentar la cola izquierda del portafolio en "
            "episodios adversos de commodities o geopolítica. En Markowitz, sin embargo, también puede "
            "mejorar la diversificación al no depender del mismo motor que retail."
        ),
        "rol_portafolio": (
            "Activo cíclico de energía que introduce riesgo global de commodities y diversificación sectorial."
        ),
    },
    "Carrefour": {
        "sector": "Retail alimentario",
        "tipo_exposicion": "Consumo básico en Europa",
        "resumen": (
            "Grupo francés de comercio alimentario y distribución minorista con enfoque multi-formato. "
            "Dentro del portafolio aporta exposición a consumo básico en Europa y funciona como una pieza "
            "más defensiva frente a activos con mayor ciclicidad."
        ),
        "tesis": [
            "Aporta exposición a retail alimentario y consumo básico.",
            "Introduce diversificación europea dentro del portafolio.",
            "Es comparable con otros formatos de retail del portafolio, pero con perfil más defensivo.",
        ],
        "riesgos": [
            "Presión competitiva en distribución minorista.",
            "Sensibilidad de márgenes a inflación de costos.",
            "Debilidad del consumo en Europa.",
        ],
        "catalizadores": [
            "Mejoras operativas y eficiencia comercial.",
            "Recuperación del consumo y tráfico en tiendas.",
            "Avances en estrategia omnicanal y formatos de proximidad.",
        ],
        "drivers_riesgo": [
            "Consumo europeo.",
            "Márgenes del retail alimentario.",
            "Competencia de precios y costos logísticos.",
        ],
        "lectura_riesgo": (
            "Este activo puede leerse como una pieza defensiva europea. En CAPM podría exhibir una beta "
            "más moderada que BP y quizá más cercana a negocios de consumo. En Markowitz aporta equilibrio "
            "al portafolio porque no comparte exactamente el mismo patrón de riesgo que energía ni que "
            "Latinoamérica."
        ),
        "rol_portafolio": (
            "Activo defensivo europeo enfocado en comercio alimentario y consumo básico."
        ),
    },
}


# -----------------------------------------------------------------------------
# Funciones auxiliares
# -----------------------------------------------------------------------------
def render_list(items: list[str], icon: str = "-") -> None:
    """Renderiza listas de texto de forma uniforme."""
    for item in items:
        st.write(f"{icon} {item}")


def render_asset_card(asset_name: str, meta: dict, ctx: dict) -> None:
    """
    Construye la tarjeta principal de contextualización para un activo.
    Mantiene coherencia con el resto del dashboard: benchmark local, benchmark
    global y relación con módulos cuantitativos.
    """
    st.subheader(f"{asset_name} ({meta['ticker']})")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Ticker", meta["ticker"])
    c2.metric("País", meta["country"])
    c3.metric("Benchmark local", meta["benchmark_local"])
    c4.metric("Benchmark global", GLOBAL_BENCHMARK)

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        [
            "Perfil general",
            "Tesis de inversión",
            "Riesgos y catalizadores",
            "Lectura de riesgo",
            "Rol en portafolio",
        ]
    )

    with tab1:
        st.markdown(f"**Sector:** {ctx['sector']}")
        st.markdown(f"**Tipo de exposición:** {ctx['tipo_exposicion']}")
        st.write(ctx["resumen"])

    with tab2:
        st.markdown("**Puntos clave de la tesis:**")
        render_list(ctx["tesis"])

    with tab3:
        col_risk, col_cat = st.columns(2)

        with col_risk:
            st.markdown("**Riesgos principales**")
            render_list(ctx["riesgos"])

        with col_cat:
            st.markdown("**Catalizadores potenciales**")
            render_list(ctx["catalizadores"])

    with tab4:
        st.markdown("**Drivers de riesgo que pueden mover el activo**")
        render_list(ctx["drivers_riesgo"])

        st.markdown("**Lectura cuantitativa esperada dentro del proyecto**")
        st.write(ctx["lectura_riesgo"])

        st.markdown(
            """
**Conexión con los módulos cuantitativos**
- **CAPM:** ayuda a interpretar la beta frente al benchmark local.
- **VaR/CVaR:** ayuda a entender qué tipo de shock puede empeorar la cola de pérdidas.
- **Markowitz:** ayuda a justificar por qué un activo diversifica o concentra riesgo.
- **Señales:** permite leer con más criterio una alerta técnica de compra o venta.
"""
        )

    with tab5:
        st.info(ctx["rol_portafolio"])
        st.markdown(
            f"""
**Lectura dentro del dashboard**
- Se compara contra el benchmark local **{meta['benchmark_local']}** en el módulo **CAPM**.
- Forma parte del portafolio global que luego se contrasta con **{GLOBAL_BENCHMARK}**.
- Su comportamiento influye en módulos como **VaR/CVaR**, **Markowitz** y **señales de trading**.
- Su valor no es solo individual: también importa por la **diversificación** que aporta al conjunto.
"""
        )


# -----------------------------------------------------------------------------
# Resumen general del portafolio
# -----------------------------------------------------------------------------
with st.expander("Ver lectura general del portafolio", expanded=True):
    st.markdown(
        """
Este portafolio combina **activos defensivos de consumo y retail** con un **activo cíclico de energía**,
y además incorpora **diversificación geográfica** entre Asia, Norteamérica, Latinoamérica y Europa.

### Lectura estratégica del conjunto
- **Bloque defensivo:** Seven & i Holdings y Carrefour.
- **Bloque mixto consumo-movilidad:** Alimentation Couche-Tard.
- **Bloque regional latinoamericano diversificado:** FEMSA.
- **Bloque cíclico global:** BP.

### Implicación para el análisis de riesgo
- En **CAPM**, no todos los activos deberían reaccionar igual frente al mercado.
- En **VaR/CVaR**, BP podría amplificar escenarios extremos más que los activos defensivos.
- En **Markowitz**, la combinación sectorial y geográfica puede mejorar diversificación.
- En **benchmark**, el portafolio completo se compara contra **ACWI**, mientras que cada activo usa su benchmark local en análisis individuales.
"""
    )


# -----------------------------------------------------------------------------
# Controles laterales
# -----------------------------------------------------------------------------
with st.sidebar:
    st.header("Configuración")
    modo = st.radio("Vista", ["Un activo", "Todos los activos"], index=0)

    st.markdown("---")
    st.markdown("**Ayuda de interpretación**")
    st.caption(
        "Esta página no reemplaza el análisis cuantitativo. "
        "Su función es dar contexto estructural para interpretar los demás módulos."
    )

asset_names = list(ASSETS.keys())

# -----------------------------------------------------------------------------
# Render principal
# -----------------------------------------------------------------------------
if modo == "Un activo":
    selected_asset = st.selectbox("Selecciona un activo", asset_names)
    render_asset_card(
        selected_asset,
        ASSETS[selected_asset],
        ASSET_CONTEXT[selected_asset],
    )
else:
    for asset_name in asset_names:
        render_asset_card(
            asset_name,
            ASSETS[asset_name],
            ASSET_CONTEXT[asset_name],
        )
        st.divider()