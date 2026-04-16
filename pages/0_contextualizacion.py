import streamlit as st

from src.config import ASSETS, GLOBAL_BENCHMARK, ensure_project_dirs

ensure_project_dirs()

st.set_page_config(page_title="Contextualización de activos", layout="wide")


# ==============================
# Estilos UI
# ==============================
def inject_ui_css():
    st.markdown(
        """
        <style>
        .section-box {
            background: #ffffff;
            border: 1px solid rgba(15, 23, 42, 0.08);
            border-radius: 18px;
            padding: 16px 18px;
            box-shadow: 0 4px 14px rgba(15, 23, 42, 0.06);
            margin-bottom: 0.8rem;
        }
        .section-title {
            font-size: 1rem;
            font-weight: 700;
            color: #0f172a;
            margin-bottom: 0.2rem;
        }
        .section-subtitle {
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
        <div class="section-box">
            <div class="section-title">{title}</div>
            <div class="section-subtitle">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


inject_ui_css()

st.title("Módulo 0 - Contextualización de activos")
st.caption(
    "Resumen cualitativo y lectura estructural de cada activo del portafolio, en coherencia con los módulos de CAPM, VaR/CVaR, Markowitz y benchmark."
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
    for item in items:
        st.write(f"{icon} {item}")


def render_asset_header(asset_name: str, meta: dict, ctx: dict) -> None:
    st.markdown(f"## {asset_name}")
    st.caption(ctx["rol_portafolio"])

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Ticker", meta["ticker"])
    c2.metric("País", meta["country"])
    c3.metric("Benchmark local", meta["benchmark_local"])
    c4.metric("Benchmark global", GLOBAL_BENCHMARK)


def render_asset_tabs(asset_name: str, meta: dict, ctx: dict) -> None:
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
        section_intro(
            "Perfil general del activo",
            "Descripción estructural del negocio, tipo de exposición y función económica principal dentro del portafolio.",
        )
        st.markdown(f"**Sector:** {ctx['sector']}")
        st.markdown(f"**Tipo de exposición:** {ctx['tipo_exposicion']}")
        st.write(ctx["resumen"])

    with tab2:
        section_intro(
            "Tesis de inversión",
            "Ideas principales que justifican por qué este activo puede tener sentido dentro del conjunto del portafolio.",
        )
        render_list(ctx["tesis"])

    with tab3:
        section_intro(
            "Riesgos y catalizadores",
            "Factores que pueden deteriorar o mejorar la lectura del activo desde una perspectiva fundamental y estratégica.",
        )
        col_risk, col_cat = st.columns(2)

        with col_risk:
            st.markdown("#### Riesgos principales")
            render_list(ctx["riesgos"])

        with col_cat:
            st.markdown("#### Catalizadores potenciales")
            render_list(ctx["catalizadores"])

    with tab4:
        section_intro(
            "Lectura de riesgo",
            "Cómo debería interpretarse este activo cuando luego se analice en los módulos cuantitativos del proyecto.",
        )
        st.markdown("#### Drivers de riesgo")
        render_list(ctx["drivers_riesgo"])

        st.markdown("#### Lectura cuantitativa esperada")
        st.write(ctx["lectura_riesgo"])

        st.markdown(
            """
#### Conexión con los módulos cuantitativos
- **CAPM:** ayuda a interpretar la beta frente al benchmark local.
- **VaR/CVaR:** ayuda a entender qué tipo de shock puede empeorar la cola de pérdidas.
- **Markowitz:** ayuda a justificar por qué un activo diversifica o concentra riesgo.
- **Señales:** permite leer con más criterio una alerta técnica de compra o venta.
"""
        )

    with tab5:
        section_intro(
            "Rol en el portafolio",
            "Función estratégica del activo dentro del conjunto total del dashboard.",
        )
        st.info(ctx["rol_portafolio"])
        st.markdown(
            f"""
**Lectura dentro del dashboard**
- Se compara contra el benchmark local **{meta['benchmark_local']}** en el módulo **CAPM**.
- Forma parte del portafolio global que luego se contrasta con **{GLOBAL_BENCHMARK}**.
- Su comportamiento influye en módulos como **VaR/CVaR**, **Markowitz** y **señales**.
- Su valor no es solo individual: también importa por la **diversificación** que aporta al conjunto.
"""
        )


def render_asset_card(asset_name: str, meta: dict, ctx: dict) -> None:
    render_asset_header(asset_name, meta, ctx)
    render_asset_tabs(asset_name, meta, ctx)


# -----------------------------------------------------------------------------
# Sidebar
# -----------------------------------------------------------------------------
asset_names = list(ASSETS.keys())

with st.sidebar:
    st.header("Parámetros")

    vista = st.radio(
        "Modo de visualización",
        ["Vista resumida", "Un activo", "Todos los activos"],
        index=0,
    )

    st.divider()
    st.subheader("Opciones de visualización")

    mostrar_lectura_general = st.checkbox(
        "Mostrar lectura general del portafolio",
        value=True,
    )

    expandir_resumen = st.checkbox(
        "Expandir resumen general",
        value=True,
    )

    st.divider()
    with st.expander("Filtros secundarios"):
        selected_asset = None
        if vista == "Un activo":
            selected_asset = st.selectbox("Selecciona un activo", asset_names)

        mostrar_roles = st.checkbox("Resaltar rol en portafolio", value=True)
        mostrar_conexion_modulos = st.checkbox("Mostrar conexión con módulos", value=True)

    st.divider()
    st.caption(
        "Esta página no reemplaza el análisis cuantitativo. Su función es dar contexto estructural para interpretar los demás módulos."
    )


# -----------------------------------------------------------------------------
# Lectura general del portafolio
# -----------------------------------------------------------------------------
if mostrar_lectura_general:
    with st.expander("Ver lectura general del portafolio", expanded=expandir_resumen):
        section_intro(
            "Lectura estratégica del conjunto",
            "Este bloque resume cómo se combinan perfiles defensivos, cíclicos y regionales dentro del portafolio.",
        )

        st.markdown(
            f"""
Este portafolio combina **activos defensivos de consumo y retail** con un **activo cíclico de energía**,
e incorpora además **diversificación geográfica** entre Asia, Norteamérica, Latinoamérica y Europa.

### Estructura del portafolio
- **Bloque defensivo:** Seven & i Holdings y Carrefour.
- **Bloque mixto consumo-movilidad:** Alimentation Couche-Tard.
- **Bloque regional latinoamericano diversificado:** FEMSA.
- **Bloque cíclico global:** BP.

### Implicación para el análisis de riesgo
- En **CAPM**, no todos los activos deberían reaccionar igual frente al mercado.
- En **VaR/CVaR**, BP podría amplificar escenarios extremos más que los activos defensivos.
- En **Markowitz**, la combinación sectorial y geográfica puede mejorar diversificación.
- En **benchmark**, el portafolio completo se compara contra **{GLOBAL_BENCHMARK}**, mientras que cada activo usa su benchmark local en análisis individuales.
"""
        )


# -----------------------------------------------------------------------------
# Vista resumida
# -----------------------------------------------------------------------------
if vista == "Vista resumida":
    st.markdown("### Vista resumida del portafolio")
    section_intro(
        "Resumen por activo",
        "Esta vista permite comparar rápidamente sector, exposición y rol estratégico de cada activo.",
    )

    for asset_name in asset_names:
        meta = ASSETS[asset_name]
        ctx = ASSET_CONTEXT[asset_name]

        with st.container(border=True):
            st.markdown(f"#### {asset_name} ({meta['ticker']})")

            c1, c2, c3 = st.columns(3)
            c1.markdown(f"**Sector:** {ctx['sector']}")
            c2.markdown(f"**País:** {meta['country']}")
            c3.markdown(f"**Benchmark local:** {meta['benchmark_local']}")

            st.write(ctx["resumen"])

            if mostrar_roles:
                st.info(f"**Rol en portafolio:** {ctx['rol_portafolio']}")

            if mostrar_conexion_modulos:
                st.caption(
                    f"Conecta principalmente con CAPM ({meta['benchmark_local']}), benchmark global ({GLOBAL_BENCHMARK}), VaR/CVaR y Markowitz."
                )

# -----------------------------------------------------------------------------
# Vista un activo
# -----------------------------------------------------------------------------
elif vista == "Un activo":
    selected_asset = selected_asset or asset_names[0]
    render_asset_card(
        selected_asset,
        ASSETS[selected_asset],
        ASSET_CONTEXT[selected_asset],
    )

# -----------------------------------------------------------------------------
# Vista todos los activos
# -----------------------------------------------------------------------------
else:
    st.markdown("### Todos los activos")
    section_intro(
        "Lectura completa del portafolio",
        "Aquí puedes recorrer cada activo con su contexto cualitativo completo y su conexión con el resto del dashboard.",
    )

    for asset_name in asset_names:
        render_asset_card(
            asset_name,
            ASSETS[asset_name],
            ASSET_CONTEXT[asset_name],
        )
        st.divider()