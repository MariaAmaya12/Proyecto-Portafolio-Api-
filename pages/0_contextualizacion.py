from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.api.market import get_market_bundle
from src.config import ASSETS, DEFAULT_END_DATE, GLOBAL_BENCHMARK, ensure_project_dirs
from src.context_events import CONTEXT_EVENTS
from src.ui_navigation import render_sidebar_navigation
from src.ui_style import apply_global_typography, render_page_title

ensure_project_dirs()

st.set_page_config(page_title="Contextualización de activos", layout="wide")
apply_global_typography()
render_sidebar_navigation()


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
            font-size: 1.12rem;
            font-weight: 800;
            color: #0f172a;
            margin-bottom: 0.2rem;
        }
        .section-subtitle {
            font-size: 0.86rem;
            color: #64748b;
            line-height: 1.45;
        }
        .section-caption {
            font-size: 0.82rem;
            color: #64748b;
            line-height: 1.45;
            margin-top: 0.45rem;
        }
        .context-event {
            background: #f8fbff;
            border: 1px solid #d5e4fb;
            border-radius: 14px;
            padding: 12px 14px;
            margin-bottom: 0.6rem;
        }
        .context-event-title {
            color: #274c77;
            font-weight: 800;
            margin-bottom: 0.2rem;
        }
        .context-event-meta {
            color: #64748b;
            font-size: 0.82rem;
            margin-bottom: 0.25rem;
        }
        .role-badge {
            display: inline-block;
            background: #eaf3ff;
            border: 1px solid #c9ddfc;
            border-radius: 999px;
            color: #0f3d75;
            font-weight: 800;
            font-size: 0.82rem;
            padding: 0.36rem 0.62rem;
            margin-top: 0.35rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def section_intro(title: str, subtitle: str, caption: str | None = None):
    caption_html = f'<div class="section-caption">{caption}</div>' if caption else ""
    st.markdown(
        f"""
        <div class="section-box">
            <div class="section-title">{title}</div>
            <div class="section-subtitle">{subtitle}</div>
            {caption_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


inject_ui_css()

render_page_title(
    "Módulo 0 - Contextualización de activos",
    "Contexto cualitativo: qué hace cada empresa, qué riesgos la mueven y qué aporta al portafolio.",
)

# -----------------------------------------------------------------------------
# Contexto cualitativo enriquecido por activo
# -----------------------------------------------------------------------------
ASSET_CONTEXT = {
    "Seven & i Holdings": {
        "tags": ["retail", "consumer", "japan"],
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
        "resumen_sencillo": "Cadena japonesa de tiendas tipo 7-Eleven. Vende productos de consumo diario y suele moverse de forma más estable que sectores cíclicos.",
        "rol_sencillo": "Rol: Defensivo (consumo diario)",
        "drivers_sencillos": [
            "Consumo diario en Japón",
            "Costos operativos y márgenes",
            "Tipo de cambio (JPY)",
        ],
        "ingresos_sencillos": [
            "Ventas en tiendas de conveniencia y retail de proximidad.",
            "Ingresos ligados a productos de consumo diario y operación comercial en Japón.",
        ],
        "opera_sencillo": "Opera principalmente en Japón, con exposición al consumidor asiático desarrollado.",
        "mueve_precio_sencillo": [
            "Cambios en consumo diario, márgenes y costos operativos.",
            "Percepción del mercado sobre Japón y movimientos del yen.",
        ],
    },
    "Alimentation Couche-Tard": {
        "tags": ["retail", "consumer", "mobility"],
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
        "resumen_sencillo": "Operador de tiendas de conveniencia (Circle K) y estaciones asociadas. Mezcla ventas en tienda y movilidad, por eso es ‘mixto’: parte defensivo, parte cíclico.",
        "rol_sencillo": "Rol: Mixto (consumo + movilidad)",
        "drivers_sencillos": [
            "Tráfico de clientes y movilidad",
            "Márgenes de combustible y tienda",
            "Consumo en Canadá/EE. UU./Europa",
        ],
        "ingresos_sencillos": [
            "Ventas en tiendas de conveniencia, especialmente bajo marcas como Circle K.",
            "Ingresos asociados a estaciones, movilidad y consumo rápido.",
        ],
        "opera_sencillo": "Opera en Canadá, Estados Unidos, Europa y otros mercados internacionales.",
        "mueve_precio_sencillo": [
            "Tráfico de clientes, movilidad y fortaleza del consumo.",
            "Márgenes de combustible, tienda y costos operativos.",
        ],
    },
    "FEMSA": {
        "tags": ["retail", "consumer", "latam", "fx"],
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
        "resumen_sencillo": "Grupo mexicano con negocios ligados a consumo (ej. retail de proximidad) y presencia regional. Su desempeño puede verse afectado por el entorno macro y el tipo de cambio.",
        "rol_sencillo": "Rol: Regional (LatAm + FX)",
        "drivers_sencillos": [
            "Consumo en México/LatAm",
            "Tipo de cambio (MXN) y tasas",
            "Costos y márgenes operativos",
        ],
        "ingresos_sencillos": [
            "Negocios ligados a consumo, retail de proximidad y operaciones regionales.",
            "Ingresos sensibles al gasto de hogares y al entorno macro de Latinoamérica.",
        ],
        "opera_sencillo": "Opera principalmente en México y Latinoamérica, con exposición regional.",
        "mueve_precio_sencillo": [
            "Consumo en México/LatAm, tasas y expectativas macro.",
            "Tipo de cambio, costos y márgenes operativos.",
        ],
    },
    "BP": {
        "tags": ["energy", "global"],
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
        "resumen_sencillo": "Empresa de energía: su precio suele depender más del petróleo/gas y del contexto global que del consumo minorista. Tiende a ser el activo más volátil del portafolio.",
        "rol_sencillo": "Rol: Cíclico (energía)",
        "drivers_sencillos": [
            "Precio del petróleo y gas",
            "Eventos geopolíticos / OPEP / conflicto",
            "Expectativas sobre transición energética",
        ],
        "ingresos_sencillos": [
            "Producción, refinación, comercialización y negocios ligados a petróleo y gas.",
            "Ingresos muy conectados al ciclo energético global.",
        ],
        "opera_sencillo": "Opera globalmente, con exposición a mercados energéticos internacionales.",
        "mueve_precio_sencillo": [
            "Precio del petróleo y gas, OPEP y eventos geopolíticos.",
            "Expectativas sobre transición energética, inversión y regulación.",
        ],
    },
    "Carrefour": {
        "tags": ["retail", "consumer", "europe"],
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
        "resumen_sencillo": "Retail alimentario en Europa. Al vender productos básicos, suele ser más defensivo; aun así, sus márgenes dependen de costos e inflación.",
        "rol_sencillo": "Rol: Defensivo (consumo básico)",
        "drivers_sencillos": [
            "Consumo e inflación en Europa",
            "Competencia de precios",
            "Costos logísticos y márgenes",
        ],
        "ingresos_sencillos": [
            "Ventas de alimentos y productos básicos en supermercados e hipermercados.",
            "Ingresos ligados al consumo frecuente de hogares europeos.",
        ],
        "opera_sencillo": "Opera sobre todo en Europa, con Carrefour como referencia francesa.",
        "mueve_precio_sencillo": [
            "Inflación, competencia de precios y consumo en Europa.",
            "Costos logísticos, márgenes y eficiencia operativa.",
        ],
    },
}


# -----------------------------------------------------------------------------
# Funciones auxiliares
# -----------------------------------------------------------------------------
def render_list(items: list[str], icon: str = "-") -> None:
    for item in items:
        st.write(f"{icon} {item}")


def render_asset_header(asset_name: str, meta: dict, ctx: dict) -> None:
    st.header(asset_name)
    st.caption(ctx["rol_portafolio"])

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Ticker", meta["ticker"])
    c2.metric("País", meta["country"])
    c3.metric("Benchmark local", meta["benchmark_local"])
    c4.metric("Benchmark global", GLOBAL_BENCHMARK)


def render_business_snapshot(asset_name: str, meta: dict, ctx: dict) -> None:
    section_intro(
        "Ficha rápida del negocio",
        ctx["resumen_sencillo"],
    )

    col_business, col_market = st.columns(2)
    with col_business:
        st.subheader("Qué hace")
        st.write(f"Empresa del sector {ctx['sector'].lower()} con exposición a {ctx['tipo_exposicion'].lower()}.")

        st.subheader("Qué vende / de dónde salen ingresos")
        render_list(ctx["ingresos_sencillos"])

        st.subheader("Dónde opera")
        st.write(f"- {ctx['opera_sencillo']}")

    with col_market:
        st.subheader("Qué suele mover el precio")
        render_list(ctx["mueve_precio_sencillo"])

        st.subheader("Rol en portafolio")
        st.markdown(
            f'<span class="role-badge">{ctx["rol_sencillo"]}</span>',
            unsafe_allow_html=True,
        )
        st.caption(f"Ticker: {meta['ticker']} | Benchmark local: {meta['benchmark_local']}")


def render_asset_tabs(
    asset_name: str,
    meta: dict,
    ctx: dict,
    show_role: bool = True,
    show_module_connection: bool = True,
) -> None:
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
            "Qué hace la empresa",
            "Lectura simple del negocio antes de interpretar métricas, gráficos o señales.",
        )
        st.markdown(f"**Sector:** {ctx['sector']}")
        st.markdown(f"**Tipo de exposición:** {ctx['tipo_exposicion']}")
        st.write(ctx["resumen_sencillo"])
        st.markdown("**De dónde salen los ingresos principales**")
        render_list(ctx["ingresos_sencillos"])
        st.markdown("**Dónde opera**")
        st.write(ctx["opera_sencillo"])

    with tab2:
        section_intro(
            "Por qué puede aportar al portafolio",
            "Puntos concretos que explican su utilidad dentro del conjunto.",
        )
        render_list(ctx["tesis"])

    with tab3:
        section_intro(
            "Qué la mueve",
            "Factores que pueden empujar el precio hacia arriba o hacia abajo.",
        )
        col_risk, col_cat = st.columns(2)

        with col_risk:
            st.subheader("Riesgos que pueden presionar el precio")
            render_list(ctx["riesgos"])

        with col_cat:
            st.subheader("Factores que pueden ayudar")
            render_list(ctx["catalizadores"])

    with tab4:
        section_intro(
            "Lectura de riesgo en lenguaje simple",
            "Cómo conectar el negocio con CAPM, VaR/CVaR, Markowitz y señales.",
        )
        st.subheader("Qué suele mover su riesgo")
        render_list(ctx["drivers_sencillos"])

        st.subheader("Lectura para los módulos cuantitativos")
        st.write(ctx["lectura_riesgo"])

        if show_module_connection:
            st.markdown(
                """
#### Conexión con los módulos cuantitativos
- **CAPM:** ayuda a interpretar la beta frente al benchmark local.
- **VaR/CVaR:** ayuda a entender qué tipo de shock puede empeorar la cola de pérdidas.
- **Markowitz:** ayuda a justificar por qué un activo diversifica o concentra riesgo.
- **Señales:** permite leer con más criterio una alerta técnica de compra o venta.
"""
            )
        else:
            st.caption("La conexión con módulos está oculta por la configuración del sidebar.")

    with tab5:
        section_intro(
            "Rol en el portafolio",
            "Qué papel cumple este activo dentro del conjunto.",
        )
        if show_role:
            st.info(ctx["rol_sencillo"])
        else:
            st.caption("El rol en portafolio está oculto por la configuración del sidebar.")
        st.markdown(
            f"""
**Lectura dentro del dashboard**
- Se compara contra el benchmark local **{meta['benchmark_local']}** en el módulo **CAPM**.
- Forma parte del portafolio global que luego se contrasta con **{GLOBAL_BENCHMARK}**.
- Su comportamiento influye en módulos como **VaR/CVaR**, **Markowitz** y **señales**.
- Su valor no es solo individual: también importa por la **diversificación** que aporta al conjunto.
"""
        )


def render_asset_card(
    asset_name: str,
    meta: dict,
    ctx: dict,
    show_role: bool = True,
    show_module_connection: bool = True,
) -> None:
    render_asset_header(asset_name, meta, ctx)
    render_asset_tabs(asset_name, meta, ctx, show_role, show_module_connection)


def _asof_price(close: pd.Series, target_date: pd.Timestamp) -> tuple[pd.Timestamp | None, float | None]:
    clean = close.dropna().sort_index()
    if clean.empty:
        return None, None

    target_date = pd.Timestamp(target_date).tz_localize(None).normalize()
    available = clean.loc[clean.index <= target_date]
    if available.empty:
        return None, None

    price_date = pd.Timestamp(available.index[-1])
    return price_date, float(available.iloc[-1])


def _format_return(value: float | None) -> str:
    if value is None:
        return "Sin datos"
    return f"{value:.2%}"


def _context_events_for_asset(ctx: dict, start_date: pd.Timestamp, end_date: pd.Timestamp) -> list[dict]:
    asset_tags = set(ctx.get("tags", []))
    selected_events = []
    fallback_events = []

    for event in CONTEXT_EVENTS:
        event_start = pd.Timestamp(event["fecha_inicio"])
        event_end = pd.Timestamp(event["fecha_fin"])
        event_tags = set(event.get("tags", []))
        overlaps_range = event_start <= end_date and event_end >= start_date
        matches_asset = bool(asset_tags & event_tags) or "global" in event_tags
        if overlaps_range:
            if matches_asset:
                selected_events.append(event)
            else:
                fallback_events.append(event)

    selected_events.sort(key=lambda item: item["fecha_inicio"])
    fallback_events.sort(key=lambda item: item["fecha_inicio"])
    if len(selected_events) < 2:
        selected_ids = {event["titulo"] for event in selected_events}
        selected_events.extend(
            event for event in fallback_events if event["titulo"] not in selected_ids
        )
    return selected_events


def _nearest_context_events(
    events: list[dict],
    reference_date: pd.Timestamp,
    window_days: int = 45,
) -> list[dict]:
    nearby = []
    for event in events:
        event_start = pd.Timestamp(event["fecha_inicio"])
        event_end = pd.Timestamp(event["fecha_fin"])
        expanded_start = event_start - pd.Timedelta(days=window_days)
        expanded_end = event_end + pd.Timedelta(days=window_days)
        if expanded_start <= reference_date <= expanded_end:
            nearby.append(event)
    return nearby[:3]


def _weekly_drop_context(series: pd.Series, events: list[dict]) -> tuple[pd.Timestamp, float, list[dict]] | None:
    weekly = series.resample("W-FRI").last().dropna()
    weekly_returns = weekly.pct_change().dropna()
    if weekly_returns.empty:
        return None

    worst_date = pd.Timestamp(weekly_returns.idxmin())
    worst_return = float(weekly_returns.loc[worst_date])
    if worst_return > -0.08:
        return None

    return worst_date, worst_return, _nearest_context_events(events, worst_date)


def _event_importance_for_asset(event: dict, ctx: dict) -> str:
    asset_tags = set(ctx.get("tags", []))
    event_tags = set(event.get("tags", []))
    matches = asset_tags & event_tags

    if "energy" in matches:
        return "Importa porque el activo depende directamente del ciclo de petróleo, gas y riesgo geopolítico."
    if "latam" in matches or "fx" in matches:
        return "Importa porque puede afectar divisas, tasas y apetito por riesgo regional."
    if "retail" in matches or "consumer" in matches:
        return "Importa porque puede cambiar consumo, costos, márgenes y tráfico de clientes."
    if "global" in event_tags:
        return "Importa porque modifica el apetito global por riesgo y la volatilidad del mercado."
    return "Importa porque ayuda a explicar cambios de contexto durante el periodo analizado."


def render_performance_context(asset_name: str, meta: dict, ctx: dict) -> None:
    ticker = meta["ticker"]
    end_date = pd.Timestamp(DEFAULT_END_DATE).normalize()
    start_date = end_date - pd.DateOffset(years=5, days=10)

    st.header("Evolución del precio y contexto de mercado")
    st.subheader("Hoy vs 2 años vs 5 años")
    st.caption("Compara el precio actual con referencias de 2 y 5 años para leer desempeño relativo y contexto.")

    try:
        bundle = get_market_bundle(
            tickers=[ticker],
            start=str(start_date.date()),
            end=str(end_date.date()),
        )
    except Exception:
        st.warning(
            "No fue posible consultar el backend para esta sección. "
            "El contexto de desempeño se oculta temporalmente."
        )
        return

    close = bundle.get("close", pd.DataFrame())
    if close.empty or ticker not in close.columns:
        st.warning("No hay precios suficientes para construir esta lectura de desempeño.")
        return

    series = close[ticker].dropna().sort_index()
    if series.empty:
        st.warning("No hay precios suficientes para construir esta lectura de desempeño.")
        return

    latest_date, latest_price = _asof_price(series, end_date)
    date_2y, price_2y = _asof_price(series, end_date - pd.DateOffset(years=2))
    date_5y, price_5y = _asof_price(series, end_date - pd.DateOffset(years=5))

    ret_2y = (latest_price / price_2y - 1) if latest_price and price_2y else None
    ret_5y = (latest_price / price_5y - 1) if latest_price and price_5y else None

    st.subheader("Puntos de referencia")
    c1, c2, c3 = st.columns(3)
    c1.metric("Precio actual", "Sin datos" if latest_price is None else f"{latest_price:,.2f}")
    c2.metric("Retorno 2 años", _format_return(ret_2y))
    c3.metric("Retorno 5 años", _format_return(ret_5y))
    st.caption(
        "Las referencias 2y y 5y usan el cierre bursátil más cercano hacia atrás si la bolsa estaba cerrada "
        "(feriados/calendario)."
    )

    st.subheader("Precio normalizado (Base 100)")
    plot_series = series.loc[series.index >= (date_5y if date_5y is not None else series.index.min())]
    if not plot_series.empty:
        normalized = plot_series / plot_series.iloc[0] * 100
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=normalized.index,
                y=normalized,
                mode="lines",
                name=f"{asset_name} ({ticker})",
                line=dict(color="#2563EB", width=2.6),
            )
        )
        if date_5y is not None:
            fig.add_vline(x=date_5y, line_dash="dot", line_color="#64748b", line_width=1.4)
            fig.add_annotation(
                x=date_5y,
                y=normalized.max(),
                text="5 años",
                showarrow=False,
                yshift=14,
                font=dict(color="#475569", size=12),
                bgcolor="rgba(255,255,255,0.85)",
            )
        if date_2y is not None:
            fig.add_vline(x=date_2y, line_dash="dot", line_color="#D97706", line_width=1.4)
            fig.add_annotation(
                x=date_2y,
                y=normalized.max(),
                text="2 años",
                showarrow=False,
                yshift=32,
                font=dict(color="#92400E", size=12),
                bgcolor="rgba(255,255,255,0.85)",
            )

        fig.update_layout(
            template="plotly_white",
            title="Precio normalizado a 100",
            height=420,
            margin=dict(l=48, r=28, t=64, b=44),
            paper_bgcolor="#ffffff",
            plot_bgcolor="#ffffff",
            font=dict(color="#0f172a", size=12),
            hovermode="x unified",
            yaxis_title="Base 100",
            xaxis_title="Fecha",
        )
        fig.update_xaxes(showgrid=True, gridcolor="rgba(15, 23, 42, 0.08)")
        fig.update_yaxes(showgrid=True, gridcolor="rgba(15, 23, 42, 0.08)")
        st.plotly_chart(fig, width="stretch")
    st.caption("Base 100: todas las series parten en 100 para comparar desempeño relativo.")

    st.subheader("Qué estaba pasando en el mercado")
    events = _context_events_for_asset(ctx, series.index.min(), series.index.max())
    if not events:
        st.caption("No hay eventos guía configurados para este activo en el rango consultado.")
    for event in events[:4]:
        sources = event.get("sources") or event.get("fuentes") or []
        source_text = ", ".join(sources) if sources else "pendiente"
        st.markdown(
            f"""
            <div class="context-event">
                <div class="context-event-title">{event['titulo']}</div>
                <div class="context-event-meta">{event['fecha_inicio']} a {event['fecha_fin']} · {', '.join(event.get('tags', []))}</div>
                <div>{event['nota']}</div>
                <div><strong>Por qué importa para este activo:</strong> {_event_importance_for_asset(event, ctx)}</div>
                <div><strong>Fuente(s):</strong> {source_text}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    movement = _weekly_drop_context(series, events)
    if movement is not None:
        movement_date, movement_return, related_events = movement
        st.info(
            f"Movimiento destacado (posibles factores): una de las caídas semanales más fuertes fue "
            f"de {movement_return:.2%} alrededor de {movement_date.date()}. Este movimiento podría estar "
            "asociado a cambios de contexto de mercado, sin afirmar causalidad directa."
        )
        if related_events:
            st.caption("Eventos curados cercanos a esa fecha que podrían ayudar a contextualizar el movimiento:")
            for event in related_events:
                st.write(f"- {event['titulo']}: {event['nota']}")


# -----------------------------------------------------------------------------
# Sidebar
# -----------------------------------------------------------------------------
asset_names = list(ASSETS.keys())

with st.sidebar:
    st.header("Parámetros")

    vista = st.radio(
        "Vista",
        ["Resumen", "Un activo"],
        index=0,
    )

    selected_asset = None
    if vista == "Un activo":
        selected_asset = st.selectbox("Selecciona un activo", asset_names)

# -----------------------------------------------------------------------------
# Lectura general del portafolio
# -----------------------------------------------------------------------------
section_intro(
    "Lectura estratégica del conjunto",
    (
        "El portafolio combina consumo defensivo, movilidad, exposición regional y energía. "
        "Esta mezcla ayuda a interpretar por qué algunos activos estabilizan el conjunto y otros aportan más riesgo."
    ),
)


# -----------------------------------------------------------------------------
# Vista resumida
# -----------------------------------------------------------------------------
if vista == "Resumen":
    st.header("Vista resumida del portafolio")
    section_intro(
        "Resumen por activo",
        "Esta vista resume qué hace cada empresa, qué riesgos la mueven y qué papel cumple dentro del portafolio.",
        caption="Puente cualitativo para interpretar CAPM, VaR/CVaR y Markowitz.",
    )

    comparison = pd.DataFrame(
        [
            {
                "Activo": asset_name,
                "Ticker": ASSETS[asset_name]["ticker"],
                "País": ASSETS[asset_name]["country"],
                "Sector": ASSET_CONTEXT[asset_name]["sector"],
                "Tipo_exposicion": ASSET_CONTEXT[asset_name]["tipo_exposicion"],
                "Benchmark_local": ASSETS[asset_name]["benchmark_local"],
            }
            for asset_name in asset_names
        ]
    )
    table_styles = [
        {
            "selector": "th",
            "props": [
                ("background-color", "#EAF3FF"),
                ("color", "#0F3D75"),
                ("font-weight", "800"),
                ("border-bottom", "1px solid #C9DDFC"),
            ],
        }
    ]
    comparison_style = comparison.style.set_table_styles(table_styles)
    try:
        comparison_style = comparison_style.hide(axis="index")
    except AttributeError:
        comparison_style = comparison_style.hide_index()
    st.table(comparison_style)

    st.markdown(
        """
**Cómo leer el rol de cada activo**
- **Defensivo:** tiende a depender de consumo básico o recurrente.
- **Cíclico:** suele moverse más con petróleo, economía global o shocks externos.
- **Mixto/Regional:** combina consumo, movilidad, divisas o exposición geográfica específica.
"""
    )

    with st.expander("Ver fichas rápidas por activo (opcional)", expanded=False):
        asset_columns = st.columns(2)
        for index, asset_name in enumerate(asset_names):
            meta = ASSETS[asset_name]
            ctx = ASSET_CONTEXT[asset_name]

            with asset_columns[index % 2]:
                with st.container(border=True):
                    st.markdown(f"#### {asset_name} ({meta['ticker']})")
                    st.caption(
                        f"Sector: {ctx['sector']} | País: {meta['country']} | "
                        f"Benchmark local: {meta['benchmark_local']}"
                    )
                    st.write(ctx["resumen_sencillo"])

                    st.markdown("**Drivers clave**")
                    render_list(ctx["drivers_sencillos"])

                    st.markdown(
                        f'<span class="role-badge">{ctx["rol_sencillo"]}</span>',
                        unsafe_allow_html=True,
                    )

# -----------------------------------------------------------------------------
# Vista un activo
# -----------------------------------------------------------------------------
elif vista == "Un activo":
    selected_asset = selected_asset or asset_names[0]
    render_business_snapshot(
        selected_asset,
        ASSETS[selected_asset],
        ASSET_CONTEXT[selected_asset],
    )
    render_asset_card(
        selected_asset,
        ASSETS[selected_asset],
        ASSET_CONTEXT[selected_asset],
        True,
        True,
    )
    render_performance_context(
        selected_asset,
        ASSETS[selected_asset],
        ASSET_CONTEXT[selected_asset],
    )
