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
            border: 1px solid rgba(37, 99, 235, 0.12);
            border-radius: 10px;
            padding: 9px 11px;
            margin-bottom: 0.45rem;
        }
        .context-event-title {
            color: #274c77;
            font-weight: 800;
            font-size: 0.94rem;
            margin-bottom: 0.08rem;
        }
        .context-event-meta {
            color: #64748b;
            font-size: 0.76rem;
            margin-bottom: 0.22rem;
        }
        .context-event-body {
            color: #334155;
            font-size: 0.84rem;
            line-height: 1.34;
            margin-bottom: 0.18rem;
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
        .asset-hero {
            background: #ffffff;
            border: 1px solid rgba(15, 23, 42, 0.08);
            border-radius: 16px;
            padding: 18px 20px 16px;
            box-shadow: 0 8px 22px rgba(15, 23, 42, 0.07);
            margin-bottom: 1.05rem;
        }
        .asset-hero-title {
            color: #0f172a;
            font-size: 1.85rem;
            font-weight: 850;
            line-height: 1.15;
            margin-bottom: 0.25rem;
        }
        .asset-hero-subtitle {
            color: #475569;
            font-size: 0.96rem;
            line-height: 1.4;
            margin-bottom: 0.45rem;
        }
        .asset-hero-top {
            display: flex;
            align-items: flex-start;
            justify-content: space-between;
            gap: 1rem;
            margin-bottom: 0.85rem;
        }
        .asset-meta-grid,
        .price-metric-grid {
            display: grid;
            gap: 0.6rem;
        }
        .asset-meta-grid {
            grid-template-columns: repeat(4, minmax(0, 1fr));
        }
        .price-metric-grid {
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 0.75rem;
            margin-top: 0.35rem;
        }
        .asset-meta-item,
        .price-metric-item {
            background: #f8fafc;
            border: 1px solid rgba(15, 23, 42, 0.06);
            border-radius: 10px;
            padding: 11px 12px;
        }
        .price-metric-item {
            background: #ffffff;
            border-color: rgba(37, 99, 235, 0.16);
            padding: 14px 16px;
        }
        .asset-meta-label,
        .price-metric-label {
            color: #475569;
            font-size: 0.82rem;
            font-weight: 800;
            letter-spacing: 0;
            margin-bottom: 0.2rem;
        }
        .price-metric-label {
            color: #334155;
            font-size: 0.94rem;
            font-weight: 800;
            margin-bottom: 0.3rem;
        }
        .asset-meta-value,
        .price-metric-value {
            color: #0f172a;
            font-size: 1.12rem;
            font-weight: 900;
            line-height: 1.15;
        }
        .price-metric-value {
            color: #0f172a;
            font-size: 1.55rem;
            font-weight: 900;
            line-height: 1.08;
        }
        .section-heading-lite {
            margin: 1.1rem 0 0.45rem;
        }
        .section-heading-lite-title {
            color: #0f172a;
            font-size: 1.15rem;
            font-weight: 850;
            line-height: 1.2;
        }
        .section-heading-lite-subtitle {
            color: #64748b;
            font-size: 0.86rem;
            line-height: 1.35;
            margin-top: 0.12rem;
        }
        .compact-card {
            background: #ffffff;
            border: 1px solid rgba(15, 23, 42, 0.07);
            border-radius: 10px;
            padding: 11px 12px;
            min-height: 100%;
            box-shadow: 0 2px 8px rgba(15, 23, 42, 0.03);
        }
        .compact-card-title {
            color: #0f172a;
            font-weight: 800;
            font-size: 0.94rem;
            margin-bottom: 0.32rem;
        }
        .compact-card-body {
            color: #334155;
            font-size: 0.86rem;
            line-height: 1.36;
        }
        .compact-card-body ul {
            margin: 0.05rem 0 0;
            padding-left: 1rem;
        }
        .compact-card-body li {
            margin-bottom: 0.18rem;
        }
        .insight-box {
            background: #eef6ff;
            border-left: 4px solid #2563eb;
            border-radius: 10px;
            color: #334155;
            padding: 9px 12px;
            margin: 0.38rem 0 0.65rem;
            font-size: 0.87rem;
            line-height: 1.35;
        }
        .market-caption {
            color: #64748b;
            font-size: 0.8rem;
            line-height: 1.35;
            margin: 0.25rem 0 0.35rem;
        }
        @media (max-width: 900px) {
            .asset-hero-top {
                display: block;
            }
            .asset-meta-grid,
            .price-metric-grid {
                grid-template-columns: repeat(2, minmax(0, 1fr));
            }
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


def _bullets_html(items: list[str], limit: int = 3) -> str:
    return "<ul>" + "".join(f"<li>{item}</li>" for item in items[:limit]) + "</ul>"


def _compact_card(title: str, body: str | list[str]) -> None:
    body_html = _bullets_html(body) if isinstance(body, list) else body
    st.markdown(
        f"""
        <div class="compact-card">
            <div class="compact-card-title">{title}</div>
            <div class="compact-card-body">{body_html}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _section_heading(title: str, subtitle: str | None = None) -> None:
    subtitle_html = f'<div class="section-heading-lite-subtitle">{subtitle}</div>' if subtitle else ""
    st.markdown(
        f"""
        <div class="section-heading-lite">
            <div class="section-heading-lite-title">{title}</div>
            {subtitle_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def _metric_item(label: str, value: str) -> str:
    return (
        '<div class="asset-meta-item">'
        f'<div class="asset-meta-label">{label}</div>'
        f'<div class="asset-meta-value">{value}</div>'
        "</div>"
    )


def _price_metric_item(label: str, value: str) -> str:
    return (
        '<div class="price-metric-item">'
        f'<div class="price-metric-label">{label}</div>'
        f'<div class="price-metric-value">{value}</div>'
        "</div>"
    )


def _first_sentence(text: str) -> str:
    clean_text = " ".join(str(text).split())
    if ". " not in clean_text:
        return clean_text
    return clean_text.split(". ", 1)[0].rstrip(".") + "."


def _asset_subtitle(ctx: dict) -> str:
    return f"{ctx['sector']} con exposición a {ctx['tipo_exposicion'].lower()}."


def render_asset_executive_header(asset_name: str, meta: dict, ctx: dict) -> None:
    st.markdown(
        f"""
        <div class="asset-hero">
            <div class="asset-hero-top">
                <div>
                    <div class="asset-hero-title">{asset_name}</div>
                    <div class="asset-hero-subtitle">{_asset_subtitle(ctx)}</div>
                </div>
                <span class="role-badge">{ctx["rol_sencillo"]}</span>
            </div>
            <div class="asset-meta-grid">
                {_metric_item("Ticker", meta["ticker"])}
                {_metric_item("País", meta["country"])}
                {_metric_item("Benchmark local", meta["benchmark_local"])}
                {_metric_item("Benchmark global", GLOBAL_BENCHMARK)}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_business_understanding(ctx: dict) -> None:
    _section_heading(
        "Entender el negocio",
        "Perfil operativo del activo antes de interpretar precios, riesgo y diversificación.",
    )

    col_left, col_right = st.columns(2)
    with col_left:
        _compact_card(
            "Qué hace",
            f"Empresa del sector {ctx['sector'].lower()} con exposición a {ctx['tipo_exposicion'].lower()}.",
        )
        _compact_card("De dónde salen los ingresos", ctx["ingresos_sencillos"])

    with col_right:
        _compact_card("Dónde opera", ctx["opera_sencillo"])
        _compact_card("Qué suele mover el precio", ctx["mueve_precio_sencillo"])


def render_investment_quick_read(ctx: dict) -> None:
    _section_heading(
        "Lectura rápida para inversión",
        "Aporte, catalizadores y riesgos principales en formato ejecutivo.",
    )

    col_contribution, col_catalysts, col_risks = st.columns(3)
    with col_contribution:
        _compact_card("Qué aporta al portafolio", ctx["tesis"])
    with col_catalysts:
        _compact_card("Catalizadores", ctx["catalizadores"])
    with col_risks:
        _compact_card("Riesgos", ctx["riesgos"])


def render_portfolio_role(meta: dict, ctx: dict) -> None:
    _section_heading(
        "Rol en el portafolio",
        "Cierre cualitativo para conectar este activo con los módulos cuantitativos.",
    )

    col_div, col_when, col_next = st.columns(3)
    with col_div:
        _compact_card(
            "Cómo diversifica",
            [
                f"Añade exposición a {ctx['tipo_exposicion'].lower()}.",
                f"Introduce el mercado de {meta['country']} frente a otros riesgos geográficos.",
                ctx["rol_portafolio"],
            ],
        )
    with col_when:
        _compact_card(
            "Cuándo ayuda",
            [
                "Cuando su motor de riesgo difiere del resto del conjunto.",
                "Cuando la diversificación geográfica o sectorial reduce concentración.",
                "Cuando el contexto favorece sus drivers operativos principales.",
            ],
        )
    with col_next:
        _compact_card(
            "Qué esperar en módulos posteriores",
            [
                f"CAPM lo contrastará contra {meta['benchmark_local']}.",
                "VaR/CVaR evaluará su contribución a pérdidas extremas.",
                "Markowitz medirá su aporte a diversificación y eficiencia.",
            ],
        )


def render_single_asset_view(asset_name: str, meta: dict, ctx: dict) -> None:
    render_asset_executive_header(asset_name, meta, ctx)
    render_business_understanding(ctx)
    render_investment_quick_read(ctx)
    render_performance_context(asset_name, meta, ctx)
    render_portfolio_role(meta, ctx)


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


def _performance_insight(ret_3y: float | None, ret_1y: float | None) -> str:
    reference_return = ret_3y if ret_3y is not None else ret_1y
    if reference_return is None:
        tone = "una trayectoria que debe interpretarse con cautela por disponibilidad de datos"
    elif reference_return > 0.15:
        tone = "una trayectoria acumulada positiva"
    elif reference_return < -0.15:
        tone = "una trayectoria acumulada negativa"
    else:
        tone = "una trayectoria mixta o relativamente lateral"

    return (
        f"En la ventana observada, el activo muestra {tone}. "
        "Las correcciones deben leerse junto con el contexto operativo y macro, sin asumir causalidad directa."
    )


def _format_price(value: float | None) -> str:
    return "Sin datos" if value is None else f"{value:,.2f}"


def _return_interpretation(ret: float | None, years: int) -> str:
    if ret is None:
        return "No hay datos suficientes para calcular este retorno."
    if abs(ret) <= 0.005:
        if years == 1:
            return (
                "El activo se encuentra cerca del nivel que tenía hace 1 año, "
                "lo que sugiere un comportamiento reciente relativamente estable."
            )
        return (
            "El activo se mantiene cerca del nivel observado hace 3 años, "
            "lo que sugiere una evolución moderada o lateral en el mediano plazo."
        )
    if ret > 0:
        if years == 1:
            return (
                "El activo está hoy por encima del nivel que tenía hace 1 año, "
                "lo que indica un desempeño positivo reciente."
            )
        return (
            "El activo está hoy por encima del nivel observado hace 3 años, "
            "lo que indica un desempeño favorable en el mediano plazo."
        )
    if years == 1:
        return (
            "El activo está hoy por debajo del nivel que tenía hace 1 año, "
            "lo que refleja una caída o corrección en el último año."
        )
    return (
        "El activo está hoy por debajo del nivel observado hace 3 años, "
        "lo que muestra un desempeño débil en el mediano plazo."
    )


def render_market_context_help(
    latest_price: float | None,
    price_1y: float | None,
    price_3y: float | None,
    ret_1y: float | None,
    ret_3y: float | None,
) -> None:
    with st.expander("¿Cómo se calcularon estos retornos?", expanded=False):
        st.markdown(
            f"""
**Retorno acumulado (1 año)**  
Se calcula comparando el precio actual con el precio observado hace 1 año.

Fórmula:  
`Retorno 1 año = ((precio_actual / precio_hace_1a) - 1) * 100`

Valores usados:  
- Precio actual: **{_format_price(latest_price)}**  
- Precio hace 1 año: **{_format_price(price_1y)}**  
- Retorno calculado: **{_format_return(ret_1y)}**

Interpretación:  
**{_return_interpretation(ret_1y, 1)}**

**Retorno acumulado (3 años)**  
Se calcula comparando el precio actual con el precio observado hace 3 años.

Fórmula:  
`Retorno 3 años = ((precio_actual / precio_hace_3a) - 1) * 100`

Valores usados:  
- Precio actual: **{_format_price(latest_price)}**  
- Precio hace 3 años: **{_format_price(price_3y)}**  
- Retorno calculado: **{_format_return(ret_3y)}**

Interpretación:  
**{_return_interpretation(ret_3y, 3)}**
"""
        )


def render_performance_context(asset_name: str, meta: dict, ctx: dict) -> None:
    ticker = meta["ticker"]
    end_date = pd.Timestamp(DEFAULT_END_DATE).normalize()
    start_date = end_date - pd.DateOffset(years=5, days=10)

    st.header("Precio y contexto de mercado")
    st.caption("Cómo ha evolucionado el activo y qué eventos ayudan a interpretar ese movimiento.")

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
    date_1y, price_1y = _asof_price(series, end_date - pd.DateOffset(years=1))
    date_3y, price_3y = _asof_price(series, end_date - pd.DateOffset(years=3))
    date_5y, price_5y = _asof_price(series, end_date - pd.DateOffset(years=5))

    ret_1y = (latest_price / price_1y - 1) if latest_price and price_1y else None
    ret_3y = (latest_price / price_3y - 1) if latest_price and price_3y else None

    st.markdown(
        f"""
        <div class="price-metric-grid">
            {_price_metric_item("Precio actual", "Sin datos" if latest_price is None else f"{latest_price:,.2f}")}
            {_price_metric_item("Retorno acumulado (1 año)", _format_return(ret_1y))}
            {_price_metric_item("Retorno acumulado (3 años)", _format_return(ret_3y))}
        </div>
        <div class="market-caption">
            Precio actual = valor observado hoy. Retorno acumulado 1y y 3y = variación porcentual desde esos horizontes hasta la fecha actual.
        </div>
        """,
        unsafe_allow_html=True,
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
                name="Precio normalizado",
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
        if date_3y is not None:
            fig.add_vline(x=date_3y, line_dash="dot", line_color="#D97706", line_width=1.4)
            fig.add_annotation(
                x=date_3y,
                y=normalized.max(),
                text="3 años",
                showarrow=False,
                yshift=32,
                font=dict(color="#92400E", size=12),
                bgcolor="rgba(255,255,255,0.85)",
            )
        if date_1y is not None:
            fig.add_vline(x=date_1y, line_dash="dot", line_color="#16A34A", line_width=1.4)
            fig.add_annotation(
                x=date_1y,
                y=normalized.max(),
                text="1 año",
                showarrow=False,
                yshift=50,
                font=dict(color="#166534", size=12),
                bgcolor="rgba(255,255,255,0.85)",
            )

        fig.update_layout(
            template="plotly_white",
            title="Precio normalizado a 100",
            height=455,
            margin=dict(l=42, r=18, t=52, b=34),
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
    render_market_context_help(latest_price, price_1y, price_3y, ret_1y, ret_3y)
    st.markdown(
        '<div class="market-caption">Base 100: la serie parte en 100 para comparar desempeño relativo.</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div class="insight-box">{_performance_insight(ret_3y, ret_1y)}</div>',
        unsafe_allow_html=True,
    )

    st.subheader("Eventos de contexto")
    events = _context_events_for_asset(ctx, series.index.min(), series.index.max())
    if not events:
        st.caption("No hay eventos guía configurados para este activo en el rango consultado.")
    for event in events[:3]:
        st.markdown(
            f"""
            <div class="context-event">
                <div class="context-event-title">{event['titulo']}</div>
                <div class="context-event-meta">{event['fecha_inicio']} a {event['fecha_fin']}</div>
                <div class="context-event-body">{_first_sentence(event['nota'])}</div>
                <div class="context-event-body"><strong>Por qué importa:</strong> {_event_importance_for_asset(event, ctx)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    movement = _weekly_drop_context(series, events)
    if movement is not None:
        movement_date, movement_return, related_events = movement
        related_context = related_events[0]["titulo"] if related_events else "el entorno de mercado del periodo"
        st.markdown(
            f"""
            <div class="insight-box">
                <strong>Movimiento destacado:</strong> una de las caídas semanales más fuertes fue de
                {movement_return:.2%} alrededor de {movement_date.date()}. Puede contextualizarse con
                {related_context}, sin afirmar causalidad directa.
            </div>
            """,
            unsafe_allow_html=True,
        )


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
# Vista resumida
# -----------------------------------------------------------------------------
if vista == "Resumen":
    section_intro(
        "Lectura estratégica del conjunto",
        (
            "El portafolio combina consumo defensivo, movilidad, exposición regional y energía. "
            "Esta mezcla ayuda a interpretar por qué algunos activos estabilizan el conjunto y otros aportan más riesgo."
        ),
    )
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
    render_single_asset_view(
        selected_asset,
        ASSETS[selected_asset],
        ASSET_CONTEXT[selected_asset],
    )
