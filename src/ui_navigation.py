import streamlit as st


NAVIGATION_PAGES = [
    ("app.py", "Inicio"),
    ("pages/0_contextualizacion.py", "Contextualización"),
    ("pages/01_tecnico.py", "Mód.1 Análisis técnico"),
    ("pages/02_rendimientos.py", "Mód.2 Rendimientos"),
    ("pages/03_garch.py", "Mód.3 Modelos GARCH"),
    ("pages/04_capm.py", "Mód.4 CAPM y Beta"),
    ("pages/05_var_cvar.py", "Mód.5 VaR/CVaR"),
    ("pages/06_markowitz.py", "Mód.6 Optimización Markowitz"),
    ("pages/07_senales.py", "Mód.7 Señales"),
    ("pages/08_macro_benchmark.py", "Mód.8 Macro y Benchmark"),
    ("pages/09_panel_decision.py", "Mód.9 Panel de decisión"),
]


def render_sidebar_navigation() -> None:
    with st.sidebar:
        st.header("Navegación")
        for page_path, label in NAVIGATION_PAGES:
            st.page_link(page_path, label=label)
        st.divider()
