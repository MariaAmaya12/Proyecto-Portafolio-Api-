import streamlit as st


NAVIGATION_PAGES = [
    ("app.py", "Inicio"),
    ("pages/0_contextualizacion.py", "Contextualización"),
    ("pages/01_tecnico.py", "M1. Análisis técnico"),
    ("pages/02_rendimientos.py", "M2. Rendimientos"),
    ("pages/03_garch.py", "M3. Modelos GARCH"),
    ("pages/04_capm.py", "M4. CAPM y Beta"),
    ("pages/05_var_cvar.py", "M5. VaR/CVaR"),
    ("pages/06_markowitz.py", "M6. Optimización Markowitz"),
    ("pages/07_senales.py", "M7. Señales"),
    ("pages/08_macro_benchmark.py", "M8. Macro y Benchmark"),
    ("pages/09_panel_decision.py", "M9. Panel de decisión"),
]


def render_sidebar_navigation() -> None:
    with st.sidebar:
        st.header("Navegación")
        for page_path, label in NAVIGATION_PAGES:
            st.page_link(page_path, label=label)
        st.divider()
