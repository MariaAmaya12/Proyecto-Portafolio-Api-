import streamlit as st


def apply_global_typography(base_px: int = 18) -> None:
    st.markdown(
        f"""
        <style>
        section.main .stMarkdown,
        section.main .stText,
        section.main p,
        section.main li,
        section.main label,
        section.main span,
        section.main div[data-testid="stCaptionContainer"] {{
            font-size: {base_px}px;
            line-height: 1.55;
        }}
        section.main h1,
        section.main div.block-container div[data-testid="stTitle"] h1,
        section.main div.block-container div[data-testid="stHeading"] h1,
        section.main div.block-container div[data-testid="stMarkdownContainer"] h1 {{
            color: #0f172a !important;
            font-size: 2.6rem !important;
            font-weight: 800 !important;
            line-height: 1.15 !important;
            letter-spacing: 0 !important;
            margin-top: 0.6rem !important;
            margin-bottom: 0.35rem !important;
        }}
        section.main h2,
        section.main div.block-container div[data-testid="stHeader"] h2,
        section.main div.block-container div[data-testid="stHeading"] h2,
        section.main div.block-container div[data-testid="stMarkdownContainer"] h2 {{
            color: #0f172a !important;
            font-size: 1.9rem !important;
            font-weight: 800 !important;
            line-height: 1.2 !important;
            margin-top: 0.85rem !important;
            margin-bottom: 0.35rem !important;
        }}
        section.main h3,
        section.main div.block-container div[data-testid="stSubheader"] h3,
        section.main div.block-container div[data-testid="stHeading"] h3,
        section.main div.block-container div[data-testid="stMarkdownContainer"] h3 {{
            color: #0f172a !important;
            font-size: 1.45rem !important;
            font-weight: 750 !important;
            line-height: 1.25 !important;
            margin-top: 0.7rem !important;
            margin-bottom: 0.25rem !important;
        }}
        section.main h4,
        section.main div.block-container div[data-testid="stHeading"] h4,
        section.main div.block-container div[data-testid="stMarkdownContainer"] h4 {{
            color: #0f172a !important;
            font-size: 1.15rem !important;
            font-weight: 700 !important;
            line-height: 1.3 !important;
            margin-top: 0.6rem !important;
            margin-bottom: 0.2rem !important;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_page_title(title: str, subtitle: str | None = None) -> None:
    st.title(title)
    if subtitle:
        st.caption(subtitle)
