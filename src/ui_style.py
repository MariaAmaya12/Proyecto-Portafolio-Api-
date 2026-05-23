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


def apply_dashboard_css() -> None:
    """Inject shared dashboard CSS used by all module pages."""
    st.markdown(
        """
        <style>
        .section-intro-box {
            background: #ffffff;
            border: 1px solid rgba(15, 23, 42, 0.08);
            border-radius: 18px;
            padding: 16px 18px;
            box-shadow: 0 4px 14px rgba(15, 23, 42, 0.06);
            margin-bottom: 0.75rem;
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
        .module-header-box {
            background: linear-gradient(135deg, #f8fbff 0%, #eff6ff 100%);
            border: 1px solid rgba(37, 99, 235, 0.12);
            border-radius: 18px;
            padding: 16px 20px;
            margin-bottom: 1rem;
        }
        .module-header-badge {
            display: inline-block;
            padding: 3px 10px;
            border-radius: 999px;
            background: rgba(37, 99, 235, 0.10);
            color: #1e40af;
            font-size: 0.76rem;
            font-weight: 700;
            margin-bottom: 0.45rem;
        }
        .module-header-title {
            color: #0f172a;
            font-size: 1.18rem;
            font-weight: 800;
            margin-bottom: 0.2rem;
            line-height: 1.2;
        }
        .module-header-desc {
            color: #64748b;
            font-size: 0.86rem;
            line-height: 1.45;
        }
        .conclusion-box {
            background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%);
            border: 1px solid rgba(22, 163, 74, 0.18);
            border-radius: 18px;
            padding: 16px 18px;
            margin: 0.35rem 0 0.75rem;
            color: #0f172a;
            font-size: 0.91rem;
            line-height: 1.55;
        }
        .conclusion-box.warn {
            background: linear-gradient(135deg, #fffbeb 0%, #fef3c7 100%);
            border-color: rgba(234, 179, 8, 0.22);
        }
        .conclusion-box.danger {
            background: linear-gradient(135deg, #fff1f2 0%, #fee2e2 100%);
            border-color: rgba(220, 38, 38, 0.20);
        }
        .conclusion-box-label {
            font-size: 0.74rem;
            font-weight: 700;
            color: #15803d;
            letter-spacing: 0.05em;
            text-transform: uppercase;
            margin-bottom: 0.35rem;
        }
        .conclusion-box.warn .conclusion-box-label { color: #a16207; }
        .conclusion-box.danger .conclusion-box-label { color: #b91c1c; }
        .soft-explain-box {
            background: #eff6ff;
            border: 1px solid rgba(37, 99, 235, 0.16);
            border-radius: 14px;
            padding: 14px 16px;
            margin: 0.65rem 0 0.9rem;
        }
        .soft-explain-title {
            color: #0f172a;
            font-size: 0.95rem;
            font-weight: 700;
            margin-bottom: 0.25rem;
        }
        .soft-explain-body {
            color: #334155;
            font-size: 0.88rem;
            line-height: 1.5;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
