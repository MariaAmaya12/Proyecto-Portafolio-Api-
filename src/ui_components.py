from __future__ import annotations

from collections.abc import Iterable

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components


def _sanitize_text(text: object) -> str:
    if text is None:
        return ""
    return str(text).replace("<", "").replace(">", "")


def render_section(title: str, subtitle: str | None = None) -> None:
    subtitle_html = ""
    if subtitle:
        subtitle_html = f'<div class="section-intro-subtitle">{_sanitize_text(subtitle)}</div>'

    st.markdown(
        f"""
        <div class="section-intro-box">
            <div class="section-intro-title">{_sanitize_text(title)}</div>
            {subtitle_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def _content_to_markdown(content: str | Iterable[str]) -> str:
    if isinstance(content, str):
        return content
    return "\n".join(f"- {_sanitize_text(item)}" for item in content)


def render_explanation_expander(title: str, content: str | Iterable[str]) -> None:
    with st.expander(title, expanded=False):
        st.markdown(_content_to_markdown(content))


def render_chart_explanation(title: str, what_it_shows: str | Iterable[str], how_to_read: str | Iterable[str]) -> None:
    with st.expander(title, expanded=False):
        st.markdown("**Qué muestra**")
        st.markdown(_content_to_markdown(what_it_shows))
        st.markdown("**Cómo interpretarlo**")
        st.markdown(_content_to_markdown(how_to_read))


def render_kpi_help(title: str, content: str | Iterable[str]) -> None:
    with st.expander(title, expanded=False):
        st.markdown(_content_to_markdown(content))


def render_insight(message: str, kind: str = "info") -> None:
    renderer = {
        "info": st.info,
        "success": st.success,
        "warning": st.warning,
        "error": st.error,
    }.get(kind, st.info)
    renderer(message)


def kpi_card(
    title: object,
    value: object,
    delta: object | None = None,
    delta_type: str = "neu",
    caption: object = "",
) -> None:
    safe_title = _sanitize_text(title)
    safe_value = _sanitize_text(value)
    safe_delta = _sanitize_text(delta) if delta is not None else ""
    safe_caption = _sanitize_text(caption)

    delta_html = ""
    if safe_delta:
        delta_html = f'<div class="kpi-delta {delta_type}">{safe_delta}</div>'

    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            body {{
                margin: 0;
                padding: 0;
                background: transparent;
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            }}

            .kpi-card {{
                background: linear-gradient(180deg, #e0f2fe 0%, #f0f9ff 100%);
                border: 1px solid rgba(14, 116, 144, 0.16);
                border-radius: 18px;
                padding: 14px 15px 12px 15px;
                box-shadow: 0 4px 14px rgba(14, 116, 144, 0.10);
                min-height: 156px;
                box-sizing: border-box;
                display: flex;
                flex-direction: column;
                justify-content: space-between;
                overflow: visible;
            }}

            .kpi-label {{
                font-size: 0.84rem;
                font-weight: 600;
                color: #0f3b57;
                margin-bottom: 0.25rem;
                letter-spacing: 0.2px;
                line-height: 1.22;
                overflow-wrap: anywhere;
            }}

            .kpi-value {{
                font-size: 1.58rem;
                font-weight: 800;
                color: #082f49;
                line-height: 1.1;
                margin-bottom: 0.28rem;
                word-break: break-word;
                overflow-wrap: anywhere;
            }}

            .kpi-delta {{
                display: inline-block;
                width: fit-content;
                font-size: 0.74rem;
                font-weight: 700;
                padding: 0.20rem 0.48rem;
                border-radius: 999px;
                margin-top: 0.02rem;
            }}

            .kpi-delta.pos {{
                background-color: rgba(22, 163, 74, 0.12);
                color: #166534;
            }}

            .kpi-delta.neg {{
                background-color: rgba(220, 38, 38, 0.12);
                color: #991b1b;
            }}

            .kpi-delta.neu {{
                background-color: rgba(8, 47, 73, 0.10);
                color: #0f3b57;
            }}

            .kpi-caption {{
                font-size: 0.74rem;
                color: #27536d;
                margin-top: 0.45rem;
                line-height: 1.28;
                overflow-wrap: anywhere;
            }}
        </style>
    </head>
    <body>
        <div class="kpi-card">
            <div>
                <div class="kpi-label">{safe_title}</div>
                <div class="kpi-value">{safe_value}</div>
                {delta_html}
            </div>
            <div class="kpi-caption">{safe_caption}</div>
        </div>
    </body>
    </html>
    """

    components.html(html, height=178)


def render_table(df: pd.DataFrame, hide_index: bool = True, width: str = "stretch") -> None:
    st.dataframe(df, hide_index=hide_index, width=width)
