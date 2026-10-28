"""Shared Streamlit theme utilities and reusable UI components for PhysAI Neural Lab."""

from __future__ import annotations

import streamlit as st

# Color tokens (kept in sync with .streamlit/config.toml and CSS below).
PALETTE = {
    "bg": "#0a1020",
    "card": "#121a30",
    "card_alt": "#0f1729",
    "border": "#243049",
    "primary": "#22d3ee",      # cyan
    "secondary": "#3b82f6",    # blue
    "accent": "#f59e0b",        # amber
    "text": "#e8eef9",
    "muted": "#8b9ab8",
    "success": "#34d399",
    "warning": "#f59e0b",
    "error": "#f87171",
    "planned": "#94a3b8",
}

THEME_CSS = f"""
<style>
    /* Base */
    .stApp {{
        background-color: {PALETTE['bg']};
        color: {PALETTE['text']};
    }}
    .stApp, .stApp p, .stApp li, .stApp span {{
        color: {PALETTE['text']} !important;
    }}
    /* Sidebar */
    section[data-testid="stSidebar"] {{
        background-color: #080d1c;
        border-right: 1px solid {PALETTE['border']};
    }}
    section[data-testid="stSidebar"] .stMarkdown, section[data-testid="stSidebar"] span {{
        color: {PALETTE['text']} !important;
    }}
    /* Headings */
    h1, h2, h3, h4 {{
        color: {PALETTE['text']} !important;
        letter-spacing: 0.2px;
    }}
    /* Cards */
    .pl-card {{
        background: linear-gradient(180deg, {PALETTE['card']} 0%, {PALETTE['card_alt']} 100%);
        border: 1px solid {PALETTE['border']};
        border-radius: 14px;
        padding: 18px 20px;
        height: 100%;
        box-shadow: 0 2px 14px rgba(0,0,0,0.25);
    }}
    .pl-card h4 {{
        margin: 0 0 6px 0;
        font-size: 16px;
        color: {PALETTE['primary']} !important;
    }}
    .pl-card .pl-sub {{
        font-size: 12.5px;
        color: {PALETTE['muted']};
        margin: 0 0 8px 0;
        text-transform: uppercase;
        letter-spacing: 1px;
    }}
    .pl-card .pl-body {{
        font-size: 13.5px;
        color: {PALETTE['text']};
        line-height: 1.5;
        margin: 6px 0 0 0;
    }}
    .pl-status {{
        display: inline-block;
        font-size: 11.5px;
        font-weight: 600;
        padding: 2px 10px;
        border-radius: 999px;
        margin-top: 8px;
    }}
    .pl-status.implemented {{ background: rgba(52,211,153,0.15); color: {PALETTE['success']} !important; border: 1px solid rgba(52,211,153,0.4); }}
    .pl-status.progress {{ background: rgba(245,158,11,0.15); color: {PALETTE['warning']} !important; border: 1px solid rgba(245,158,11,0.4); }}
    .pl-status.planned {{ background: rgba(148,163,184,0.15); color: {PALETTE['planned']} !important; border: 1px solid rgba(148,163,184,0.4); }}
    /* Metric chips */
    .pl-metric {{
        background: {PALETTE['card']};
        border: 1px solid {PALETTE['border']};
        border-radius: 12px;
        padding: 14px 16px;
        text-align: center;
    }}
    .pl-metric .pl-metric-val {{ font-size: 26px; font-weight: 700; color: {PALETTE['primary']} !important; }}
    .pl-metric .pl-metric-label {{ font-size: 11.5px; color: {PALETTE['muted']}; text-transform: uppercase; letter-spacing: 1px; }}
    /* Step pill */
    .pl-step {{
        background: {PALETTE['card']};
        border: 1px solid {PALETTE['border']};
        border-radius: 10px;
        padding: 10px 12px;
        font-size: 13px;
        color: {PALETTE['text']};
        height: 100%;
    }}
    .pl-step .pl-step-num {{ color: {PALETTE['primary']} !important; font-weight: 700; font-size: 12px; letter-spacing: 1px; }}
    /* Banner */
    .pl-banner {{
        border: 1px solid {PALETTE['border']};
        border-left: 3px solid {PALETTE['primary']};
        background: {PALETTE['card_alt']};
        border-radius: 10px;
        padding: 12px 16px;
        font-size: 13.5px;
    }}
    .pl-note {{
        border: 1px solid rgba(245,158,11,0.4);
        border-left: 3px solid {PALETTE['warning']};
        background: rgba(245,158,11,0.06);
        border-radius: 10px;
        padding: 12px 16px;
        font-size: 13px;
    }}
    /* Inputs */
    .stNumberInput label, .stSelectbox label, .stSlider label, .stTextInput label {{
        color: {PALETTE['text']} !important;
    }}
    .stDownloadButton > button {{
        background-color: {PALETTE['secondary']} !important;
        border-color: {PALETTE['secondary']} !important;
        color: #fff !important;
    }}
    .stButton > button {{
        border-color: {PALETTE['border']} !important;
        color: {PALETTE['text']} !important;
    }}
    /* Plotly modebar */
    .modebar-bg {{ background: transparent !important; }}
    hr {{ border-color: {PALETTE['border']} !important; }}
</style>
"""


def inject_theme() -> None:
    """Inject the dark-navy theme CSS and page config."""
    st.markdown(THEME_CSS, unsafe_allow_html=True)


def card(title: str, subtitle: str, body: str, status: str | None = None) -> None:
    """Render a summary card with an optional status badge.

    ``status`` is one of "Implemented", "In Progress", or "Planned".
    """
    status_html = ""
    if status:
        cls = {
            "Implemented": "implemented",
            "In Progress": "progress",
            "Planned": "planned",
        }.get(status, "planned")
        status_html = f'<div class="pl-status {cls}">{status}</div>'
    st.markdown(
        f"""<div class="pl-card">
            <div class="pl-sub">{subtitle}</div>
            <h4>{title}</h4>
            <div class="pl-body">{body}</div>
            {status_html}
        </div>""",
        unsafe_allow_html=True,
    )


def metric_chip(value: str, label: str) -> None:
    st.markdown(
        f"""<div class="pl-metric">
            <div class="pl-metric-val">{value}</div>
            <div class="pl-metric-label">{label}</div>
        </div>""",
        unsafe_allow_html=True,
    )


def step(num: str, title: str) -> None:
    st.markdown(
        f"""<div class="pl-step">
            <div class="pl-step-num">STEP {num}</div>
            <div>{title}</div>
        </div>""",
        unsafe_allow_html=True,
    )


def banner(html: str) -> None:
    st.markdown(f'<div class="pl-banner">{html}</div>', unsafe_allow_html=True)


def note(html: str) -> None:
    st.markdown(f'<div class="pl-note">{html}</div>', unsafe_allow_html=True)


def page_title(title: str, subtitle: str = "") -> None:
    parts = [f"<h2 style='margin-bottom:2px'>{title}</h2>"]
    if subtitle:
        parts.append(f"<div style='color:{PALETTE['muted']};font-size:14px;margin-bottom:14px'>{subtitle}</div>")
    st.markdown("".join(parts), unsafe_allow_html=True)