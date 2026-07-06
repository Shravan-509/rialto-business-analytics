"""Streamlit entry point for the Rialto Business Analytics Platform."""

from __future__ import annotations

from pathlib import Path
import sys
from datetime import datetime

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import config
from src.data_pipeline import load_clean_export


APP_VERSION = "v0.1.0"


def load_css() -> None:
    """Apply the local Streamlit theme overrides."""
    css_path = Path(__file__).parent / "assets" / "style.css"
    if css_path.exists():
        st.markdown(f"<style>{css_path.read_text()}</style>", unsafe_allow_html=True)


def render_sidebar(cleaned_df, last_refresh: datetime) -> None:
    """Render project-level navigation guidance in the sidebar."""
    st.sidebar.markdown(
        f"""
        <div class="sidebar-brand">
            <div class="sidebar-logo">
                <span>RI</span>
                <span>Rialto Analytics</span>
            </div>
            <div class="sidebar-version">Application version {APP_VERSION}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.sidebar.markdown(
        '<div class="sidebar-section-title">Navigation</div>',
        unsafe_allow_html=True,
    )
    st.sidebar.markdown(
        """
        <div class="sidebar-nav-item"><span class="sidebar-nav-icon">OV</span><span class="sidebar-nav-label">Executive Overview</span></div>
        <div class="sidebar-nav-item"><span class="sidebar-nav-icon">SC</span><span class="sidebar-nav-label">Sales and Customers</span></div>
        <div class="sidebar-nav-item"><span class="sidebar-nav-icon">RS</span><span class="sidebar-nav-label">Returns and Satisfaction</span></div>
        <div class="sidebar-nav-item"><span class="sidebar-nav-icon">SA</span><span class="sidebar-nav-label">Sentiment Analysis</span></div>
        <div class="sidebar-nav-item"><span class="sidebar-nav-icon">PA</span><span class="sidebar-nav-label">Predictive Analytics</span></div>
        <div class="sidebar-nav-item"><span class="sidebar-nav-icon">FC</span><span class="sidebar-nav-label">Forecasting</span></div>
        <div class="sidebar-nav-item"><span class="sidebar-nav-icon">AI</span><span class="sidebar-nav-label">GenAI Insights</span></div>
        <div class="sidebar-nav-item"><span class="sidebar-nav-icon">BI</span><span class="sidebar-nav-label">Power BI Export</span></div>
        """,
        unsafe_allow_html=True,
    )
    st.sidebar.markdown(
        '<div class="sidebar-section-title">Dataset</div>',
        unsafe_allow_html=True,
    )
    st.sidebar.markdown(
        f"""
        <div class="sidebar-data-point">
            <div class="sidebar-data-label">Records</div>
            <div class="sidebar-data-value">{len(cleaned_df):,}</div>
        </div>
        <div class="sidebar-data-point">
            <div class="sidebar-data-label">Date range</div>
            <div class="sidebar-data-value">{cleaned_df[config.DATE_COLUMN].min().strftime("%b %d, %Y")} - {cleaned_df[config.DATE_COLUMN].max().strftime("%b %d, %Y")}</div>
        </div>
        <div class="sidebar-data-point">
            <div class="sidebar-data-label">Last refresh</div>
            <div class="sidebar-data-value">{last_refresh.strftime("%Y-%m-%d %H:%M:%S")}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    """Render the platform landing page and validate the data pipeline."""
    st.set_page_config(
        page_title=config.APP_TITLE,
        layout="wide",
        initial_sidebar_state="expanded",
    )
    load_css()

    try:
        cleaned_df = load_clean_export()
    except Exception as exc:
        st.error(f"Unable to load the Rialto dataset: {exc}")
        st.stop()

    last_refresh = datetime.now()
    render_sidebar(cleaned_df, last_refresh)

    st.markdown(
        f"""
        <div class="hero-panel">
            <div class="hero-eyebrow">Business Analytics Platform</div>
            <div class="hero-title">{config.APP_TITLE}</div>
            <p class="hero-subtitle">{config.APP_SUBTITLE}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        f"""
        <div class="summary-card">
            Dataset loaded, validated, cleaned, feature engineered, and exported
            to <strong>{config.CLEANED_DATA_FILE.name}</strong>. Last refresh:
            <strong>{last_refresh.strftime("%Y-%m-%d %H:%M:%S")}</strong>.
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("Dataset preview", expanded=False):
        st.dataframe(cleaned_df.head(20), width="stretch")


if __name__ == "__main__":
    main()
