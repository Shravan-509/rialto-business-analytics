"""Executive AI Advisor dashboard page for Milestone 5."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import sys

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analytics import apply_filters
from src.data_pipeline import load_clean_export
from src.forecasting import run_all_forecasts
from src.genai_layer import generate_executive_ai_advisor
from src.visualizations import (
    load_css,
    render_footer,
    render_page_header,
    render_sidebar,
)


@st.cache_data(show_spinner=False)
def cached_clean_data():
    """Load and clean the Rialto dataset."""
    return load_clean_export()


@st.cache_data(show_spinner=False)
def cached_forecasts(df):
    """Generate cached advisor forecast context."""
    return run_all_forecasts(df, periods=6)


@st.cache_data(show_spinner=False)
def cached_advisor(df):
    """Generate cached Executive AI Advisor output."""
    forecasts = cached_forecasts(df)
    return generate_executive_ai_advisor(df, forecasts)


def render_advisor_section(title: str, content: str) -> None:
    """Render a single consultant-report section."""
    st.markdown(
        f"""
        <div class="advisor-section">
            <div class="advisor-section-title">{title}</div>
            <div class="advisor-section-body">{content}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    """Render the Executive AI Advisor page."""
    st.set_page_config(page_title="Executive AI Advisor | Rialto", layout="wide")
    load_css()

    df = cached_clean_data()
    last_refresh = datetime.now()
    filters = render_sidebar(df, last_refresh)
    filtered_df = apply_filters(df, filters)

    render_page_header(
        "Executive AI Advisor",
        "Executive AI Advisor",
        "Gemini-powered executive decision support based exclusively on verified Rialto analytics.",
    )

    if filtered_df.empty:
        st.warning("No records match the selected filters.")
        render_footer(last_refresh)
        st.stop()

    with st.spinner("Preparing executive advisor report..."):
        advisor, source = cached_advisor(filtered_df)

    st.markdown(
        f"""
        <div class="consultant-report">
            <div class="consultant-kicker">{source}</div>
            <div class="consultant-title">CEO Decision Intelligence Brief</div>
            <p>{advisor.get("executive_summary", "")}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)
    with col1:
        render_advisor_section("Overall Business Health", advisor["overall_business_health"])
        render_advisor_section("Key Risks", advisor["key_risks"])
        render_advisor_section("Revenue Outlook", advisor["revenue_outlook"])
    with col2:
        render_advisor_section("Emerging Opportunities", advisor["emerging_opportunities"])
        render_advisor_section("Customer Outlook", advisor["customer_outlook"])
        render_advisor_section("Return Outlook", advisor["return_outlook"])

    actions = "".join(f"<li>{action}</li>" for action in advisor["recommended_actions"])
    st.markdown(
        f"""
        <div class="summary-card">
            <strong>Recommended Actions</strong>
            <ul>{actions}</ul>
        </div>
        """,
        unsafe_allow_html=True,
    )
    render_footer(last_refresh)


main()
