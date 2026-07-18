"""Application About page for the Rialto platform."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import sys

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import config
from src.data_pipeline import load_clean_export
from src.visualizations import load_css, render_footer, render_page_header, render_sidebar


@st.cache_data(show_spinner=False)
def cached_clean_data():
    """Load and clean the Rialto dataset."""
    return load_clean_export()


def render_info_card(title: str, lines: list[str]) -> None:
    """Render a compact About-page information card."""
    body = "<br/>".join(lines)
    st.markdown(
        f"""
        <div class="summary-card">
            <strong>{title}</strong><br/>
            {body}
        </div>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    """Render application metadata and model information."""
    st.set_page_config(page_title="About | Rialto", layout="wide")
    load_css()

    df = cached_clean_data()
    last_refresh = datetime.now()
    render_sidebar(df, last_refresh)
    render_page_header(
        "About",
        "About the Rialto Platform",
        "Version, purpose, dataset, model, and repository information for the final capstone application.",
    )

    col1, col2 = st.columns(2)
    with col1:
        render_info_card(
            "Project",
            [
                f"Name: {config.PROJECT_NAME}",
                f"Application version: {config.APP_VERSION}",
                f"Author: {config.APP_AUTHOR}",
                "Purpose: Interactive decision intelligence for business analytics, forecasting, and executive reporting.",
            ],
        )
        render_info_card(
            "Dataset",
            [
                "Source: Rialto Data.csv",
                f"Records: {len(df):,}",
                f"Date range: {df[config.DATE_COLUMN].min().date()} to {df[config.DATE_COLUMN].max().date()}",
                "Refresh model: Replace the CSV with the same schema to regenerate analytics.",
            ],
        )
        render_info_card(
            "Technology Stack",
            [
                "Python, Streamlit, Pandas, NumPy, Plotly",
                "Scikit-learn, Joblib, XGBoost where available",
                "NLTK, Transformers, WordCloud, BERTopic where available",
                "Prophet, Statsmodels ARIMA, Gemini API, ReportLab or PDF fallback",
            ],
        )

    with col2:
        render_info_card(
            "ML Models Used",
            [
                "Return prediction: Logistic Regression, Decision Tree, Random Forest, XGBoost where available",
                "Satisfaction prediction: Linear Regression, Random Forest Regressor, Gradient Boosting Regressor",
                "Customer segment prediction: Random Forest and Decision Tree classifiers",
            ],
        )
        render_info_card(
            "Forecast Models",
            [
                "Primary: Prophet",
                "Fallback: ARIMA",
                "Safety fallback: Naive last-value forecast",
            ],
        )
        render_info_card(
            "AI Provider",
            [
                f"Configured provider: {config.LLM_PROVIDER}",
                f"Gemini model: {config.GEMINI_MODEL}",
                f"OpenAI model: {config.OPENAI_MODEL}",
                "Fallback: deterministic rule-based summaries when API access is unavailable.",
            ],
        )
        render_info_card(
            "Repository and License",
            [
                "Repository: rialto-business-analytics",
                "GitHub Repository: publish this folder as the project repository.",
                "License: MIT",
            ],
        )

    render_info_card(
        "Repository Structure",
        [
            "app/: Streamlit entry point, pages, and CSS",
            "src/: data, analytics, NLP, ML, forecasting, GenAI, export, and reporting modules",
            "data/: raw, processed, and Power BI-ready datasets",
            "models/: persisted predictive model artifacts",
            "reports/: generated Markdown and PDF reports",
            "docs/: architecture documentation",
            "tests/: data-pipeline tests",
        ],
    )
    render_footer(last_refresh)


main()
