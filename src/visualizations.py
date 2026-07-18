"""Reusable Streamlit and Plotly presentation helpers."""

from __future__ import annotations

from datetime import datetime
from typing import Iterable

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src import config
from src.analytics import FilterState


APP_VERSION = config.APP_VERSION
COLOR_REVENUE = "#0f766e"
COLOR_BLUE = "#1d4ed8"
COLOR_TEAL = "#14b8a6"
COLOR_AMBER = "#f59e0b"
COLOR_RED = "#dc2626"
COLOR_MUTED = "#64748b"
COLOR_GRID = "rgba(100, 116, 139, 0.16)"


def load_css() -> None:
    """Load shared Streamlit CSS."""
    css_path = config.APP_DIR / "assets" / "style.css"
    if css_path.exists():
        st.markdown(f"<style>{css_path.read_text()}</style>", unsafe_allow_html=True)


def format_currency(value: float) -> str:
    """Format a number as currency."""
    return f"${value:,.2f}"


def format_number(value: float | int) -> str:
    """Format a number with thousands separators."""
    return f"{value:,.0f}"


def format_percent(value: float) -> str:
    """Format a decimal as a percentage."""
    return f"{value:.1%}"


def format_metric(value: float | int | str, metric_type: str = "number") -> str:
    """Format a KPI value by type."""
    if isinstance(value, str):
        return value
    if metric_type == "currency":
        return format_currency(float(value))
    if metric_type == "percent":
        return format_percent(float(value))
    if metric_type == "decimal":
        return f"{float(value):.2f}"
    return format_number(value)


def apply_chart_theme(fig: go.Figure, title: str) -> go.Figure:
    """Apply the shared executive dashboard Plotly theme."""
    fig.update_layout(
        template="plotly_white",
        title=dict(text=title, font=dict(size=18, color="#14213d"), x=0.02),
        margin=dict(l=24, r=24, t=70, b=40),
        hovermode="x unified",
        legend_title_text="",
        font=dict(family="Inter, Arial, sans-serif", color="#14213d"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#ffffff",
    )
    fig.update_xaxes(showgrid=False, linecolor="#dbe4ef", tickfont=dict(size=12))
    fig.update_yaxes(gridcolor=COLOR_GRID, linecolor="#dbe4ef", tickfont=dict(size=12))
    return fig


def render_page_header(eyebrow: str, title: str, subtitle: str) -> None:
    """Render a consistent page header."""
    st.markdown(
        f"""
        <div class="hero-panel">
            <div class="hero-eyebrow">{eyebrow}</div>
            <div class="hero-title">{title}</div>
            <p class="hero-subtitle">{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_kpi_cards(cards: list[dict[str, str]], columns_per_row: int = 4) -> None:
    """Render dashboard KPI cards in a responsive column grid."""
    for start in range(0, len(cards), columns_per_row):
        columns = st.columns(columns_per_row)
        for column, card in zip(columns, cards[start : start + columns_per_row]):
            column.markdown(
                f"""
                <div class="kpi-card">
                    <div class="kpi-header">
                        <div class="kpi-icon">{card["icon"]}</div>
                        <div class="kpi-title">{card["title"]}</div>
                    </div>
                    <div class="kpi-value">{card["value"]}</div>
                    <div class="kpi-description">{card["description"]}</div>
                    <div class="kpi-trend">{card.get("trend", "Dynamic view")}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_chart(fig: go.Figure) -> None:
    """Render a Plotly figure inside the shared chart card."""
    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    plotly_config = {"displayModeBar": False, "responsive": True}
    st.plotly_chart(
        fig,
        width="stretch",
        config=plotly_config,
    )
    st.markdown("</div>", unsafe_allow_html=True)


def render_insight(text: str) -> None:
    """Render a business insight below a chart."""
    st.markdown(
        f"""
        <div class="insight-card">
            <strong>Business Insight.</strong> {text}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_executive_insights(insights: Iterable[str]) -> None:
    """Render page-level executive business insights."""
    items = "".join(f"<li>{insight}</li>" for insight in insights)
    st.markdown(
        f"""
        <div class="summary-card">
            <strong>Executive Business Insights</strong>
            <ul>{items}</ul>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar_brand() -> None:
    """Render shared sidebar brand treatment."""
    st.sidebar.markdown(
        f"""
        <div class="sidebar-brand">
            <div class="sidebar-logo">
                <span>RI</span>
                <span>Rialto Decision Intelligence</span>
            </div>
            <div class="sidebar-version">Application version {APP_VERSION}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar_filters(df: pd.DataFrame) -> FilterState:
    """Render reusable filters and return the selected filter state."""
    st.sidebar.markdown(
        '<div class="sidebar-section-title">Filters</div>',
        unsafe_allow_html=True,
    )

    min_date = df[config.DATE_COLUMN].min().date()
    max_date = df[config.DATE_COLUMN].max().date()
    date_input = st.sidebar.date_input(
        "Date range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
    )
    if isinstance(date_input, (tuple, list)) and len(date_input) == 2:
        date_range = (date_input[0], date_input[1])
    elif isinstance(date_input, (tuple, list)) and len(date_input) == 1:
        date_range = (date_input[0], date_input[0])
    else:
        date_range = (date_input, date_input)

    months = sorted(df["Month"].dropna().astype(str).unique())
    quarters = sorted(df["Quarter"].dropna().astype(str).unique())
    revenue_bands = sorted(df["Revenue_Band"].dropna().astype(str).unique())
    return_statuses = sorted(df[config.RETURN_COLUMN].dropna().astype(str).unique())
    customers = sorted(df[config.CUSTOMER_ID_COLUMN].dropna().astype(str).unique())

    selected_months = st.sidebar.multiselect("Month", months, default=months)
    selected_quarters = st.sidebar.multiselect("Quarter", quarters, default=quarters)
    selected_bands = st.sidebar.multiselect(
        "Revenue band", revenue_bands, default=revenue_bands
    )
    selected_returns = st.sidebar.multiselect(
        "Returned", return_statuses, default=return_statuses
    )
    selected_customers = st.sidebar.multiselect(
        "Customer", customers, default=customers
    )

    return FilterState(
        date_range=date_range,
        months=selected_months,
        quarters=selected_quarters,
        revenue_bands=selected_bands,
        return_statuses=selected_returns,
        customers=selected_customers,
    )


def render_sidebar_metadata(df: pd.DataFrame, last_refresh: datetime) -> None:
    """Render source dataset metadata in the sidebar."""
    st.sidebar.markdown(
        '<div class="sidebar-section-title">Dataset</div>',
        unsafe_allow_html=True,
    )
    st.sidebar.markdown(
        f"""
        <div class="sidebar-data-point">
            <div class="sidebar-data-label">Records</div>
            <div class="sidebar-data-value">{len(df):,}</div>
        </div>
        <div class="sidebar-data-point">
            <div class="sidebar-data-label">Date range</div>
            <div class="sidebar-data-value">{df[config.DATE_COLUMN].min().strftime("%b %d, %Y")} - {df[config.DATE_COLUMN].max().strftime("%b %d, %Y")}</div>
        </div>
        <div class="sidebar-data-point">
            <div class="sidebar-data-label">Last refresh</div>
            <div class="sidebar-data-value">{last_refresh.strftime("%Y-%m-%d %H:%M:%S")}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar_navigation() -> None:
    """Render compact navigation labels with icon badges."""
    st.sidebar.markdown(
        '<div class="sidebar-section-title">Navigation</div>',
        unsafe_allow_html=True,
    )
    st.sidebar.markdown(
        """
        <div class="sidebar-nav-item"><span class="sidebar-nav-icon">OV</span><span class="sidebar-nav-label">Executive Overview</span></div>
        <div class="sidebar-nav-item"><span class="sidebar-nav-icon">SA</span><span class="sidebar-nav-label">Sales Analytics</span></div>
        <div class="sidebar-nav-item"><span class="sidebar-nav-icon">CA</span><span class="sidebar-nav-label">Customer Analytics</span></div>
        <div class="sidebar-nav-item"><span class="sidebar-nav-icon">RA</span><span class="sidebar-nav-label">Returns Analytics</span></div>
        <div class="sidebar-nav-item"><span class="sidebar-nav-icon">CS</span><span class="sidebar-nav-label">Customer Satisfaction</span></div>
        <div class="sidebar-nav-item"><span class="sidebar-nav-icon">TX</span><span class="sidebar-nav-label">Sentiment Analysis</span></div>
        <div class="sidebar-nav-item"><span class="sidebar-nav-icon">ML</span><span class="sidebar-nav-label">Predictive Analytics</span></div>
        <div class="sidebar-nav-item"><span class="sidebar-nav-icon">FC</span><span class="sidebar-nav-label">Forecasting</span></div>
        <div class="sidebar-nav-item"><span class="sidebar-nav-icon">AI</span><span class="sidebar-nav-label">Executive AI Advisor</span></div>
        <div class="sidebar-nav-item"><span class="sidebar-nav-icon">BI</span><span class="sidebar-nav-label">Power BI Export</span></div>
        <div class="sidebar-nav-item"><span class="sidebar-nav-icon">AB</span><span class="sidebar-nav-label">About</span></div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar(df: pd.DataFrame, last_refresh: datetime) -> FilterState:
    """Render the complete reusable sidebar."""
    render_sidebar_brand()
    filters = render_sidebar_filters(df)
    render_sidebar_metadata(df, last_refresh)
    render_sidebar_navigation()
    return filters


def render_footer(last_refresh: datetime) -> None:
    """Render the shared dashboard footer."""
    st.markdown(
        f"""
        <div class="footer-card">
            {config.APP_TITLE} | Generated from Rialto Data.csv |
            Last refresh: {last_refresh.strftime("%Y-%m-%d %H:%M:%S")}
        </div>
        """,
        unsafe_allow_html=True,
    )
