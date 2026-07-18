"""Executive Overview dashboard page."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import sys

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import config
from src.data_pipeline import (
    build_monthly_summary,
    calculate_executive_kpis,
    load_clean_export,
)


APP_VERSION = config.APP_VERSION
COLOR_REVENUE = "#0f766e"
COLOR_TRANSACTIONS = "#1d4ed8"
COLOR_DISTRIBUTION = "#14b8a6"
COLOR_GRID = "rgba(100, 116, 139, 0.16)"


def load_css() -> None:
    """Apply shared dashboard CSS styles."""
    css_path = PROJECT_ROOT / "app" / "assets" / "style.css"
    if css_path.exists():
        st.markdown(f"<style>{css_path.read_text()}</style>", unsafe_allow_html=True)


def format_currency(value: float) -> str:
    """Format revenue values for dashboard display."""
    return f"${value:,.2f}"


def format_percent(value: float) -> str:
    """Format decimal rates as percentages."""
    return f"{value:.1%}"


def format_date(value: datetime) -> str:
    """Format date-like values for display."""
    return value.strftime("%b %d, %Y")


def apply_chart_theme(fig: go.Figure) -> go.Figure:
    """Apply a consistent clean Plotly theme."""
    fig.update_layout(
        template="plotly_white",
        margin=dict(l=24, r=24, t=68, b=36),
        hovermode="x unified",
        legend_title_text="",
        title=dict(font=dict(size=18, color="#14213d"), x=0.02),
        font=dict(family="Inter, Arial, sans-serif", color="#14213d"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#ffffff",
    )
    fig.update_xaxes(showgrid=False, linecolor="#dbe4ef", tickfont=dict(size=12))
    fig.update_yaxes(gridcolor=COLOR_GRID, linecolor="#dbe4ef", tickfont=dict(size=12))
    return fig


def render_sidebar(df, last_refresh: datetime):
    """Render sidebar metadata and interactive filters."""
    min_date = df[config.DATE_COLUMN].min().date()
    max_date = df[config.DATE_COLUMN].max().date()
    date_range = st.sidebar.date_input(
        "Date range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
    )

    revenue_bands = sorted(df["Revenue_Band"].dropna().astype(str).unique())
    selected_bands = st.sidebar.multiselect(
        "Revenue band",
        options=revenue_bands,
        default=revenue_bands,
    )

    return_statuses = sorted(df[config.RETURN_COLUMN].dropna().astype(str).unique())
    selected_returns = st.sidebar.multiselect(
        "Return status",
        options=return_statuses,
        default=return_statuses,
    )

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
            <div class="sidebar-data-value">{format_date(df[config.DATE_COLUMN].min())} - {format_date(df[config.DATE_COLUMN].max())}</div>
        </div>
        <div class="sidebar-data-point">
            <div class="sidebar-data-label">Last refresh</div>
            <div class="sidebar-data-value">{last_refresh.strftime("%Y-%m-%d %H:%M:%S")}</div>
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
        <div class="sidebar-nav-item"><span class="sidebar-nav-icon">AI</span><span class="sidebar-nav-label">Executive AI Advisor</span></div>
        <div class="sidebar-nav-item"><span class="sidebar-nav-icon">BI</span><span class="sidebar-nav-label">Power BI Export</span></div>
        <div class="sidebar-nav-item"><span class="sidebar-nav-icon">AB</span><span class="sidebar-nav-label">About</span></div>
        """,
        unsafe_allow_html=True,
    )

    return date_range, selected_bands, selected_returns


def filter_data(df, date_range, revenue_bands, return_statuses):
    """Apply dashboard filters without changing the underlying KPI formulas."""
    filtered = df.copy()
    if isinstance(date_range, (tuple, list)) and len(date_range) == 2:
        start_date, end_date = date_range
    elif isinstance(date_range, (tuple, list)) and len(date_range) == 1:
        start_date = end_date = date_range[0]
    else:
        start_date = end_date = date_range

    filtered = filtered[
        (filtered[config.DATE_COLUMN].dt.date >= start_date)
        & (filtered[config.DATE_COLUMN].dt.date <= end_date)
    ]

    if revenue_bands:
        filtered = filtered[filtered["Revenue_Band"].astype(str).isin(revenue_bands)]
    if return_statuses:
        filtered = filtered[
            filtered[config.RETURN_COLUMN].astype(str).isin(return_statuses)
        ]
    return filtered


def render_brand_sidebar_header() -> None:
    """Render app identity and version in the sidebar."""
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
        '<div class="sidebar-section-title">Filters</div>',
        unsafe_allow_html=True,
    )


def render_hero() -> None:
    """Render the executive page header."""
    st.markdown(
        """
        <div class="hero-panel">
            <div class="hero-eyebrow">Executive Dashboard</div>
            <div class="hero-title">Executive Overview</div>
            <p class="hero-subtitle">
                Dynamic business performance summary generated from Rialto Data.csv.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_kpis(kpis: dict[str, float]) -> None:
    """Render the six required executive KPI cards."""
    cards = [
        (
            "$",
            "Total Revenue",
            format_currency(kpis["total_revenue"]),
            "Revenue generated by the filtered transaction set.",
        ),
        (
            "#",
            "Total Transactions",
            f"{kpis['total_transactions']:,}",
            "Number of transaction records included in the view.",
        ),
        (
            "ID",
            "Total Customers",
            f"{kpis['total_customers']:,}",
            "Unique customers represented in the selected period.",
        ),
        (
            "AOV",
            "Average Order Value",
            format_currency(kpis["average_order_value"]),
            "Average revenue earned per transaction.",
        ),
        (
            "%",
            "Return Rate",
            format_percent(kpis["return_rate"]),
            "Share of transactions marked as returned.",
        ),
        (
            "CS",
            "Average Satisfaction",
            f"{kpis['average_satisfaction']:.2f}",
            "Mean customer satisfaction score in the filtered data.",
        ),
    ]

    for row_start in range(0, len(cards), 3):
        columns = st.columns(3)
        for column, (icon, title, value, description) in zip(
            columns, cards[row_start : row_start + 3]
        ):
            column.markdown(
                f"""
                <div class="kpi-card">
                    <div class="kpi-header">
                        <div class="kpi-icon">{icon}</div>
                        <div class="kpi-title">{title}</div>
                    </div>
                    <div class="kpi-value">{value}</div>
                    <div class="kpi-description">{description}</div>
                    <div class="kpi-trend">Trend baseline pending</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_executive_summary(kpis: dict[str, float]) -> None:
    """Render a dynamic written summary from calculated KPI values."""
    summary = (
        f"Revenue generated this period was "
        f"{format_currency(kpis['total_revenue'])} across "
        f"{kpis['total_transactions']:,} transactions from "
        f"{kpis['total_customers']:,} customers. The return rate is "
        f"{format_percent(kpis['return_rate'])}, while the average customer "
        f"satisfaction score is {kpis['average_satisfaction']:.2f}. This provides "
        "a quick overview of current business performance for the selected filters."
    )
    st.markdown(
        f"""
        <div class="summary-card">
            <strong>Executive Summary.</strong> {summary}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_charts(df) -> None:
    """Render the Executive Overview Plotly charts."""
    monthly = build_monthly_summary(df)

    revenue_fig = go.Figure()
    revenue_fig.add_trace(
        go.Scatter(
            x=monthly["Month"],
            y=monthly["monthly_revenue"],
            mode="lines+markers",
            name="Revenue",
            line=dict(color=COLOR_REVENUE, width=3),
            marker=dict(size=8),
            hovertemplate="<b>%{x}</b><br>Revenue: $%{y:,.2f}<extra></extra>",
        )
    )
    revenue_fig.update_layout(title="Monthly Revenue Trend")
    revenue_fig.update_yaxes(tickprefix="$", separatethousands=True)
    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    plotly_config = {"displayModeBar": False, "responsive": True}
    st.plotly_chart(
        apply_chart_theme(revenue_fig),
        width="stretch",
        config=plotly_config,
    )
    st.markdown("</div>", unsafe_allow_html=True)

    left, right = st.columns(2)

    transactions_fig = px.bar(
        monthly,
        x="Month",
        y="monthly_transactions",
        title="Monthly Transaction Volume",
        labels={"Month": "Month", "monthly_transactions": "Transactions"},
        color_discrete_sequence=[COLOR_TRANSACTIONS],
    )
    transactions_fig.update_traces(
        hovertemplate="<b>%{x}</b><br>Transactions: %{y:,}<extra></extra>"
    )
    transactions_fig.update_yaxes(separatethousands=True)
    left.markdown('<div class="chart-card">', unsafe_allow_html=True)
    left.plotly_chart(
        apply_chart_theme(transactions_fig),
        width="stretch",
        config=plotly_config,
    )
    left.markdown("</div>", unsafe_allow_html=True)

    distribution_fig = px.histogram(
        df,
        x=config.REVENUE_COLUMN,
        nbins=25,
        title="Transaction Revenue Distribution",
        labels={config.REVENUE_COLUMN: "Transaction Revenue"},
        color_discrete_sequence=[COLOR_DISTRIBUTION],
    )
    distribution_fig.update_traces(
        hovertemplate="Revenue range: %{x}<br>Transactions: %{y:,}<extra></extra>"
    )
    distribution_fig.update_xaxes(tickprefix="$", separatethousands=True)
    right.markdown('<div class="chart-card">', unsafe_allow_html=True)
    right.plotly_chart(
        apply_chart_theme(distribution_fig),
        width="stretch",
        config=plotly_config,
    )
    right.markdown("</div>", unsafe_allow_html=True)


def render_footer(last_refresh: datetime) -> None:
    """Render a professional dashboard footer."""
    st.markdown(
        f"""
        <div class="footer-card">
            {config.APP_TITLE} | Generated from Rialto Data.csv |
            Last refresh: {last_refresh.strftime("%Y-%m-%d %H:%M:%S")}
        </div>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    """Render the Executive Overview page."""
    st.set_page_config(
        page_title="Executive Overview | Rialto",
        layout="wide",
    )
    load_css()

    try:
        df = load_clean_export()
    except Exception as exc:
        st.error(f"Unable to prepare dashboard data: {exc}")
        st.stop()

    last_refresh = datetime.now()
    render_brand_sidebar_header()
    date_range, selected_bands, selected_returns = render_sidebar(df, last_refresh)
    filtered_df = filter_data(df, date_range, selected_bands, selected_returns)

    render_hero()
    if filtered_df.empty:
        st.warning("No records match the selected filters. Adjust filters to continue.")
        render_footer(last_refresh)
        st.stop()

    kpis = calculate_executive_kpis(filtered_df)
    render_kpis(kpis)
    render_executive_summary(kpis)
    render_charts(filtered_df)

    with st.expander("Cleaned data preview", expanded=False):
        st.dataframe(filtered_df.head(25), width="stretch")

    render_footer(last_refresh)


main()
