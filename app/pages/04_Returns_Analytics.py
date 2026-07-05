"""Returns Analytics dashboard page for Milestone 2."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import sys

import plotly.express as px
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import config
from src.analytics import (
    apply_filters,
    calculate_returns_kpis,
    monthly_summary,
    returns_by_revenue_band,
)
from src.data_pipeline import load_clean_export
from src.visualizations import (
    COLOR_BLUE,
    COLOR_RED,
    COLOR_REVENUE,
    COLOR_TEAL,
    apply_chart_theme,
    format_currency,
    format_metric,
    format_percent,
    load_css,
    render_chart,
    render_executive_insights,
    render_footer,
    render_insight,
    render_kpi_cards,
    render_page_header,
    render_sidebar,
)


def main() -> None:
    """Render the Returns Analytics page."""
    st.set_page_config(page_title="Returns Analytics | Rialto", layout="wide")
    load_css()

    df = load_clean_export()
    last_refresh = datetime.now()
    filtered_df = apply_filters(df, render_sidebar(df, last_refresh))

    render_page_header(
        "Descriptive Analytics",
        "Returns Analytics",
        "Return rates, returned revenue, and operating risk patterns.",
    )

    if filtered_df.empty:
        st.warning("No records match the selected filters.")
        render_footer(last_refresh)
        st.stop()

    kpis = calculate_returns_kpis(filtered_df)
    cards = [
        {
            "icon": "%",
            "title": "Return Rate",
            "value": format_percent(kpis["return_rate"]),
            "description": "Share of selected transactions returned.",
        },
        {
            "icon": "$R",
            "title": "Returned Revenue",
            "value": format_currency(kpis["returned_revenue"]),
            "description": "Revenue associated with returned transactions.",
        },
        {
            "icon": "$N",
            "title": "Non Returned Revenue",
            "value": format_currency(kpis["non_returned_revenue"]),
            "description": "Revenue associated with non-returned transactions.",
        },
        {
            "icon": "#",
            "title": "Returned Transactions",
            "value": format_metric(kpis["returned_transactions"]),
            "description": "Count of transactions marked returned.",
        },
    ]
    render_kpi_cards(cards)

    monthly = monthly_summary(filtered_df)
    band_returns = returns_by_revenue_band(filtered_df)

    fig = px.line(monthly, x="Month", y="return_rate", markers=True, color_discrete_sequence=[COLOR_RED])
    fig.update_traces(hovertemplate="<b>%{x}</b><br>Return rate: %{y:.1%}<extra></extra>")
    fig.update_yaxes(tickformat=".0%")
    render_chart(apply_chart_theme(fig, "Return Rate by Month"))
    render_insight(f"Overall return rate is {format_percent(kpis['return_rate'])} for the selected data.")

    revenue_status = (
        filtered_df.groupby(config.RETURN_COLUMN, as_index=False)[config.REVENUE_COLUMN]
        .sum()
        .rename(columns={config.REVENUE_COLUMN: "revenue"})
    )
    fig = px.bar(revenue_status, x=config.RETURN_COLUMN, y="revenue", color=config.RETURN_COLUMN)
    fig.update_traces(hovertemplate="Status: %{x}<br>Revenue: $%{y:,.2f}<extra></extra>")
    fig.update_yaxes(tickprefix="$", separatethousands=True)
    render_chart(apply_chart_theme(fig, "Returned vs Non Returned Revenue"))
    render_insight(f"Returned revenue totals {format_currency(kpis['returned_revenue'])}, compared with {format_currency(kpis['non_returned_revenue'])} non-returned revenue.")

    fig = px.bar(monthly, x="Month", y="monthly_returns", color_discrete_sequence=[COLOR_RED])
    fig.update_traces(hovertemplate="<b>%{x}</b><br>Returned transactions: %{y:,}<extra></extra>")
    render_chart(apply_chart_theme(fig, "Return Trend"))
    render_insight("The return trend highlights months with the highest volume of returned transactions.")

    fig = px.bar(band_returns, x="Revenue_Band", y="return_rate", color_discrete_sequence=[COLOR_TEAL])
    fig.update_traces(hovertemplate="Revenue band: %{x}<br>Return rate: %{y:.1%}<extra></extra>")
    fig.update_yaxes(tickformat=".0%")
    render_chart(apply_chart_theme(fig, "Return Rate by Revenue Band"))
    render_insight("Return rate by revenue band shows whether returns concentrate in lower or higher value orders.")

    returned_monthly = (
        filtered_df[filtered_df["Return_Flag"] == 1]
        .groupby("Month", as_index=False)[config.REVENUE_COLUMN]
        .sum()
        .rename(columns={config.REVENUE_COLUMN: "returned_revenue"})
    )
    fig = px.line(returned_monthly, x="Month", y="returned_revenue", markers=True, color_discrete_sequence=[COLOR_REVENUE])
    fig.update_traces(hovertemplate="<b>%{x}</b><br>Returned revenue: $%{y:,.2f}<extra></extra>")
    fig.update_yaxes(tickprefix="$", separatethousands=True)
    render_chart(apply_chart_theme(fig, "Returned Revenue Trend"))
    render_insight("Returned revenue trend shows the financial exposure associated with returned transactions over time.")

    fig = px.histogram(filtered_df, x="Return_Flag", color=config.RETURN_COLUMN, nbins=2)
    fig.update_traces(hovertemplate="Return flag: %{x}<br>Transactions: %{y:,}<extra></extra>")
    render_chart(apply_chart_theme(fig, "Returns Distribution"))
    render_insight(f"{format_metric(kpis['returned_transactions'])} transactions were returned in the selected view.")

    render_executive_insights(
        [
            f"Return rate is {format_percent(kpis['return_rate'])}.",
            f"Returned revenue totals {format_currency(kpis['returned_revenue'])}.",
            "Revenue band analysis can help prioritize where return prevention should be focused.",
        ]
    )
    render_footer(last_refresh)


main()
