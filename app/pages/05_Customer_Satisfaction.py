"""Customer Satisfaction dashboard page for Milestone 2."""

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
    calculate_satisfaction_kpis,
    low_satisfaction_table,
    monthly_summary,
)
from src.data_pipeline import load_clean_export
from src.visualizations import (
    COLOR_BLUE,
    COLOR_RED,
    COLOR_TEAL,
    apply_chart_theme,
    format_metric,
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
    """Render the Customer Satisfaction page."""
    st.set_page_config(page_title="Customer Satisfaction | Rialto", layout="wide")
    load_css()

    df = load_clean_export()
    last_refresh = datetime.now()
    filtered_df = apply_filters(df, render_sidebar(df, last_refresh))

    render_page_header(
        "Descriptive Analytics",
        "Customer Satisfaction",
        "Satisfaction score patterns, return-status comparison, and low-score transactions.",
    )

    if filtered_df.empty:
        st.warning("No records match the selected filters.")
        render_footer(last_refresh)
        st.stop()

    kpis = calculate_satisfaction_kpis(filtered_df)
    cards = [
        {
            "icon": "AVG",
            "title": "Average Satisfaction",
            "value": f"{kpis['average_satisfaction']:.2f}",
            "description": "Mean satisfaction score in the selected data.",
        },
        {
            "icon": "HI",
            "title": "Highest Satisfaction",
            "value": f"{kpis['highest_satisfaction']:.2f}",
            "description": "Highest recorded satisfaction score.",
        },
        {
            "icon": "LO",
            "title": "Lowest Satisfaction",
            "value": f"{kpis['lowest_satisfaction']:.2f}",
            "description": "Lowest recorded satisfaction score.",
        },
        {
            "icon": "LS",
            "title": "Low Satisfaction Transactions",
            "value": format_metric(kpis["low_satisfaction_transactions"]),
            "description": "Transactions with satisfaction score of 2 or below.",
        },
    ]
    render_kpi_cards(cards)

    monthly = monthly_summary(filtered_df)

    fig = px.histogram(filtered_df, x=config.SATISFACTION_COLUMN, nbins=5, color_discrete_sequence=[COLOR_TEAL])
    fig.update_traces(hovertemplate="Satisfaction: %{x}<br>Transactions: %{y:,}<extra></extra>")
    render_chart(apply_chart_theme(fig, "Satisfaction Distribution"))
    render_insight(f"Average satisfaction is {kpis['average_satisfaction']:.2f} for the selected transactions.")

    fig = px.bar(monthly, x="Month", y="average_satisfaction", color_discrete_sequence=[COLOR_BLUE])
    fig.update_traces(hovertemplate="<b>%{x}</b><br>Avg satisfaction: %{y:.2f}<extra></extra>")
    render_chart(apply_chart_theme(fig, "Satisfaction by Month"))
    render_insight("Monthly satisfaction identifies periods where customer experience was stronger or weaker.")

    fig = px.box(filtered_df, x=config.RETURN_COLUMN, y=config.SATISFACTION_COLUMN, color=config.RETURN_COLUMN)
    fig.update_traces(hovertemplate="Returned: %{x}<br>Satisfaction: %{y}<extra></extra>")
    render_chart(apply_chart_theme(fig, "Satisfaction by Return Status"))
    render_insight("Comparing satisfaction by return status shows whether returned transactions are associated with lower experience scores.")

    fig = px.box(filtered_df, x="Revenue_Band", y=config.SATISFACTION_COLUMN, color="Revenue_Band")
    fig.update_traces(hovertemplate="Revenue band: %{x}<br>Satisfaction: %{y}<extra></extra>")
    render_chart(apply_chart_theme(fig, "Satisfaction by Revenue Band"))
    render_insight("Satisfaction by revenue band shows whether experience varies across transaction value tiers.")

    fig = px.line(monthly, x="Month", y="average_satisfaction", markers=True, color_discrete_sequence=[COLOR_RED])
    fig.update_traces(hovertemplate="<b>%{x}</b><br>Avg satisfaction: %{y:.2f}<extra></extra>")
    render_chart(apply_chart_theme(fig, "Satisfaction Trend"))
    render_insight("The satisfaction trend helps spot changes in customer experience over time.")

    heatmap_data = (
        filtered_df.groupby(["Revenue_Band", "Month"], observed=False)[config.SATISFACTION_COLUMN]
        .mean()
        .reset_index()
        .pivot(index="Revenue_Band", columns="Month", values=config.SATISFACTION_COLUMN)
    )
    fig = px.imshow(
        heatmap_data,
        color_continuous_scale="Teal",
        aspect="auto",
        labels=dict(color="Avg Satisfaction"),
    )
    fig.update_traces(hovertemplate="Revenue band: %{y}<br>Month: %{x}<br>Avg satisfaction: %{z:.2f}<extra></extra>")
    render_chart(apply_chart_theme(fig, "Satisfaction Heatmap"))
    render_insight("The satisfaction heatmap highlights where lower experience scores appear by time and revenue band.")

    st.markdown("### Low Satisfaction Transactions")
    st.dataframe(low_satisfaction_table(filtered_df), use_container_width=True)

    render_executive_insights(
        [
            f"Average satisfaction is {kpis['average_satisfaction']:.2f}.",
            f"{format_metric(kpis['low_satisfaction_transactions'])} transactions have satisfaction scores of {config.LOW_SATISFACTION_THRESHOLD} or below.",
            "Low satisfaction transactions should be reviewed alongside return status and customer feedback.",
        ]
    )
    render_footer(last_refresh)


main()
