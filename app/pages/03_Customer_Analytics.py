"""Customer Analytics dashboard page for Milestone 2."""

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
from src.analytics import apply_filters, calculate_customer_kpis, customer_summary
from src.data_pipeline import load_clean_export
from src.visualizations import (
    COLOR_BLUE,
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
    """Render the Customer Analytics page."""
    st.set_page_config(page_title="Customer Analytics | Rialto", layout="wide")
    load_css()

    df = load_clean_export()
    last_refresh = datetime.now()
    filtered_df = apply_filters(df, render_sidebar(df, last_refresh))

    render_page_header(
        "Descriptive Analytics",
        "Customer Analytics",
        "Customer value, purchase frequency, RFM segments, and satisfaction context.",
    )

    if filtered_df.empty:
        st.warning("No records match the selected filters.")
        render_footer(last_refresh)
        st.stop()

    customers = customer_summary(filtered_df)
    kpis = calculate_customer_kpis(filtered_df)
    cards = [
        {
            "icon": "CU",
            "title": "Total Customers",
            "value": format_metric(kpis["total_customers"]),
            "description": "Unique customers in the selected dataset.",
        },
        {
            "icon": "RP",
            "title": "Repeat Customers",
            "value": format_metric(kpis["repeat_customers"]),
            "description": "Customers with more than one transaction.",
        },
        {
            "icon": "%",
            "title": "Repeat Customer Rate",
            "value": format_percent(kpis["repeat_customer_rate"]),
            "description": "Share of customers who purchased repeatedly.",
        },
        {
            "icon": "$",
            "title": "Avg Spend per Customer",
            "value": format_currency(kpis["average_spend_per_customer"]),
            "description": "Average customer-level revenue.",
        },
        {
            "icon": "TOP",
            "title": "Highest Spending Customer",
            "value": str(kpis["highest_spending_customer"]),
            "description": f"Revenue: {format_currency(kpis['highest_spending_customer_revenue'])}",
        },
    ]
    render_kpi_cards(cards)

    top_revenue = customers.nlargest(10, "total_revenue")
    fig = px.bar(
        top_revenue,
        x="total_revenue",
        y=config.CUSTOMER_ID_COLUMN,
        orientation="h",
        color_discrete_sequence=[COLOR_REVENUE],
    )
    fig.update_layout(yaxis=dict(autorange="reversed"))
    fig.update_traces(hovertemplate="Customer: %{y}<br>Revenue: $%{x:,.2f}<extra></extra>")
    fig.update_xaxes(tickprefix="$", separatethousands=True)
    render_chart(apply_chart_theme(fig, "Top 10 Customers by Revenue"))
    render_insight(f"{top_revenue.iloc[0][config.CUSTOMER_ID_COLUMN]} is the highest revenue customer in the selected data.")

    top_tx = customers.nlargest(10, "transaction_count")
    fig = px.bar(
        top_tx,
        x="transaction_count",
        y=config.CUSTOMER_ID_COLUMN,
        orientation="h",
        color_discrete_sequence=[COLOR_BLUE],
    )
    fig.update_layout(yaxis=dict(autorange="reversed"))
    fig.update_traces(hovertemplate="Customer: %{y}<br>Transactions: %{x:,}<extra></extra>")
    render_chart(apply_chart_theme(fig, "Top 10 Customers by Transactions"))
    render_insight(f"{top_tx.iloc[0][config.CUSTOMER_ID_COLUMN]} has the highest purchase frequency in the selected view.")

    fig = px.histogram(customers, x="total_revenue", nbins=25, color_discrete_sequence=[COLOR_TEAL])
    fig.update_traces(hovertemplate="Revenue range: %{x}<br>Customers: %{y:,}<extra></extra>")
    fig.update_xaxes(tickprefix="$", separatethousands=True)
    render_chart(apply_chart_theme(fig, "Revenue per Customer"))
    render_insight(f"Average spend per customer is {format_currency(kpis['average_spend_per_customer'])}.")

    fig = px.histogram(customers, x="transaction_count", nbins=15, color_discrete_sequence=[COLOR_BLUE])
    fig.update_traces(hovertemplate="Transactions: %{x}<br>Customers: %{y:,}<extra></extra>")
    render_chart(apply_chart_theme(fig, "Customer Purchase Frequency"))
    render_insight(f"The repeat customer rate is {format_percent(kpis['repeat_customer_rate'])}, indicating the current depth of repeat purchasing.")

    segment_counts = customers["segment"].value_counts().reset_index()
    segment_counts.columns = ["segment", "customers"]
    fig = px.pie(segment_counts, names="segment", values="customers", hole=0.45)
    fig.update_traces(hovertemplate="%{label}: %{value:,} customers<extra></extra>")
    render_chart(apply_chart_theme(fig, "Customer Segmentation"))
    render_insight("RFM segmentation groups customers by recency, frequency, and monetary value to prioritize engagement.")

    fig = px.scatter(
        customers,
        x="recency",
        y="monetary",
        size="frequency",
        color="segment",
        hover_data=[config.CUSTOMER_ID_COLUMN, "frequency"],
    )
    fig.update_yaxes(tickprefix="$", separatethousands=True)
    fig.update_traces(hovertemplate="Recency: %{x} days<br>Revenue: $%{y:,.2f}<extra></extra>")
    render_chart(apply_chart_theme(fig, "RFM Analysis"))
    render_insight("Customers with low recency, high frequency, and high monetary value represent the strongest relationship segment.")

    fig = px.scatter(
        customers,
        x="total_revenue",
        y="average_satisfaction",
        size="transaction_count",
        color="segment",
        hover_data=[config.CUSTOMER_ID_COLUMN],
    )
    fig.update_xaxes(tickprefix="$", separatethousands=True)
    render_chart(apply_chart_theme(fig, "Revenue vs Satisfaction Scatter Plot"))
    render_insight("This view compares customer value with satisfaction to identify high-value customers needing service attention.")

    fig = px.box(customers, y="total_revenue", x="segment", color="segment")
    fig.update_yaxes(tickprefix="$", separatethousands=True)
    render_chart(apply_chart_theme(fig, "Customer Distribution"))
    render_insight("Customer distribution by segment shows how customer value varies across relationship groups.")

    render_executive_insights(
        [
            f"Rialto has {format_metric(kpis['total_customers'])} customers in the selected view.",
            f"Repeat customer rate is {format_percent(kpis['repeat_customer_rate'])}.",
            f"The highest spending customer is {kpis['highest_spending_customer']} with {format_currency(kpis['highest_spending_customer_revenue'])} in revenue.",
        ]
    )
    render_footer(last_refresh)


main()
