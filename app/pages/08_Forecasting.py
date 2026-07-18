"""Forecasting dashboard page for Milestone 5."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import sys

import plotly.graph_objects as go
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import config
from src.analytics import apply_filters
from src.data_pipeline import load_clean_export
from src.forecasting import ForecastResult, run_all_forecasts
from src.visualizations import (
    COLOR_BLUE,
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


@st.cache_data(show_spinner=False)
def cached_clean_data():
    """Load and clean the Rialto dataset."""
    return load_clean_export()


@st.cache_data(show_spinner=False)
def cached_forecasts(df, periods: int):
    """Run and cache forecast outputs."""
    return run_all_forecasts(df, periods=periods)


def render_forecast_cards(results: dict[str, ForecastResult]) -> None:
    """Render top-line forecast cards."""
    cards = []
    for name, result in results.items():
        next_value = (
            float(result.forecast["forecast"].iloc[0])
            if not result.forecast.empty
            else 0.0
        )
        value = (
            format_currency(next_value)
            if "Revenue" in name
            else format_metric(next_value, "decimal")
        )
        cards.append(
            {
                "icon": "FC",
                "title": name,
                "value": value,
                "description": f"Next-period forecast using {result.model_name}.",
                "trend": f"MAPE {format_percent(result.accuracy.get('mape', 0.0))}",
            }
        )
    render_kpi_cards(cards, columns_per_row=4)


def render_forecast_chart(result: ForecastResult) -> None:
    """Render historical, forecast, and confidence interval chart."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=result.historical["period"],
            y=result.historical["value"],
            mode="lines+markers",
            name="Historical",
            line=dict(color=COLOR_BLUE, width=3),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=result.forecast["period"],
            y=result.forecast["forecast"],
            mode="lines+markers",
            name="Forecast",
            line=dict(color=COLOR_TEAL, width=3),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=list(result.forecast["period"]) + list(result.forecast["period"])[::-1],
            y=list(result.forecast["upper_bound"]) + list(result.forecast["lower_bound"])[::-1],
            fill="toself",
            fillcolor="rgba(20, 184, 166, 0.18)",
            line=dict(color="rgba(255,255,255,0)"),
            name="Confidence interval",
            hoverinfo="skip",
        )
    )
    render_chart(apply_chart_theme(fig, result.metric_name))
    accuracy = result.accuracy
    render_insight(
        f"{result.metric_name} uses {result.model_name}. "
        f"Holdout RMSE is {accuracy.get('rmse', 0.0):,.2f} and "
        f"MAPE is {format_percent(accuracy.get('mape', 0.0))}."
    )


def render_accuracy_table(results: dict[str, ForecastResult]) -> None:
    """Render forecast accuracy metrics."""
    rows = []
    for result in results.values():
        rows.append(
            {
                "Metric": result.metric_name,
                "Model": result.model_name,
                "MAE": result.accuracy.get("mae", 0.0),
                "RMSE": result.accuracy.get("rmse", 0.0),
                "MAPE": result.accuracy.get("mape", 0.0),
            }
        )
    st.markdown("### Forecast Accuracy")
    st.dataframe(rows, width="stretch")


def main() -> None:
    """Render the Forecasting page."""
    st.set_page_config(page_title="Forecasting | Rialto", layout="wide")
    load_css()

    df = cached_clean_data()
    last_refresh = datetime.now()
    filters = render_sidebar(df, last_refresh)
    filtered_df = apply_filters(df, filters)

    render_page_header(
        "Forecasting",
        "Time Series Forecasting",
        "Monthly revenue, transactions, customer growth, and satisfaction outlook.",
    )

    if filtered_df.empty:
        st.warning("No records match the selected filters.")
        render_footer(last_refresh)
        st.stop()

    forecast_periods = st.slider("Forecast horizon in months", 3, 12, 6)
    with st.spinner("Generating forecasts..."):
        results = cached_forecasts(filtered_df, forecast_periods)

    render_forecast_cards(results)
    for result in results.values():
        render_forecast_chart(result)
    render_accuracy_table(results)
    render_executive_insights(
        [
            "Forecasts update dynamically when filters or the source dataset change.",
            "Prophet is used when available, with ARIMA or naive fallback for reliability.",
            "Confidence intervals communicate forecast uncertainty for executive planning.",
        ]
    )
    render_footer(last_refresh)


main()
