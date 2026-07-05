"""Descriptive analytics helpers for the Rialto dashboard pages."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

from src import config


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FilterState:
    """Reusable dashboard filter state."""

    date_range: tuple[date, date]
    months: list[str]
    quarters: list[str]
    revenue_bands: list[str]
    return_statuses: list[str]
    customers: list[str]


def apply_filters(df: pd.DataFrame, filters: FilterState) -> pd.DataFrame:
    """Apply shared dashboard filters to the transaction-level dataset."""
    logger.info("Applying dashboard filters")
    filtered = df.copy()
    start_date, end_date = filters.date_range

    filtered = filtered[
        (filtered[config.DATE_COLUMN].dt.date >= start_date)
        & (filtered[config.DATE_COLUMN].dt.date <= end_date)
    ]

    if filters.months:
        filtered = filtered[filtered["Month"].isin(filters.months)]
    if filters.quarters:
        filtered = filtered[filtered["Quarter"].isin(filters.quarters)]
    if filters.revenue_bands:
        filtered = filtered[
            filtered["Revenue_Band"].astype(str).isin(filters.revenue_bands)
        ]
    if filters.return_statuses:
        filtered = filtered[
            filtered[config.RETURN_COLUMN].astype(str).isin(filters.return_statuses)
        ]
    if filters.customers:
        filtered = filtered[filtered[config.CUSTOMER_ID_COLUMN].isin(filters.customers)]

    return filtered


def monthly_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Return monthly revenue, transactions, returns, and satisfaction metrics."""
    if df.empty:
        return pd.DataFrame(
            columns=[
                "Month",
                "monthly_revenue",
                "monthly_transactions",
                "monthly_returns",
                "return_rate",
                "average_satisfaction",
            ]
        )

    summary = (
        df.groupby("Month", as_index=False)
        .agg(
            monthly_revenue=(config.REVENUE_COLUMN, "sum"),
            monthly_transactions=(config.TRANSACTION_ID_COLUMN, "count"),
            monthly_returns=("Return_Flag", "sum"),
            average_satisfaction=(config.SATISFACTION_COLUMN, "mean"),
        )
        .sort_values("Month")
    )
    summary["return_rate"] = summary["monthly_returns"] / summary[
        "monthly_transactions"
    ].replace(0, np.nan)
    summary["cumulative_revenue"] = summary["monthly_revenue"].cumsum()
    return summary


def quarter_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Return quarterly revenue and transaction metrics."""
    return (
        df.groupby("Quarter", as_index=False)
        .agg(
            quarterly_revenue=(config.REVENUE_COLUMN, "sum"),
            quarterly_transactions=(config.TRANSACTION_ID_COLUMN, "count"),
        )
        .sort_values("Quarter")
    )


def calculate_sales_kpis(df: pd.DataFrame) -> dict[str, float]:
    """Calculate sales analytics KPIs from the filtered dataset."""
    monthly = monthly_summary(df)
    total_transactions = len(df)
    total_revenue = float(df[config.REVENUE_COLUMN].sum()) if total_transactions else 0.0
    latest_revenue = monthly["monthly_revenue"].iloc[-1] if len(monthly) else np.nan
    previous_revenue = monthly["monthly_revenue"].iloc[-2] if len(monthly) > 1 else np.nan
    first_revenue = monthly["monthly_revenue"].iloc[0] if len(monthly) else np.nan

    return {
        "total_revenue": total_revenue,
        "total_transactions": int(total_transactions),
        "average_order_value": total_revenue / total_transactions
        if total_transactions
        else 0.0,
        "highest_transaction": float(df[config.REVENUE_COLUMN].max())
        if total_transactions
        else 0.0,
        "lowest_transaction": float(df[config.REVENUE_COLUMN].min())
        if total_transactions
        else 0.0,
        "monthly_growth_pct": percentage_change(latest_revenue, previous_revenue),
        "revenue_growth_pct": percentage_change(latest_revenue, first_revenue),
    }


def customer_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Build customer-level summary metrics."""
    if df.empty:
        return pd.DataFrame()

    max_date = df[config.DATE_COLUMN].max()
    customers = (
        df.groupby(config.CUSTOMER_ID_COLUMN)
        .agg(
            total_revenue=(config.REVENUE_COLUMN, "sum"),
            transaction_count=(config.TRANSACTION_ID_COLUMN, "count"),
            average_satisfaction=(config.SATISFACTION_COLUMN, "mean"),
            returned_transactions=("Return_Flag", "sum"),
            first_purchase=(config.DATE_COLUMN, "min"),
            last_purchase=(config.DATE_COLUMN, "max"),
        )
        .reset_index()
    )
    customers["recency"] = (max_date - customers["last_purchase"]).dt.days
    customers["frequency"] = customers["transaction_count"]
    customers["monetary"] = customers["total_revenue"]
    customers["return_rate"] = customers["returned_transactions"] / customers[
        "transaction_count"
    ].replace(0, np.nan)
    customers["segment"] = assign_rfm_segments(customers)
    return customers


def assign_rfm_segments(customer_df: pd.DataFrame) -> pd.Series:
    """Assign RFM customer segments using dynamic quantile-based scores."""
    if customer_df.empty:
        return pd.Series(dtype=str)

    scored = customer_df.copy()
    scored["recency_score"] = quantile_score(scored["recency"], high_is_good=False)
    scored["frequency_score"] = quantile_score(scored["frequency"], high_is_good=True)
    scored["monetary_score"] = quantile_score(scored["monetary"], high_is_good=True)
    scored["rfm_score"] = (
        scored["recency_score"] + scored["frequency_score"] + scored["monetary_score"]
    )

    conditions = [
        scored["rfm_score"] >= scored["rfm_score"].quantile(0.8),
        (scored["frequency_score"] >= 3) & (scored["monetary_score"] >= 3),
        scored["recency_score"] <= 2,
        (scored["recency_score"] >= 3) & (scored["frequency_score"] <= 2),
    ]
    choices = ["Champions", "Loyal", "Lost", "New"]
    return pd.Series(
        np.select(conditions, choices, default="At Risk"), index=customer_df.index
    )


def quantile_score(series: pd.Series, high_is_good: bool) -> pd.Series:
    """Score a numeric series from 1 to 4 using dataset-driven quantiles."""
    if series.nunique(dropna=True) <= 1:
        return pd.Series(2, index=series.index)

    ranked = series.rank(method="first")
    labels = [1, 2, 3, 4] if high_is_good else [4, 3, 2, 1]
    return pd.qcut(ranked, q=4, labels=labels, duplicates="drop").astype(int)


def calculate_customer_kpis(df: pd.DataFrame) -> dict[str, float | str]:
    """Calculate customer analytics KPIs from the filtered dataset."""
    customers = customer_summary(df)
    total_customers = len(customers)
    repeat_customers = int((customers["transaction_count"] > 1).sum())
    highest = (
        customers.sort_values("total_revenue", ascending=False).iloc[0]
        if total_customers
        else None
    )

    return {
        "total_customers": total_customers,
        "repeat_customers": repeat_customers,
        "repeat_customer_rate": repeat_customers / total_customers
        if total_customers
        else 0.0,
        "average_spend_per_customer": float(customers["total_revenue"].mean())
        if total_customers
        else 0.0,
        "highest_spending_customer": highest[config.CUSTOMER_ID_COLUMN]
        if highest is not None
        else "N/A",
        "highest_spending_customer_revenue": float(highest["total_revenue"])
        if highest is not None
        else 0.0,
    }


def calculate_returns_kpis(df: pd.DataFrame) -> dict[str, float]:
    """Calculate returns analytics KPIs from the filtered dataset."""
    returned = df[df["Return_Flag"] == 1]
    non_returned = df[df["Return_Flag"] == 0]
    return {
        "return_rate": float(df["Return_Flag"].mean()) if len(df) else 0.0,
        "returned_revenue": float(returned[config.REVENUE_COLUMN].sum()),
        "non_returned_revenue": float(non_returned[config.REVENUE_COLUMN].sum()),
        "returned_transactions": int(len(returned)),
    }


def returns_by_revenue_band(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate return rate by dynamic revenue band."""
    result = (
        df.groupby("Revenue_Band", observed=False)
        .agg(
            transactions=(config.TRANSACTION_ID_COLUMN, "count"),
            returned_transactions=("Return_Flag", "sum"),
            revenue=(config.REVENUE_COLUMN, "sum"),
        )
        .reset_index()
    )
    result["return_rate"] = result["returned_transactions"] / result[
        "transactions"
    ].replace(0, np.nan)
    return result


def calculate_satisfaction_kpis(df: pd.DataFrame) -> dict[str, float]:
    """Calculate customer satisfaction KPIs from the filtered dataset."""
    return {
        "average_satisfaction": float(df[config.SATISFACTION_COLUMN].mean())
        if len(df)
        else 0.0,
        "highest_satisfaction": float(df[config.SATISFACTION_COLUMN].max())
        if len(df)
        else 0.0,
        "lowest_satisfaction": float(df[config.SATISFACTION_COLUMN].min())
        if len(df)
        else 0.0,
        "low_satisfaction_transactions": int(df["Low_Satisfaction_Flag"].sum()),
    }


def low_satisfaction_table(df: pd.DataFrame) -> pd.DataFrame:
    """Return the table of low-satisfaction transactions requested for the page."""
    columns = [
        config.TRANSACTION_ID_COLUMN,
        config.CUSTOMER_ID_COLUMN,
        config.REVENUE_COLUMN,
        config.RETURN_COLUMN,
        config.TEXT_COLUMN,
    ]
    return (
        df[df[config.SATISFACTION_COLUMN] <= config.LOW_SATISFACTION_THRESHOLD][
            columns
        ]
        .sort_values(config.REVENUE_COLUMN, ascending=False)
        .rename(
            columns={
                config.TRANSACTION_ID_COLUMN: "Transaction ID",
                config.CUSTOMER_ID_COLUMN: "Customer",
                config.REVENUE_COLUMN: "Revenue",
                config.RETURN_COLUMN: "Returned",
                config.TEXT_COLUMN: "Feedback",
            }
        )
    )


def percentage_change(current: float, previous: float) -> float:
    """Calculate percentage change safely."""
    if pd.isna(current) or pd.isna(previous) or previous == 0:
        return 0.0
    return float((current - previous) / previous)


def trend_phrase(value: float, metric_name: str) -> str:
    """Create a compact natural-language trend statement."""
    if value > 0:
        return f"{metric_name} increased by {value:.1%}."
    if value < 0:
        return f"{metric_name} declined by {abs(value):.1%}."
    return f"{metric_name} is flat for the selected data."
