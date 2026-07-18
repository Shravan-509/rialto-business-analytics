"""Power BI export utilities for final platform outputs."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from src import config
from src.analytics import (
    calculate_customer_kpis,
    calculate_returns_kpis,
    calculate_sales_kpis,
    customer_summary,
    monthly_summary,
    returns_by_revenue_band,
)
from src.data_pipeline import calculate_executive_kpis
from src.forecasting import ForecastResult, forecasts_to_frame


logger = logging.getLogger(__name__)


def export_power_bi_datasets(
    df: pd.DataFrame,
    forecasts: dict[str, ForecastResult] | None = None,
) -> dict[str, str]:
    """Generate all Power BI CSV datasets from current platform outputs."""
    config.POWER_BI_DIR.mkdir(parents=True, exist_ok=True)
    paths = {
        "executive_kpi": config.POWER_BI_DIR / "executive_kpi.csv",
        "sales": config.POWER_BI_DIR / "sales.csv",
        "customer": config.POWER_BI_DIR / "customer.csv",
        "returns": config.POWER_BI_DIR / "returns.csv",
        "forecast": config.POWER_BI_DIR / "forecast.csv",
        "prediction": config.POWER_BI_DIR / "prediction.csv",
        "sentiment": config.POWER_BI_DIR / "sentiment.csv",
    }

    build_executive_kpi_export(df).to_csv(paths["executive_kpi"], index=False)
    build_sales_export(df).to_csv(paths["sales"], index=False)
    build_customer_export(df).to_csv(paths["customer"], index=False)
    build_returns_export(df).to_csv(paths["returns"], index=False)
    build_forecast_export(forecasts).to_csv(paths["forecast"], index=False)
    load_optional_csv(config.PROCESSED_DATA_DIR / "predictions.csv").to_csv(
        paths["prediction"], index=False
    )
    load_optional_csv(config.PROCESSED_DATA_DIR / "sentiment_results.csv").to_csv(
        paths["sentiment"], index=False
    )

    logger.info("Power BI exports generated in %s", config.POWER_BI_DIR)
    return {key: str(path) for key, path in paths.items()}


def build_executive_kpi_export(df: pd.DataFrame) -> pd.DataFrame:
    """Build a two-column executive KPI table."""
    executive = calculate_executive_kpis(df)
    sales = calculate_sales_kpis(df)
    customers = calculate_customer_kpis(df)
    returns = calculate_returns_kpis(df)
    payload = {
        **executive,
        "monthly_growth_pct": sales["monthly_growth_pct"],
        "repeat_customer_rate": customers["repeat_customer_rate"],
        "returned_revenue": returns["returned_revenue"],
    }
    return pd.DataFrame(
        [{"kpi": key, "value": value} for key, value in payload.items()]
    )


def build_sales_export(df: pd.DataFrame) -> pd.DataFrame:
    """Build Power BI sales fact table."""
    sales = monthly_summary(df)
    sales["Month"] = pd.to_datetime(sales["Month"])
    return sales


def build_customer_export(df: pd.DataFrame) -> pd.DataFrame:
    """Build Power BI customer metrics table."""
    return customer_summary(df)


def build_returns_export(df: pd.DataFrame) -> pd.DataFrame:
    """Build Power BI returns table by revenue band."""
    return returns_by_revenue_band(df)


def build_forecast_export(
    forecasts: dict[str, ForecastResult] | None,
) -> pd.DataFrame:
    """Build Power BI forecast export table."""
    if not forecasts:
        return pd.DataFrame(
            columns=[
                "period",
                "forecast",
                "lower_bound",
                "upper_bound",
                "metric",
                "model",
                "mae",
                "rmse",
                "mape",
            ]
        )
    return forecasts_to_frame(forecasts)


def load_optional_csv(path: Path) -> pd.DataFrame:
    """Load an optional processed CSV without crashing export generation."""
    if path.exists():
        return pd.read_csv(path)
    logger.info("Optional export source missing: %s", path)
    return pd.DataFrame()
