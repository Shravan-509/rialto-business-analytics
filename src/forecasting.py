"""Time-series forecasting utilities for the Rialto platform."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from src import config


logger = logging.getLogger(__name__)


@dataclass
class ForecastResult:
    """Container for one forecast model output."""

    metric_name: str
    value_column: str
    model_name: str
    historical: pd.DataFrame
    forecast: pd.DataFrame
    accuracy: dict[str, float]


def build_monthly_forecast_inputs(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Build monthly time-series inputs for all required forecast metrics."""
    monthly_base = df.copy()
    monthly_base["period"] = monthly_base[config.DATE_COLUMN].dt.to_period("M").dt.to_timestamp()

    revenue = (
        monthly_base.groupby("period", as_index=False)
        .agg(value=(config.REVENUE_COLUMN, "sum"))
        .sort_values("period")
    )
    transactions = (
        monthly_base.groupby("period", as_index=False)
        .agg(value=(config.TRANSACTION_ID_COLUMN, "count"))
        .sort_values("period")
    )
    satisfaction = (
        monthly_base.groupby("period", as_index=False)
        .agg(value=(config.SATISFACTION_COLUMN, "mean"))
        .sort_values("period")
    )
    customers = (
        monthly_base.groupby("period")
        .agg(new_customers=(config.CUSTOMER_ID_COLUMN, pd.Series.nunique))
        .sort_index()
        .reset_index()
    )
    customers["value"] = customers["new_customers"].cumsum()

    return {
        "Monthly Revenue Forecast": revenue,
        "Transaction Forecast": transactions,
        "Customer Growth Forecast": customers[["period", "value"]],
        "Satisfaction Forecast": satisfaction,
    }


def run_all_forecasts(df: pd.DataFrame, periods: int = 6) -> dict[str, ForecastResult]:
    """Run all required forecasts with Prophet preferred and ARIMA fallback."""
    inputs = build_monthly_forecast_inputs(df)
    return {
        metric_name: forecast_monthly_series(metric_name, frame, periods)
        for metric_name, frame in inputs.items()
    }


def forecast_monthly_series(
    metric_name: str,
    series_df: pd.DataFrame,
    periods: int = 6,
) -> ForecastResult:
    """Forecast one monthly series using Prophet when available, then ARIMA."""
    historical = normalize_monthly_series(series_df)
    train, test = split_train_test(historical)

    try:
        result = prophet_forecast(metric_name, historical, train, test, periods)
        logger.info("Forecasted %s with Prophet.", metric_name)
        return result
    except Exception as exc:
        logger.info("Prophet unavailable for %s; using ARIMA fallback: %s", metric_name, exc)

    try:
        result = arima_forecast(metric_name, historical, train, test, periods)
        logger.info("Forecasted %s with ARIMA.", metric_name)
        return result
    except Exception as exc:
        logger.warning("ARIMA failed for %s; using naive forecast fallback: %s", metric_name, exc)
        return naive_forecast(metric_name, historical, train, test, periods)


def normalize_monthly_series(series_df: pd.DataFrame) -> pd.DataFrame:
    """Ensure a complete monthly series with numeric values."""
    if series_df.empty:
        return pd.DataFrame(columns=["period", "value"])

    monthly = series_df.copy()
    monthly["period"] = pd.to_datetime(monthly["period"])
    monthly["value"] = pd.to_numeric(monthly["value"], errors="coerce").fillna(0.0)
    full_index = pd.date_range(
        monthly["period"].min(),
        monthly["period"].max(),
        freq="MS",
    )
    monthly = (
        monthly.set_index("period")
        .reindex(full_index)
        .rename_axis("period")
        .reset_index()
    )
    monthly["value"] = monthly["value"].interpolate().bfill().ffill().fillna(0.0)
    return monthly


def split_train_test(historical: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create a small holdout split for forecast accuracy."""
    if len(historical) < 4:
        return historical, historical.iloc[0:0].copy()

    test_size = max(1, min(3, int(round(len(historical) * 0.2))))
    return historical.iloc[:-test_size].copy(), historical.iloc[-test_size:].copy()


def prophet_forecast(
    metric_name: str,
    historical: pd.DataFrame,
    train: pd.DataFrame,
    test: pd.DataFrame,
    periods: int,
) -> ForecastResult:
    """Forecast with Prophet."""
    from prophet import Prophet

    if len(historical) < 3:
        raise ValueError("Prophet requires at least three monthly observations.")

    prophet_train = historical.rename(columns={"period": "ds", "value": "y"})
    model = Prophet(interval_width=0.8, yearly_seasonality=False, weekly_seasonality=False)
    model.fit(prophet_train)
    future = model.make_future_dataframe(periods=periods, freq="MS")
    prediction = model.predict(future)
    forecast = format_prophet_output(prediction, historical["period"].max())

    accuracy = {}
    if not test.empty and len(train) >= 3:
        accuracy = prophet_accuracy(train, test)
    return ForecastResult(metric_name, "value", "Prophet", historical, forecast, accuracy)


def prophet_accuracy(train: pd.DataFrame, test: pd.DataFrame) -> dict[str, float]:
    """Calculate Prophet holdout accuracy."""
    from prophet import Prophet

    model = Prophet(interval_width=0.8, yearly_seasonality=False, weekly_seasonality=False)
    model.fit(train.rename(columns={"period": "ds", "value": "y"}))
    future = pd.DataFrame({"ds": test["period"]})
    prediction = model.predict(future)
    return calculate_accuracy(test["value"].to_numpy(), prediction["yhat"].to_numpy())


def format_prophet_output(prediction: pd.DataFrame, latest_period: pd.Timestamp) -> pd.DataFrame:
    """Convert Prophet output into shared forecast schema."""
    forecast = prediction[prediction["ds"] > latest_period][
        ["ds", "yhat", "yhat_lower", "yhat_upper"]
    ].copy()
    forecast.columns = ["period", "forecast", "lower_bound", "upper_bound"]
    return clip_forecast_values(forecast)


def arima_forecast(
    metric_name: str,
    historical: pd.DataFrame,
    train: pd.DataFrame,
    test: pd.DataFrame,
    periods: int,
) -> ForecastResult:
    """Forecast with a conservative ARIMA model."""
    from statsmodels.tsa.arima.model import ARIMA

    if len(historical) < 3:
        raise ValueError("ARIMA requires at least three monthly observations.")

    values = historical["value"].astype(float)
    model = ARIMA(values, order=(1, 1, 1)).fit()
    forecast_obj = model.get_forecast(steps=periods)
    predicted = forecast_obj.predicted_mean.to_numpy()
    intervals = forecast_obj.conf_int(alpha=0.2).to_numpy()
    future_periods = pd.date_range(
        historical["period"].max() + pd.offsets.MonthBegin(1),
        periods=periods,
        freq="MS",
    )
    forecast = pd.DataFrame(
        {
            "period": future_periods,
            "forecast": predicted,
            "lower_bound": intervals[:, 0],
            "upper_bound": intervals[:, 1],
        }
    )

    accuracy = {}
    if not test.empty and len(train) >= 3:
        train_model = ARIMA(train["value"].astype(float), order=(1, 1, 1)).fit()
        holdout = train_model.forecast(steps=len(test))
        accuracy = calculate_accuracy(test["value"].to_numpy(), np.asarray(holdout))
    return ForecastResult(metric_name, "value", "ARIMA", historical, clip_forecast_values(forecast), accuracy)


def naive_forecast(
    metric_name: str,
    historical: pd.DataFrame,
    train: pd.DataFrame,
    test: pd.DataFrame,
    periods: int,
) -> ForecastResult:
    """Forecast with a stable last-value fallback."""
    last_value = float(historical["value"].iloc[-1]) if len(historical) else 0.0
    std = float(historical["value"].std()) if len(historical) > 1 else 0.0
    future_periods = pd.date_range(
        historical["period"].max() + pd.offsets.MonthBegin(1),
        periods=periods,
        freq="MS",
    )
    forecast = pd.DataFrame(
        {
            "period": future_periods,
            "forecast": last_value,
            "lower_bound": last_value - std,
            "upper_bound": last_value + std,
        }
    )
    accuracy = {}
    if not test.empty:
        accuracy = calculate_accuracy(test["value"].to_numpy(), np.repeat(train["value"].iloc[-1], len(test)))
    return ForecastResult(metric_name, "value", "Naive fallback", historical, clip_forecast_values(forecast), accuracy)


def clip_forecast_values(forecast: pd.DataFrame) -> pd.DataFrame:
    """Clip forecast outputs to valid non-negative ranges."""
    clipped = forecast.copy()
    for column in ["forecast", "lower_bound", "upper_bound"]:
        clipped[column] = pd.to_numeric(clipped[column], errors="coerce").fillna(0.0).clip(lower=0.0)
    return clipped


def calculate_accuracy(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    """Calculate forecast accuracy metrics."""
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    errors = actual - predicted
    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(errors**2)))
    denominator = np.where(actual == 0, np.nan, actual)
    mape = float(np.nanmean(np.abs(errors / denominator))) if len(actual) else 0.0
    if np.isnan(mape):
        mape = 0.0
    return {"mae": mae, "rmse": rmse, "mape": mape}


def forecasts_to_frame(results: dict[str, ForecastResult]) -> pd.DataFrame:
    """Flatten forecast results for export and reporting."""
    rows = []
    for result in results.values():
        forecast = result.forecast.copy()
        forecast["metric"] = result.metric_name
        forecast["model"] = result.model_name
        forecast["mae"] = result.accuracy.get("mae", 0.0)
        forecast["rmse"] = result.accuracy.get("rmse", 0.0)
        forecast["mape"] = result.accuracy.get("mape", 0.0)
        rows.append(forecast)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)
