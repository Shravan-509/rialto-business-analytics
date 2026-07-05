"""Data loading, validation, cleaning, and feature engineering pipeline."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src import config


def ensure_project_directories() -> None:
    """Create the standard platform directories if they are missing."""
    for directory in [
        config.RAW_DATA_DIR,
        config.PROCESSED_DATA_DIR,
        config.POWER_BI_DIR,
        config.MODELS_DIR,
        config.NOTEBOOKS_DIR,
        config.REPORTS_DIR,
    ]:
        directory.mkdir(parents=True, exist_ok=True)


def load_csv(file_path: Path | str = config.RAW_DATA_FILE) -> pd.DataFrame:
    """Load the Rialto CSV from disk.

    Parameters
    ----------
    file_path:
        Path to a CSV file that follows the Rialto schema.

    Returns
    -------
    pandas.DataFrame
        Raw transaction-level data.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Source CSV was not found at {path}. Add the dataset to data/raw/."
        )
    return pd.read_csv(path)


def validate_schema(df: pd.DataFrame) -> None:
    """Validate that the input dataframe contains every required column."""
    missing_columns = [
        column for column in config.REQUIRED_COLUMNS if column not in df.columns
    ]
    if missing_columns:
        missing = ", ".join(missing_columns)
        required = ", ".join(config.REQUIRED_COLUMNS)
        raise ValueError(
            f"Dataset schema is invalid. Missing columns: {missing}. "
            f"Required columns: {required}."
        )


def clean_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Handle missing values using deterministic dataset-driven rules."""
    cleaned = df.copy()

    cleaned[config.REVENUE_COLUMN] = pd.to_numeric(
        cleaned[config.REVENUE_COLUMN], errors="coerce"
    )
    cleaned[config.SATISFACTION_COLUMN] = pd.to_numeric(
        cleaned[config.SATISFACTION_COLUMN], errors="coerce"
    )

    revenue_median = cleaned[config.REVENUE_COLUMN].median()
    satisfaction_median = cleaned[config.SATISFACTION_COLUMN].median()

    cleaned[config.REVENUE_COLUMN] = cleaned[config.REVENUE_COLUMN].fillna(
        revenue_median
    )
    cleaned[config.SATISFACTION_COLUMN] = cleaned[
        config.SATISFACTION_COLUMN
    ].fillna(satisfaction_median)
    cleaned[config.TEXT_COLUMN] = cleaned[config.TEXT_COLUMN].fillna(
        "No feedback provided"
    )
    cleaned[config.RETURN_COLUMN] = cleaned[config.RETURN_COLUMN].fillna("Unknown")
    return cleaned


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create reusable analytics features from the cleaned transaction table."""
    engineered = df.copy()

    engineered[config.DATE_COLUMN] = pd.to_datetime(
        engineered[config.DATE_COLUMN], errors="coerce"
    )
    engineered = engineered.dropna(subset=[config.DATE_COLUMN]).copy()

    engineered[config.RETURN_COLUMN] = (
        engineered[config.RETURN_COLUMN].astype(str).str.strip().str.title()
    )

    engineered["Month"] = engineered[config.DATE_COLUMN].dt.to_period("M").astype(str)
    engineered["Quarter"] = (
        engineered[config.DATE_COLUMN].dt.to_period("Q").astype(str)
    )
    engineered["Year"] = engineered[config.DATE_COLUMN].dt.year
    engineered["Return_Flag"] = (
        engineered[config.RETURN_COLUMN].str.lower().eq("yes").astype(int)
    )
    engineered["Feedback_Length"] = (
        engineered[config.TEXT_COLUMN].astype(str).str.len()
    )
    engineered["Low_Satisfaction_Flag"] = (
        engineered[config.SATISFACTION_COLUMN]
        <= config.LOW_SATISFACTION_THRESHOLD
    ).astype(int)

    engineered["Revenue_Band"] = _build_revenue_bands(
        engineered[config.REVENUE_COLUMN]
    )
    return engineered


def _build_revenue_bands(revenue: pd.Series) -> pd.Series:
    """Build quartile-based revenue bands that adapt to the current dataset."""
    unique_values = revenue.nunique(dropna=True)
    if unique_values <= 1:
        return pd.Series(["Single Band"] * len(revenue), index=revenue.index)

    quantiles = min(len(config.REVENUE_BAND_LABELS), unique_values)
    labels = config.REVENUE_BAND_LABELS[:quantiles]
    try:
        return pd.qcut(revenue, q=quantiles, labels=labels, duplicates="drop")
    except ValueError:
        return pd.Series(["Unbanded"] * len(revenue), index=revenue.index)


def prepare_data(file_path: Path | str = config.RAW_DATA_FILE) -> pd.DataFrame:
    """Run the full data pipeline from raw CSV to enriched dataset."""
    ensure_project_directories()
    raw_df = load_csv(file_path)
    raw_df.columns = [str(column).strip() for column in raw_df.columns]
    validate_schema(raw_df)

    cleaned = raw_df.drop_duplicates().copy()
    cleaned = clean_missing_values(cleaned)
    cleaned = engineer_features(cleaned)
    return cleaned


def export_cleaned_data(
    df: pd.DataFrame, output_path: Path | str = config.CLEANED_DATA_FILE
) -> Path:
    """Export the cleaned dataset to the processed data directory."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def load_clean_export(file_path: Path | str = config.RAW_DATA_FILE) -> pd.DataFrame:
    """Prepare data and persist the latest cleaned export in one call."""
    cleaned = prepare_data(file_path)
    export_cleaned_data(cleaned)
    return cleaned


def calculate_executive_kpis(df: pd.DataFrame) -> dict[str, float]:
    """Calculate headline KPIs for the Executive Overview page."""
    total_transactions = len(df)
    total_revenue = float(df[config.REVENUE_COLUMN].sum())

    return {
        "total_revenue": total_revenue,
        "total_transactions": int(total_transactions),
        "total_customers": int(df[config.CUSTOMER_ID_COLUMN].nunique()),
        "average_order_value": (
            total_revenue / total_transactions if total_transactions else np.nan
        ),
        "return_rate": float(df["Return_Flag"].mean()) if total_transactions else np.nan,
        "average_satisfaction": float(df[config.SATISFACTION_COLUMN].mean()),
    }


def build_monthly_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate transaction data into monthly revenue and transaction metrics."""
    monthly = (
        df.groupby("Month", as_index=False)
        .agg(
            monthly_revenue=(config.REVENUE_COLUMN, "sum"),
            monthly_transactions=(config.TRANSACTION_ID_COLUMN, "count"),
        )
        .sort_values("Month")
    )
    return monthly
