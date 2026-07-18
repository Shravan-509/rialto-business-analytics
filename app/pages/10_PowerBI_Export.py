"""Power BI export dashboard page for Milestone 5."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import sys

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analytics import apply_filters
from src.data_pipeline import load_clean_export
from src.export_engine import export_power_bi_datasets
from src.forecasting import run_all_forecasts
from src.reporting import generate_final_report
from src.visualizations import (
    load_css,
    render_footer,
    render_page_header,
    render_sidebar,
)


@st.cache_data(show_spinner=False)
def cached_clean_data():
    """Load and clean the Rialto dataset."""
    return load_clean_export()


@st.cache_data(show_spinner=False)
def cached_forecasts(df):
    """Generate cached forecast outputs for export."""
    return run_all_forecasts(df, periods=6)


def render_download_button(label: str, path: str) -> None:
    """Render a download button for an export artifact."""
    file_path = Path(path)
    if not file_path.exists():
        return
    st.download_button(
        label=label,
        data=file_path.read_bytes(),
        file_name=file_path.name,
        mime="text/csv" if file_path.suffix == ".csv" else "application/octet-stream",
        width="stretch",
    )


def main() -> None:
    """Render the Power BI Export page."""
    st.set_page_config(page_title="Power BI Export | Rialto", layout="wide")
    load_css()

    df = cached_clean_data()
    last_refresh = datetime.now()
    filters = render_sidebar(df, last_refresh)
    filtered_df = apply_filters(df, filters)

    render_page_header(
        "Power BI Export",
        "Power BI Export Center",
        "Generate curated CSV datasets and final report files for business reporting.",
    )

    if filtered_df.empty:
        st.warning("No records match the selected filters.")
        render_footer(last_refresh)
        st.stop()

    with st.spinner("Generating Power BI exports and final report..."):
        forecasts = cached_forecasts(filtered_df)
        exports = export_power_bi_datasets(filtered_df, forecasts)
        report_paths = generate_final_report(filtered_df, forecasts)

    st.markdown("### Power BI CSV Files")
    st.dataframe(
        [{"Export": key, "Path": value} for key, value in exports.items()],
        width="stretch",
    )

    columns = st.columns(3)
    for index, (key, path) in enumerate(exports.items()):
        with columns[index % 3]:
            render_download_button(f"Download {key}.csv", path)

    st.markdown("### Final Report Files")
    st.dataframe(
        [{"Report": key, "Path": value} for key, value in report_paths.items()],
        width="stretch",
    )
    for key, path in report_paths.items():
        if path:
            render_download_button(f"Download {key}", path)

    render_footer(last_refresh)


main()
