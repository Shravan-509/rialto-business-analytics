"""Basic tests for the Milestone 1 data pipeline."""

from src.data_pipeline import calculate_executive_kpis, prepare_data


def test_prepare_data_creates_required_features():
    """The cleaned dataset should include all Milestone 1 engineered features."""
    df = prepare_data()
    expected_features = {
        "Month",
        "Quarter",
        "Year",
        "Return_Flag",
        "Revenue_Band",
        "Feedback_Length",
        "Low_Satisfaction_Flag",
    }
    assert expected_features.issubset(df.columns)


def test_executive_kpis_are_dynamic():
    """Executive KPI calculation should return the six required metrics."""
    df = prepare_data()
    kpis = calculate_executive_kpis(df)
    assert set(kpis) == {
        "total_revenue",
        "total_transactions",
        "total_customers",
        "average_order_value",
        "return_rate",
        "average_satisfaction",
    }
