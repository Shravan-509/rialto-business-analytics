"""Predictive Analytics dashboard page for Milestone 4."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import sys

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import config
from src.analytics import apply_filters
from src.data_pipeline import load_clean_export
from src.ml_pipeline import (
    PredictiveDependencyError,
    generate_predictive_summary,
    predict_what_if_return_probability,
    run_predictive_pipeline,
)
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


@st.cache_data(show_spinner=False)
def cached_clean_data():
    """Load and clean the Rialto dataset."""
    return load_clean_export()


@st.cache_resource(show_spinner=False)
def cached_predictive_results(df):
    """Train and cache predictive models."""
    return run_predictive_pipeline(df)


def render_summary_cards(results) -> None:
    """Render top-line predictive summary cards."""
    st.markdown("### Prediction Summary")
    return_best = results.return_result.metrics.sort_values(
        ["roc_auc", "f1", "accuracy"], ascending=False
    ).iloc[0]
    satisfaction_best = results.satisfaction_result.metrics.sort_values("rmse").iloc[0]
    segment_best = results.segment_result.metrics.sort_values(
        ["f1", "accuracy"], ascending=False
    ).iloc[0]
    top_risk = results.high_risk_transactions.iloc[0]
    cards = [
        {
            "icon": "RET",
            "title": "Best Return Model",
            "value": results.return_result.best_model_name,
            "description": f"F1: {return_best['f1']:.2f} | ROC AUC: {return_best['roc_auc']:.2f}",
        },
        {
            "icon": "SAT",
            "title": "Best Satisfaction Model",
            "value": results.satisfaction_result.best_model_name,
            "description": f"RMSE: {satisfaction_best['rmse']:.2f} | MAE: {satisfaction_best['mae']:.2f}",
        },
        {
            "icon": "SEG",
            "title": "Segment Model",
            "value": results.segment_result.best_model_name,
            "description": f"Accuracy: {segment_best['accuracy']:.2f}",
        },
        {
            "icon": "RISK",
            "title": "Highest Risk Transaction",
            "value": str(top_risk["Transaction ID"]),
            "description": f"Return probability: {format_percent(top_risk['Predicted Return Probability'])}",
        },
    ]
    render_kpi_cards(cards, columns_per_row=4)


def render_model_tables(results) -> None:
    """Render model performance tables."""
    st.markdown("### Model Performance")
    st.markdown("**Return Prediction Models**")
    st.dataframe(results.return_result.metrics, width="stretch")
    st.markdown("**Satisfaction Prediction Models**")
    st.dataframe(results.satisfaction_result.metrics, width="stretch")
    st.markdown("**Customer Segment Prediction Models**")
    st.dataframe(results.segment_result.metrics, width="stretch")
    st.markdown("**Classification Report: Return Prediction**")
    st.dataframe(results.return_result.classification_report, width="stretch")
    st.markdown("**Classification Report: Customer Segment Prediction**")
    st.dataframe(results.segment_result.classification_report, width="stretch")


def render_return_charts(results) -> None:
    """Render return prediction evaluation charts."""
    return_result = results.return_result

    prediction_counts = (
        return_result.predictions.groupby(["actual", "predicted"], as_index=False)
        .size()
        .rename(columns={"size": "count"})
    )
    fig = px.bar(
        prediction_counts,
        x="actual",
        y="count",
        color="predicted",
        barmode="group",
    )
    render_chart(apply_chart_theme(fig, "Actual vs Predicted Returns"))
    render_insight("Actual vs predicted returns shows how often the selected model separates returned and non-returned orders.")

    if not return_result.roc_curve.empty:
        fig = px.line(return_result.roc_curve, x="fpr", y="tpr")
        fig.add_trace(
            go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash"))
        )
        render_chart(apply_chart_theme(fig, "ROC Curve"))
        render_insight("The ROC curve shows the tradeoff between identifying returned orders and false positives.")

    if not return_result.precision_recall_curve.empty:
        fig = px.line(
            return_result.precision_recall_curve,
            x="recall",
            y="precision",
        )
        render_chart(apply_chart_theme(fig, "Precision Recall Curve"))
        render_insight("Precision-recall performance is useful when returned orders are less frequent than non-returned orders.")

    fig = px.imshow(
        return_result.confusion_matrix,
        text_auto=True,
        color_continuous_scale="Teal",
        aspect="auto",
    )
    render_chart(apply_chart_theme(fig, "Return Prediction Confusion Matrix"))
    render_insight("The confusion matrix identifies correct predictions and the main error types.")

    fig = px.histogram(
        return_result.predictions,
        x="probability",
        color="actual",
        nbins=20,
        color_discrete_sequence=[COLOR_BLUE, COLOR_RED],
    )
    render_chart(apply_chart_theme(fig, "Prediction Probability Distribution"))
    render_insight("Probability distribution highlights how confidently the model scores return risk.")

    importance = return_result.feature_importance.head(15)
    fig = px.bar(
        importance,
        x="importance",
        y="feature",
        orientation="h",
        color_discrete_sequence=[COLOR_REVENUE],
    )
    fig.update_layout(yaxis=dict(autorange="reversed"))
    render_chart(apply_chart_theme(fig, "Return Model Feature Importance"))
    top_feature = importance.iloc[0]["feature"] if not importance.empty else "available features"
    render_insight(f"{top_feature} is the strongest available driver in the selected return model.")

    if not return_result.shap_importance.empty:
        shap_df = return_result.shap_importance.head(15)
        fig = px.bar(
            shap_df,
            x="mean_abs_shap",
            y="feature",
            orientation="h",
            color_discrete_sequence=[COLOR_TEAL],
        )
        fig.update_layout(yaxis=dict(autorange="reversed"))
        render_chart(apply_chart_theme(fig, "SHAP Summary Importance"))
        render_insight("SHAP importance estimates average contribution size where SHAP is available.")


def render_regression_charts(results) -> None:
    """Render satisfaction regression evaluation charts."""
    predictions = results.satisfaction_result.predictions
    fig = px.scatter(
        predictions,
        x="actual",
        y="predicted",
        color_discrete_sequence=[COLOR_TEAL],
    )
    fig.add_trace(
        go.Scatter(
            x=[predictions["actual"].min(), predictions["actual"].max()],
            y=[predictions["actual"].min(), predictions["actual"].max()],
            mode="lines",
            line=dict(dash="dash", color=COLOR_RED),
            name="Perfect prediction",
        )
    )
    render_chart(apply_chart_theme(fig, "Actual vs Predicted Satisfaction"))
    render_insight("Actual vs predicted satisfaction compares model estimates with observed customer scores.")

    fig = px.scatter(
        predictions,
        x="predicted",
        y="residual",
        color_discrete_sequence=[COLOR_BLUE],
    )
    fig.add_hline(y=0, line_dash="dash", line_color=COLOR_RED)
    render_chart(apply_chart_theme(fig, "Residual Plot"))
    render_insight("Residuals show where satisfaction predictions are over- or under-estimated.")

    fig = px.histogram(
        predictions,
        x="residual",
        nbins=20,
        color_discrete_sequence=[COLOR_TEAL],
    )
    render_chart(apply_chart_theme(fig, "Error Distribution"))
    render_insight("Error distribution shows the spread of satisfaction prediction errors.")

    importance = results.satisfaction_result.feature_importance.head(15)
    if not importance.empty:
        fig = px.bar(
            importance,
            x="importance",
            y="feature",
            orientation="h",
            color_discrete_sequence=[COLOR_REVENUE],
        )
        fig.update_layout(yaxis=dict(autorange="reversed"))
        render_chart(apply_chart_theme(fig, "Satisfaction Model Feature Importance"))
        render_insight("Satisfaction feature importance highlights variables most associated with predicted satisfaction scores.")


def render_segment_charts(results) -> None:
    """Render customer segment prediction evaluation charts."""
    fig = px.imshow(
        results.segment_result.confusion_matrix,
        text_auto=True,
        color_continuous_scale="Teal",
        aspect="auto",
    )
    render_chart(apply_chart_theme(fig, "Customer Segment Confusion Matrix"))
    render_insight("Segment confusion matrix shows how well transaction features map to customer relationship segments.")

    importance = results.segment_result.feature_importance.head(15)
    if not importance.empty:
        fig = px.bar(
            importance,
            x="importance",
            y="feature",
            orientation="h",
            color_discrete_sequence=[COLOR_TEAL],
        )
        fig.update_layout(yaxis=dict(autorange="reversed"))
        render_chart(apply_chart_theme(fig, "Segment Model Feature Importance"))
        render_insight("Segment feature importance identifies which transaction and customer features drive segment assignment.")


def render_high_risk_table(results) -> None:
    """Render high-risk transaction table."""
    st.markdown("### High Risk Transactions")
    high_risk = results.high_risk_transactions.head(25).copy()
    high_risk["Predicted Return Probability"] = high_risk[
        "Predicted Return Probability"
    ].map(lambda value: f"{value:.1%}")
    high_risk["Revenue"] = high_risk["Revenue"].map(lambda value: f"${value:,.2f}")
    st.dataframe(high_risk, width="stretch")


def render_what_if(results) -> None:
    """Render interactive what-if controls."""
    st.markdown("### What-If Return Risk Analysis")
    model_df = results.model_data
    col1, col2, col3, col4 = st.columns(4)
    revenue = col1.slider(
        "Revenue",
        float(model_df[config.REVENUE_COLUMN].min()),
        float(model_df[config.REVENUE_COLUMN].max()),
        float(model_df[config.REVENUE_COLUMN].median()),
    )
    satisfaction = col2.slider(
        "Satisfaction",
        1.0,
        5.0,
        float(model_df[config.SATISFACTION_COLUMN].median()),
        0.1,
    )
    purchase_frequency = col3.slider(
        "Purchase Frequency",
        float(model_df["customer_purchase_frequency"].min()),
        float(model_df["customer_purchase_frequency"].max()),
        float(model_df["customer_purchase_frequency"].median()),
        1.0,
    )
    sentiment = col4.slider("Sentiment", -1.0, 1.0, float(model_df["vader_compound"].median()), 0.05)
    probability = predict_what_if_return_probability(
        results.return_result.best_model,
        revenue,
        satisfaction,
        purchase_frequency,
        sentiment,
        model_df,
    )
    st.metric("Predicted Return Probability", format_percent(probability))


def render_predictive_summary(results) -> None:
    """Render Gemini or rule-based predictive executive summary."""
    summary, source = generate_predictive_summary(results)
    recommendations = "".join(f"<li>{item}</li>" for item in summary["recommendations"])
    st.markdown("### Business Recommendations")
    st.markdown(
        f"""
        <div class="summary-card">
            <strong>{source}.</strong>
            <p><strong>Model Performance.</strong> {summary.get("executive_overview", "")}</p>
            <p><strong>Highest Risk Customers.</strong> {summary.get("risk_customers", "")}</p>
            <p><strong>Important Variables.</strong> {summary.get("important_variables", "")}</p>
            <p><strong>Business Risks.</strong> {summary.get("business_risks", "")}</p>
            <p><strong>Recommended Actions.</strong></p>
            <ul>{recommendations}</ul>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_exports(results) -> None:
    """Render export file paths."""
    st.markdown("### Predictive Exports")
    st.dataframe(
        [{"Export": key, "Path": value} for key, value in results.exports.items()],
        width="stretch",
    )


def main() -> None:
    """Render the Predictive Analytics page."""
    st.set_page_config(page_title="Predictive Analytics | Rialto", layout="wide")
    load_css()

    df = cached_clean_data()
    last_refresh = datetime.now()
    filters = render_sidebar(df, last_refresh)
    filtered_df = apply_filters(df, filters)

    render_page_header(
        "Predictive Analytics",
        "Predictive Analytics",
        "Return risk, satisfaction prediction, customer segment prediction, and what-if analysis.",
    )

    if filtered_df.empty:
        st.warning("No records match the selected filters.")
        render_footer(last_refresh)
        st.stop()

    try:
        with st.spinner("Training predictive models and generating outputs..."):
            results = cached_predictive_results(filtered_df)
    except PredictiveDependencyError as exc:
        st.error(str(exc))
        render_footer(last_refresh)
        st.stop()
    except Exception as exc:
        st.error(f"Predictive Analytics could not be generated: {exc}")
        render_footer(last_refresh)
        st.stop()

    render_summary_cards(results)
    render_model_tables(results)
    render_return_charts(results)
    render_regression_charts(results)
    render_segment_charts(results)
    render_high_risk_table(results)
    render_what_if(results)
    render_predictive_summary(results)
    render_executive_insights(
        [
            f"{results.return_result.best_model_name} was selected for return prediction.",
            f"{results.satisfaction_result.best_model_name} was selected for satisfaction prediction.",
            "High-risk transactions should be reviewed before operational changes are made.",
        ]
    )
    render_exports(results)
    render_footer(last_refresh)


main()
