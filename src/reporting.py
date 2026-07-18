"""Automated final business analytics report generation."""

from __future__ import annotations

import logging
import textwrap
from pathlib import Path
from typing import Any

import pandas as pd

from src import config
from src.analytics import calculate_customer_kpis, calculate_returns_kpis, calculate_sales_kpis
from src.data_pipeline import calculate_executive_kpis
from src.forecasting import ForecastResult
from src.genai_layer import generate_executive_ai_advisor


logger = logging.getLogger(__name__)


def generate_final_report(
    df: pd.DataFrame,
    forecasts: dict[str, ForecastResult] | None = None,
) -> dict[str, str]:
    """Generate Markdown and PDF final report artifacts."""
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    markdown = build_report_markdown(df, forecasts)
    config.REPORT_MARKDOWN_FILE.write_text(markdown, encoding="utf-8")
    pdf_path = write_report_pdf(markdown, config.REPORT_PDF_FILE)
    return {
        "markdown": str(config.REPORT_MARKDOWN_FILE),
        "pdf": str(pdf_path) if pdf_path else "",
    }


def build_report_markdown(
    df: pd.DataFrame,
    forecasts: dict[str, ForecastResult] | None = None,
) -> str:
    """Build the final report Markdown from dynamic platform analytics."""
    executive = calculate_executive_kpis(df)
    sales = calculate_sales_kpis(df)
    customers = calculate_customer_kpis(df)
    returns = calculate_returns_kpis(df)
    advisor, advisor_source = generate_executive_ai_advisor(df, forecasts)
    sentiment = read_optional_processed("sentiment_results.csv")
    metrics = read_optional_processed("model_metrics.csv")

    forecast_lines = []
    for name, result in (forecasts or {}).items():
        next_value = (
            float(result.forecast["forecast"].iloc[0])
            if not result.forecast.empty
            else 0.0
        )
        forecast_lines.append(
            f"- {name}: {next_value:,.2f} next period using {result.model_name}."
        )
    if not forecast_lines:
        forecast_lines.append("- Forecast evidence is unavailable.")

    sentiment_summary = summarize_sentiment(sentiment)
    predictive_summary = summarize_model_metrics(metrics)

    return f"""# Business Analytics Report

## Executive Summary
{advisor.get("executive_summary", "Evidence is insufficient.")}

Source: {advisor_source}

## Descriptive Analytics
- Total revenue: ${executive["total_revenue"]:,.2f}
- Total transactions: {executive["total_transactions"]:,}
- Total customers: {executive["total_customers"]:,}
- Average order value: ${executive["average_order_value"]:,.2f}
- Monthly revenue growth: {sales["monthly_growth_pct"]:.1%}
- Repeat customer rate: {customers["repeat_customer_rate"]:.1%}
- Return rate: {returns["return_rate"]:.1%}

## Predictive Analytics
{predictive_summary}

## Sentiment Analysis
{sentiment_summary}

## Forecasting
{chr(10).join(forecast_lines)}

## Business Recommendations
1. {advisor["recommended_actions"][0]}
2. {advisor["recommended_actions"][1]}
3. {advisor["recommended_actions"][2]}

## Limitations
- The analysis is based only on the supplied Rialto dataset.
- Forecasts are constrained by the available monthly history.
- AI-generated narratives are explanations of calculated analytics, not replacements for the calculations.

## Future Improvements
- Add more transaction history to strengthen forecasting accuracy.
- Add controlled operational cost and margin data.
- Monitor model performance after each new dataset refresh.
"""


def summarize_sentiment(sentiment: pd.DataFrame) -> str:
    """Summarize processed sentiment output for the report."""
    if sentiment.empty or "vader_label" not in sentiment.columns:
        return "Sentiment output is unavailable. Run the Sentiment Analysis page to refresh NLP outputs."
    counts = sentiment["vader_label"].value_counts()
    total = int(counts.sum())
    details = ", ".join(f"{label}: {count}" for label, count in counts.items())
    return f"Sentiment analysis covers {total:,} feedback records. Distribution: {details}."


def summarize_model_metrics(metrics: pd.DataFrame) -> str:
    """Summarize processed predictive metrics for the report."""
    if metrics.empty or "task" not in metrics.columns or "model" not in metrics.columns:
        return "Predictive model metrics are unavailable. Run the Predictive Analytics page to refresh outputs."
    lines = []
    for task, task_metrics in metrics.groupby("task"):
        if "f1" in task_metrics.columns and task_metrics["f1"].notna().any():
            row = task_metrics.sort_values(["f1", "accuracy"], ascending=False).iloc[0]
            lines.append(f"- {task}: best available model is {row['model']}.")
        elif "rmse" in task_metrics.columns and task_metrics["rmse"].notna().any():
            row = task_metrics.sort_values("rmse").iloc[0]
            lines.append(f"- {task}: best available model is {row['model']} with RMSE {row['rmse']:.2f}.")
    return "\n".join(lines) if lines else "Predictive evidence is insufficient."


def read_optional_processed(filename: str) -> pd.DataFrame:
    """Read a processed CSV if available."""
    path = config.PROCESSED_DATA_DIR / filename
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception as exc:
        logger.info("Could not read %s for final report: %s", path, exc)
        return pd.DataFrame()


def write_report_pdf(markdown: str, output_path: Path) -> Path | None:
    """Write the report PDF with reportlab when available."""
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer
    except Exception as exc:
        logger.info("reportlab unavailable; using matplotlib PDF fallback: %s", exc)
        return write_report_pdf_with_matplotlib(markdown, output_path)

    styles = getSampleStyleSheet()
    story: list[Any] = []
    for block in markdown.split("\n\n"):
        block = block.strip()
        if not block:
            continue
        if block.startswith("# "):
            story.append(Paragraph(block[2:], styles["Title"]))
        elif block.startswith("## "):
            story.append(Paragraph(block[3:], styles["Heading2"]))
        else:
            safe_block = "<br/>".join(
                textwrap.wrap(block.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"), width=95)
            )
            story.append(Paragraph(safe_block, styles["BodyText"]))
        story.append(Spacer(1, 10))

    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=letter,
        rightMargin=54,
        leftMargin=54,
        topMargin=54,
        bottomMargin=54,
    )
    doc.build(story)
    return output_path


def write_report_pdf_with_matplotlib(markdown: str, output_path: Path) -> Path | None:
    """Write a simple readable PDF report using matplotlib."""
    try:
        from matplotlib.backends.backend_pdf import PdfPages
        import matplotlib.pyplot as plt
    except Exception as exc:
        logger.warning("PDF dependencies unavailable; PDF report was not generated: %s", exc)
        return None

    lines = markdown_to_plain_lines(markdown)
    lines_per_page = 42
    with PdfPages(str(output_path)) as pdf:
        for start in range(0, len(lines), lines_per_page):
            page_lines = lines[start : start + lines_per_page]
            fig = plt.figure(figsize=(8.5, 11))
            fig.patch.set_facecolor("white")
            fig.text(
                0.07,
                0.95,
                "\n".join(page_lines),
                ha="left",
                va="top",
                fontsize=10.5,
                family="DejaVu Sans",
                color="#14213d",
                linespacing=1.35,
            )
            fig.text(
                0.5,
                0.035,
                f"{config.APP_TITLE} | Page {start // lines_per_page + 1}",
                ha="center",
                va="bottom",
                fontsize=8,
                color="#64748b",
            )
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
    return output_path


def markdown_to_plain_lines(markdown: str) -> list[str]:
    """Convert report Markdown into wrapped plain text lines for PDF fallback."""
    plain_lines: list[str] = []
    for raw_line in markdown.splitlines():
        line = raw_line.strip()
        if not line:
            plain_lines.append("")
            continue
        line = line.replace("# ", "").replace("## ", "")
        wrapped = textwrap.wrap(line, width=92) or [""]
        plain_lines.extend(wrapped)
    return plain_lines
