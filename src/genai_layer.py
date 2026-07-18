"""Executive decision intelligence layer powered by Gemini with fallback."""

from __future__ import annotations

import json
import logging
import re
from functools import lru_cache
from typing import Any

import pandas as pd

from src import config
from src.analytics import (
    calculate_customer_kpis,
    calculate_returns_kpis,
    calculate_sales_kpis,
)
from src.data_pipeline import calculate_executive_kpis
from src.forecasting import ForecastResult
from src.nlp_pipeline import get_provider_api_key


logger = logging.getLogger(__name__)


def build_decision_payload(
    df: pd.DataFrame,
    forecasts: dict[str, ForecastResult] | None = None,
) -> dict[str, Any]:
    """Build the aggregated analytics payload for Executive AI Advisor."""
    sentiment = load_processed_summary("sentiment_results.csv", "vader_label")
    predictions = load_processed_summary("predictions.csv", "Actual Return")
    forecast_payload = {}
    for name, result in (forecasts or {}).items():
        latest = result.forecast.iloc[0].to_dict() if not result.forecast.empty else {}
        forecast_payload[name] = {
            "model": result.model_name,
            "next_period_forecast": latest,
            "accuracy": result.accuracy,
        }

    return {
        "executive_kpis": calculate_executive_kpis(df),
        "sales_kpis": calculate_sales_kpis(df),
        "customer_kpis": calculate_customer_kpis(df),
        "returns_kpis": calculate_returns_kpis(df),
        "sentiment_summary": sentiment,
        "prediction_summary": predictions,
        "forecast_summary": forecast_payload,
        "record_count": len(df),
        "date_range": {
            "start": str(df[config.DATE_COLUMN].min().date()),
            "end": str(df[config.DATE_COLUMN].max().date()),
        },
    }


def load_processed_summary(filename: str, category_column: str) -> dict[str, Any]:
    """Load compact counts from a processed dataset if available."""
    path = config.PROCESSED_DATA_DIR / filename
    if not path.exists():
        return {"source": filename, "available": False}
    try:
        frame = pd.read_csv(path)
        counts = (
            frame[category_column].value_counts().to_dict()
            if category_column in frame.columns
            else {}
        )
        return {
            "source": filename,
            "available": True,
            "rows": len(frame),
            "counts": counts,
        }
    except Exception as exc:
        logger.info("Could not summarize %s: %s", path, exc)
        return {"source": filename, "available": False}


def generate_executive_ai_advisor(
    df: pd.DataFrame,
    forecasts: dict[str, ForecastResult] | None = None,
) -> tuple[dict[str, Any], str]:
    """Generate an executive AI advisor report with Gemini and fallback."""
    payload = build_decision_payload(df, forecasts)
    payload_key = json.dumps(payload, sort_keys=True, default=str)
    return cached_gemini_advisor(payload_key)


@lru_cache(maxsize=8)
def cached_gemini_advisor(payload_key: str) -> tuple[dict[str, Any], str]:
    """Cache Gemini advisor responses for identical analytics payloads."""
    payload = json.loads(payload_key)
    fallback = build_rule_based_advisor(payload)
    api_key = get_provider_api_key("GEMINI_API_KEY")
    if not api_key:
        logger.info("GEMINI_API_KEY missing; using Executive AI Advisor fallback.")
        return fallback, "Rule-based fallback"

    try:
        from google import genai

        prompt = build_advisor_prompt(payload)
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model=config.GEMINI_MODEL,
            contents=prompt,
        )
        return parse_advisor_response(response.text or ""), "AI-generated with Gemini"
    except Exception as exc:
        logger.exception("Gemini Executive AI Advisor failed; using fallback: %s", exc)
        return fallback, "Rule-based fallback"


def build_advisor_prompt(payload: dict[str, Any]) -> str:
    """Build the Executive AI Advisor Gemini prompt."""
    return f"""
You are a Senior Business Intelligence Consultant advising a CEO.

Use ONLY the supplied Rialto analytics. Never invent numbers. If evidence is
insufficient, say so clearly. Return only valid JSON.

Create an executive decision intelligence report with these exact keys:
overall_business_health, key_risks, emerging_opportunities, recommended_actions,
revenue_outlook, customer_outlook, return_outlook, executive_summary.

recommended_actions must contain exactly three concise actions.

Analytics:
{json.dumps(payload, default=str)}
"""


def parse_advisor_response(text: str) -> dict[str, Any]:
    """Parse the Gemini JSON response for Executive AI Advisor."""
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            raise
        parsed = json.loads(match.group(0))

    required_keys = [
        "overall_business_health",
        "key_risks",
        "emerging_opportunities",
        "recommended_actions",
        "revenue_outlook",
        "customer_outlook",
        "return_outlook",
        "executive_summary",
    ]
    for key in required_keys:
        parsed.setdefault(key, "Evidence is insufficient for this section.")
    actions = parsed.get("recommended_actions", [])
    if not isinstance(actions, list):
        actions = [str(actions)]
    parsed["recommended_actions"] = [str(item) for item in actions[:3]]
    while len(parsed["recommended_actions"]) < 3:
        parsed["recommended_actions"].append(
            "Continue monitoring the metric as new Rialto data becomes available."
        )
    return parsed


def build_rule_based_advisor(payload: dict[str, Any]) -> dict[str, Any]:
    """Build a deterministic Executive AI Advisor fallback."""
    executive = payload["executive_kpis"]
    sales = payload["sales_kpis"]
    returns = payload["returns_kpis"]
    customers = payload["customer_kpis"]
    return_rate = returns.get("return_rate", 0.0)
    revenue = executive.get("total_revenue", 0.0)
    satisfaction = executive.get("average_satisfaction", 0.0)

    return {
        "overall_business_health": (
            f"Rialto generated ${revenue:,.2f} across "
            f"{executive.get('total_transactions', 0):,} transactions with "
            f"average satisfaction of {satisfaction:.2f}."
        ),
        "key_risks": (
            f"The current return rate is {return_rate:.1%}; transactions with "
            "high return probability should remain an operational focus."
        ),
        "emerging_opportunities": (
            f"Repeat customer rate is {customers.get('repeat_customer_rate', 0.0):.1%}, "
            "which can guide targeted retention and loyalty decisions."
        ),
        "recommended_actions": [
            "Review high-return-risk transactions before fulfillment and service decisions.",
            "Protect revenue growth by monitoring monthly revenue and customer repeat behavior.",
            "Use satisfaction and sentiment findings to prioritize customer experience fixes.",
        ],
        "revenue_outlook": (
            f"Monthly revenue growth is {sales.get('monthly_growth_pct', 0.0):.1%} "
            "based only on the current analytics."
        ),
        "customer_outlook": (
            f"The platform identified {executive.get('total_customers', 0):,} unique customers."
        ),
        "return_outlook": (
            f"Returned revenue totals ${returns.get('returned_revenue', 0.0):,.2f}."
        ),
        "executive_summary": (
            "This advisor report is generated from the supplied Rialto analytics and "
            "does not assume external market conditions."
        ),
    }
