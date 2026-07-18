# Business Analytics Report

## Executive Summary

The Rialto Decision Intelligence Platform analyzes transaction, customer, return, satisfaction, feedback, predictive, and forecasting data from `Rialto Data.csv`. It is designed as an executive decision-support platform rather than a static report. The system combines descriptive analytics, natural language processing, machine learning, forecasting, GenAI-based explanation, Power BI exports, and automated reporting.

The analysis identifies revenue performance, customer behavior, return risk, satisfaction patterns, and feedback sentiment as the primary decision areas. Predictive and forecasting components extend the platform from retrospective reporting into forward-looking decision intelligence.

## Business Objectives

- Build a dynamic analytics application from the Rialto transaction dataset.
- Provide executive KPIs and business-facing dashboards.
- Identify sales, customer, return, satisfaction, and sentiment patterns.
- Predict return risk, satisfaction outcomes, and customer segments.
- Forecast revenue, transaction activity, customer growth, and satisfaction.
- Export curated datasets for Power BI.
- Generate business recommendations through a GenAI explanation layer grounded in calculated analytics.

## Dataset Overview

The source dataset is `Rialto Data.csv`. Required fields include transaction ID, customer ID, transaction date, revenue amount, return status, satisfaction score, and feedback text.

The pipeline validates the schema, removes duplicates, handles missing values, converts dates, and engineers reusable business features including Month, Quarter, Year, Return Flag, Revenue Band, Feedback Length, and Low Satisfaction Flag.

## Descriptive Analytics

The Executive Dashboard summarizes total revenue, total transactions, total customers, average order value, return rate, and average satisfaction. These KPIs refresh dynamically from the current dataset and active dashboard filters.

## Customer Insights

Customer analytics highlight unique customers, repeat behavior, average spend, satisfaction patterns, and RFM-style customer segments. These insights support retention planning, customer value analysis, and prioritization of high-value or at-risk customer groups.

## Sales Insights

Sales analytics evaluate monthly revenue, transaction volume, growth patterns, revenue distribution, and quarterly performance. The dashboard helps leadership identify stronger and weaker periods and understand revenue concentration across transaction values.

## Returns Analysis

Returns analytics quantify return rate, returned revenue, non-returned revenue, and return behavior by month and revenue band. These outputs support operational investigation into fulfillment, product quality, expectation mismatch, and service issues.

## Customer Satisfaction

Satisfaction analytics track average, highest, and lowest satisfaction scores, low-satisfaction transactions, score distributions, and satisfaction trends. Low-score records are surfaced to support targeted service recovery and root-cause analysis.

## Sentiment Analysis

The NLP pipeline preprocesses feedback text, applies VADER sentiment, uses BERT when available, generates word clouds, calculates n-grams, extracts topics, and tracks monthly sentiment trends. Fallback logic keeps the dashboard usable when optional NLP dependencies or models are unavailable.

## Predictive Analytics

Predictive analytics include return classification, satisfaction regression, and customer segment prediction. The system compares multiple models, reports performance metrics, displays confusion matrices and feature importance, supports SHAP where available, and exports high-risk transaction records.

## Forecasting

Forecasting covers monthly revenue, transaction volume, customer growth, and satisfaction. Prophet is the preferred model, with ARIMA and naive fallback methods used when needed. Forecast views include historical performance, forecast values, confidence intervals, and holdout accuracy metrics.

## Executive AI Insights

The Executive AI Advisor uses Gemini to generate a CEO-oriented decision brief covering business health, key risks, emerging opportunities, revenue outlook, customer outlook, return outlook, and recommended actions. The system falls back to deterministic rule-based recommendations if Gemini access fails.

## Business Recommendations

1. Use the high-risk transaction output to prioritize operational review of transactions most likely to be returned.
2. Monitor satisfaction and sentiment together to identify recurring customer experience issues before they become larger revenue or retention risks.
3. Use forecast and Power BI exports in recurring leadership reviews to track revenue, customer growth, and return-rate movement over time.

## Risks

- The dataset is compact, so predictive and forecasting outputs should be interpreted with appropriate caution.
- Forecast quality depends on the amount and stability of historical monthly data.
- AI-generated summaries are advisory explanations of calculated analytics, not independent evidence.
- Optional dependencies may use fallback behavior depending on the local runtime environment.

## Future Scope

- Add automated model monitoring and scheduled retraining.
- Add database ingestion and scheduled data refresh.
- Add authentication and role-based access control.
- Add cost, margin, inventory, and channel fields for profitability analysis.
- Add CI/CD, Docker deployment, and dashboard smoke tests.

## Conclusion

The Rialto Decision Intelligence Platform is a complete capstone-ready analytics product. It demonstrates practical business analytics, software engineering, NLP, machine learning, forecasting, GenAI integration, export automation, and executive reporting in one cohesive application.
