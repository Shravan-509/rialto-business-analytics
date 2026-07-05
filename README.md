# Rialto AI-Powered Business Analytics Platform

## Project Overview

This Master's Business Analytics capstone is an interactive Streamlit platform built entirely from `Rialto Data.csv`. Milestone 1 implements the dynamic data pipeline and the Executive Overview dashboard.

The application currently loads the raw CSV, validates the schema, handles missing values, removes duplicates, converts dates, engineers business features, exports a cleaned dataset, and displays executive KPIs and Plotly charts.

All analytics are calculated dynamically from the CSV. Replacing `data/raw/Rialto Data.csv` with another file using the same schema will refresh the dashboard outputs.

## Architecture

```text
Rialto Data.csv
  ↓
Data Loading & Schema Validation
  ↓
Cleaning & Feature Engineering
  ↓
Cleaned Dataset Export
  ↓
Streamlit Executive Overview
```

Approved full-platform architecture:

```text
app/
├── main.py
├── pages/
└── assets/
data/
├── raw/
├── processed/
└── power_bi/
src/
├── config.py
├── data_pipeline.py
├── analytics.py
├── visualizations.py
├── ml_pipeline.py
├── nlp_pipeline.py
├── forecasting.py
├── genai_layer.py
├── export_engine.py
└── utils.py
```

Milestone 1 scope:

- Streamlit app shell
- Sidebar navigation
- Dynamic CSV data pipeline
- Cleaned data export to `data/processed/`
- Executive Overview page
- Total Revenue
- Total Transactions
- Total Customers
- Average Order Value
- Return Rate
- Average Satisfaction
- Monthly Revenue chart
- Monthly Transactions chart
- Revenue Distribution chart

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## How to Run

Place the source file here:

```text
data/raw/Rialto Data.csv
```

Then start the Streamlit app:

```bash
streamlit run app/main.py
```

The app will automatically export the cleaned dataset to:

```text
data/processed/rialto_cleaned.csv
```

## Current Status

Milestone 1 is complete. Sentiment analysis, machine learning, forecasting, GenAI, and Power BI export are intentionally reserved for later milestones.
