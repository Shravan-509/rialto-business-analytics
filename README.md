# Rialto Decision Intelligence Platform

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red)
![Machine Learning](https://img.shields.io/badge/ML-Scikit--learn%20%7C%20XGBoost-teal)
![Forecasting](https://img.shields.io/badge/Forecasting-Prophet%20%7C%20ARIMA-purple)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

## Project Overview

Rialto Decision Intelligence Platform is a Master's Business Analytics capstone project built as a production-style Streamlit application. It transforms `Rialto Data.csv` into a full decision intelligence workflow covering descriptive analytics, NLP, predictive modeling, forecasting, executive AI recommendations, Power BI exports, and automated reporting.

Every KPI, chart, model result, export, and recommendation is generated dynamically from the source dataset. Replacing `data/raw/Rialto Data.csv` with another file using the same schema refreshes the platform outputs.

## Features

| Area | Capability |
| --- | --- |
| Executive Dashboard | Revenue, transactions, customers, AOV, return rate, satisfaction |
| Sales Analytics | Revenue trends, growth, order distribution, quarterly analysis |
| Customer Analytics | Customer value, repeat behavior, RFM-style segmentation |
| Returns Analytics | Return rate, returned revenue, return-risk patterns |
| Satisfaction Analytics | Score distribution, low-satisfaction transactions, satisfaction trends |
| NLP | VADER, BERT fallback, word clouds, n-grams, topics, monthly sentiment |
| Predictive Analytics | Return prediction, satisfaction prediction, customer segment prediction |
| Forecasting | Monthly revenue, transaction, customer growth, and satisfaction forecasts |
| AI Advisor | Gemini-powered executive decision brief with rule-based fallback |
| Exports | Power BI CSVs, model outputs, Markdown report, PDF report |

## Technology Stack

- Python
- Streamlit
- Pandas and NumPy
- Plotly
- Scikit-learn
- XGBoost where available
- SHAP where available
- NLTK
- Transformers
- WordCloud
- BERTopic where available
- Prophet
- Statsmodels ARIMA
- Google Gemini API
- OpenAI API support for earlier summary layer
- Joblib
- ReportLab with matplotlib PDF fallback

## Folder Structure

```text
app/
  main.py
  assets/style.css
  pages/
    01_Executive_Overview.py
    02_Sales_Analytics.py
    03_Customer_Analytics.py
    04_Returns_Analytics.py
    05_Customer_Satisfaction.py
    06_Sentiment_Analysis.py
    07_Predictive_Analytics.py
    08_Forecasting.py
    09_Executive_AI_Advisor.py
    10_PowerBI_Export.py
    11_About.py
data/
  raw/
  processed/
  power_bi/
docs/
models/
notebooks/
reports/
src/
  analytics.py
  config.py
  data_pipeline.py
  export_engine.py
  forecasting.py
  genai_layer.py
  ml_pipeline.py
  nlp_pipeline.py
  reporting.py
  utils.py
  visualizations.py
tests/
```

## Architecture

```mermaid
flowchart LR
    A["Rialto Data.csv"] --> B["Data Pipeline"]
    B --> C["Processed Analytics Dataset"]
    C --> D["Descriptive Dashboards"]
    C --> E["NLP Pipeline"]
    C --> F["Predictive ML Pipeline"]
    C --> G["Forecast Pipeline"]
    E --> H["Executive AI Advisor"]
    F --> H
    G --> H
    C --> I["Power BI Export"]
    H --> J["Business Report Generator"]
```

See [docs/Architecture.md](docs/Architecture.md) for detailed pipeline diagrams.

## Machine Learning Models

Return prediction:

- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier
- XGBoost when installed

Customer satisfaction prediction:

- Linear Regression
- Random Forest Regressor
- Gradient Boosting Regressor

Customer segment prediction:

- Random Forest Segment Classifier
- Decision Tree Segment Classifier

The platform compares models using accuracy, precision, recall, F1, ROC AUC, RMSE, MAE, R2, confusion matrices, feature importance, and SHAP when available.

## Forecasting

Forecasting covers:

- Monthly revenue
- Monthly transactions
- Customer growth
- Satisfaction score

Prophet is preferred. If Prophet fails or is unavailable, the system falls back to ARIMA, then to a naive last-value forecast for constrained environments.

## NLP

The NLP pipeline includes:

- Text preprocessing
- Automatic NLTK resource handling
- VADER sentiment
- BERT sentiment when available
- Word clouds
- Word frequency
- N-grams
- Topic extraction using BERTopic or fallback methods
- Monthly sentiment trends

## AI Features

Generative AI is used only as an explanation and recommendation layer. It does not replace calculations, models, or forecasts.

Supported provider configuration:

```bash
export LLM_PROVIDER=gemini
export GEMINI_API_KEY="your-gemini-key"
export OPENAI_API_KEY="your-openai-key"
```

Models:

- Gemini: `gemini-2.5-flash`
- OpenAI: `gpt-4.1-mini`

If API keys, packages, network access, or provider calls fail, the platform uses rule-based executive summaries.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Place the dataset at:

```text
data/raw/Rialto Data.csv
```

## Running Locally

```bash
streamlit run app/main.py
```

Open the local Streamlit URL shown in the terminal.

## Environment Variables

| Variable | Required | Purpose |
| --- | --- | --- |
| `LLM_PROVIDER` | No | `gemini`, `openai`, or fallback behavior |
| `GEMINI_API_KEY` | No | Enables Gemini Executive AI Advisor |
| `OPENAI_API_KEY` | No | Enables OpenAI-backed summary support |
| `GEMINI_MODEL` | No | Override Gemini model |
| `OPENAI_MODEL` | No | Override OpenAI model |

## 📸 Application Screenshots

### 🏠 Home

The landing page introduces the **Rialto Decision Intelligence Platform**, highlighting the application's purpose, key capabilities, and navigation to various analytics modules.

<p align="center">
  <img src="images/1_home.png" alt="Home" width="1000">
</p>

---

### 📊 Executive Dashboard

Provides a high-level overview of business performance through interactive KPI cards, revenue trends, transaction metrics, customer insights, return rate, and customer satisfaction. This dashboard enables executives to monitor overall business health at a glance.

<p align="center">
  <img src="images/2_dashboard.png" alt="Executive Dashboard" width="1000">
</p>

---

### 💰 Sales Analytics

Analyzes sales performance using interactive visualizations, including monthly revenue trends, quarterly sales, revenue distribution, cumulative revenue, and sales heatmaps to identify growth opportunities.

<p align="center">
  <img src="images/3_sales.png" alt="Sales Analytics" width="1000">
</p>

---

### 👥 Customer Analytics

Explores customer behavior through segmentation, purchase frequency, customer lifetime value, RFM analysis, and revenue contribution to better understand customer engagement and loyalty.

<p align="center">
  <img src="images/4_customer.png" alt="Customer Analytics" width="1000">
</p>

---

### 📦 Returns Analytics

Examines return patterns, return rates, returned revenue, and category-level return trends to identify operational challenges and opportunities for improving product quality and customer satisfaction.

<p align="center">
  <img src="images/5_returns.png" alt="Returns Analytics" width="1000">
</p>

---

### ⭐ Customer Satisfaction

Visualizes customer satisfaction scores using distributions, trends, heatmaps, and detailed insights to help evaluate customer experience across different segments.

<p align="center">
  <img src="images/6_descriptive.png" alt="Customer Satisfaction" width="1000">
</p>

---

### 😊 Sentiment Analysis

Uses Natural Language Processing (NLP) to analyze customer reviews, classify sentiment, identify key themes, generate word clouds, and produce AI-assisted summaries for actionable business insights.

<p align="center">
  <img src="images/7_sentiment.png" alt="Sentiment Analysis" width="1000">
</p>

---

### 🤖 Predictive Analytics

Applies machine learning models to predict customer behavior, return likelihood, and satisfaction levels while presenting feature importance, evaluation metrics, and business recommendations.

<p align="center">
  <img src="images/8_predictive.png" alt="Predictive Analytics" width="1000">
</p>

---

### 📈 Forecasting

Generates future forecasts for revenue, customer growth, transactions, and satisfaction using Prophet and ARIMA models to support strategic planning and business decision-making.

<p align="center">
  <img src="images/9_forecast.png" alt="Forecasting" width="1000">
</p>

---

### 🧠 Executive AI Advisor

Leverages Google Gemini to transform analytical results into executive-ready business recommendations, highlighting opportunities, risks, strategic actions, and overall business health.

<p align="center">
  <img src="images/10_ai_advisor.png" alt="Executive AI Advisor" width="1000">
</p>

---

### 📄 Power BI Export

Allows users to export processed analytics datasets and business reports for further visualization and reporting in Microsoft Power BI.

<p align="center">
  <img src="images/11_powerbi_export.png" alt="Power BI Export" width="1000">
</p>

---

### ℹ️ About

Provides comprehensive project information including architecture, technology stack, machine learning models, AI capabilities, forecasting approach, version information, and author details.

<p align="center">
  <img src="images/12_about.png" alt="About" width="1000">
</p>


## Generated Outputs

Processed datasets:

```text
data/processed/rialto_cleaned.csv
data/processed/sentiment_results.csv
data/processed/predictions.csv
data/processed/model_metrics.csv
data/processed/feature_importance.csv
```

Power BI exports:

```text
data/power_bi/executive_kpi.csv
data/power_bi/sales.csv
data/power_bi/customer.csv
data/power_bi/returns.csv
data/power_bi/forecast.csv
data/power_bi/prediction.csv
data/power_bi/sentiment.csv
```

Reports:

```text
reports/Business_Analytics_Report.md
reports/business_analytics_report.md
reports/business_analytics_report.pdf
```

## Deployment

Recommended deployment options:

- Streamlit Community Cloud for portfolio/public capstone demos
- Render or Railway for lightweight hosted deployments
- Docker-based deployment for controlled enterprise environments

Deployment checklist:

- Add `data/raw/Rialto Data.csv` or configure secure data loading
- Set secrets in the hosting platform, not in source control
- Install dependencies from `requirements.txt`
- Run `streamlit run app/main.py`
- Verify AI fallbacks work when API keys are unavailable

## Quality Assurance

Run tests:

```bash
pytest
```

Compile-check Python files:

```bash
python -m py_compile app/main.py src/*.py app/pages/*.py
```

## Future Enhancements

- Add authentication for multi-user access
- Add Docker and CI/CD workflows
- Add richer scenario planning for executive decision support
- Add automated model drift monitoring
- Add database ingestion alongside CSV ingestion
- Add screenshots and a hosted demo link

## Acknowledgements

This capstone uses open-source Python analytics, visualization, NLP, forecasting, and machine learning libraries. Generative AI summaries are grounded in calculated platform analytics and include deterministic fallbacks.

## License

This project is released under the MIT License. See [LICENSE](LICENSE).
