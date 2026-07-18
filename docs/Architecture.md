# Architecture

## High-Level Architecture

```mermaid
flowchart TD
    A["Raw CSV: Rialto Data.csv"] --> B["Data Pipeline"]
    B --> C["Cleaned Feature Dataset"]
    C --> D["Descriptive Analytics"]
    C --> E["NLP Pipeline"]
    C --> F["Machine Learning Pipeline"]
    C --> G["Forecast Pipeline"]
    D --> H["Streamlit Dashboards"]
    E --> H
    F --> H
    G --> H
    D --> I["Power BI Export"]
    E --> I
    F --> I
    G --> I
    D --> J["Executive AI Advisor"]
    E --> J
    F --> J
    G --> J
    J --> K["Business Report Generator"]
```

The platform keeps calculations in `src/` modules and presentation in `app/`. Streamlit pages call reusable pipelines instead of embedding business logic directly in the UI.

## Data Pipeline

```mermaid
flowchart LR
    A["data/raw/Rialto Data.csv"] --> B["load_csv"]
    B --> C["validate_schema"]
    C --> D["clean_missing_values"]
    D --> E["engineer_features"]
    E --> F["data/processed/rialto_cleaned.csv"]
    F --> G["Dashboards and Exports"]
```

Engineered fields include `Month`, `Quarter`, `Year`, `Return_Flag`, `Revenue_Band`, `Feedback_Length`, and `Low_Satisfaction_Flag`.

## Analytics Pipeline

```mermaid
flowchart TD
    A["Cleaned Feature Dataset"] --> B["Executive KPI Calculations"]
    A --> C["Sales Aggregations"]
    A --> D["Customer Aggregations"]
    A --> E["Returns Aggregations"]
    A --> F["Satisfaction Aggregations"]
    B --> G["Executive Overview"]
    C --> H["Sales Analytics"]
    D --> I["Customer Analytics"]
    E --> J["Returns Analytics"]
    F --> K["Customer Satisfaction"]
```

The analytics layer uses reusable Pandas aggregations from `src/analytics.py` and `src/data_pipeline.py`. Dashboard pages render these outputs without changing the calculation rules.

## ML Pipeline

```mermaid
flowchart TD
    A["Cleaned Transactions"] --> B["Feature Preparation"]
    B --> C["Return Classification"]
    B --> D["Satisfaction Regression"]
    B --> E["Customer Segment Prediction"]
    C --> F["Model Comparison"]
    D --> F
    E --> F
    F --> G["Feature Importance and SHAP"]
    G --> H["Predictions and High-Risk Customers"]
    H --> I["Saved Joblib Models"]
```

The ML layer compares multiple candidate models and selects the best performer using task-appropriate metrics.

## NLP Pipeline

```mermaid
flowchart TD
    A["Feedback Text"] --> B["Preprocessing"]
    B --> C["VADER Sentiment"]
    B --> D["BERT Sentiment if Available"]
    B --> E["Word Frequency and N-grams"]
    B --> F["Topic Modeling"]
    C --> G["Sentiment Dashboard"]
    D --> G
    E --> G
    F --> G
    G --> H["Executive Sentiment Summary"]
```

The NLP pipeline uses automatic fallbacks for unavailable NLTK resources, BERT models, BERTopic, and WordCloud.

## Forecast Pipeline

```mermaid
flowchart LR
    A["Monthly Aggregations"] --> B["Prophet"]
    B --> C{"Success?"}
    C -- "Yes" --> D["Forecast Output"]
    C -- "No" --> E["ARIMA"]
    E --> F{"Success?"}
    F -- "Yes" --> D
    F -- "No" --> G["Naive Fallback"]
    G --> D
    D --> H["Forecast Dashboard and Power BI Export"]
```

Forecasts cover revenue, transactions, customer growth, and satisfaction.

## GenAI Pipeline

```mermaid
flowchart TD
    A["Calculated KPIs"] --> D["Structured Analytics Payload"]
    B["Sentiment Outputs"] --> D
    C["ML and Forecast Outputs"] --> D
    D --> E["Gemini Advisor Prompt"]
    E --> F{"Gemini Available?"}
    F -- "Yes" --> G["AI Executive Advisor"]
    F -- "No" --> H["Rule-Based Fallback"]
    G --> I["Dashboard and Report"]
    H --> I
```

GenAI is constrained to explanation and recommendation. It is not used for primary analytical calculations.

## Export Pipeline

```mermaid
flowchart LR
    A["Cleaned Data"] --> B["Executive KPI CSV"]
    A --> C["Sales CSV"]
    A --> D["Customer CSV"]
    A --> E["Returns CSV"]
    F["Forecasts"] --> G["Forecast CSV"]
    H["Prediction Outputs"] --> I["Prediction CSV"]
    J["Sentiment Outputs"] --> K["Sentiment CSV"]
    B --> L["data/power_bi"]
    C --> L
    D --> L
    E --> L
    G --> L
    I --> L
    K --> L
```
