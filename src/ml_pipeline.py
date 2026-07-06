"""Predictive analytics pipeline for the Rialto platform."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src import config
from src.analytics import customer_summary
from src.nlp_pipeline import get_provider_api_key


logger = logging.getLogger(__name__)

NUMERIC_RETURN_FEATURES = [
    config.REVENUE_COLUMN,
    config.SATISFACTION_COLUMN,
    "customer_purchase_frequency",
    "recency",
    "monetary",
    "vader_compound",
    "Month_Number",
    "Quarter_Number",
]

NUMERIC_SATISFACTION_FEATURES = [
    config.REVENUE_COLUMN,
    "Return_Flag",
    "customer_purchase_frequency",
    "recency",
    "monetary",
    "vader_compound",
    "Month_Number",
    "Quarter_Number",
]

NUMERIC_SEGMENT_FEATURES = [
    config.REVENUE_COLUMN,
    config.SATISFACTION_COLUMN,
    "Return_Flag",
    "customer_purchase_frequency",
    "recency",
    "monetary",
    "vader_compound",
    "Month_Number",
    "Quarter_Number",
]

CATEGORICAL_FEATURES = ["Revenue_Band", "Quarter"]


@dataclass
class ClassificationResult:
    """Container for classification model output."""

    task_name: str
    best_model_name: str
    best_model: Any
    metrics: pd.DataFrame
    predictions: pd.DataFrame
    confusion_matrix: pd.DataFrame
    classification_report: pd.DataFrame
    feature_importance: pd.DataFrame
    roc_curve: pd.DataFrame
    precision_recall_curve: pd.DataFrame
    shap_importance: pd.DataFrame


@dataclass
class RegressionResult:
    """Container for regression model output."""

    task_name: str
    best_model_name: str
    best_model: Any
    metrics: pd.DataFrame
    predictions: pd.DataFrame
    feature_importance: pd.DataFrame


@dataclass
class PredictiveResults:
    """Container for all Milestone 4 predictive outputs."""

    model_data: pd.DataFrame
    return_result: ClassificationResult
    satisfaction_result: RegressionResult
    segment_result: ClassificationResult
    high_risk_transactions: pd.DataFrame
    exports: dict[str, str]


class PredictiveDependencyError(RuntimeError):
    """Raised when required predictive analytics dependencies are unavailable."""


def require_sklearn() -> dict[str, Any]:
    """Import required scikit-learn objects lazily with a helpful error."""
    try:
        from joblib import dump
        from sklearn.compose import ColumnTransformer
        from sklearn.ensemble import (
            GradientBoostingRegressor,
            RandomForestClassifier,
            RandomForestRegressor,
        )
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import LinearRegression, LogisticRegression
        from sklearn.metrics import (
            accuracy_score,
            classification_report,
            confusion_matrix,
            f1_score,
            mean_absolute_error,
            mean_squared_error,
            precision_recall_curve,
            precision_score,
            r2_score,
            recall_score,
            roc_auc_score,
            roc_curve,
        )
        from sklearn.model_selection import train_test_split
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import OneHotEncoder, StandardScaler
        from sklearn.tree import DecisionTreeClassifier
    except Exception as exc:
        raise PredictiveDependencyError(
            "Predictive Analytics requires scikit-learn and joblib. "
            "Install dependencies with `pip install -r requirements.txt`."
        ) from exc

    return locals()


def prepare_modeling_data(df: pd.DataFrame) -> pd.DataFrame:
    """Create a transaction-level modeling table from cleaned Rialto data."""
    model_df = df.copy()
    model_df["Month_Number"] = model_df[config.DATE_COLUMN].dt.month
    model_df["Quarter_Number"] = model_df[config.DATE_COLUMN].dt.quarter

    customer_features = customer_summary(model_df)[
        [
            config.CUSTOMER_ID_COLUMN,
            "frequency",
            "recency",
            "monetary",
            "segment",
        ]
    ].rename(columns={"frequency": "customer_purchase_frequency"})

    model_df = model_df.merge(customer_features, on=config.CUSTOMER_ID_COLUMN, how="left")
    model_df["vader_compound"] = load_sentiment_scores(model_df)
    model_df["Revenue_Band"] = model_df["Revenue_Band"].astype(str)
    model_df["Quarter"] = model_df["Quarter"].astype(str)
    return model_df


def load_sentiment_scores(model_df: pd.DataFrame) -> pd.Series:
    """Load sentiment scores from exports when available, otherwise use a fallback."""
    sentiment_path = config.PROCESSED_DATA_DIR / "sentiment_results.csv"
    if sentiment_path.exists():
        try:
            sentiment = pd.read_csv(sentiment_path)
            if {
                config.TRANSACTION_ID_COLUMN,
                "vader_compound",
            }.issubset(sentiment.columns):
                merged = model_df[[config.TRANSACTION_ID_COLUMN]].merge(
                    sentiment[[config.TRANSACTION_ID_COLUMN, "vader_compound"]],
                    on=config.TRANSACTION_ID_COLUMN,
                    how="left",
                )
                return merged["vader_compound"].fillna(0.0)
        except Exception as exc:
            logger.info("Could not load sentiment export; using neutral score: %s", exc)
    return pd.Series(0.0, index=model_df.index)


def build_preprocessor(numeric_features: list[str], categorical_features: list[str]):
    """Create a preprocessing transformer for numeric and categorical features."""
    deps = require_sklearn()
    numeric_pipeline = deps["Pipeline"](
        [
            ("imputer", deps["SimpleImputer"](strategy="median")),
            ("scaler", deps["StandardScaler"]()),
        ]
    )
    categorical_pipeline = deps["Pipeline"](
        [
            ("imputer", deps["SimpleImputer"](strategy="most_frequent")),
            ("encoder", deps["OneHotEncoder"](handle_unknown="ignore")),
        ]
    )
    return deps["ColumnTransformer"](
        [
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ]
    )


def split_data(X: pd.DataFrame, y: pd.Series, classification: bool = True):
    """Split data while using stratification only when safe."""
    deps = require_sklearn()
    stratify = None
    if classification and y.nunique() > 1 and y.value_counts().min() >= 2:
        stratify = y
    return deps["train_test_split"](
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=stratify,
    )


def train_return_models(model_df: pd.DataFrame) -> ClassificationResult:
    """Train and evaluate return prediction classification models."""
    deps = require_sklearn()
    features = NUMERIC_RETURN_FEATURES + CATEGORICAL_FEATURES
    X = model_df[features]
    y = model_df["Return_Flag"].astype(int)
    X_train, X_test, y_train, y_test = split_data(X, y)
    preprocessor = build_preprocessor(NUMERIC_RETURN_FEATURES, CATEGORICAL_FEATURES)

    models = {
        "Logistic Regression": deps["LogisticRegression"](
            max_iter=1000, class_weight="balanced"
        ),
        "Decision Tree": deps["DecisionTreeClassifier"](
            max_depth=4, random_state=42, class_weight="balanced"
        ),
        "Random Forest": deps["RandomForestClassifier"](
            n_estimators=250, random_state=42, class_weight="balanced"
        ),
    }
    try:
        from xgboost import XGBClassifier

        models["XGBoost"] = XGBClassifier(
            n_estimators=150,
            max_depth=3,
            learning_rate=0.05,
            eval_metric="logloss",
            random_state=42,
        )
    except Exception as exc:
        logger.info("XGBoost unavailable; skipping return model: %s", exc)

    return evaluate_classification_models(
        task_name="Return Prediction",
        models=models,
        preprocessor=preprocessor,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        feature_frame=X,
        positive_label=1,
    )


def evaluate_classification_models(
    task_name: str,
    models: dict[str, Any],
    preprocessor: Any,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    feature_frame: pd.DataFrame,
    positive_label: int | str,
) -> ClassificationResult:
    """Train, compare, and evaluate classification models."""
    deps = require_sklearn()
    metric_rows = []
    fitted = {}

    for name, estimator in models.items():
        pipeline = deps["Pipeline"](
            [("preprocessor", preprocessor), ("model", estimator)]
        )
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)
        probabilities = get_positive_probabilities(pipeline, X_test, positive_label)
        metric_rows.append(
            {
                "task": task_name,
                "model": name,
                "accuracy": deps["accuracy_score"](y_test, predictions),
                "precision": deps["precision_score"](
                    y_test, predictions, average="weighted", zero_division=0
                ),
                "recall": deps["recall_score"](
                    y_test, predictions, average="weighted", zero_division=0
                ),
                "f1": deps["f1_score"](
                    y_test, predictions, average="weighted", zero_division=0
                ),
                "roc_auc": safe_roc_auc(deps, y_test, probabilities),
            }
        )
        fitted[name] = pipeline

    metrics = pd.DataFrame(metric_rows)
    metrics["_rank_auc"] = metrics["roc_auc"].fillna(-1)
    best_name = (
        metrics.sort_values(["_rank_auc", "f1", "accuracy"], ascending=False)
        .iloc[0]["model"]
    )
    metrics = metrics.drop(columns=["_rank_auc"])
    best_model = fitted[str(best_name)]
    best_predictions = best_model.predict(X_test)
    best_probabilities = get_positive_probabilities(best_model, X_test, positive_label)
    labels = sorted(pd.Series(y_test).dropna().unique().tolist())

    prediction_frame = X_test.copy()
    prediction_frame["actual"] = y_test.values
    prediction_frame["predicted"] = best_predictions
    prediction_frame["probability"] = best_probabilities

    cm = pd.DataFrame(
        deps["confusion_matrix"](y_test, best_predictions, labels=labels),
        index=[f"Actual {label}" for label in labels],
        columns=[f"Predicted {label}" for label in labels],
    )
    report = pd.DataFrame(
        deps["classification_report"](
            y_test, best_predictions, zero_division=0, output_dict=True
        )
    ).T.reset_index(names="class")

    roc_df = build_roc_curve(deps, y_test, best_probabilities)
    pr_df = build_precision_recall_curve(deps, y_test, best_probabilities)
    importance = feature_importance(best_model)
    shap_df = shap_importance(best_model, feature_frame)

    return ClassificationResult(
        task_name=task_name,
        best_model_name=str(best_name),
        best_model=best_model,
        metrics=metrics,
        predictions=prediction_frame,
        confusion_matrix=cm,
        classification_report=report,
        feature_importance=importance,
        roc_curve=roc_df,
        precision_recall_curve=pr_df,
        shap_importance=shap_df,
    )


def safe_roc_auc(deps: dict[str, Any], y_true: pd.Series, probability: np.ndarray) -> float:
    """Calculate ROC AUC safely."""
    try:
        if pd.Series(y_true).nunique() <= 1:
            return np.nan
        return float(deps["roc_auc_score"](y_true, probability))
    except Exception:
        return np.nan


def get_positive_probabilities(model: Any, X: pd.DataFrame, positive_label: int | str) -> np.ndarray:
    """Return positive-class probabilities when available."""
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X)
        classes = list(model.named_steps["model"].classes_)
        if positive_label in classes:
            return probabilities[:, classes.index(positive_label)]
        return probabilities[:, -1]
    return np.zeros(len(X))


def build_roc_curve(deps: dict[str, Any], y_true: pd.Series, probability: np.ndarray) -> pd.DataFrame:
    """Build ROC curve data."""
    try:
        if pd.Series(y_true).nunique() <= 1:
            return pd.DataFrame(columns=["fpr", "tpr", "threshold"])
        fpr, tpr, threshold = deps["roc_curve"](y_true, probability)
        return pd.DataFrame({"fpr": fpr, "tpr": tpr, "threshold": threshold})
    except Exception:
        return pd.DataFrame(columns=["fpr", "tpr", "threshold"])


def build_precision_recall_curve(deps: dict[str, Any], y_true: pd.Series, probability: np.ndarray) -> pd.DataFrame:
    """Build precision-recall curve data."""
    try:
        precision, recall, threshold = deps["precision_recall_curve"](
            y_true, probability
        )
        threshold_values = np.append(threshold, np.nan)
        return pd.DataFrame(
            {"precision": precision, "recall": recall, "threshold": threshold_values}
        )
    except Exception:
        return pd.DataFrame(columns=["precision", "recall", "threshold"])


def feature_importance(model: Any) -> pd.DataFrame:
    """Extract native feature importance or coefficients from a fitted pipeline."""
    try:
        names = model.named_steps["preprocessor"].get_feature_names_out()
        estimator = model.named_steps["model"]
        if hasattr(estimator, "feature_importances_"):
            values = estimator.feature_importances_
        elif hasattr(estimator, "coef_"):
            values = np.abs(np.ravel(estimator.coef_))
        else:
            return pd.DataFrame(columns=["feature", "importance"])
        return (
            pd.DataFrame({"feature": names, "importance": values})
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )
    except Exception as exc:
        logger.info("Could not extract feature importance: %s", exc)
        return pd.DataFrame(columns=["feature", "importance"])


def shap_importance(model: Any, feature_frame: pd.DataFrame) -> pd.DataFrame:
    """Generate optional SHAP mean absolute importance when available."""
    try:
        import shap

        sample = feature_frame.head(min(len(feature_frame), 100))
        transformed = model.named_steps["preprocessor"].transform(sample)
        if hasattr(transformed, "toarray"):
            transformed = transformed.toarray()
        estimator = model.named_steps["model"]
        explainer = shap.Explainer(estimator, transformed)
        values = explainer(transformed)
        feature_names = model.named_steps["preprocessor"].get_feature_names_out()
        return (
            pd.DataFrame(
                {
                    "feature": feature_names,
                    "mean_abs_shap": np.abs(values.values).mean(axis=0),
                }
            )
            .sort_values("mean_abs_shap", ascending=False)
            .reset_index(drop=True)
        )
    except Exception as exc:
        logger.info("SHAP unavailable or unsupported for selected model: %s", exc)
        return pd.DataFrame(columns=["feature", "mean_abs_shap"])


def train_satisfaction_models(model_df: pd.DataFrame) -> RegressionResult:
    """Train and evaluate satisfaction regression models."""
    deps = require_sklearn()
    features = NUMERIC_SATISFACTION_FEATURES + CATEGORICAL_FEATURES
    X = model_df[features]
    y = model_df[config.SATISFACTION_COLUMN].astype(float)
    X_train, X_test, y_train, y_test = split_data(X, y, classification=False)
    preprocessor = build_preprocessor(NUMERIC_SATISFACTION_FEATURES, CATEGORICAL_FEATURES)
    models = {
        "Linear Regression": deps["LinearRegression"](),
        "Random Forest Regressor": deps["RandomForestRegressor"](
            n_estimators=250, random_state=42
        ),
        "Gradient Boosting Regressor": deps["GradientBoostingRegressor"](
            random_state=42
        ),
    }

    metric_rows = []
    fitted = {}
    for name, estimator in models.items():
        pipeline = deps["Pipeline"](
            [("preprocessor", preprocessor), ("model", estimator)]
        )
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)
        mse = deps["mean_squared_error"](y_test, predictions)
        metric_rows.append(
            {
                "task": "Satisfaction Prediction",
                "model": name,
                "rmse": float(np.sqrt(mse)),
                "mae": deps["mean_absolute_error"](y_test, predictions),
                "r2": deps["r2_score"](y_test, predictions),
            }
        )
        fitted[name] = pipeline

    metrics = pd.DataFrame(metric_rows)
    best_name = metrics.sort_values("rmse").iloc[0]["model"]
    best_model = fitted[str(best_name)]
    best_predictions = best_model.predict(X_test)
    prediction_frame = X_test.copy()
    prediction_frame["actual"] = y_test.values
    prediction_frame["predicted"] = best_predictions
    prediction_frame["residual"] = y_test.values - best_predictions

    return RegressionResult(
        task_name="Satisfaction Prediction",
        best_model_name=str(best_name),
        best_model=best_model,
        metrics=metrics,
        predictions=prediction_frame,
        feature_importance=feature_importance(best_model),
    )


def train_segment_model(model_df: pd.DataFrame) -> ClassificationResult:
    """Train a customer segment classification model."""
    deps = require_sklearn()
    features = NUMERIC_SEGMENT_FEATURES + CATEGORICAL_FEATURES
    X = model_df[features]
    y = model_df["segment"].astype(str)
    X_train, X_test, y_train, y_test = split_data(X, y)
    preprocessor = build_preprocessor(NUMERIC_SEGMENT_FEATURES, CATEGORICAL_FEATURES)
    models = {
        "Random Forest Segment Classifier": deps["RandomForestClassifier"](
            n_estimators=250, random_state=42, class_weight="balanced"
        ),
        "Decision Tree Segment Classifier": deps["DecisionTreeClassifier"](
            max_depth=5, random_state=42, class_weight="balanced"
        ),
    }
    positive_label = y.value_counts().index[0]
    return evaluate_classification_models(
        task_name="Customer Segment Prediction",
        models=models,
        preprocessor=preprocessor,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        feature_frame=X,
        positive_label=positive_label,
    )


def score_return_risk(model_df: pd.DataFrame, return_model: Any) -> pd.DataFrame:
    """Score all transactions for return probability and build risk table."""
    features = NUMERIC_RETURN_FEATURES + CATEGORICAL_FEATURES
    probabilities = get_positive_probabilities(return_model, model_df[features], 1)
    scored = model_df.copy()
    scored["predicted_return_probability"] = probabilities
    scored["predicted_return_label"] = (probabilities >= 0.5).astype(int)
    risk = scored[
        [
            config.TRANSACTION_ID_COLUMN,
            config.CUSTOMER_ID_COLUMN,
            "predicted_return_probability",
            "Return_Flag",
            config.REVENUE_COLUMN,
            config.SATISFACTION_COLUMN,
            "vader_compound",
        ]
    ].rename(
        columns={
            config.TRANSACTION_ID_COLUMN: "Transaction ID",
            config.CUSTOMER_ID_COLUMN: "Customer",
            "predicted_return_probability": "Predicted Return Probability",
            "Return_Flag": "Actual Return",
            config.REVENUE_COLUMN: "Revenue",
            config.SATISFACTION_COLUMN: "Satisfaction",
            "vader_compound": "Sentiment",
        }
    )
    return risk.sort_values("Predicted Return Probability", ascending=False)


def run_predictive_pipeline(df: pd.DataFrame) -> PredictiveResults:
    """Run the full Milestone 4 predictive analytics pipeline."""
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_df = prepare_modeling_data(df)
    return_result = train_return_models(model_df)
    satisfaction_result = train_satisfaction_models(model_df)
    segment_result = train_segment_model(model_df)
    high_risk = score_return_risk(model_df, return_result.best_model)
    exports = export_predictive_outputs(
        model_df, return_result, satisfaction_result, segment_result, high_risk
    )
    save_models(return_result, satisfaction_result, segment_result)
    return PredictiveResults(
        model_data=model_df,
        return_result=return_result,
        satisfaction_result=satisfaction_result,
        segment_result=segment_result,
        high_risk_transactions=high_risk,
        exports=exports,
    )


def export_predictive_outputs(
    model_df: pd.DataFrame,
    return_result: ClassificationResult,
    satisfaction_result: RegressionResult,
    segment_result: ClassificationResult,
    high_risk: pd.DataFrame,
) -> dict[str, str]:
    """Export Milestone 4 prediction outputs."""
    config.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    paths = {
        "predictions": config.PROCESSED_DATA_DIR / "predictions.csv",
        "feature_importance": config.PROCESSED_DATA_DIR / "feature_importance.csv",
        "model_metrics": config.PROCESSED_DATA_DIR / "model_metrics.csv",
        "high_risk_customers": config.PROCESSED_DATA_DIR / "high_risk_customers.csv",
    }
    scored = high_risk.merge(
        model_df[[config.TRANSACTION_ID_COLUMN, "segment"]].rename(
            columns={config.TRANSACTION_ID_COLUMN: "Transaction ID"}
        ),
        on="Transaction ID",
        how="left",
    )
    scored.to_csv(paths["predictions"], index=False)

    importance = pd.concat(
        [
            return_result.feature_importance.assign(task="Return Prediction"),
            satisfaction_result.feature_importance.assign(task="Satisfaction Prediction"),
            segment_result.feature_importance.assign(task="Segment Prediction"),
        ],
        ignore_index=True,
    )
    importance.to_csv(paths["feature_importance"], index=False)

    metrics = pd.concat(
        [
            return_result.metrics,
            satisfaction_result.metrics,
            segment_result.metrics,
        ],
        ignore_index=True,
        sort=False,
    )
    metrics.to_csv(paths["model_metrics"], index=False)
    high_risk.head(25).to_csv(paths["high_risk_customers"], index=False)
    return {key: str(value) for key, value in paths.items()}


def save_models(
    return_result: ClassificationResult,
    satisfaction_result: RegressionResult,
    segment_result: ClassificationResult,
) -> None:
    """Save trained model artifacts using joblib."""
    try:
        from joblib import dump

        dump(return_result.best_model, config.MODELS_DIR / "return_prediction_model.joblib")
        dump(
            satisfaction_result.best_model,
            config.MODELS_DIR / "satisfaction_prediction_model.joblib",
        )
        dump(segment_result.best_model, config.MODELS_DIR / "segment_prediction_model.joblib")
    except Exception as exc:
        logger.info("Could not save model artifacts: %s", exc)


def predict_what_if_return_probability(
    return_model: Any,
    revenue: float,
    satisfaction: float,
    purchase_frequency: float,
    sentiment: float,
    model_df: pd.DataFrame,
) -> float:
    """Predict return probability for interactive what-if inputs."""
    row = pd.DataFrame(
        [
            {
                config.REVENUE_COLUMN: revenue,
                config.SATISFACTION_COLUMN: satisfaction,
                "customer_purchase_frequency": purchase_frequency,
                "recency": float(model_df["recency"].median()),
                "monetary": float(model_df["monetary"].median()),
                "vader_compound": sentiment,
                "Month_Number": int(model_df["Month_Number"].median()),
                "Quarter_Number": int(model_df["Quarter_Number"].median()),
                "Revenue_Band": infer_revenue_band(revenue, model_df),
                "Quarter": str(model_df["Quarter"].mode().iloc[0]),
            }
        ]
    )
    return float(get_positive_probabilities(return_model, row, 1)[0])


def infer_revenue_band(revenue: float, model_df: pd.DataFrame) -> str:
    """Infer the revenue band for a what-if revenue value."""
    quantiles = model_df[config.REVENUE_COLUMN].quantile([0.25, 0.5, 0.75])
    labels = ["Low", "Mid", "High", "Premium"]
    if revenue <= quantiles.iloc[0]:
        return labels[0]
    if revenue <= quantiles.iloc[1]:
        return labels[1]
    if revenue <= quantiles.iloc[2]:
        return labels[2]
    return labels[3]


def predictive_rule_summary(results: PredictiveResults) -> dict[str, Any]:
    """Generate rule-based predictive recommendations."""
    return_metrics = results.return_result.metrics.sort_values(
        ["roc_auc", "f1", "accuracy"], ascending=False
    ).iloc[0]
    top_feature = (
        results.return_result.feature_importance.iloc[0]["feature"]
        if not results.return_result.feature_importance.empty
        else "available model features"
    )
    top_risk = results.high_risk_transactions.head(3)
    risk_customers = ", ".join(top_risk["Customer"].astype(str).tolist())
    return {
        "executive_overview": (
            f"The best return model is {results.return_result.best_model_name} "
            f"with F1 score {return_metrics['f1']:.2f}."
        ),
        "risk_customers": risk_customers or "Evidence is insufficient to identify high-risk customers.",
        "important_variables": str(top_feature),
        "business_risks": "High predicted return probability indicates transactions requiring operational review.",
        "recommendations": [
            "Prioritize outreach for transactions with the highest predicted return probability.",
            "Review the strongest predictive variables before changing fulfillment or service workflows.",
            "Monitor model performance as new Rialto transactions are added.",
        ],
    }


def generate_predictive_summary(results: PredictiveResults) -> tuple[dict[str, Any], str]:
    """Use Gemini for predictive summary, with rule-based fallback."""
    fallback = predictive_rule_summary(results)
    api_key = get_provider_api_key("GEMINI_API_KEY")
    if not api_key:
        return fallback, "Rule-based fallback"
    try:
        from google import genai

        payload = {
            "return_model_metrics": results.return_result.metrics.to_dict("records"),
            "satisfaction_model_metrics": results.satisfaction_result.metrics.to_dict("records"),
            "segment_model_metrics": results.segment_result.metrics.to_dict("records"),
            "top_features": results.return_result.feature_importance.head(10).to_dict("records"),
            "high_risk_transactions": results.high_risk_transactions.head(10).to_dict("records"),
        }
        prompt = (
            "You are a Senior Business Intelligence Consultant. Use only the supplied "
            "predictive analytics. Return JSON with executive_overview, risk_customers, "
            "important_variables, business_risks, and exactly three recommendations. "
            f"Analytics: {json.dumps(payload, default=str)}"
        )
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model=config.GEMINI_MODEL,
            contents=prompt,
        )
        parsed = parse_predictive_summary(response.text or "")
        return parsed, "AI-generated with Gemini"
    except Exception as exc:
        logger.exception("Gemini predictive summary failed; using fallback: %s", exc)
        return fallback, "Rule-based fallback"


def parse_predictive_summary(text: str) -> dict[str, Any]:
    """Parse Gemini predictive summary JSON."""
    match = None
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        import re

        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            raise
        parsed = json.loads(match.group(0))
    recommendations = parsed.get("recommendations", [])
    if not isinstance(recommendations, list):
        recommendations = [str(recommendations)]
    parsed["recommendations"] = [str(item) for item in recommendations[:3]]
    while len(parsed["recommendations"]) < 3:
        parsed["recommendations"].append(
            "Evidence is insufficient to provide an additional supported recommendation."
        )
    return parsed
