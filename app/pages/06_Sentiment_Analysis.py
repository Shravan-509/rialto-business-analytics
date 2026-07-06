"""Sentiment Analysis and NLP dashboard page for Milestone 3."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import sys

import plotly.express as px
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import config
from src.analytics import apply_filters
from src.data_pipeline import load_clean_export
from src.nlp_pipeline import (
    build_topic_model,
    build_word_frequency,
    build_wordcloud_image,
    export_nlp_outputs,
    generate_executive_summary,
    model_comparison,
    prepare_sentiment_dataset,
    top_ngrams,
)
from src.visualizations import (
    COLOR_BLUE,
    COLOR_RED,
    COLOR_TEAL,
    apply_chart_theme,
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
    """Load and clean source data with Streamlit caching."""
    return load_clean_export()


@st.cache_data(show_spinner=False)
def cached_sentiment_data(df):
    """Run and cache the sentiment pipeline."""
    return prepare_sentiment_dataset(df)


@st.cache_data(show_spinner=False)
def cached_topics(df):
    """Run and cache topic modeling."""
    return build_topic_model(df)


@st.cache_data(show_spinner=False)
def cached_word_frequency(df):
    """Build and cache word frequencies."""
    return build_word_frequency(df)


@st.cache_data(show_spinner=False)
def cached_wordcloud(df, sentiment):
    """Build and cache word cloud bytes."""
    image = build_wordcloud_image(df, sentiment)
    return None if image is None else image.getvalue()


@st.cache_data(show_spinner=False)
def cached_ngrams(df, ngram_range, top_n):
    """Build and cache n-gram frequency tables."""
    return top_ngrams(df, ngram_range, top_n=top_n)


@st.cache_data(show_spinner=False)
def cached_executive_summary(sentiment_df, topic_df, word_frequency_df):
    """Build and cache the executive summary."""
    return generate_executive_summary(sentiment_df, topic_df, word_frequency_df)


def sentiment_kpis(sentiment_df):
    """Calculate sentiment dashboard KPIs."""
    total_reviews = len(sentiment_df)
    counts = sentiment_df["vader_label"].value_counts()
    return {
        "total_reviews": total_reviews,
        "positive_reviews": int(counts.get("Positive", 0)),
        "neutral_reviews": int(counts.get("Neutral", 0)),
        "negative_reviews": int(counts.get("Negative", 0)),
        "average_sentiment_score": float(sentiment_df["vader_compound"].mean())
        if total_reviews
        else 0.0,
        "highest_confidence": float(sentiment_df["bert_confidence"].max())
        if total_reviews
        else 0.0,
        "lowest_confidence": float(sentiment_df["bert_confidence"].min())
        if total_reviews
        else 0.0,
    }


def render_wordclouds(sentiment_df) -> None:
    """Render overall, positive, and negative word clouds."""
    st.markdown("### Word Clouds")
    columns = st.columns(3)
    clouds = [
        ("Overall Word Cloud", None),
        ("Positive Reviews Word Cloud", "Positive"),
        ("Negative Reviews Word Cloud", "Negative"),
    ]
    for column, (title, sentiment) in zip(columns, clouds):
        column.markdown(f"**{title}**")
        image = cached_wordcloud(sentiment_df, sentiment)
        if image is None:
            column.info("Not enough review text to generate this word cloud.")
        else:
            column.image(image, width=280)


def render_executive_ai_brief(summary, summary_source) -> None:
    """Render the structured executive AI brief."""
    recommendations = summary.get("recommendations", [])
    recommendation_items = "".join(f"<li>{item}</li>" for item in recommendations)
    st.markdown("### GenAI Executive Summary")
    st.markdown(
        f"""
        <div class="summary-card">
            <strong>{summary_source}.</strong>
            <h4>Executive AI Brief</h4>
            <p><strong>Executive Overview.</strong> {summary.get("executive_overview", "")}</p>
            <p><strong>Positive Drivers.</strong> {summary.get("positive_drivers", "")}</p>
            <p><strong>Complaint Themes.</strong> {summary.get("complaint_themes", "")}</p>
            <p><strong>Business Risks.</strong> {summary.get("business_risks", "")}</p>
            <p><strong>Recommendations.</strong></p>
            <ul>{recommendation_items}</ul>
            <p><strong>Confidence.</strong> {summary.get("confidence_statement", "")}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_top_words(word_frequency_df) -> None:
    """Render top word bar charts."""
    st.markdown("### Top Words")
    charts = [
        ("Top 20 Positive Words", "Positive"),
        ("Top 20 Negative Words", "Negative"),
        ("Top 20 Most Frequent Words", "All"),
    ]
    for title, sentiment in charts:
        subset = word_frequency_df[word_frequency_df["sentiment"] == sentiment].head(20)
        fig = px.bar(
            subset,
            x="count",
            y="word",
            orientation="h",
            color_discrete_sequence=[COLOR_TEAL if sentiment != "Negative" else COLOR_RED],
        )
        fig.update_layout(yaxis=dict(autorange="reversed"))
        fig.update_traces(hovertemplate="Word: %{y}<br>Count: %{x:,}<extra></extra>")
        render_chart(apply_chart_theme(fig, title))
        render_insight(
            f"The leading {sentiment.lower()} terms show the most common language patterns in the selected reviews."
        )


def render_review_explorer(sentiment_df) -> None:
    """Render interactive review explorer table."""
    st.markdown("### Review Explorer")
    col1, col2, col3, col4 = st.columns(4)
    sentiments = sorted(sentiment_df["vader_label"].dropna().unique())
    customers = sorted(sentiment_df[config.CUSTOMER_ID_COLUMN].dropna().unique())
    bands = sorted(sentiment_df["Revenue_Band"].dropna().astype(str).unique())
    returns = sorted(sentiment_df[config.RETURN_COLUMN].dropna().astype(str).unique())

    selected_sentiments = col1.multiselect("Sentiment", sentiments, default=sentiments)
    selected_customers = col2.multiselect("Customer", customers, default=customers)
    selected_bands = col3.multiselect("Revenue Band", bands, default=bands)
    selected_returns = col4.multiselect("Return Status", returns, default=returns)

    explorer = sentiment_df[
        sentiment_df["vader_label"].isin(selected_sentiments)
        & sentiment_df[config.CUSTOMER_ID_COLUMN].isin(selected_customers)
        & sentiment_df["Revenue_Band"].astype(str).isin(selected_bands)
        & sentiment_df[config.RETURN_COLUMN].astype(str).isin(selected_returns)
    ]
    display_columns = [
        config.TRANSACTION_ID_COLUMN,
        config.CUSTOMER_ID_COLUMN,
        config.TEXT_COLUMN,
        "vader_label",
        "bert_confidence",
        config.SATISFACTION_COLUMN,
        config.RETURN_COLUMN,
    ]
    st.dataframe(
        explorer[display_columns].rename(
            columns={
                config.TRANSACTION_ID_COLUMN: "Transaction ID",
                config.CUSTOMER_ID_COLUMN: "Customer",
                config.TEXT_COLUMN: "Feedback",
                "vader_label": "Sentiment",
                "bert_confidence": "Confidence",
                config.SATISFACTION_COLUMN: "Satisfaction",
                config.RETURN_COLUMN: "Returned",
            }
        ),
        width="stretch",
    )


def main() -> None:
    """Render the Sentiment Analysis page."""
    st.set_page_config(page_title="Sentiment Analysis | Rialto", layout="wide")
    load_css()

    source_df = cached_clean_data()
    last_refresh = datetime.now()
    filters = render_sidebar(source_df, last_refresh)
    filtered_source = apply_filters(source_df, filters)

    render_page_header(
        "NLP Analytics",
        "Sentiment Analysis",
        "Customer review sentiment, topics, themes, emotions, and review exploration.",
    )

    if filtered_source.empty:
        st.warning("No records match the selected filters.")
        render_footer(last_refresh)
        st.stop()

    with st.spinner("Running NLP pipeline and sentiment models..."):
        sentiment_df, metadata = cached_sentiment_data(filtered_source)
        word_frequency_df = cached_word_frequency(sentiment_df)
        topic_df, topic_source = cached_topics(sentiment_df)
        exports = export_nlp_outputs(sentiment_df, topic_df, word_frequency_df)

    kpis = sentiment_kpis(sentiment_df)
    cards = [
        {
            "icon": "RV",
            "title": "Total Reviews",
            "value": format_metric(kpis["total_reviews"]),
            "description": "Customer feedback records analyzed.",
        },
        {
            "icon": "+",
            "title": "Positive Reviews",
            "value": format_metric(kpis["positive_reviews"]),
            "description": "Reviews classified as positive by VADER.",
        },
        {
            "icon": "=",
            "title": "Neutral Reviews",
            "value": format_metric(kpis["neutral_reviews"]),
            "description": "Reviews classified as neutral by VADER.",
        },
        {
            "icon": "-",
            "title": "Negative Reviews",
            "value": format_metric(kpis["negative_reviews"]),
            "description": "Reviews classified as negative by VADER.",
        },
        {
            "icon": "AVG",
            "title": "Avg Sentiment Score",
            "value": f"{kpis['average_sentiment_score']:.2f}",
            "description": "Average VADER compound score.",
        },
        {
            "icon": "HI",
            "title": "Highest Confidence",
            "value": format_percent(kpis["highest_confidence"]),
            "description": "Highest BERT sentiment confidence.",
        },
        {
            "icon": "LO",
            "title": "Lowest Confidence",
            "value": format_percent(kpis["lowest_confidence"]),
            "description": "Lowest BERT sentiment confidence.",
        },
    ]
    render_kpi_cards(cards)

    st.caption(
        f"VADER source: {metadata['vader_source']} | BERT source: {metadata['bert_source']}"
    )

    sentiment_counts = (
        sentiment_df["vader_label"].value_counts().rename_axis("sentiment").reset_index(name="reviews")
    )
    fig = px.bar(
        sentiment_counts,
        x="sentiment",
        y="reviews",
        color="sentiment",
        color_discrete_map={"Positive": COLOR_TEAL, "Neutral": COLOR_BLUE, "Negative": COLOR_RED},
    )
    fig.update_traces(hovertemplate="Sentiment: %{x}<br>Reviews: %{y:,}<extra></extra>")
    render_chart(apply_chart_theme(fig, "Sentiment Distribution"))
    render_insight("Sentiment distribution shows the overall balance of customer perception in the selected review set.")

    monthly_sentiment = (
        sentiment_df.groupby(["Month", "vader_label"], as_index=False)
        .size()
        .rename(columns={"size": "reviews"})
    )
    fig = px.bar(monthly_sentiment, x="Month", y="reviews", color="vader_label", barmode="stack")
    fig.update_traces(hovertemplate="Month: %{x}<br>Reviews: %{y:,}<extra></extra>")
    render_chart(apply_chart_theme(fig, "Monthly Sentiment Trend"))
    render_insight("Monthly sentiment trend highlights how positive, neutral, and negative reviews shift over time.")

    band_sentiment = (
        sentiment_df.groupby(["Revenue_Band", "vader_label"], observed=False, as_index=False)
        .size()
        .rename(columns={"size": "reviews"})
    )
    fig = px.bar(band_sentiment, x="Revenue_Band", y="reviews", color="vader_label", barmode="group")
    render_chart(apply_chart_theme(fig, "Sentiment by Revenue Band"))
    render_insight("Revenue band sentiment shows whether customer perception differs across transaction value tiers.")

    return_sentiment = (
        sentiment_df.groupby([config.RETURN_COLUMN, "vader_label"], as_index=False)
        .size()
        .rename(columns={"size": "reviews"})
    )
    fig = px.bar(return_sentiment, x=config.RETURN_COLUMN, y="reviews", color="vader_label", barmode="group")
    render_chart(apply_chart_theme(fig, "Sentiment by Return Status"))
    render_insight("Sentiment by return status helps identify whether negative reviews concentrate among returned orders.")

    satisfaction_sentiment = (
        sentiment_df.groupby([config.SATISFACTION_COLUMN, "vader_label"], as_index=False)
        .size()
        .rename(columns={"size": "reviews"})
    )
    fig = px.bar(satisfaction_sentiment, x=config.SATISFACTION_COLUMN, y="reviews", color="vader_label", barmode="group")
    render_chart(apply_chart_theme(fig, "Sentiment by Satisfaction Score"))
    render_insight("Sentiment by satisfaction score checks whether text sentiment aligns with numeric customer ratings.")

    monthly_avg = (
        sentiment_df.groupby("Month", as_index=False)["vader_compound"]
        .mean()
        .rename(columns={"vader_compound": "average_sentiment"})
    )
    fig = px.line(monthly_avg, x="Month", y="average_sentiment", markers=True, color_discrete_sequence=[COLOR_TEAL])
    fig.update_traces(hovertemplate="Month: %{x}<br>Avg sentiment: %{y:.2f}<extra></extra>")
    render_chart(apply_chart_theme(fig, "Average Sentiment by Month"))
    render_insight("Average sentiment by month summarizes review tone as a continuous trend.")

    heatmap_data = (
        sentiment_df.groupby(["Revenue_Band", "Month"], observed=False)["vader_compound"]
        .mean()
        .reset_index()
        .pivot(index="Revenue_Band", columns="Month", values="vader_compound")
    )
    fig = px.imshow(
        heatmap_data,
        color_continuous_scale="Teal",
        aspect="auto",
        labels=dict(color="Avg Sentiment"),
    )
    fig.update_traces(hovertemplate="Revenue band: %{y}<br>Month: %{x}<br>Avg sentiment: %{z:.2f}<extra></extra>")
    render_chart(apply_chart_theme(fig, "Sentiment Heatmap"))
    render_insight("The heatmap identifies month and revenue-band combinations with stronger or weaker sentiment.")

    fig = px.histogram(sentiment_df, x="bert_confidence", nbins=20, color_discrete_sequence=[COLOR_BLUE])
    fig.update_traces(hovertemplate="Confidence: %{x:.2f}<br>Reviews: %{y:,}<extra></extra>")
    render_chart(apply_chart_theme(fig, "Confidence Distribution"))
    render_insight("Confidence distribution shows how strongly the BERT or fallback classifier assigned review sentiment labels.")

    render_wordclouds(sentiment_df)
    render_top_words(word_frequency_df)

    st.markdown("### N-Gram Analysis")
    for title, ngram_range in [("Top Bigrams", (2, 2)), ("Top Trigrams", (3, 3))]:
        ngram_df = cached_ngrams(sentiment_df, ngram_range, top_n=15)
        fig = px.bar(ngram_df, x="count", y="term", orientation="h", color_discrete_sequence=[COLOR_TEAL])
        fig.update_layout(yaxis=dict(autorange="reversed"))
        fig.update_traces(hovertemplate="Phrase: %{y}<br>Count: %{x:,}<extra></extra>")
        render_chart(apply_chart_theme(fig, title))
        render_insight(f"{title.lower()} reveal repeated customer language patterns beyond individual words.")

    st.markdown(f"### Topic Modeling ({topic_source})")
    st.dataframe(topic_df, width="stretch")
    render_insight("Topic modeling groups reviews into recurring themes using BERTopic when available, otherwise LDA.")

    emotion_counts = (
        sentiment_df["emotion"].value_counts().rename_axis("emotion").reset_index(name="reviews")
    )
    fig = px.bar(emotion_counts, x="emotion", y="reviews", color="emotion")
    render_chart(apply_chart_theme(fig, "Emotion Distribution"))
    render_insight("Emotion distribution provides a lightweight view of the dominant emotional signals in feedback.")

    emotion_trend = (
        sentiment_df.groupby(["Month", "emotion"], as_index=False)
        .size()
        .rename(columns={"size": "reviews"})
    )
    fig = px.line(emotion_trend, x="Month", y="reviews", color="emotion", markers=True)
    render_chart(apply_chart_theme(fig, "Emotion Trend"))
    render_insight("Emotion trend shows how review emotions vary across time.")

    render_review_explorer(sentiment_df)

    summary, summary_source = cached_executive_summary(
        sentiment_df, topic_df, word_frequency_df
    )
    render_executive_ai_brief(summary, summary_source)

    st.markdown("### Sentiment Model Comparison")
    st.dataframe(
        model_comparison(str(metadata["bert_source"]), float(metadata["bert_seconds"])),
        width="stretch",
    )
    render_insight("Recommendation: use VADER as the primary dashboard baseline, BERT when available for contextual comparison, and the rule-based method as a transparent fallback.")

    st.markdown("### Exported NLP Files")
    st.dataframe(
        [{"Export": key, "Path": value} for key, value in exports.items()],
        width="stretch",
    )

    render_executive_insights(
        [
            f"Total analyzed reviews: {format_metric(kpis['total_reviews'])}.",
            f"Positive review share is {format_percent(kpis['positive_reviews'] / kpis['total_reviews']) if kpis['total_reviews'] else '0.0%'}.",
            f"Negative review share is {format_percent(kpis['negative_reviews'] / kpis['total_reviews']) if kpis['total_reviews'] else '0.0%'}.",
            "Review themes, sentiment by returns, and low-confidence classifications should be monitored before operational decisions.",
        ]
    )
    render_footer(last_refresh)


main()
