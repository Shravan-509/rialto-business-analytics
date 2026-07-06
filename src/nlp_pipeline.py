"""NLP, sentiment analysis, topic modeling, and review insight utilities."""

from __future__ import annotations

from collections import Counter
from io import BytesIO
import json
import logging
import os
import re
import string
import time
from functools import lru_cache
from typing import Iterable

import numpy as np
import pandas as pd

from src import config

logger = logging.getLogger(__name__)

DEFAULT_STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "but",
    "by",
    "for",
    "from",
    "has",
    "have",
    "in",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "was",
    "were",
    "with",
    "product",
    "item",
    "order",
}

POSITIVE_WORDS = {
    "amazing",
    "excellent",
    "fast",
    "good",
    "great",
    "happy",
    "love",
    "perfect",
    "quality",
    "recommend",
    "satisfied",
    "smooth",
}

NEGATIVE_WORDS = {
    "bad",
    "damaged",
    "delay",
    "disappointed",
    "late",
    "mislabelled",
    "poor",
    "problem",
    "return",
    "slow",
    "wrong",
}

EMOTION_LEXICON = {
    "Joy": {"amazing", "excellent", "fast", "good", "great", "happy", "love", "perfect", "satisfied"},
    "Anger": {"bad", "damaged", "wrong", "poor", "problem", "return"},
    "Sadness": {"disappointed", "poor", "bad", "issue"},
    "Fear": {"concern", "risk", "problem", "delay"},
    "Surprise": {"unexpected", "surprise", "wow", "sudden"},
}


def ensure_nltk_resource(resource: str, download_name: str) -> bool:
    """Ensure an NLTK resource exists, downloading it when possible."""
    try:
        import nltk

        nltk.data.find(resource)
        return True
    except Exception:
        try:
            import nltk

            nltk.download(download_name, quiet=True)
            nltk.data.find(resource)
            logger.info("Downloaded missing NLTK resource: %s", download_name)
            return True
        except Exception as exc:
            logger.info(
                "NLTK resource %s unavailable; using local fallback where possible: %s",
                download_name,
                exc,
            )
            return False


def ensure_core_nltk_resources() -> None:
    """Attempt to download the core NLTK resources used by the NLP module."""
    resources = [
        ("tokenizers/punkt", "punkt"),
        ("corpora/stopwords", "stopwords"),
        ("corpora/wordnet", "wordnet"),
        ("corpora/omw-1.4", "omw-1.4"),
    ]
    for resource, download_name in resources:
        ensure_nltk_resource(resource, download_name)


@lru_cache(maxsize=1)
def get_stop_words() -> set[str]:
    """Return stop words from NLTK when available, otherwise use fallback words."""
    if ensure_nltk_resource("corpora/stopwords", "stopwords"):
        try:
            from nltk.corpus import stopwords

            return set(stopwords.words("english")).union(DEFAULT_STOP_WORDS)
        except Exception as exc:
            logger.info("Could not load NLTK stopwords; using fallback set: %s", exc)
    return DEFAULT_STOP_WORDS


@lru_cache(maxsize=1)
def get_lemmatizer():
    """Return an NLTK lemmatizer when available."""
    if ensure_nltk_resource("corpora/wordnet", "wordnet"):
        ensure_nltk_resource("corpora/omw-1.4", "omw-1.4")
        try:
            from nltk.stem import WordNetLemmatizer

            lemmatizer = WordNetLemmatizer()
            lemmatizer.lemmatize("tests")
            return lemmatizer
        except Exception as exc:
            logger.info("Could not load lemmatizer; continuing without it: %s", exc)
    return None


def tokenize_text(text: str) -> list[str]:
    """Lowercase, clean, tokenize, remove stop words, and lemmatize review text."""
    lowered = str(text).lower()
    without_punctuation = lowered.translate(str.maketrans("", "", string.punctuation))
    without_numbers = re.sub(r"\d+", " ", without_punctuation)
    normalized = re.sub(r"\s+", " ", without_numbers).strip()
    tokens = normalized.split()
    stop_words = get_stop_words()
    lemmatizer = get_lemmatizer()
    cleaned_tokens = []

    for token in tokens:
        if token in stop_words or len(token) <= 1:
            continue
        if lemmatizer is not None:
            try:
                token = lemmatizer.lemmatize(token)
            except LookupError as exc:
                logger.info(
                    "WordNet lookup failed; continuing without lemmatization: %s",
                    exc,
                )
                get_lemmatizer.cache_clear()
                lemmatizer = None
        cleaned_tokens.append(token)
    return cleaned_tokens


def tokens_to_text(tokens: Iterable[str]) -> str:
    """Convert token iterables into a cache-friendly string."""
    return " ".join(str(token) for token in tokens)


def token_values(tokens: str | Iterable[str]) -> list[str]:
    """Return token values from either a cache-friendly string or an iterable."""
    if isinstance(tokens, str):
        return tokens.split()
    return list(tokens)


def preprocess_reviews(df: pd.DataFrame) -> pd.DataFrame:
    """Create tokenized and cleaned review text columns."""
    processed = df.copy()
    ensure_core_nltk_resources()
    processed["review_text"] = processed[config.TEXT_COLUMN].fillna(
        "No feedback provided"
    )
    processed["tokens"] = processed["review_text"].apply(
        lambda text: tokens_to_text(tokenize_text(text))
    )
    processed["cleaned_review"] = processed["tokens"]
    return processed


def classify_sentiment(score: float) -> str:
    """Classify a compound sentiment score."""
    if score >= 0.05:
        return "Positive"
    if score <= -0.05:
        return "Negative"
    return "Neutral"


def rule_based_score(text: str) -> dict[str, float | str]:
    """Score sentiment with a transparent fallback lexicon."""
    tokens = tokenize_text(text)
    if not tokens:
        compound = 0.0
    else:
        positive_hits = sum(token in POSITIVE_WORDS for token in tokens)
        negative_hits = sum(token in NEGATIVE_WORDS for token in tokens)
        compound = (positive_hits - negative_hits) / max(len(tokens), 1)
    positive = max(compound, 0.0)
    negative = abs(min(compound, 0.0))
    neutral = max(0.0, 1.0 - positive - negative)
    return {
        "rule_compound": compound,
        "rule_positive": positive,
        "rule_neutral": neutral,
        "rule_negative": negative,
        "rule_label": classify_sentiment(compound),
    }


def run_vader_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    """Run VADER sentiment analysis with a rule-based fallback."""
    result = df.copy()
    analyzer = None
    source = "Fallback rule-based sentiment"

    if ensure_nltk_resource("sentiment/vader_lexicon.zip", "vader_lexicon"):
        try:
            from nltk.sentiment import SentimentIntensityAnalyzer

            analyzer = SentimentIntensityAnalyzer()
            source = "NLTK VADER"
        except Exception as exc:
            logger.info("VADER unavailable; using rule-based fallback: %s", exc)

    if analyzer is not None:
        scores = result["review_text"].apply(lambda text: analyzer.polarity_scores(str(text)))
        result["vader_compound"] = scores.apply(lambda item: item["compound"])
        result["vader_positive"] = scores.apply(lambda item: item["pos"])
        result["vader_neutral"] = scores.apply(lambda item: item["neu"])
        result["vader_negative"] = scores.apply(lambda item: item["neg"])
        result["vader_label"] = result["vader_compound"].apply(classify_sentiment)
    else:
        fallback = pd.DataFrame(result["review_text"].apply(rule_based_score).tolist())
        result["vader_compound"] = fallback["rule_compound"]
        result["vader_positive"] = fallback["rule_positive"]
        result["vader_neutral"] = fallback["rule_neutral"]
        result["vader_negative"] = fallback["rule_negative"]
        result["vader_label"] = fallback["rule_label"]

    result["vader_source"] = source
    return result


@lru_cache(maxsize=1)
def get_bert_pipeline():
    """Load and cache a Hugging Face sentiment pipeline.

    Transformers will automatically use the local model cache or download the
    model when network access is available.
    """
    try:
        from transformers import pipeline

        return pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
            truncation=True,
        )
    except Exception as exc:
        logger.info("BERT sentiment pipeline unavailable; using fallback: %s", exc)
        return None


def run_bert_sentiment(df: pd.DataFrame) -> tuple[pd.DataFrame, str, float]:
    """Run BERT sentiment or provide a deterministic fallback."""
    result = df.copy()
    start = time.perf_counter()
    bert_pipeline = get_bert_pipeline()

    if bert_pipeline is None:
        result["bert_label"] = result["vader_label"]
        result["bert_confidence"] = result["vader_compound"].abs().clip(0.5, 0.99)
        result["bert_source"] = "Fallback from VADER because BERT was unavailable"
        return result, "Fallback from VADER because BERT was unavailable", time.perf_counter() - start

    try:
        predictions = bert_pipeline(result["review_text"].astype(str).tolist())
        result["bert_label"] = [
            "Positive" if item["label"].upper() == "POSITIVE" else "Negative"
            for item in predictions
        ]
        result["bert_confidence"] = [float(item["score"]) for item in predictions]
        result["bert_source"] = "Hugging Face DistilBERT SST-2"
        return result, "Hugging Face DistilBERT SST-2", time.perf_counter() - start
    except Exception as exc:
        logger.info("BERT prediction failed; using fallback labels: %s", exc)
        result["bert_label"] = result["vader_label"]
        result["bert_confidence"] = result["vader_compound"].abs().clip(0.5, 0.99)
        result["bert_source"] = "Fallback from VADER because BERT prediction failed"
        return result, "Fallback from VADER because BERT prediction failed", time.perf_counter() - start


def classify_emotion(tokens: Iterable[str]) -> str:
    """Classify review emotion with a compact transparent lexicon."""
    token_set = set(tokens)
    emotion_scores = {
        emotion: len(token_set.intersection(words))
        for emotion, words in EMOTION_LEXICON.items()
    }
    if max(emotion_scores.values(), default=0) == 0:
        return "Neutral"
    return max(emotion_scores, key=emotion_scores.get)


def add_emotion_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """Add rule-based emotion labels."""
    result = df.copy()
    result["emotion"] = result["tokens"].apply(lambda tokens: classify_emotion(token_values(tokens)))
    return result


def build_word_frequency(df: pd.DataFrame) -> pd.DataFrame:
    """Build word frequencies overall and by sentiment."""
    rows = []
    for sentiment in ["All", "Positive", "Neutral", "Negative"]:
        if sentiment == "All":
            subset = df
        else:
            subset = df[df["vader_label"] == sentiment]
        counter = Counter(
            token for tokens in subset["tokens"] for token in token_values(tokens)
        )
        rows.extend(
            {"sentiment": sentiment, "word": word, "count": count}
            for word, count in counter.items()
        )
    return pd.DataFrame(rows).sort_values(["sentiment", "count"], ascending=[True, False])


def top_ngrams(df: pd.DataFrame, ngram_range: tuple[int, int], top_n: int = 15) -> pd.DataFrame:
    """Return top n-grams from cleaned review text."""
    try:
        from sklearn.feature_extraction.text import CountVectorizer

        vectorizer = CountVectorizer(ngram_range=ngram_range, min_df=1)
        matrix = vectorizer.fit_transform(df["cleaned_review"])
        counts = np.asarray(matrix.sum(axis=0)).ravel()
        return (
            pd.DataFrame({"term": vectorizer.get_feature_names_out(), "count": counts})
            .sort_values("count", ascending=False)
            .head(top_n)
        )
    except Exception as exc:
        logger.info("N-gram analysis unavailable; returning empty result: %s", exc)
        return pd.DataFrame(columns=["term", "count"])


def build_topic_model(df: pd.DataFrame, min_topics: int = 5, max_topics: int = 8) -> tuple[pd.DataFrame, str]:
    """Build topic summary with BERTopic when available, otherwise LDA."""
    texts = df["cleaned_review"].replace("", np.nan).dropna().tolist()
    if not texts:
        return pd.DataFrame(columns=["topic_name", "top_keywords", "review_count", "example_reviews"]), "No topic model"

    topic_count = int(min(max_topics, max(min_topics, max(2, len(texts) // 8))))

    try:
        from bertopic import BERTopic

        model = BERTopic(nr_topics=topic_count, verbose=False)
        topics, _ = model.fit_transform(texts)
        info = model.get_topic_info()
        rows = []
        for _, topic_row in info[info["Topic"] != -1].head(topic_count).iterrows():
            topic_id = int(topic_row["Topic"])
            keywords = [word for word, _ in model.get_topic(topic_id)[:8]]
            examples = [texts[i] for i, topic in enumerate(topics) if topic == topic_id][:3]
            rows.append(
                {
                    "topic_name": f"Topic {topic_id}",
                    "top_keywords": ", ".join(keywords),
                    "review_count": int(topic_row["Count"]),
                    "example_reviews": " | ".join(examples),
                }
            )
        return pd.DataFrame(rows), "BERTopic"
    except Exception as exc:
        logger.info("BERTopic unavailable; trying LDA fallback: %s", exc)

    try:
        from sklearn.decomposition import LatentDirichletAllocation
        from sklearn.feature_extraction.text import CountVectorizer

        vectorizer = CountVectorizer(max_features=500, min_df=1)
        matrix = vectorizer.fit_transform(texts)
        topic_count = min(topic_count, max(2, matrix.shape[0]))
        model = LatentDirichletAllocation(n_components=topic_count, random_state=42)
        doc_topics = model.fit_transform(matrix)
        assigned_topics = doc_topics.argmax(axis=1)
        terms = vectorizer.get_feature_names_out()
        rows = []
        for topic_idx, topic_weights in enumerate(model.components_):
            top_terms = [terms[i] for i in topic_weights.argsort()[-8:][::-1]]
            examples = [texts[i] for i, topic in enumerate(assigned_topics) if topic == topic_idx][:3]
            rows.append(
                {
                    "topic_name": f"Topic {topic_idx + 1}",
                    "top_keywords": ", ".join(top_terms),
                    "review_count": int((assigned_topics == topic_idx).sum()),
                    "example_reviews": " | ".join(examples),
                }
            )
        return pd.DataFrame(rows), "LDA"
    except Exception as exc:
        logger.info("LDA topic model unavailable; using keyword fallback: %s", exc)
        return build_keyword_topics(df, topic_count), "Keyword topic fallback"


def build_keyword_topics(df: pd.DataFrame, topic_count: int) -> pd.DataFrame:
    """Build a lightweight topic table when advanced topic models are unavailable."""
    counter = Counter(token for tokens in df["tokens"] for token in token_values(tokens))
    common_words = [word for word, _ in counter.most_common(max(topic_count * 6, 1))]
    if not common_words:
        return pd.DataFrame(
            columns=["topic_name", "top_keywords", "review_count", "example_reviews"]
        )

    topics = []
    for index in range(topic_count):
        keywords = common_words[index * 6 : (index + 1) * 6]
        if not keywords:
            continue
        keyword_set = set(keywords)
        matching = df[
            df["tokens"].apply(
                lambda tokens: bool(keyword_set.intersection(token_values(tokens)))
            )
        ]
        topics.append(
            {
                "topic_name": f"Topic {index + 1}",
                "top_keywords": ", ".join(keywords),
                "review_count": int(len(matching)),
                "example_reviews": " | ".join(
                    matching["review_text"].astype(str).head(3).tolist()
                ),
            }
        )
    return pd.DataFrame(topics)


def build_wordcloud_image(df: pd.DataFrame, sentiment: str | None = None) -> BytesIO | None:
    """Generate a word cloud image buffer."""
    subset = df if sentiment is None else df[df["vader_label"] == sentiment]
    text = " ".join(subset["cleaned_review"].dropna().tolist()).strip()
    if not text:
        return None
    try:
        from wordcloud import WordCloud

        image = WordCloud(
            width=900,
            height=420,
            background_color="white",
            colormap="viridis",
            max_words=80,
        ).generate(text)
        buffer = BytesIO()
        image.to_image().save(buffer, format="PNG")
        buffer.seek(0)
        return buffer
    except Exception as exc:
        logger.info("Word cloud unavailable; skipping word cloud image: %s", exc)
        return None


def prepare_sentiment_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str | float]]:
    """Run the full NLP pipeline and return results plus metadata."""
    processed = preprocess_reviews(df)
    processed = run_vader_sentiment(processed)
    processed, bert_source, bert_seconds = run_bert_sentiment(processed)
    processed = add_emotion_analysis(processed)
    processed["sentiment_agreement"] = processed["vader_label"] == processed["bert_label"]
    metadata = {
        "bert_source": bert_source,
        "bert_seconds": bert_seconds,
        "vader_source": str(processed["vader_source"].iloc[0]) if len(processed) else "N/A",
    }
    return processed, metadata


def export_nlp_outputs(
    sentiment_df: pd.DataFrame,
    topic_df: pd.DataFrame,
    word_frequency_df: pd.DataFrame,
) -> dict[str, str]:
    """Export Milestone 3 processed NLP datasets."""
    config.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    paths = {
        "sentiment_results": config.PROCESSED_DATA_DIR / "sentiment_results.csv",
        "topic_summary": config.PROCESSED_DATA_DIR / "topic_summary.csv",
        "word_frequency": config.PROCESSED_DATA_DIR / "word_frequency.csv",
        "bert_predictions": config.PROCESSED_DATA_DIR / "bert_predictions.csv",
    }
    sentiment_df.to_csv(paths["sentiment_results"], index=False)
    topic_df.to_csv(paths["topic_summary"], index=False)
    word_frequency_df.to_csv(paths["word_frequency"], index=False)
    sentiment_df[
        [
            config.TRANSACTION_ID_COLUMN,
            config.CUSTOMER_ID_COLUMN,
            "review_text",
            "bert_label",
            "bert_confidence",
            "bert_source",
        ]
    ].to_csv(paths["bert_predictions"], index=False)
    return {key: str(path) for key, path in paths.items()}


def generate_rule_based_summary(
    sentiment_df: pd.DataFrame,
    topic_df: pd.DataFrame,
    word_frequency_df: pd.DataFrame,
) -> dict[str, str | list[str]]:
    """Generate a structured executive summary from calculated NLP outputs."""
    total_reviews = len(sentiment_df)
    sentiment_counts = sentiment_df["vader_label"].value_counts()
    dominant_sentiment = sentiment_counts.idxmax() if not sentiment_counts.empty else "Neutral"
    negative_rate = sentiment_counts.get("Negative", 0) / total_reviews if total_reviews else 0
    positive_terms = ", ".join(
        word_frequency_df[word_frequency_df["sentiment"] == "Positive"]
        .head(5)["word"]
        .tolist()
    )
    negative_terms = ", ".join(
        word_frequency_df[word_frequency_df["sentiment"] == "Negative"]
        .head(5)["word"]
        .tolist()
    )
    top_topic = topic_df.iloc[0]["top_keywords"] if not topic_df.empty else "not enough text for topics"
    risk = (
        "Returned orders and low satisfaction reviews should be reviewed closely."
        if negative_rate > 0
        else "Current review sentiment shows limited negative language in the selected data."
    )
    return {
        "executive_overview": (
            f"Overall customer perception is mostly {dominant_sentiment.lower()} "
            f"across {total_reviews:,} reviews."
        ),
        "positive_drivers": positive_terms
        or "Evidence is insufficient to identify strong positive drivers.",
        "complaint_themes": negative_terms
        or "Evidence is insufficient to identify recurring negative themes.",
        "business_risks": risk,
        "recommendations": [
            "Monitor negative reviews by return status.",
            "Protect the positive experience drivers visible in customer feedback.",
            "Review recurring complaint terms in operational meetings.",
        ],
        "confidence_statement": (
            "This summary is based only on the supplied sentiment, topic, and word-frequency analytics."
        ),
    }


def generate_executive_summary(
    sentiment_df: pd.DataFrame,
    topic_df: pd.DataFrame,
    word_frequency_df: pd.DataFrame,
) -> tuple[dict[str, str | list[str]], str]:
    """Generate a provider-based executive summary with rule-based fallback."""
    fallback = generate_rule_based_summary(sentiment_df, topic_df, word_frequency_df)
    provider = get_llm_provider()

    if provider == "openai":
        return generate_openai_summary(
            sentiment_df, topic_df, word_frequency_df, fallback
        )
    if provider == "gemini":
        return generate_gemini_summary(
            sentiment_df, topic_df, word_frequency_df, fallback
        )

    logger.info("Unsupported LLM_PROVIDER=%s; using rule-based fallback.", provider)
    return fallback, "Rule-based fallback"


def generate_openai_summary(
    sentiment_df: pd.DataFrame,
    topic_df: pd.DataFrame,
    word_frequency_df: pd.DataFrame,
    fallback: dict[str, str | list[str]],
) -> tuple[dict[str, str | list[str]], str]:
    """Generate an executive summary using OpenAI."""
    api_key = get_provider_api_key("OPENAI_API_KEY")
    if not api_key:
        logger.info("OPENAI_API_KEY missing; using rule-based fallback.")
        return fallback, "Rule-based fallback"

    try:
        from openai import OpenAI, RateLimitError
    except Exception as exc:
        logger.info("OpenAI package unavailable; using rule-based fallback: %s", exc)
        return fallback, "Rule-based fallback"

    try:
        client = OpenAI(api_key=api_key)
        prompt = build_summary_prompt_from_data(
            sentiment_df, topic_df, word_frequency_df
        )
        response = client.responses.create(
            model=config.OPENAI_MODEL,
            input=prompt,
            max_output_tokens=config.MAX_OUTPUT_TOKENS,
        )
        return parse_llm_summary(response.output_text), "AI-generated with OpenAI"
    except RateLimitError:
        logger.warning("OpenAI quota exceeded. Using rule-based fallback.")
        return fallback, "Rule-based fallback"

    except Exception as exc:
        logger.exception("OpenAI executive summary failed; using rule-based fallback: %s", exc)
        return fallback, "Rule-based fallback"


def generate_gemini_summary(
    sentiment_df: pd.DataFrame,
    topic_df: pd.DataFrame,
    word_frequency_df: pd.DataFrame,
    fallback: dict[str, str | list[str]],
) -> tuple[dict[str, str | list[str]], str]:
    """Generate an executive summary using Gemini."""
    api_key = get_provider_api_key("GEMINI_API_KEY")
    if not api_key:
        logger.info("GEMINI_API_KEY missing; using rule-based fallback.")
        return fallback, "Rule-based fallback"

    try:
        from google import genai

        client = genai.Client(api_key=api_key)
        prompt = build_summary_prompt_from_data(
            sentiment_df, topic_df, word_frequency_df
        )
        response = client.models.generate_content(
            model=config.GEMINI_MODEL,
            contents=prompt,
        )
        return parse_llm_summary(response.text or ""), "AI-generated with Gemini"
    except Exception as exc:
        logger.exception("Gemini executive summary failed; using rule-based fallback: %s", exc)
        return fallback, "Rule-based fallback"


def build_summary_prompt_from_data(
    sentiment_df: pd.DataFrame,
    topic_df: pd.DataFrame,
    word_frequency_df: pd.DataFrame,
) -> str:
    """Build the structured executive-summary prompt from calculated analytics."""
    sentiment_counts = sentiment_df["vader_label"].value_counts().to_dict()
    topic_records = topic_df.head(5).to_dict(orient="records")
    positive_words = (
        word_frequency_df[word_frequency_df["sentiment"] == "Positive"]
        .head(10)
        .to_dict(orient="records")
    )
    negative_words = (
        word_frequency_df[word_frequency_df["sentiment"] == "Negative"]
        .head(10)
        .to_dict(orient="records")
    )
    return build_summary_prompt(
        sentiment_counts=sentiment_counts,
        topic_records=topic_records,
        positive_words=positive_words,
        negative_words=negative_words,
    )


def build_summary_prompt(
    sentiment_counts: dict[str, int],
    topic_records: list[dict],
    positive_words: list[dict],
    negative_words: list[dict],
) -> str:
    """Build the structured Senior BI Consultant prompt for LLM providers."""
    analytics_payload = {
        "sentiment_counts": sentiment_counts,
        "topics": topic_records,
        "positive_words": positive_words,
        "negative_words": negative_words,
    }
    return f"""
You are a Senior Business Intelligence Consultant.

You have been provided with aggregated business analytics generated from an e-commerce company.

Important Rules

• Use ONLY the analytics provided.
• Never invent numbers.
• Never make assumptions that are not supported by the data.
• If evidence is insufficient, explicitly state that.

Write a concise executive report suitable for the CEO.

The report should contain the following sections.

1. Executive Overview
Summarize the overall customer sentiment.

2. Positive Drivers
Identify what customers appreciate most.

3. Primary Complaint Themes
Summarize the most common negative issues.

4. Business Risks
Explain the biggest operational or customer experience risks.

5. Strategic Recommendations
Provide exactly three actionable recommendations.

6. Confidence Statement
Briefly explain that the recommendations are based only on the supplied analytics.

Limit the response to approximately 180 words.

Professional business tone.

Do not use markdown tables.

Do not fabricate insights.

Return only valid JSON with this schema:
{{
  "executive_overview": "...",
  "positive_drivers": "...",
  "complaint_themes": "...",
  "business_risks": "...",
  "recommendations": [
    "...",
    "...",
    "..."
  ],
  "confidence_statement": "..."
}}

Aggregated analytics:
{json.dumps(analytics_payload, default=str)}
""".strip()


def parse_llm_summary(output_text: str) -> dict[str, str | list[str]]:
    """Parse and validate the structured LLM executive summary."""
    try:
        parsed = json.loads(output_text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", output_text, re.DOTALL)
        if not match:
            raise
        parsed = json.loads(match.group(0))

    required_fields = {
        "executive_overview",
        "positive_drivers",
        "complaint_themes",
        "business_risks",
        "recommendations",
        "confidence_statement",
    }
    missing = required_fields.difference(parsed)
    if missing:
        raise ValueError(f"OpenAI summary missing required fields: {sorted(missing)}")

    recommendations = parsed.get("recommendations", [])
    if not isinstance(recommendations, list):
        parsed["recommendations"] = [str(recommendations)]
    parsed["recommendations"] = [str(item) for item in parsed["recommendations"][:3]]
    while len(parsed["recommendations"]) < 3:
        parsed["recommendations"].append(
            "Evidence is insufficient to provide an additional supported recommendation."
        )
    return parsed


def get_llm_provider() -> str:
    """Read selected LLM provider from environment or Streamlit secrets."""
    provider = os.getenv("LLM_PROVIDER")
    if provider:
        return provider.strip().lower()

    secret_provider = get_streamlit_secret("LLM_PROVIDER")
    return (secret_provider or config.LLM_PROVIDER or "openai").strip().lower()


def get_provider_api_key(key_name: str) -> str | None:
    """Read a provider API key from environment or Streamlit secrets."""
    api_key = os.getenv(key_name)
    if api_key:
        return api_key
    return get_streamlit_secret(key_name)


def get_streamlit_secret(key_name: str) -> str | None:
    """Read a value from Streamlit secrets when available."""
    try:
        import streamlit as st

        secret_value = st.secrets.get(key_name)
        if secret_value:
            return str(secret_value)
    except Exception as exc:
        logger.info("Streamlit secrets unavailable for %s: %s", key_name, exc)

    return None


def get_openai_api_key() -> str | None:
    """Backward-compatible OpenAI key helper."""
    return get_provider_api_key("OPENAI_API_KEY")


def model_comparison(bert_source: str, bert_seconds: float) -> pd.DataFrame:
    """Return a business-oriented model comparison table."""
    return pd.DataFrame(
        [
            {
                "Approach": "VADER",
                "Speed": "Fast",
                "Confidence": "Uses compound and component lexicon scores",
                "Interpretability": "High",
                "Business Suitability": "Strong baseline for dashboards",
                "Advantages": "Transparent, lightweight, repeatable",
                "Limitations": "Can miss context-specific meaning",
            },
            {
                "Approach": "BERT",
                "Speed": f"Model dependent ({bert_seconds:.2f}s this run)",
                "Confidence": "Classifier confidence score",
                "Interpretability": "Medium",
                "Business Suitability": "Best when model is available and validated",
                "Advantages": f"Contextual transformer sentiment via {bert_source}",
                "Limitations": "Requires model download/cache and governance",
            },
            {
                "Approach": "Rule-based",
                "Speed": "Very fast",
                "Confidence": "Keyword score proxy",
                "Interpretability": "Very high",
                "Business Suitability": "Reliable fallback and audit layer",
                "Advantages": "No external model dependency",
                "Limitations": "Simpler vocabulary and weaker nuance",
            },
        ]
    )
