import re
import nltk
import spacy

from nltk.sentiment import SentimentIntensityAnalyzer


# Download VADER lexicon if required
try:
    nltk.data.find("sentiment/vader_lexicon.zip")
except LookupError:
    nltk.download("vader_lexicon")


# Initialize NLP tools once
analyzer = SentimentIntensityAnalyzer()
nlp = spacy.load("en_core_web_sm")


# Negation words carry sentiment, so never remove them
KEEP = {"not", "no", "never", "nor", "none", "nt"}


# 1. Create sentiment labels using VADER
def create_sentiment(text):

    score = analyzer.polarity_scores(str(text))["compound"]

    if score >= 0.05:
        return "positive"

    elif score <= -0.05:
        return "negative"

    else:
        return "neutral"


# 2. Clean a single review
def clean_text(text):

    text = str(text).lower()

    # Remove URLs and non-alphabetic characters
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z ]", "", text)

    doc = nlp(text)

    words = []

    for token in doc:

        # Preserve negations because they change sentiment
        if token.lemma_ in KEEP or not token.is_stop:
            words.append(token.lemma_)

    return " ".join(words)


# 3. Clean multiple reviews efficiently
def clean_batch(texts):

    cleaned = []

    # Process reviews in batches for better performance
    docs = nlp.pipe(
        texts,
        disable=["parser", "ner"],
        batch_size=200
    )

    for doc in docs:

        words = [
            token.lemma_
            for token in doc
            if token.lemma_ in KEEP or not token.is_stop
        ]

        cleaned.append(" ".join(words))

    return cleaned