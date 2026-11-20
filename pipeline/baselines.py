"""Non-LLM sentiment baselines, all mapped to positive / neutral / negative.

- stars:      1-2 stars negative, 3 neutral, 4-5 positive (the usual weak-label rule)
- vader:      lexicon compound score; >= 0.05 positive, <= -0.05 negative (VADER's standard cutoffs)
- distilbert: distilbert-base-uncased-finetuned-sst-2-english is binary, so a review is
              neutral when P(positive) falls between NEUTRAL_BAND[0] and NEUTRAL_BAND[1]
All three get the same input as the LLM: review title + text.
"""

STAR_MAP = {1: "negative", 2: "negative", 3: "neutral", 4: "positive", 5: "positive"}
DISTILBERT = "distilbert-base-uncased-finetuned-sst-2-english"
NEUTRAL_BAND = (0.2, 0.8)


def stars(ratings):
    return [STAR_MAP[int(r)] for r in ratings]


def vader(texts):
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

    sia = SentimentIntensityAnalyzer()
    out = []
    for t in texts:
        c = sia.polarity_scores(t)["compound"]
        out.append("positive" if c >= 0.05 else "negative" if c <= -0.05 else "neutral")
    return out


def distilbert(texts, batch_size=32):
    from transformers import pipeline

    clf = pipeline("text-classification", model=DISTILBERT, top_k=None, truncation=True, max_length=512)
    out = []
    for res in clf(list(texts), batch_size=batch_size):
        p_pos = next(r["score"] for r in res if r["label"] == "POSITIVE")
        lo, hi = NEUTRAL_BAND
        out.append("positive" if p_pos >= hi else "negative" if p_pos <= lo else "neutral")
    return out


def review_input(df):
    """Title + text, the same string the LLM sees."""
    return [f"{t}. {x}" if t else x for t, x in zip(df["title"], df["text"])]
