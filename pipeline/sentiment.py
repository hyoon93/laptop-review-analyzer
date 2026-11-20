"""Stage 1: overall review sentiment with an LLM (zero-shot by default, few-shot optional).

The model sees the review title and text but never the star rating, so the star rating
can serve as an independent baseline instead of leaking into the prediction.
"""

from .config import SENTIMENTS
from .llm import chat_json, map_parallel

SYSTEM = """You classify the overall sentiment of Amazon customer reviews of laptops.

Labels:
- positive: the reviewer is satisfied with the laptop overall (would keep or recommend it), even if they mention minor complaints.
- negative: the reviewer is dissatisfied overall (returned it, regrets buying it, warns others), even if they mention some good points.
- neutral: there is no clear overall verdict: purely factual text, a question, or pros and cons that are balanced with no conclusion.

Also report "mixed": true if the review contains at least one clearly positive AND at least one clearly negative opinion about the product or the purchase, whatever the overall label."""

SCHEMA = {
    "name": "review_sentiment",
    "schema": {
        "type": "object",
        "properties": {
            "sentiment": {"type": "string", "enum": SENTIMENTS},
            "mixed": {"type": "boolean"},
        },
        "required": ["sentiment", "mixed"],
        "additionalProperties": False,
    },
}


def format_review(title, text):
    return f"Review title: {title}\nReview text: {text}"


def build_messages(title, text, examples=None):
    """examples: optional list of (title, text, sentiment, mixed) for few-shot prompting."""
    messages = [{"role": "system", "content": SYSTEM}]
    for ex_title, ex_text, ex_label, ex_mixed in examples or []:
        messages.append({"role": "user", "content": format_review(ex_title, ex_text)})
        messages.append({"role": "assistant",
                         "content": f'{{"sentiment": "{ex_label}", "mixed": {str(ex_mixed).lower()}}}'})
    messages.append({"role": "user", "content": format_review(title, text)})
    return messages


def classify(title, text, examples=None, stage=None, **kw):
    stage = stage or ("sentiment_fewshot" if examples else "sentiment")
    return chat_json(build_messages(title, text, examples), SCHEMA, stage=stage, **kw)


def classify_frame(df, examples=None, workers=8, **kw):
    """Classify every row of a DataFrame with title/text columns; returns a list of dicts."""
    rows = list(zip(df["title"], df["text"]))
    return map_parallel(lambda r: classify(r[0], r[1], examples, **kw), rows,
                        workers=workers, desc="sentiment")
