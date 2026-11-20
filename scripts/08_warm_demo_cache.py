"""Pre-run the app's example inputs through the Batch API so their buttons answer from the
cache: instant, free, and still working if the live API is slow or rate-limited during
the demo. Typed-in questions and reviews still call the API live.
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline import aspects, batch, qa, sentiment  # noqa: E402
from pipeline.config import PROCESSED  # noqa: E402
from pipeline.examples import EXAMPLE_QUESTIONS, EXAMPLE_REVIEWS  # noqa: E402


def run_examples():
    products = pd.read_csv(PROCESSED / "products.csv")
    out = []
    for p in products.itertuples():
        for q in EXAMPLE_QUESTIONS:
            res = qa.answer(q, p.product_id, p.short_name)
            out.append((p.product_id, q, res["answerable"], len(res["kept"]), len(res["removed"])))
    for title, text in EXAMPLE_REVIEWS.values():
        sentiment.classify(title, text, stage="live_sentiment")
        aspects.extract(text, title, stage="live_aspects")
    return out


if __name__ == "__main__":
    for row in batch.run_rounds(run_examples, stage="demo_examples"):
        print(row)
