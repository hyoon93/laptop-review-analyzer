"""Run the per-review LLM stages (sentiment, aspects) on every review.

Outputs (data/processed/):
    sentiment_llm.csv   review_id, sentiment, mixed
    aspects.csv         one row per extracted aspect: review_id, product_id, rating, term,
                        category, polarity, evidence, evidence_found

Every call goes through the disk cache, so the script can be stopped and rerun for free.
Use --product L01 to run a single product.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline import aspects, batch, sentiment  # noqa: E402
from pipeline.config import PROCESSED  # noqa: E402
from pipeline.llm import usage_table  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--product")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--batch", action="store_true",
                    help="send uncached calls through the Batch API first (50%% cheaper)")
    args = ap.parse_args()

    reviews = pd.read_csv(PROCESSED / "reviews.csv", keep_default_na=False)
    if args.product:
        reviews = reviews[reviews["product_id"] == args.product]
    suffix = f"_{args.product}" if args.product else ""

    if args.batch:
        rows = list(zip(reviews["title"], reviews["text"]))
        calls = ([{"messages": sentiment.build_messages(t, x), "schema": sentiment.SCHEMA,
                   "stage": "sentiment"} for t, x in rows]
                 + [{"messages": aspects.build_messages(x, t), "schema": aspects.SCHEMA,
                     "stage": "aspects"} for t, x in rows])
        missing = batch.run(calls, stage="reviews")
        print(f"batch done, {missing} calls missing (they run synchronously below)")

    sent = sentiment.classify_frame(reviews, workers=args.workers)
    pd.DataFrame({
        "review_id": reviews["review_id"].to_numpy(),
        "sentiment": [s["sentiment"] for s in sent],
        "mixed": [s["mixed"] for s in sent],
    }).to_csv(PROCESSED / f"sentiment_llm{suffix}.csv", index=False)

    found = aspects.extract_frame(reviews, workers=args.workers)
    rows = []
    for r, items in zip(reviews.itertuples(), found):
        for a in items:
            rows.append({"review_id": r.review_id, "product_id": r.product_id, "rating": r.rating,
                         **a, "evidence_found": aspects.evidence_found(a["evidence"], r.title, r.text)})
    asp = pd.DataFrame(rows)
    asp.to_csv(PROCESSED / f"aspects{suffix}.csv", index=False)

    print(f"{len(reviews)} reviews: {len(asp)} aspects, "
          f"{asp['evidence_found'].mean():.1%} evidence spans found verbatim")
    print(usage_table().to_string(index=False))


if __name__ == "__main__":
    main()
