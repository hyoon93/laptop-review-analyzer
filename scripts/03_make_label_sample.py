"""Draw the hand-labeling sample for evaluation A (application dataset).

300 reviews from data/processed/reviews.csv, 60 per star level, spread across products as
evenly as each star level allows. Stratifying by stars keeps 2- and 3-star reviews (the
hard, often mixed ones) from being swamped by 5-star reviews. The order is shuffled so
the labeler cannot infer the star level from position.

The first 100 rows (by the "aspect_subset" flag) are also used for the aspect judgments.

Output: data/labels/label_sample.csv (review_id, product_id, rating, title, text, aspect_subset)
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline.config import LABELS, PROCESSED  # noqa: E402


def stratified(df, per_star, seed):
    parts = []
    for star, group in df.groupby("rating"):
        # round-robin over products so no single product dominates a star level
        group = group.sample(frac=1, random_state=seed)
        group["_rank"] = group.groupby("product_id").cumcount()
        parts.append(group.sort_values(["_rank", "product_id"]).head(per_star).drop(columns="_rank"))
    return pd.concat(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-star", type=int, default=60)
    ap.add_argument("--aspect-n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=661)
    args = ap.parse_args()

    out = LABELS / "label_sample.csv"
    if out.exists():
        sys.exit(f"{out} already exists; delete it on purpose if you really want a new sample")

    reviews = pd.read_csv(PROCESSED / "reviews.csv")
    sample = stratified(reviews, args.per_star, args.seed).sample(frac=1, random_state=args.seed)
    sample["aspect_subset"] = False
    # aspect subset: 20 per star, taken from the shuffled sample
    aspect_ids = sample.groupby("rating").head(args.aspect_n // 5)["review_id"]
    sample.loc[sample["review_id"].isin(aspect_ids), "aspect_subset"] = True

    LABELS.mkdir(parents=True, exist_ok=True)
    cols = ["review_id", "product_id", "rating", "title", "text", "aspect_subset"]
    sample[cols].to_csv(out, index=False)
    print(f"{len(sample)} reviews -> {out}")
    print(sample.groupby(["rating", "product_id"]).size().unstack(fill_value=0))


if __name__ == "__main__":
    main()
