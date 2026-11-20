"""Run the non-LLM sentiment baselines (stars, VADER, DistilBERT SST-2) on every review.

Output: data/processed/baselines.csv (review_id, stars, vader, distilbert)
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline import baselines  # noqa: E402
from pipeline.config import PROCESSED  # noqa: E402

reviews = pd.read_csv(PROCESSED / "reviews.csv", keep_default_na=False)
texts = baselines.review_input(reviews)
out = pd.DataFrame({
    "review_id": reviews["review_id"],
    "stars": baselines.stars(reviews["rating"]),
    "vader": baselines.vader(texts),
    "distilbert": baselines.distilbert(texts),
})
out.to_csv(PROCESSED / "baselines.csv", index=False)
print(f"{len(out)} reviews -> baselines.csv")
for col in ["stars", "vader", "distilbert"]:
    print(col, out[col].value_counts().to_dict())
