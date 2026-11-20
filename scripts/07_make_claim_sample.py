"""Draw the summary claims I judge by hand (evaluation C: checks the LLM judge).

40 random claims per condition (ungrounded, grounded, grounded_verified), spread across
products. A grounded claim that was kept unchanged appears with the same text in
grounded_verified; such a claim is shown once and its verdict counts for every sampled
row with that text. So the page shows somewhat fewer than 120 items.

The page shows only the claim and its evidence, never the condition, the citations or
any verdict: evidence = 12 reviews: the ones the claim cites, then the most similar
(the same evidence the judge sees), in shuffled order.

Output: data/labels/claim_sample.csv
    item_id, product_id, claim, evidence_ids, sampled_rows (claim_ids drawn for each condition)
Refuses to overwrite an existing sample.
"""

import json
import random
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval.judge import CONDITIONS, evidence_ids, load_claims  # noqa: E402
from pipeline.config import LABELS  # noqa: E402

SEED = 661
PER_CONDITION = 40
OUT = LABELS / "claim_sample.csv"


def spread(df, n, rng):
    """n random rows spread across products as evenly as possible."""
    groups = [g.sample(frac=1, random_state=rng.randint(0, 10**6)) for _, g in df.groupby("product_id")]
    picks, i = [], 0
    while len(picks) < min(n, len(df)):
        for g in groups:
            if i < len(g) and len(picks) < n:
                picks.append(g.iloc[i])
        i += 1
    return pd.DataFrame(picks)


def main():
    if OUT.exists():
        sys.exit(f"{OUT} exists; delete it first if you really want a new sample.")
    rng = random.Random(SEED)
    df = load_claims()
    drawn = pd.concat([spread(df[df["condition"] == c], PER_CONDITION, rng) for c in CONDITIONS])

    items = []
    for (pid, claim), g in drawn.groupby(["product_id", "claim"], sort=False):
        ids = evidence_ids(claim, pid, g.iloc[0]["citations"])
        rng.shuffle(ids)
        items.append({"product_id": pid, "claim": claim, "evidence_ids": json.dumps(ids),
                      "sampled_rows": json.dumps(g["claim_id"].tolist())})
    rng.shuffle(items)
    out = pd.DataFrame(items)
    out.insert(0, "item_id", [f"C{i:03d}" for i in range(1, len(out) + 1)])
    out.to_csv(OUT, index=False)
    print(f"{len(drawn)} sampled rows ({PER_CONDITION} per condition) -> {len(out)} distinct claims to judge -> {OUT}")


if __name__ == "__main__":
    main()
