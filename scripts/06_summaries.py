"""Generate the product summaries (ungrounded + grounded) and run the claim checker.

Output: data/processed/summaries.csv, one row per claim and condition:
    claim_id, product_id, condition (ungrounded / grounded / grounded_verified), aspect,
    claim, citations; for checked claims also verdict, reason, supporting_ids,
    invalid_citations, kept, repaired, original_claim (before the rewrite)

Needs data/processed/aspects.csv (scripts/05) and the ChromaDB index. All LLM calls go
through the Batch API (rounds: generation, check, repair, re-check). Use --sync to call
the API directly instead.
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline import batch, summarize  # noqa: E402
from pipeline.config import PROCESSED  # noqa: E402


def run_all(product_ids):
    products, asp = summarize.load_tables()
    rows = []
    for pid in product_ids:
        rows += summarize.summarize_product(pid, products, asp)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--product", nargs="*")
    ap.add_argument("--sync", action="store_true")
    args = ap.parse_args()

    pids = args.product or pd.read_csv(PROCESSED / "products.csv")["product_id"].tolist()
    rows = run_all(pids) if args.sync else batch.run_rounds(lambda: run_all(pids), stage="summaries")
    df = pd.DataFrame(rows)
    for col in ["citations", "supporting_ids", "invalid_citations"]:
        if col in df:
            df[col] = df[col].map(lambda v: json.dumps(v) if isinstance(v, list) else "[]")
    df.to_csv(PROCESSED / "summaries.csv", index=False)

    g = df[df["condition"] == "grounded"]
    v = df[df["condition"] == "grounded_verified"]
    print(f"{len(df)} rows: {(df['condition'] == 'ungrounded').sum()} ungrounded, {len(g)} grounded, "
          f"{len(v)} grounded_verified ({int(v['repaired'].sum())} of them rewritten)")
    print("first check of grounded claims:", g["verdict"].value_counts().to_dict())


if __name__ == "__main__":
    main()
