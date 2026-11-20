"""Evaluation C: the one comparison (ungrounded vs grounded vs grounded + verified).

An independent judge (gpt-4.1, evaluation only, never used in the app) grades every
summary claim against evidence gathered the same way for every condition: exactly
JUDGE_K = 12 reviews per claim, the ones it cites (grounded conditions) first, then the
most similar reviews (local retrieval). The judge's reliability is checked against my
own blind judgments of a claim subset (data/labels/claim_judgments.csv) with Cohen's kappa.

Metrics per condition:
    supported / partially supported / unsupported rate (judge)
    citation precision: share of cited reviews the judge says support the claim (grounded only)
    aspect coverage: share of the product's top aspects with at least one supported claim

Run: python -m eval.judge   (Batch API; writes data/processed/judge.csv, eval_summaries.csv)
"""

import json

import pandas as pd

from pipeline import batch, index, verify
from pipeline.config import JUDGE_MODEL, LABELS, PROCESSED
from pipeline.llm import chat_json

JUDGE_K = 12
CONDITIONS = ["ungrounded", "grounded", "grounded_verified"]

SYSTEM = """You are evaluating a summary of customer reviews of a laptop. Decide whether one claim from the summary is supported by the customer reviews shown.

Verdicts:
- supported: the reviews clearly state what the claim says. A claim about "several", "many" or "most" reviewers needs at least two reviews that say it.
- partially_supported: some of the claim is backed, but it overstates how many reviewers say it, or adds details the reviews do not give.
- unsupported: no review says it, or the reviews contradict it.

Judge only from the review text shown, not from what you know about the product. List the IDs of every shown review that supports the claim."""


def evidence_ids(claim, product_id, citations):
    """Exactly JUDGE_K reviews for every claim: the cited ones first, then the most similar.

    A fixed count keeps the human judgments blind: a varying number of evidence reviews
    would hint at whether the claim came from a condition with citations.
    """
    reviews = verify._reviews()
    cited = [c for c in dict.fromkeys(citations)
             if c in reviews.index and reviews.loc[c, "product_id"] == product_id][:JUDGE_K]
    retrieved = [h["review_id"] for h in index.search(claim, product_id=product_id, k=JUDGE_K + len(cited))]
    return list(dict.fromkeys(cited + retrieved))[:JUDGE_K]


def judge(claim, product_id, citations):
    ids = evidence_ids(claim, product_id, citations)
    body = "\n\n".join(f"[{rid}] {verify.review_text(rid, max_words=300)}" for rid in ids)
    messages = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": f"Claim: {claim}\n\nReviews:\n\n{body}"}]
    out = chat_json(messages, verify.SCHEMA, stage="judge", model=JUDGE_MODEL)
    return {"judge_verdict": out["verdict"], "judge_reason": out["reason"],
            "judge_supporting_ids": [i for i in out["supporting_ids"] if i in ids]}


def load_claims():
    df = pd.read_csv(PROCESSED / "summaries.csv", keep_default_na=False)
    for col in ["citations", "supporting_ids", "invalid_citations"]:
        df[col] = df[col].map(lambda s: json.loads(s) if s else [])
    for col in ["kept", "repaired"]:
        df[col] = df[col].astype(str).str.lower() == "true"
    return df


def run_judge(df):
    # Identical claims (a grounded claim kept unchanged in grounded_verified) hit the cache.
    return [judge(r.claim, r.product_id, r.citations) for r in df.itertuples()]


def by_condition(df):
    return df.assign(cond=df["condition"])


def citation_precision(row):
    if not row["citations"]:
        return None
    return len(set(row["citations"]) & set(row["judge_supporting_ids"])) / len(set(row["citations"]))


def coverage(d, verdict_col):
    out = []
    for pid, g in d.groupby("product_id"):
        top = g["top_aspects"].iloc[0].split(",")
        ok = set(g.loc[g[verdict_col] == "supported", "aspect"])
        out.append(len(ok & set(top)) / len(top))
    return sum(out) / len(out)


def summarize(df, verdict_col="judge_verdict"):
    d = by_condition(df)
    rows = []
    for cond in CONDITIONS:
        g = d[d["cond"] == cond]
        v = g[verdict_col].value_counts(normalize=True)
        rows.append({
            "condition": cond, "claims": len(g),
            "claims_per_product": len(g) / g["product_id"].nunique(),
            "supported": v.get("supported", 0.0),
            "partially_supported": v.get("partially_supported", 0.0),
            "unsupported": v.get("unsupported", 0.0),
            "citation_precision": g["citation_precision"].dropna().mean() if cond != "ungrounded" else None,
            "aspect_coverage": coverage(g, verdict_col),
        })
    return pd.DataFrame(rows)


def human_tables(df):
    """My blind judgments vs the judge, if they exist.

    Returns (kappa dict over distinct claims, per-condition table over the sampled rows
    with both the human and the judge rates), or (None, None).
    """
    sample_path, judged_path = LABELS / "claim_sample.csv", LABELS / "claim_judgments.csv"
    if not (sample_path.exists() and judged_path.exists()):
        return None, None
    from sklearn.metrics import cohen_kappa_score

    items = pd.read_csv(sample_path).merge(pd.read_csv(judged_path), on="item_id")
    items["sampled_rows"] = items["sampled_rows"].map(json.loads)
    rows = items.explode("sampled_rows").rename(columns={"sampled_rows": "claim_id"})
    rows = rows.merge(df[["claim_id", "condition", "judge_verdict"]], on="claim_id")
    per_item = rows.drop_duplicates("item_id")
    binary = lambda s: (s == "supported").astype(int)
    kappa = {"n": len(per_item),
             "agreement": float((per_item["human_verdict"] == per_item["judge_verdict"]).mean()),
             "kappa_3way": float(cohen_kappa_score(per_item["human_verdict"], per_item["judge_verdict"])),
             "kappa_supported_vs_not": float(cohen_kappa_score(binary(per_item["human_verdict"]),
                                                               binary(per_item["judge_verdict"])))}
    table = []
    for cond in CONDITIONS:
        g = rows[rows["condition"] == cond]
        table.append({"condition": cond, "n": len(g),
                      **{f"human_{v}": (g["human_verdict"] == v).mean() for v in verify.VERDICTS},
                      **{f"judge_{v}": (g["judge_verdict"] == v).mean() for v in verify.VERDICTS}})
    return kappa, pd.DataFrame(table)


def main():
    df = load_claims()
    verdicts = batch.run_rounds(lambda: run_judge(df), stage="judge")
    df = pd.concat([df, pd.DataFrame(verdicts)], axis=1)
    df["citation_precision"] = df.apply(citation_precision, axis=1)
    out = df.copy()
    for col in ["citations", "supporting_ids", "invalid_citations", "judge_supporting_ids"]:
        out[col] = out[col].map(json.dumps)
    out.to_csv(PROCESSED / "judge.csv", index=False)

    table = summarize(df)
    table.to_csv(PROCESSED / "eval_summaries.csv", index=False)
    print(table.round(3).to_string(index=False))
    k, human = human_tables(df)
    if k:
        print("\njudge vs human:", json.dumps(k, indent=2))
        (PROCESSED / "eval_judge_kappa.json").write_text(json.dumps(k, indent=2), encoding="utf-8")
        human.to_csv(PROCESSED / "eval_summaries_human.csv", index=False)
        print(human.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
