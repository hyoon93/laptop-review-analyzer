"""Evaluation A (application data): aspect extraction, evidence spans, star/text disagreement.

1. Aspect quality on the 100 reviews I judged (data/labels/aspect_judgments.csv,
   aspect_missed.csv):
       precision           share of extracted aspects that are real
       polarity accuracy   share of real aspects with the right polarity
       approximate recall  real / (real + missed), where "missed" are aspects I listed
2. Evidence spans (all reviews): share of quoted evidence found verbatim in the review.
3. Star/text disagreement (all reviews): share of 4-5 star reviews with at least one
   negative aspect, and of 1-2 star reviews with at least one positive aspect; checked
   against my sentiment labels on the 300-review sample.

Run: python -m eval.aspect_eval  (writes data/processed/eval_aspects.json)
"""

import json

import pandas as pd

from pipeline.config import LABELS, PROCESSED


def aspect_quality():
    path, missed_path = LABELS / "aspect_judgments.csv", LABELS / "aspect_missed.csv"
    if not (path.exists() and missed_path.exists()):
        return None
    j = pd.read_csv(path, keep_default_na=False)
    missed = pd.read_csv(missed_path, keep_default_na=False)
    real = j[j["is_real"] == "yes"]
    n_missed = int(missed["n_missed"].sum())
    by_cat = j.groupby("category").agg(extracted=("is_real", "size"),
                                       precision=("is_real", lambda s: (s == "yes").mean()))
    return {
        "reviews_judged": int(missed["review_id"].nunique()),
        "aspects_extracted": len(j),
        "precision": float((j["is_real"] == "yes").mean()),
        "polarity_accuracy": float((real["polarity_ok"] == "yes").mean()) if len(real) else None,
        "missed": n_missed,
        "approx_recall": len(real) / (len(real) + n_missed) if len(real) + n_missed else None,
        "by_category": by_cat.round(3).reset_index().to_dict("records"),
    }


def evidence_and_disagreement():
    asp = pd.read_csv(PROCESSED / "aspects.csv", keep_default_na=False)
    asp["evidence_found"] = asp["evidence_found"].astype(str).str.lower() == "true"
    reviews = pd.read_csv(PROCESSED / "reviews.csv", keep_default_na=False)[["review_id", "rating"]]
    has = asp.groupby("review_id")["polarity"].agg(lambda s: set(s)).rename("pols")
    r = reviews.merge(has, on="review_id", how="left")
    r["pols"] = r["pols"].map(lambda v: v if isinstance(v, set) else set())
    r["neg_aspect"] = r["pols"].map(lambda s: "negative" in s)
    r["pos_aspect"] = r["pols"].map(lambda s: "positive" in s)
    high, low = r[r["rating"] >= 4], r[r["rating"] <= 2]

    # Check on the hand-labeled sample: 4-5 star reviews that I labeled mixed or negative.
    labels = pd.read_csv(LABELS / "sentiment_labels.csv", keep_default_na=False)
    labels["mixed"] = labels["mixed"].astype(str).str.lower() == "true"
    lab = r.merge(labels, on="review_id")
    lab_high = lab[lab["rating"] >= 4]
    human_neg_or_mixed = (lab_high["sentiment"] == "negative") | lab_high["mixed"]
    return {
        "aspects_total": len(asp),
        "aspects_per_review": len(asp) / len(reviews),
        "evidence_verbatim": float(asp["evidence_found"].mean()),
        "high_star_reviews": len(high),
        "high_star_with_negative_aspect": float(high["neg_aspect"].mean()),
        "low_star_reviews": len(low),
        "low_star_with_positive_aspect": float(low["pos_aspect"].mean()),
        "check_on_labeled_4_5_star": {
            "n": len(lab_high),
            "model_negative_aspect": float(lab_high["neg_aspect"].mean()),
            "human_negative_or_mixed": float(human_neg_or_mixed.mean()),
            "agreement": float((lab_high["neg_aspect"] == human_neg_or_mixed).mean()),
        },
    }


def main():
    out = {"aspect_quality": aspect_quality(), **evidence_and_disagreement()}
    (PROCESSED / "eval_aspects.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k != "aspect_quality"}, indent=2))
    q = out["aspect_quality"]
    if q:
        print(json.dumps({k: v for k, v in q.items() if k != "by_category"}, indent=2))
        print(pd.DataFrame(q["by_category"]).to_string(index=False))
    else:
        print("aspect judgments not done yet")


if __name__ == "__main__":
    main()
