"""Evaluation A (application data): overall sentiment against the 300 hand labels.

Ground truth: data/labels/sentiment_labels.csv (labeled blind to the star rating).
Systems: every column of data/processed/baselines.csv (stars, VADER, DistilBERT) plus,
when present, LLM predictions in data/processed/sentiment_llm*.csv
(review_id, sentiment, mixed); the file name after "sentiment_" becomes the system name.

The sample is stratified (60 reviews per star level), so low-star reviews are
over-represented compared with the dataset. Besides plain sample metrics, accuracy is
also reported re-weighted to the dataset's star distribution, which estimates accuracy
on all 1,538 reviews.

Reviews whose note says "not English" are excluded.
Run: python -m eval.sentiment_eval  (writes data/processed/eval_sentiment*.csv)
"""

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from pipeline.config import LABELS, PROCESSED, SENTIMENTS

N_BOOT = 2000
SEED = 661


def load():
    sample = pd.read_csv(LABELS / "label_sample.csv")[["review_id", "product_id", "rating"]]
    gold = pd.read_csv(LABELS / "sentiment_labels.csv", keep_default_na=False)
    gold = gold[~gold["note"].str.contains("not english", case=False)]
    df = sample.merge(gold.rename(columns={"sentiment": "gold"}), on="review_id")
    df["mixed"] = df["mixed"].astype(str).str.lower() == "true"

    systems = pd.read_csv(PROCESSED / "baselines.csv")
    for path in sorted(PROCESSED.glob("sentiment_llm*.csv")):
        name = path.stem.removeprefix("sentiment_")
        pred = pd.read_csv(path)[["review_id", "sentiment"]].rename(columns={"sentiment": name})
        systems = systems.merge(pred, on="review_id", how="left")
    df = df.merge(systems, on="review_id", how="left")
    names = [c for c in systems.columns if c != "review_id"]
    return df, names


def star_weights(df):
    """Weight each sampled review so star levels match the full dataset's mix."""
    pop = pd.read_csv(PROCESSED / "reviews.csv", usecols=["rating"])["rating"].value_counts(normalize=True)
    smp = df["rating"].value_counts(normalize=True)
    return df["rating"].map(pop / smp).to_numpy()


def metrics(gold, pred, weights=None):
    return {
        "accuracy": accuracy_score(gold, pred),
        "macro_f1": f1_score(gold, pred, labels=SENTIMENTS, average="macro", zero_division=0),
        **{f"f1_{s}": f1_score(gold, pred, labels=[s], average="macro", zero_division=0)
           for s in SENTIMENTS},
        "accuracy_star_weighted": accuracy_score(gold, pred, sample_weight=weights),
    }


def bootstrap_ci(gold, pred, rng):
    """95% percentile CI for accuracy and macro-F1 (resampling reviews)."""
    gold, pred = np.asarray(gold), np.asarray(pred)
    acc, f1 = [], []
    for _ in range(N_BOOT):
        i = rng.integers(0, len(gold), len(gold))
        acc.append((gold[i] == pred[i]).mean())
        f1.append(f1_score(gold[i], pred[i], labels=SENTIMENTS, average="macro", zero_division=0))
    return np.percentile(acc, [2.5, 97.5]), np.percentile(f1, [2.5, 97.5])


def evaluate(df, names):
    w = star_weights(df)
    rng = np.random.default_rng(SEED)
    rows, confusions, by_star = [], [], []
    for name in names:
        ok = df[name].notna().to_numpy()
        d, wd = df[ok], w[ok]
        m = metrics(d["gold"], d[name], wd)
        (a_lo, a_hi), (f_lo, f_hi) = bootstrap_ci(d["gold"], d[name], rng)
        mixed, plain = d[d["mixed"]], d[~d["mixed"]]
        rows.append({
            "system": name, "n": len(d), **m,
            "accuracy_ci_low": a_lo, "accuracy_ci_high": a_hi,
            "macro_f1_ci_low": f_lo, "macro_f1_ci_high": f_hi,
            "accuracy_mixed": accuracy_score(mixed["gold"], mixed[name]),
            "accuracy_not_mixed": accuracy_score(plain["gold"], plain[name]),
        })
        cm = confusion_matrix(d["gold"], d[name], labels=SENTIMENTS)
        confusions.append(pd.DataFrame(cm, index=[f"gold_{s}" for s in SENTIMENTS],
                                       columns=[f"pred_{s}" for s in SENTIMENTS])
                          .assign(system=name).reset_index(names="gold"))
        by_star.append((d["gold"] == d[name]).groupby(d["rating"]).mean().rename(name))
    return (pd.DataFrame(rows), pd.concat(confusions, ignore_index=True),
            pd.concat(by_star, axis=1).rename_axis("stars"))


def main():
    df, names = load()
    summary, confusions, by_star = evaluate(df, names)
    summary.to_csv(PROCESSED / "eval_sentiment.csv", index=False)
    confusions.to_csv(PROCESSED / "eval_sentiment_confusion.csv", index=False)
    by_star.to_csv(PROCESSED / "eval_sentiment_by_star.csv")

    n_mixed = int(df["mixed"].sum())
    print(f"Evaluation A: {len(df)} hand-labeled reviews "
          f"({df['gold'].value_counts().to_dict()}; {n_mixed} mixed)\n")
    cols = ["system", "accuracy", "accuracy_ci_low", "accuracy_ci_high", "macro_f1",
            "f1_positive", "f1_neutral", "f1_negative", "accuracy_star_weighted",
            "accuracy_mixed", "accuracy_not_mixed"]
    print(summary[cols].round(3).to_string(index=False))
    print("\nAccuracy by star level")
    print(by_star.round(3).to_string())
    print("\nConfusion matrices (rows = gold)")
    for name, cm in confusions.groupby("system", sort=False):
        print(f"\n{name}\n" + cm.drop(columns="system").set_index("gold").to_string())


if __name__ == "__main__":
    main()
