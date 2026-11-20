"""Clean the raw laptop reviews, profile candidate products, and build the app dataset.

Step 1 (always): clean every candidate's reviews and write per-product statistics to
    data/processed/candidate_products.csv
Step 2 (when data/processed/selected_products.json exists): keep the selected products,
    sample up to --per-product reviews from each, assign short review IDs, and write
    data/processed/products.csv and data/processed/reviews.csv

Cleaning: strip HTML tags and entities, collapse whitespace, drop reviews with fewer than
--min-words words, drop exact duplicates within a product.

English only: about 4% of the sampled reviews are in Spanish. The baselines (VADER,
DistilBERT SST-2) and the SemEval benchmark are English-only, so these are dropped. The
drop happens AFTER sampling and ID assignment, so the remaining review IDs (and any hand
labels attached to them) stay the same; dropped IDs go to excluded_non_english.csv.
"""

import argparse
import gzip
import html
import json
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline.config import PROCESSED, RAW  # noqa: E402

TAG = re.compile(r"<[^>]+>")
SPACE = re.compile(r"\s+")
VIDEO = re.compile(r"\[\[VIDEOID:[0-9a-f]+\]\]")
WORD = re.compile(r"[a-záéíóúñü']+")
# Function words that are common in one language and rare in the other. Words shared by
# both languages ("no", "a") are left out so short English reviews are not flagged.
EN_WORDS = set("the an and or but is was it this that i my to of for in on with not very "
               "have has be are so its it's you they at as just great good works".split())
ES_WORDS = set("el la los las de que y es muy por para con una un lo se mi pero bien "
               "buena bueno buen excelente".split())


def clean_text(s):
    s = html.unescape(TAG.sub(" ", s or ""))
    return SPACE.sub(" ", s).strip()


def is_spanish(title, text):
    """More Spanish than English function words (all 62 hits were checked by hand)."""
    words = WORD.findall(f"{title} {text}".lower())
    en = sum(w in EN_WORDS for w in words)
    es = sum(w in ES_WORDS for w in words)
    return es > en and (es >= 2 or en == 0)


def load_jsonl_gz(path):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def product_type(title, categories):
    t = title.lower()
    if "renewed" in t or "refurbished" in t:
        return "renewed"
    if "2 in 1 Laptops" in categories or "2-in-1" in t or "2 in 1" in t:
        return "2-in-1"
    if "chromebook" in t:
        return "chromebook"
    if "macbook" in t:
        return "macbook"
    if "gaming" in t:
        return "gaming"
    return "traditional"


def brand(meta):
    details = meta.get("details") or {}
    b = details.get("Brand") or meta.get("store") or ""
    return b.strip().title()


def load_clean_reviews(min_words):
    reviews = pd.DataFrame(load_jsonl_gz(RAW / "laptop_reviews.jsonl.gz"))
    n_raw = len(reviews)
    reviews["title"] = reviews["title"].map(clean_text)
    reviews["text"] = reviews["text"].map(clean_text)
    reviews = reviews[reviews["text"].str.split().str.len() >= min_words]
    reviews = reviews.drop_duplicates(subset=["parent_asin", "text"])
    reviews["date"] = pd.to_datetime(reviews["timestamp"], unit="ms").dt.date
    reviews["rating"] = reviews["rating"].astype(int)
    print(f"reviews: {n_raw} raw -> {len(reviews)} after cleaning")
    return reviews


def profile(meta, reviews):
    rows = []
    by_product = reviews.groupby("parent_asin")
    for m in meta:
        pid = m["parent_asin"]
        if pid not in by_product.groups:
            continue
        r = by_product.get_group(pid)
        stars = r["rating"].value_counts()
        rows.append({
            "parent_asin": pid,
            "brand": brand(m),
            "type": product_type(m["title"], m.get("categories") or []),
            "n_reviews": len(r),
            "n_neg": int(stars.get(1, 0) + stars.get(2, 0)),
            "n_3star": int(stars.get(3, 0)),
            "mean_rating": round(r["rating"].mean(), 2),
            "median_words": int(r["text"].str.split().str.len().median()),
            "first": r["date"].min(),
            "last": r["date"].max(),
            "price": m.get("price"),
            "title": m["title"],
        })
    return pd.DataFrame(rows).sort_values("n_reviews", ascending=False)


def build_dataset(meta, reviews, selected, per_product, seed):
    meta_by_id = {m["parent_asin"]: m for m in meta}
    products, parts = [], []
    for i, item in enumerate(selected, start=1):
        pid, code = item["parent_asin"], f"L{i:02d}"
        m = meta_by_id[pid]
        r = reviews[reviews["parent_asin"] == pid]
        if len(r) > per_product:
            r = r.sample(n=per_product, random_state=seed)
        r = r.sort_values("timestamp").reset_index(drop=True)
        r.insert(0, "review_id", [f"{code}-{j:03d}" for j in range(1, len(r) + 1)])
        r.insert(1, "product_id", code)
        parts.append(r)
        products.append({
            "product_id": code,
            "parent_asin": pid,
            "short_name": item["short_name"],
            "type": item.get("type") or product_type(m["title"], m.get("categories") or []),
            "brand": brand(m),
            "title": m["title"],
            "price": m.get("price"),
            "average_rating_amazon": m.get("average_rating"),
            "rating_number_amazon": m.get("rating_number"),
            "n_reviews_sampled": len(r),
            "n_reviews_available": int((reviews["parent_asin"] == pid).sum()),
        })
    cols = ["review_id", "product_id", "parent_asin", "asin", "rating", "title", "text",
            "date", "verified_purchase", "helpful_vote"]
    data = pd.concat(parts)[cols]
    data["text"] = data["text"].str.replace(VIDEO, "", regex=True).str.strip()
    spanish = [is_spanish(t, x) for t, x in zip(data["title"], data["text"])]
    excluded, data = data[spanish], data[[not s for s in spanish]]
    products = pd.DataFrame(products)
    products["n_reviews_sampled"] = products["product_id"].map(data["product_id"].value_counts())
    return products, data, excluded


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--min-words", type=int, default=5)
    ap.add_argument("--per-product", type=int, default=250)
    ap.add_argument("--seed", type=int, default=661)
    args = ap.parse_args()

    PROCESSED.mkdir(parents=True, exist_ok=True)
    meta = load_jsonl_gz(RAW / "laptops_meta.jsonl.gz")
    reviews = load_clean_reviews(args.min_words)

    prof = profile(meta, reviews)
    prof.to_csv(PROCESSED / "candidate_products.csv", index=False)
    print(f"profiled {len(prof)} candidate products -> candidate_products.csv")

    sel_path = PROCESSED / "selected_products.json"
    if not sel_path.exists():
        print("no selected_products.json yet; pick products from candidate_products.csv first")
        return
    selected = json.loads(sel_path.read_text(encoding="utf-8"))["products"]
    products, data, excluded = build_dataset(meta, reviews, selected, args.per_product, args.seed)
    products.to_csv(PROCESSED / "products.csv", index=False)
    data.to_csv(PROCESSED / "reviews.csv", index=False)
    excluded.to_csv(PROCESSED / "excluded_non_english.csv", index=False)
    print(f"{len(products)} products, {len(data)} reviews -> products.csv, reviews.csv "
          f"({len(excluded)} Spanish reviews -> excluded_non_english.csv)")
    print(products[["product_id", "short_name", "type", "n_reviews_sampled", "n_reviews_available"]]
          .to_string(index=False))
    print(data.groupby("product_id")["rating"].value_counts().unstack(fill_value=0))


if __name__ == "__main__":
    main()
