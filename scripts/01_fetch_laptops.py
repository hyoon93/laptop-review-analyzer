"""Extract a laptop-only subset of Amazon Reviews 2023 (Electronics category).

The Electronics files are large (metadata 5.3 GB, reviews 22.6 GB) and the review
file is ordered by user rather than by product, so the only way to get every review
of a product is one full pass. Both files are streamed over HTTP and filtered on the
fly; nothing large is written to disk. Interrupted downloads resume with a Range
request from the last complete line.

Outputs (data/raw/):
    laptops_meta.jsonl.gz    every product whose category path contains "Laptops"
    laptop_reviews.jsonl.gz  every review of the candidate laptops (see --min-ratings, --top)

Usage:
    python scripts/01_fetch_laptops.py                 # both passes
    python scripts/01_fetch_laptops.py --skip-meta     # reuse laptops_meta.jsonl.gz
"""

import argparse
import gzip
import json
import time
from pathlib import Path

import requests
from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
HF_BASE = "https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/resolve/main/raw"
META_URL = f"{HF_BASE}/meta_categories/meta_Electronics.jsonl"
REVIEWS_URL = f"{HF_BASE}/review_categories/Electronics.jsonl"

# Metadata fields worth keeping; images, videos and long descriptions are dropped.
META_FIELDS = ["parent_asin", "title", "main_category", "categories", "average_rating",
               "rating_number", "price", "store", "features", "details"]
# user_id is dropped on purpose: the app never needs reviewer identity.
REVIEW_FIELDS = ["parent_asin", "asin", "rating", "title", "text", "timestamp",
                 "helpful_vote", "verified_purchase"]


def hf_headers():
    """Send the course HF token if there is one (higher rate limits)."""
    for env in (ROOT.parents[1] / ".env", ROOT / ".env"):
        if env.exists():
            token = dotenv_values(env).get("HF_TOKEN")
            if token:
                return {"Authorization": f"Bearer {token}"}
    return {}


def stream_lines(url, max_retries=30):
    """Yield the file's lines as bytes, resuming from the last full line on errors."""
    pos, done, retries = 0, False, 0
    t0, last_report = time.time(), 0
    while not done:
        headers = hf_headers()
        if pos:
            headers["Range"] = f"bytes={pos}-"
        buf = b""
        try:
            with requests.get(url, headers=headers, stream=True, timeout=60) as r:
                r.raise_for_status()
                for chunk in r.iter_content(chunk_size=1 << 20):
                    buf += chunk
                    lines = buf.split(b"\n")
                    buf = lines.pop()
                    for line in lines:
                        pos += len(line) + 1
                        yield line
                    if pos - last_report >= 1 << 30:
                        last_report = pos
                        mins = (time.time() - t0) / 60
                        print(f"  {pos / 1e9:6.1f} GB read, {mins:5.1f} min, "
                              f"{pos / 1e6 / max(time.time() - t0, 1):5.1f} MB/s", flush=True)
            if buf:
                pos += len(buf)
                yield buf
            done = True
        except (requests.RequestException, ConnectionError) as e:
            retries += 1
            if retries > max_retries:
                raise
            print(f"  stream error at {pos / 1e9:.2f} GB ({type(e).__name__}); "
                  f"retry {retries} in {5 * retries}s", flush=True)
            time.sleep(5 * retries)


def fetch_meta(out_path):
    n = 0
    with gzip.open(out_path, "wt", encoding="utf-8") as out:
        for line in stream_lines(META_URL):
            if b'"Laptops"' not in line:      # cheap prefilter before json parsing
                continue
            item = json.loads(line)
            if "Laptops" not in (item.get("categories") or []):
                continue
            out.write(json.dumps({k: item.get(k) for k in META_FIELDS}, ensure_ascii=False) + "\n")
            n += 1
    print(f"meta: {n} laptop products -> {out_path}", flush=True)


def load_meta(path):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def fetch_reviews(candidates, out_path):
    wanted = {c.encode() for c in candidates}
    key = b'"parent_asin": "'
    n = 0
    with gzip.open(out_path, "wt", encoding="utf-8") as out:
        for line in stream_lines(REVIEWS_URL):
            i = line.find(key)
            if i < 0 or line[i + len(key): i + len(key) + 10] not in wanted:
                continue
            review = json.loads(line)
            if review["parent_asin"] not in candidates:
                continue
            out.write(json.dumps({k: review.get(k) for k in REVIEW_FIELDS}, ensure_ascii=False) + "\n")
            n += 1
    print(f"reviews: {n} reviews of {len(candidates)} candidate laptops -> {out_path}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--skip-meta", action="store_true", help="reuse an existing laptops_meta.jsonl.gz")
    ap.add_argument("--min-ratings", type=int, default=200, help="minimum Amazon rating count for a candidate")
    ap.add_argument("--top", type=int, default=300, help="keep at most this many candidates (most-rated first)")
    args = ap.parse_args()

    RAW.mkdir(parents=True, exist_ok=True)
    meta_path = RAW / "laptops_meta.jsonl.gz"
    if not (args.skip_meta and meta_path.exists()):
        print("pass 1/2: metadata", flush=True)
        fetch_meta(meta_path)

    meta = load_meta(meta_path)
    eligible = [m for m in meta if (m.get("rating_number") or 0) >= args.min_ratings]
    eligible.sort(key=lambda m: -m["rating_number"])
    candidates = {m["parent_asin"] for m in eligible[: args.top]}
    print(f"{len(meta)} laptops, {len(eligible)} with >= {args.min_ratings} ratings, "
          f"{len(candidates)} candidates", flush=True)
    (RAW / "candidates.json").write_text(json.dumps(sorted(candidates)), encoding="utf-8")

    print("pass 2/2: reviews", flush=True)
    fetch_reviews(candidates, RAW / "laptop_reviews.jsonl.gz")


if __name__ == "__main__":
    main()
