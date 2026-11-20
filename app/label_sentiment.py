"""Hand-labeling page for overall review sentiment (evaluation A, ground truth).

Run:  streamlit run app/label_sentiment.py

Shows one review at a time from data/labels/label_sample.csv, without its star rating
or any model prediction, and saves every answer immediately to
data/labels/sentiment_labels.csv, so you can stop and resume at any point.
Definitions: data/labels/GUIDELINES.md
"""

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline.config import LABELS, SENTIMENTS  # noqa: E402

SAMPLE = LABELS / "label_sample.csv"
OUT = LABELS / "sentiment_labels.csv"
COLUMNS = ["review_id", "sentiment", "mixed", "note"]

st.set_page_config(page_title="Sentiment labeling", layout="centered")

if not SAMPLE.exists():
    st.error("No label sample yet. Run `python scripts/03_make_label_sample.py` first.")
    st.stop()

sample = pd.read_csv(SAMPLE)
labels = pd.read_csv(OUT, keep_default_na=False) if OUT.exists() else pd.DataFrame(columns=COLUMNS)
done = set(labels["review_id"])


def first_unlabeled():
    remaining = [i for i, rid in enumerate(sample["review_id"]) if rid not in done]
    return remaining[0] if remaining else len(sample) - 1


if "idx" not in st.session_state:
    st.session_state.idx = first_unlabeled()

with st.sidebar:
    st.markdown("**Guidelines**")
    st.markdown(
        "- **positive**: satisfied overall, even with minor complaints\n"
        "- **negative**: dissatisfied overall, even with some good points\n"
        "- **neutral**: no clear overall verdict\n"
        "- **mixed**: has both a clear positive and a clear negative opinion"
    )
    st.caption("Star ratings are hidden on purpose. Full text: data/labels/GUIDELINES.md")
    if st.button("Jump to first unlabeled"):
        st.session_state.idx = first_unlabeled()
        st.rerun()

idx = st.session_state.idx
row = sample.iloc[idx]
prev = labels[labels["review_id"] == row["review_id"]]
prev = prev.iloc[0] if len(prev) else None

st.progress(len(done) / len(sample), text=f"{len(done)} of {len(sample)} labeled")
st.caption(f"Review {idx + 1} of {len(sample)} · {row['review_id']}" + (" · already labeled" if prev is not None else ""))
st.subheader(row["title"] if isinstance(row["title"], str) and row["title"] else "(no title)")
st.write(row["text"])

# Keyed by review ID, not position: if the sample file changes under an open page,
# a stale "Save" can never attach its label to a different review.
with st.form(key=f"form-{row['review_id']}"):
    sentiment = st.radio("Overall sentiment", SENTIMENTS, horizontal=True,
                         index=SENTIMENTS.index(prev["sentiment"]) if prev is not None else None)
    mixed = st.checkbox("Mixed (has both a clear positive and a clear negative opinion)",
                        value=bool(prev is not None and str(prev["mixed"]) == "True"))
    note = st.text_input("Note (optional: sarcasm, unsure, off-topic, ...)",
                         value=prev["note"] if prev is not None else "")
    saved = st.form_submit_button("Save and next", type="primary")

if saved:
    if sentiment is None:
        st.error("Pick a sentiment label first.")
    else:
        labels = labels[labels["review_id"] != row["review_id"]]
        new = pd.DataFrame([{"review_id": row["review_id"], "sentiment": sentiment,
                             "mixed": mixed, "note": note}])
        pd.concat([labels, new])[COLUMNS].to_csv(OUT, index=False)
        done.add(row["review_id"])
        st.session_state.idx = first_unlabeled() if len(done) < len(sample) else idx
        st.rerun()

left, right = st.columns(2)
if left.button("◀ Previous", disabled=idx == 0):
    st.session_state.idx = idx - 1
    st.rerun()
if right.button("Next ▶", disabled=idx == len(sample) - 1):
    st.session_state.idx = idx + 1
    st.rerun()

if len(done) == len(sample):
    st.success("All reviews labeled. Thank you!")
