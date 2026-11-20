"""Hand-judging page for summary claims (evaluation C: checks the LLM judge).

Run:  streamlit run app/label_claims.py

Shows one claim at a time from data/labels/claim_sample.csv with its evidence reviews.
It never shows which condition produced the claim, its citations, or any model verdict.
Saves immediately to data/labels/claim_judgments.csv.
"""

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline.config import LABELS, PROCESSED  # noqa: E402

SAMPLE = LABELS / "claim_sample.csv"
OUT = LABELS / "claim_judgments.csv"
VERDICTS = ["supported", "partially_supported", "unsupported"]
COLS = ["item_id", "human_verdict", "note"]

st.set_page_config(page_title="Claim judging", layout="centered")
if not SAMPLE.exists():
    st.error("No claim sample yet. Run `python scripts/07_make_claim_sample.py` first.")
    st.stop()

sample = pd.read_csv(SAMPLE)
reviews = pd.read_csv(PROCESSED / "reviews.csv", keep_default_na=False).set_index("review_id")
products = pd.read_csv(PROCESSED / "products.csv").set_index("product_id")
judged = pd.read_csv(OUT, keep_default_na=False) if OUT.exists() else pd.DataFrame(columns=COLS)
done = set(judged["item_id"])


def first_open():
    left = [i for i, c in enumerate(sample["item_id"]) if c not in done]
    return left[0] if left else len(sample) - 1


if "idx" not in st.session_state:
    st.session_state.idx = first_open()

with st.sidebar:
    st.markdown("**Verdicts**")
    st.markdown("- **supported**: the reviews below clearly say it. *Several / many / most* needs at least two reviews.\n"
                "- **partially supported**: part is backed, but it overstates how many say it or adds details.\n"
                "- **unsupported**: no review says it, or they contradict it.")
    st.caption("Judge only from the reviews shown, not from what you know about the laptop.")
    if st.button("Jump to first unjudged"):
        st.session_state.idx = first_open()
        st.rerun()

idx = st.session_state.idx
row = sample.iloc[idx]
prev = judged[judged["item_id"] == row["item_id"]]
prev = prev.iloc[0] if len(prev) else None

st.progress(len(done) / len(sample), text=f"{len(done)} of {len(sample)} claims judged")
st.caption(f"Claim {idx + 1} of {len(sample)} · {products.loc[row['product_id'], 'short_name']}"
           + (" · already judged" if prev is not None else ""))
st.markdown(f"### “{row['claim']}”")

with st.form(key=f"claim-{row['item_id']}"):
    verdict = st.radio("Is the claim supported by the reviews below?", VERDICTS, horizontal=True,
                       format_func=lambda v: v.replace("_", " "),
                       index=VERDICTS.index(prev["human_verdict"]) if prev is not None else None)
    note = st.text_input("Note (optional)", value=prev["note"] if prev is not None else "")
    saved = st.form_submit_button("Save and next", type="primary")

if saved:
    if verdict is None:
        st.error("Pick a verdict first.")
    else:
        judged = pd.concat([judged[judged["item_id"] != row["item_id"]],
                            pd.DataFrame([{"item_id": row["item_id"], "human_verdict": verdict, "note": note}])])
        judged[COLS].to_csv(OUT, index=False)
        done.add(row["item_id"])
        st.session_state.idx = first_open() if len(done) < len(sample) else idx
        st.rerun()

left, right = st.columns(2)
if left.button("◀ Previous", disabled=idx == 0):
    st.session_state.idx = idx - 1
    st.rerun()
if right.button("Next ▶", disabled=idx == len(sample) - 1):
    st.session_state.idx = idx + 1
    st.rerun()
if len(done) == len(sample):
    st.success("All claims judged. Thank you!")

st.divider()
st.markdown("**Evidence reviews**")
for rid in json.loads(row["evidence_ids"]):
    r = reviews.loc[rid]
    with st.container(border=True):
        st.markdown(f"`{rid}` **{r['title']}**")
        st.write(r["text"])
