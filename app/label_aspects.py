"""Hand-judging page for aspect extraction (evaluation A, aspect precision and recall).

Run:  streamlit run app/label_aspects.py

For the 100 reviews flagged aspect_subset in data/labels/label_sample.csv (20 per star
level), shows the review and every aspect the model extracted (data/processed/aspects.csv).
For each aspect: is it really discussed? is the polarity right? Then list what was missed.
Saves immediately to data/labels/aspect_judgments.csv and data/labels/aspect_missed.csv.
"""

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline.config import LABELS, PROCESSED  # noqa: E402

OUT = LABELS / "aspect_judgments.csv"
MISSED = LABELS / "aspect_missed.csv"
COLS = ["review_id", "aspect_no", "category", "term", "polarity", "is_real", "polarity_ok"]
MISSED_COLS = ["review_id", "n_missed", "missed"]

st.set_page_config(page_title="Aspect judging", layout="centered")

sample = pd.read_csv(LABELS / "label_sample.csv")
sample = sample[sample["aspect_subset"].astype(str).str.lower() == "true"].reset_index(drop=True)
asp = pd.read_csv(PROCESSED / "aspects.csv", keep_default_na=False)
asp["aspect_no"] = asp.groupby("review_id").cumcount()
judged = pd.read_csv(OUT, keep_default_na=False) if OUT.exists() else pd.DataFrame(columns=COLS)
missed = pd.read_csv(MISSED, keep_default_na=False) if MISSED.exists() else pd.DataFrame(columns=MISSED_COLS)
done = set(missed["review_id"])


def first_open():
    left = [i for i, rid in enumerate(sample["review_id"]) if rid not in done]
    return left[0] if left else len(sample) - 1


if "idx" not in st.session_state:
    st.session_state.idx = first_open()

with st.sidebar:
    st.markdown("**For each extracted aspect**")
    st.markdown("- **Real aspect?** Does the review really give an opinion on (or mention) this?\n"
                "- **Polarity right?** Is positive / negative / neutral correct for *this aspect*?\n\n"
                "**Then** list aspects the model missed (one per line), or leave empty.")
    st.caption("Categories: performance, battery, display, keyboard_touchpad, build_design, portability, "
               "audio_camera, ports_connectivity, storage_memory, software_os, thermals_noise, reliability, "
               "price_value, seller_support, other")
    if st.button("Jump to first unjudged"):
        st.session_state.idx = first_open()
        st.rerun()

idx = st.session_state.idx
row = sample.iloc[idx]
rid = row["review_id"]
items = asp[asp["review_id"] == rid]
prev = judged[judged["review_id"] == rid].set_index("aspect_no") if len(judged) else pd.DataFrame()
prev_m = missed[missed["review_id"] == rid]

st.progress(len(done) / len(sample), text=f"{len(done)} of {len(sample)} reviews judged")
st.caption(f"Review {idx + 1} of {len(sample)} · {rid}" + (" · already judged" if rid in done else ""))
st.subheader(row["title"] if isinstance(row["title"], str) and row["title"] else "(no title)")
st.write(row["text"])
st.divider()

with st.form(key=f"aspects-{rid}"):
    answers = []
    if items.empty:
        st.info("The model extracted no aspects from this review.")
    for a in items.itertuples():
        st.markdown(f"**{a.aspect_no + 1}. {a.category.replace('_', ' ')}**"
                    + (f" · term: *{a.term}*" if a.term else " · (implied, no term)")
                    + f" · **{a.polarity}**")
        st.caption(f"Evidence: “{a.evidence}”")
        c = st.columns(2)
        p = prev.loc[a.aspect_no] if a.aspect_no in prev.index else None
        real = c[0].radio("Real aspect?", ["yes", "no"], horizontal=True, key=f"r-{rid}-{a.aspect_no}",
                          index=["yes", "no"].index(p["is_real"]) if p is not None else None)
        pol = c[1].radio("Polarity right?", ["yes", "no"], horizontal=True, key=f"p-{rid}-{a.aspect_no}",
                         index=["yes", "no"].index(p["polarity_ok"]) if p is not None and p["polarity_ok"] in ("yes", "no") else None)
        answers.append((a, real, pol))
    miss = st.text_area("Missed aspects (one per line, e.g. 'keyboard: negative')",
                        value=prev_m.iloc[0]["missed"] if len(prev_m) else "")
    saved = st.form_submit_button("Save and next", type="primary")

if saved:
    if any(real is None or (real == "yes" and pol is None) for _, real, pol in answers):
        st.error("Answer both questions for every aspect (polarity can be skipped if the aspect is not real).")
    else:
        rows = [{"review_id": rid, "aspect_no": a.aspect_no, "category": a.category, "term": a.term,
                 "polarity": a.polarity, "is_real": real, "polarity_ok": pol if real == "yes" else ""}
                for a, real, pol in answers]
        judged = pd.concat([judged[judged["review_id"] != rid], pd.DataFrame(rows, columns=COLS)])
        judged[COLS].to_csv(OUT, index=False)
        lines = [m for m in miss.splitlines() if m.strip()]
        missed = pd.concat([missed[missed["review_id"] != rid],
                            pd.DataFrame([{"review_id": rid, "n_missed": len(lines), "missed": "\n".join(lines)}])])
        missed[MISSED_COLS].to_csv(MISSED, index=False)
        done.add(rid)
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
    st.success("All 100 reviews judged. Thank you!")
