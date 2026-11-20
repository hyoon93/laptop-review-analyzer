"""Laptop Review Analyzer: the interactive app (streamlit run app/app.py).

Everything shown for the 8 products comes from the offline batch pipeline
(data/processed/*.csv), so browsing costs nothing. Two tabs call gpt-4.1-mini live:
"Ask the reviews" (retrieval + cited answer + claim check) and "Analyze a review".
"""

import html
import json
import sys
from datetime import date
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pipeline.aspects import CATEGORIES  # noqa: E402
from pipeline.config import LABELS, MODEL, PROCESSED  # noqa: E402
from pipeline.examples import EXAMPLE_QUESTIONS, EXAMPLE_REVIEWS  # noqa: E402

st.set_page_config(page_title="Laptop Review Analyzer", page_icon="💻", layout="wide")

# Diverging pair for polarity (blue = positive, red = negative, gray = neutral).
COLORS = {"positive": "#2a78d6", "neutral": "#a8a69e", "negative": "#e34948"}
TINT = {"positive": "rgba(42,120,214,.20)", "neutral": "rgba(140,138,130,.22)",
        "negative": "rgba(227,73,72,.20)"}
SERIES = ["#2a78d6", "#eb6834"]  # categorical slots 1-2 for product A / B
LIVE_PER_SESSION = 12
LIVE_PER_DAY = 200


# ---------------------------------------------------------------- data
@st.cache_data
def load():
    products = pd.read_csv(PROCESSED / "products.csv")
    reviews = pd.read_csv(PROCESSED / "reviews.csv", keep_default_na=False)
    reviews["date"] = pd.to_datetime(reviews["date"]).dt.date
    reviews["verified_purchase"] = reviews["verified_purchase"].astype(str).str.lower() == "true"
    sent = PROCESSED / "sentiment_llm.csv"
    if sent.exists():
        s = pd.read_csv(sent)
        reviews = reviews.merge(s, on="review_id", how="left")
    else:
        reviews["sentiment"], reviews["mixed"] = None, False
    reviews["mixed"] = reviews["mixed"].astype(str).str.lower() == "true"
    star_label = reviews["rating"].map({1: "negative", 2: "negative", 3: "neutral", 4: "positive", 5: "positive"})
    reviews["disagree"] = ((reviews["rating"] >= 4) & (reviews["sentiment"] == "negative")) | \
                          ((reviews["rating"] <= 2) & (reviews["sentiment"] == "positive"))
    reviews["star_label"] = star_label
    asp_path = PROCESSED / "aspects.csv"
    asp = pd.read_csv(asp_path, keep_default_na=False) if asp_path.exists() else pd.DataFrame(
        columns=["review_id", "product_id", "rating", "term", "category", "polarity", "evidence", "evidence_found"])
    summ_path = PROCESSED / "summaries.csv"
    summ = pd.read_csv(summ_path, keep_default_na=False) if summ_path.exists() else None
    if summ is not None:
        for col in ["citations", "supporting_ids", "invalid_citations"]:
            summ[col] = summ[col].map(lambda v: json.loads(v) if v else [])
        for col in ["kept", "repaired"]:
            summ[col] = summ[col].astype(str).str.lower() == "true" if col in summ else False
    return products.set_index("product_id"), reviews, asp, summ


products, reviews, aspects, summaries = load()


@st.cache_resource(show_spinner="Preparing the search index (first start only)...")
def search_index():
    from pipeline import index

    index.ensure(reviews[["review_id", "product_id", "rating", "title", "text"]])
    return index


search_index()
by_id = reviews.set_index("review_id")


# ---------------------------------------------------------------- helpers
def stars(n):
    return "★" * int(n) + "☆" * (5 - int(n))


def badge(label, color):
    return (f"<span style='background:{color};color:white;border-radius:4px;padding:1px 6px;"
            f"font-size:0.8em;margin-right:4px'>{html.escape(label)}</span>")


def highlight(text, spans):
    """Escape text and wrap each evidence span (case-insensitive exact match) in <mark>."""
    low, marks = text.lower(), []
    for ev, pol in spans:
        i = low.find(ev.lower().strip()) if ev.strip() else -1
        if i >= 0:
            marks.append((i, i + len(ev.strip()), pol))
    marks.sort()
    out, pos = [], 0
    for s, e, pol in marks:
        if s < pos:
            continue
        out.append(html.escape(text[pos:s]))
        out.append(f"<mark style='background:{TINT[pol]};color:inherit;padding:0 2px;border-radius:3px'>"
                   f"{html.escape(text[s:e])}</mark>")
        pos = e
    out.append(html.escape(text[pos:]))
    return "".join(out)


def review_card(rid, extra=None):
    r = by_id.loc[rid]
    a = aspects[aspects["review_id"] == rid]
    head = f"**{stars(r['rating'])}** &nbsp; `{rid}` &nbsp; {r['date']}"
    if r["verified_purchase"]:
        head += " &nbsp; verified purchase"
    tags = ""
    if isinstance(r["sentiment"], str):
        tags += badge(r["sentiment"], COLORS[r["sentiment"]])
    if r["mixed"]:
        tags += badge("mixed", "#6b6a64")
    if r["disagree"]:
        tags += badge("stars disagree with text", "#8a5a00")
    body = highlight(r["text"], list(zip(a["evidence"], a["polarity"])))
    asp_line = " · ".join(f"<span style='color:{COLORS[p]}'>{html.escape(c.replace('_', ' '))}"
                          f"{(': ' + html.escape(t)) if t else ''}</span>"
                          for c, p, t in zip(a["category"], a["polarity"], a["term"]))
    with st.container(border=True):
        st.markdown(head + (f" &nbsp; {extra}" if extra else ""), unsafe_allow_html=True)
        st.markdown(f"{tags}**{html.escape(r['title'])}**", unsafe_allow_html=True)
        st.markdown(f"<div style='line-height:1.55'>{body}</div>", unsafe_allow_html=True)
        if asp_line:
            st.markdown(f"<div style='font-size:0.85em;margin-top:6px'>Aspects: {asp_line}</div>",
                        unsafe_allow_html=True)


def sentiment_bar(df, y, order, height=260):
    """Horizontal stacked bars: count of each sentiment per row of y."""
    fig = go.Figure()
    for s in ["positive", "neutral", "negative"]:
        counts = df[df["sentiment"] == s].groupby(y).size().reindex(order, fill_value=0)
        fig.add_bar(y=[str(o) for o in order], x=counts.values, name=s, orientation="h",
                    marker=dict(color=COLORS[s], line=dict(width=1, color="rgba(255,255,255,0.9)")),
                    hovertemplate=f"%{{y}}: %{{x}} {s}<extra></extra>")
    fig.update_layout(barmode="stack", height=height, margin=dict(l=10, r=10, t=10, b=10),
                      legend=dict(orientation="h", y=1.12, x=0), bargap=0.35,
                      xaxis=dict(title="reviews", gridcolor="rgba(128,128,128,0.15)"))
    return fig


def aspect_chart(a, height=420):
    """Diverging bars: negative mentions to the left, positive to the right, per category."""
    if a.empty:
        return None, []
    g = a.groupby(["category", "polarity"]).size().unstack(fill_value=0)
    for p in ["positive", "negative", "neutral"]:
        if p not in g:
            g[p] = 0
    g["total"] = g.sum(axis=1)
    g = g.sort_values("total")
    cats = g.index.tolist()
    labels = [c.replace("_", " ") for c in cats]
    fig = go.Figure()
    fig.add_bar(y=labels, x=-g["negative"], name="negative", orientation="h", customdata=cats,
                marker=dict(color=COLORS["negative"]),
                hovertemplate="%{y}: %{text} negative<extra></extra>", text=g["negative"], textposition="none")
    fig.add_bar(y=labels, x=g["positive"], name="positive", orientation="h", customdata=cats,
                marker=dict(color=COLORS["positive"]),
                hovertemplate="%{y}: %{x} positive<extra></extra>")
    fig.update_layout(barmode="relative", height=height, margin=dict(l=10, r=10, t=10, b=10),
                      legend=dict(orientation="h", y=1.08, x=0), bargap=0.3,
                      xaxis=dict(title="mentions (negative ← | → positive)", zeroline=True,
                                 zerolinecolor="rgba(128,128,128,0.6)", gridcolor="rgba(128,128,128,0.15)"))
    return fig, cats


def live_allowed():
    """Per-session and per-day caps on live gpt-4.1-mini calls (the URL is public)."""
    day = _daily_counter()
    today = date.today().isoformat()
    if day["date"] != today:
        day.update(date=today, n=0)
    used = st.session_state.get("live_used", 0)
    if used >= LIVE_PER_SESSION:
        st.warning(f"This session has used its {LIVE_PER_SESSION} live requests. Reload the page to start a new one.")
        return False
    if day["n"] >= LIVE_PER_DAY:
        st.warning("The app has reached today's limit of live requests. Cached features still work.")
        return False
    return True


def count_live():
    st.session_state["live_used"] = st.session_state.get("live_used", 0) + 1
    _daily_counter()["n"] += 1


@st.cache_resource
def _daily_counter():
    return {"date": None, "n": 0}


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.title("💻 Laptop Review Analyzer")
    pid = st.selectbox("Laptop", products.index,
                       format_func=lambda p: f"{p} · {products.loc[p, 'short_name']}")
    p = products.loc[pid]
    st.caption(f"{p['type']} · ${p['price']:,.0f} · Amazon rating {p['average_rating_amazon']} "
               f"({p['rating_number_amazon']:,} ratings)")
    st.caption(f"{int(p['n_reviews_sampled'])} reviews analyzed (random sample of "
               f"{int(p['n_reviews_available']):,} with text)")
    with st.expander("How it works"):
        st.markdown(
            f"1. **Sentiment** and **aspect extraction** with {MODEL} (JSON-schema structured output; "
            "every aspect carries a verbatim evidence quote).\n"
            "2. **Embeddings** (all-MiniLM-L6-v2) of review chunks in **ChromaDB** for retrieval.\n"
            "3. **Grounded summary**: per-aspect retrieval (RAG); every claim cites review IDs.\n"
            "4. **Claim checker**: each claim is checked against the reviews it cites; unsupported claims are removed.\n\n"
            "Reviews: Amazon Reviews 2023 (McAuley Lab), Electronics → Laptops, English only. "
            "All generated text is labeled AI-generated.")

prod_reviews = reviews[reviews["product_id"] == pid]
prod_aspects = aspects[aspects["product_id"] == pid]

tabs = st.tabs(["Overview", "Summary", "Reviews", "Ask the reviews", "Analyze a review", "Compare", "Evaluation"])

# ---------------------------------------------------------------- Overview
with tabs[0]:
    st.subheader(p["short_name"])
    st.caption(p["title"])
    n = len(prod_reviews)
    share = prod_reviews["sentiment"].value_counts(normalize=True)
    c = st.columns(5)
    c[0].metric("Reviews analyzed", n)
    c[1].metric("Positive", f"{share.get('positive', 0):.0%}")
    c[2].metric("Negative", f"{share.get('negative', 0):.0%}")
    c[3].metric("Mixed opinions", f"{prod_reviews['mixed'].mean():.0%}",
                help="Reviews with at least one clear positive and one clear negative opinion.")
    c[4].metric("Stars disagree with text", int(prod_reviews["disagree"].sum()),
                help="4–5★ reviews whose text is negative, or 1–2★ reviews whose text is positive.")

    left, right = st.columns([2, 3])
    with left:
        st.markdown("**Text sentiment by star rating**")
        st.plotly_chart(sentiment_bar(prod_reviews, "rating", [5, 4, 3, 2, 1]), width="stretch")
        st.caption("Stars are a weak label: 3★ reviews are rarely neutral, and some 4–5★ reviews are complaints.")
    with right:
        st.markdown("**What reviewers talk about** (click a bar to see those reviews)")
        fig, cats = aspect_chart(prod_aspects)
        picked = None
        if fig is not None:
            event = st.plotly_chart(fig, width="stretch", on_select="rerun",
                                    selection_mode="points", key=f"aspects-{pid}")
            pts = (event or {}).get("selection", {}).get("points", []) if event else []
            if pts:
                picked = pts[0].get("customdata")
        else:
            st.info("Aspect results are not available yet.")
    if fig is not None:
        options = ["(choose an aspect)"] + sorted(prod_aspects["category"].unique())
        idx = options.index(picked) if picked in options else 0
        choice = st.selectbox("Aspect", options, index=idx, key=f"aspect-pick-{pid}-{picked}")
        if choice != options[0]:
            sub = prod_aspects[prod_aspects["category"] == choice]
            pol = sub["polarity"].value_counts()
            st.markdown(f"**{choice.replace('_', ' ')}**: {pol.get('positive', 0)} positive, "
                        f"{pol.get('negative', 0)} negative, {pol.get('neutral', 0)} neutral mentions "
                        f"in {sub['review_id'].nunique()} reviews")
            polarity = st.radio("Show", ["negative", "positive", "neutral"], horizontal=True,
                                key=f"pol-{pid}-{choice}")
            ids = sub[sub["polarity"] == polarity]["review_id"].drop_duplicates().tolist()
            for rid in ids[:15]:
                review_card(rid)
            if len(ids) > 15:
                st.caption(f"Showing 15 of {len(ids)}. Use the Reviews tab for the full list.")

# ---------------------------------------------------------------- Summary
with tabs[1]:
    st.subheader("What reviewers say")
    if summaries is None or summaries[summaries["product_id"] == pid].empty:
        st.info("Summaries have not been generated yet.")
    else:
        s = summaries[summaries["product_id"] == pid]
        mode = st.radio("Version", ["Grounded + verified", "Grounded (before the claim check)",
                                    "Ungrounded (no retrieval, no citations)"], horizontal=True,
                        help="The three conditions of the project's comparison.")
        if mode.startswith("Ungrounded"):
            shown = s[s["condition"] == "ungrounded"]
            st.caption("Baseline: the model saw random review snippets without IDs and wrote claims "
                       "with no citations. Nothing was checked.")
        elif mode.startswith("Grounded +"):
            shown = s[s["condition"] == "grounded_verified"]
            st.caption("Each claim was checked against the full reviews it cites. Partially supported claims "
                       "were rewritten to match the evidence (✏️) and checked again; anything not fully "
                       "supported was removed.")
        else:
            shown = s[s["condition"] == "grounded"]
            st.caption("Before the check: ✅ supported, ⚠️ partially supported, ❌ unsupported (per the claim checker).")
        st.caption("🤖 AI-generated from customer reviews. Click a claim to see the reviews it cites.")
        order = s["top_aspects"].iloc[0].split(",") + ["overall"]
        for asp_name in order + sorted(set(shown["aspect"]) - set(order)):
            group = shown[shown["aspect"] == asp_name]
            if group.empty:
                continue
            st.markdown(f"##### {asp_name.replace('_', ' ').capitalize()}")
            for r in group.itertuples():
                mark = ""
                if r.condition == "grounded":
                    mark = {"supported": "✅", "partially_supported": "⚠️", "unsupported": "❌"}.get(r.verdict, "")
                elif r.condition == "grounded_verified" and r.repaired:
                    mark = "✏️"
                cites = " ".join(f"`{c}`" for c in r.citations)
                with st.expander(f"{mark} {r.claim}  {cites}".strip()):
                    if r.condition == "grounded_verified" and r.repaired:
                        st.caption(f"Rewritten from: “{r.original_claim}”")
                    if r.condition != "ungrounded":
                        st.caption(f"Claim check: {r.verdict.replace('_', ' ')}: {r.reason}")
                        if r.invalid_citations:
                            st.caption(f"Invalid citations: {', '.join(r.invalid_citations)}")
                    if not r.citations:
                        st.caption("No citations: this version cannot be traced back to specific reviews.")
                    for cid in r.citations:
                        if cid in by_id.index:
                            review_card(cid)
        if mode.startswith("Grounded +"):
            removed = s[(s["condition"] == "grounded") & ~s["kept"]]
            if len(removed):
                with st.expander(f"Removed by the claim checker ({len(removed)})"):
                    for r in removed.itertuples():
                        st.markdown(f"❌ ~~{r.claim}~~ " + " ".join(f"`{c}`" for c in r.citations))
                        st.caption(r.reason)

# ---------------------------------------------------------------- Reviews
with tabs[2]:
    st.subheader("Review explorer")
    query = st.text_input("Semantic search (runs locally, no API cost)",
                          placeholder="e.g. screen flickering, good for college, overheating while gaming")
    f = st.columns(4)
    star_sel = f[0].multiselect("Stars", [5, 4, 3, 2, 1], default=[5, 4, 3, 2, 1])
    sent_sel = f[1].multiselect("Text sentiment", ["positive", "neutral", "negative"],
                                default=["positive", "neutral", "negative"])
    asp_sel = f[2].selectbox("Mentions aspect", ["any"] + list(CATEGORIES))
    pol_sel = f[3].selectbox("Aspect polarity", ["any", "positive", "negative", "neutral"])
    g = st.columns(4)
    verified_only = g[0].checkbox("Verified purchases only")
    disagree_only = g[1].checkbox("Stars disagree with text")
    mixed_only = g[2].checkbox("Mixed opinions only")
    dmin, dmax = prod_reviews["date"].min(), prod_reviews["date"].max()
    dates = g[3].date_input("Date range", (dmin, dmax), min_value=dmin, max_value=dmax)

    df = prod_reviews[prod_reviews["rating"].isin(star_sel) & prod_reviews["sentiment"].isin(sent_sel)]
    if verified_only:
        df = df[df["verified_purchase"]]
    if disagree_only:
        df = df[df["disagree"]]
    if mixed_only:
        df = df[df["mixed"]]
    if isinstance(dates, tuple) and len(dates) == 2:
        df = df[(df["date"] >= dates[0]) & (df["date"] <= dates[1])]
    if asp_sel != "any" or pol_sel != "any":
        a = prod_aspects
        if asp_sel != "any":
            a = a[a["category"] == asp_sel]
        if pol_sel != "any":
            a = a[a["polarity"] == pol_sel]
        df = df[df["review_id"].isin(set(a["review_id"]))]

    scores = {}
    if query.strip():
        from pipeline import index

        hits = index.search(query, product_id=pid, k=60)
        scores = {h["review_id"]: h["score"] for h in hits}
        df = df[df["review_id"].isin(scores)].copy()
        df["score"] = df["review_id"].map(scores)
        df = df.sort_values("score", ascending=False)
    else:
        df = df.sort_values("date", ascending=False)
    st.caption(f"{len(df)} reviews match" + (" (top 60 search results, then filters)" if scores else ""))
    page_size = 20
    pages = max(1, -(-len(df) // page_size))
    page = st.number_input("Page", 1, pages, 1) if pages > 1 else 1
    for rid in df["review_id"].iloc[(page - 1) * page_size: page * page_size]:
        review_card(rid, extra=f"similarity {scores[rid]:.2f}" if rid in scores else None)

# ---------------------------------------------------------------- Ask the reviews
with tabs[3]:
    st.subheader(f"Ask the reviews of the {p['short_name']}")
    st.caption(f"Live: retrieves the 8 most relevant reviews, {MODEL} answers with citations, then every "
               "claim is checked against the full cited reviews. If the reviews do not answer the question, "
               "it says so instead of guessing.")
    ex = st.columns(len(EXAMPLE_QUESTIONS))
    for i, q in enumerate(EXAMPLE_QUESTIONS):
        if ex[i].button(q, key=f"exq-{i}", width="stretch"):
            st.session_state["question"] = q
    question = st.text_input("Your question", key="question", placeholder="Is the keyboard comfortable?")
    if st.button("Ask", type="primary", disabled=not question.strip()) and live_allowed():
        from pipeline import qa

        with st.spinner("Retrieving reviews and checking the answer..."):
            try:
                res = qa.answer(question.strip(), pid, p["short_name"])
                count_live()
                st.session_state["qa_result"] = (pid, question.strip(), res)
            except Exception as e:  # rate limit or network error: keep the app usable
                st.error(f"The live model call failed ({type(e).__name__}). Cached tabs still work.")
    if st.session_state.get("qa_result") and st.session_state["qa_result"][0] == pid:
        _, q, res = st.session_state["qa_result"]
        st.markdown(f"**Q: {q}**")
        st.caption("🤖 AI-generated answer")
        if not res["answerable"]:
            st.info("The retrieved reviews don't answer this question, so no answer is given. "
                    "(The model was told not to use outside knowledge about the product.)")
        for r in res["kept"]:
            mark = "✅" if r["verdict"] == "supported" else "⚠️"
            with st.expander(f"{mark} {r['claim']}  " + " ".join(f"`{c}`" for c in r["citations"])):
                st.caption(f"Claim check: {r['verdict'].replace('_', ' ')}: {r['reason']}")
                for cid in r["citations"]:
                    if cid in by_id.index:
                        review_card(cid)
        if res["removed"]:
            with st.expander(f"Removed by the claim checker ({len(res['removed'])})"):
                for r in res["removed"]:
                    st.markdown(f"❌ ~~{r['claim']}~~")
                    st.caption(r["reason"])
        with st.expander("Retrieved review snippets"):
            for h in res["hits"]:
                st.markdown(f"`{h['review_id']}` (similarity {h['score']:.2f}): {h['chunk']}")

# ---------------------------------------------------------------- Analyze a review
with tabs[4]:
    st.subheader("Analyze a new review")
    st.caption(f"Live: the same sentiment and aspect prompts the pipeline uses ({MODEL}), next to the VADER baseline.")
    ex = st.columns(len(EXAMPLE_REVIEWS))
    for i, (name, (t, x)) in enumerate(EXAMPLE_REVIEWS.items()):
        if ex[i].button(name, key=f"exr-{i}", width="stretch"):
            st.session_state["rev_title"], st.session_state["rev_text"] = t, x
    title = st.text_input("Title (optional)", key="rev_title")
    text = st.text_area("Review text", key="rev_text", height=140)
    if st.button("Analyze", type="primary", disabled=len(text.split()) < 3) and live_allowed():
        from pipeline import aspects as asp_mod
        from pipeline import sentiment as sent_mod

        with st.spinner("Analyzing..."):
            try:
                s = sent_mod.classify(title, text, stage="live_sentiment")
                a = asp_mod.extract(text, title or None, stage="live_aspects")
                count_live()
                st.session_state["rev_result"] = (title, text, s, a)
            except Exception as e:
                st.error(f"The live model call failed ({type(e).__name__}).")
    if st.session_state.get("rev_result"):
        title_r, text_r, s, a = st.session_state["rev_result"]
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

        compound = SentimentIntensityAnalyzer().polarity_scores(f"{title_r}. {text_r}")["compound"]
        vader = "positive" if compound >= 0.05 else "negative" if compound <= -0.05 else "neutral"
        c = st.columns(3)
        c[0].markdown(f"**{MODEL}** 🤖<br>{badge(s['sentiment'], COLORS[s['sentiment']])}"
                      + (badge("mixed", "#6b6a64") if s["mixed"] else ""), unsafe_allow_html=True)
        c[1].markdown(f"**VADER baseline**<br>{badge(vader, COLORS[vader])} compound {compound:+.2f}",
                      unsafe_allow_html=True)
        found = sum(e["evidence"].lower().strip() in f"{title_r} {text_r}".lower() for e in a)
        c[2].markdown(f"**Evidence quotes found verbatim**<br>{found} of {len(a)}", unsafe_allow_html=True)
        st.markdown(f"<div style='line-height:1.6'>{highlight(text_r, [(e['evidence'], e['polarity']) for e in a])}</div>",
                    unsafe_allow_html=True)
        if a:
            st.dataframe(pd.DataFrame(a)[["category", "term", "polarity", "evidence"]],
                         hide_index=True, width="stretch")
        else:
            st.caption("No specific aspects found.")

# ---------------------------------------------------------------- Compare
with tabs[5]:
    st.subheader("Compare two laptops")
    c = st.columns(2)
    others = [x for x in products.index if x != pid]
    pa = c[0].selectbox("Laptop A", products.index, index=list(products.index).index(pid),
                        format_func=lambda x: products.loc[x, "short_name"], key="cmp-a")
    pb = c[1].selectbox("Laptop B", products.index, index=list(products.index).index(others[0]),
                        format_func=lambda x: products.loc[x, "short_name"], key="cmp-b")
    rows = []
    for x in (pa, pb):
        r = reviews[reviews["product_id"] == x]
        a = aspects[aspects["product_id"] == x]
        for cat, grp in a.groupby("category"):
            pos, neg = (grp["polarity"] == "positive").sum(), (grp["polarity"] == "negative").sum()
            rows.append({"product": products.loc[x, "short_name"], "aspect": cat, "mentions": len(grp),
                         "share_positive": pos / (pos + neg) if pos + neg else None,
                         "per_100_reviews": 100 * grp["review_id"].nunique() / len(r)})
    cmp = pd.DataFrame(rows)
    m = st.columns(2)
    for col, x in zip(m, (pa, pb)):
        r = reviews[reviews["product_id"] == x]
        sh = r["sentiment"].value_counts(normalize=True)
        col.metric(products.loc[x, "short_name"], f"{sh.get('positive', 0):.0%} positive",
                   f"{sh.get('negative', 0):.0%} negative", delta_color="off")
    if not cmp.empty:
        top = (cmp.groupby("aspect")["mentions"].sum().sort_values(ascending=False).index[:10]).tolist()
        fig = go.Figure()
        for color, x in zip(SERIES, (pa, pb)):
            d = cmp[(cmp["product"] == products.loc[x, "short_name"])].set_index("aspect").reindex(top)
            fig.add_bar(x=[t.replace("_", " ") for t in top], y=d["share_positive"], name=products.loc[x, "short_name"],
                        marker=dict(color=color), customdata=d["mentions"].fillna(0),
                        hovertemplate="%{x}: %{y:.0%} positive (%{customdata} mentions)<extra></extra>")
        fig.update_layout(barmode="group", height=380, margin=dict(l=10, r=10, t=30, b=10), bargap=0.3,
                          bargroupgap=0.08, yaxis=dict(title="share of opinions that are positive",
                                                       tickformat=".0%", range=[0, 1],
                                                       gridcolor="rgba(128,128,128,0.15)"),
                          legend=dict(orientation="h", y=1.12, x=0))
        st.plotly_chart(fig, width="stretch")
        st.caption("Share positive = positive / (positive + negative) aspect mentions, for the 10 most discussed aspects.")
        with st.expander("Table"):
            st.dataframe(cmp.pivot_table(index="aspect", columns="product",
                                         values=["share_positive", "per_100_reviews"]).round(2),
                         width="stretch")

# ---------------------------------------------------------------- Evaluation
with tabs[6]:
    st.subheader("Evaluation")
    st.markdown("Results on the **application data** (Amazon laptop reviews, my hand labels) and on the "
                "**external benchmark** (SemEval-2014 Laptops) are reported separately and never averaged.")

    st.markdown("#### A. Overall sentiment vs. 300 hand-labeled reviews (application data)")
    ev = PROCESSED / "eval_sentiment.csv"
    if ev.exists():
        e = pd.read_csv(ev)
        e["system"] = e["system"].replace({"llm": f"{MODEL} (zero-shot)", "stars": "star rating → label",
                                           "vader": "VADER", "distilbert": "DistilBERT SST-2"})
        cols = {"system": "System", "accuracy": "Accuracy", "macro_f1": "Macro-F1",
                "f1_positive": "F1 pos", "f1_neutral": "F1 neu", "f1_negative": "F1 neg",
                "accuracy_mixed": "Acc. on mixed", "accuracy_star_weighted": "Acc. (dataset star mix)"}
        t = e[list(cols)].rename(columns=cols)
        t["95% CI (acc.)"] = [f"{lo:.2f}–{hi:.2f}" for lo, hi in zip(e["accuracy_ci_low"], e["accuracy_ci_high"])]
        st.dataframe(t.round(3), hide_index=True, width="stretch")
        labels = pd.read_csv(LABELS / "sentiment_labels.csv")
        st.caption(f"300 reviews, 60 per star level, labeled by me without seeing the stars "
                   f"({(labels['sentiment'] == 'negative').sum()} negative, {(labels['sentiment'] == 'positive').sum()} positive, "
                   f"{(labels['sentiment'] == 'neutral').sum()} neutral; {labels['mixed'].sum()} mixed).")
    else:
        st.info("Not computed yet.")

    ea = PROCESSED / "eval_aspects.json"
    if ea.exists():
        r = json.loads(ea.read_text())
        st.markdown("**Aspects (application data)**")
        q = r.get("aspect_quality")
        if q:
            c = st.columns(3)
            c[0].metric("Aspect precision", f"{q['precision']:.0%}", help=f"{q['aspects_extracted']} aspects in {q['reviews_judged']} reviews, judged by me")
            c[1].metric("Polarity accuracy", f"{q['polarity_accuracy']:.0%}")
            c[2].metric("Approx. recall", f"{q['approx_recall']:.0%}", help=f"{q['missed']} aspects I found that the model missed")
        st.caption(f"{r['aspects_total']:,} aspects extracted ({r['aspects_per_review']:.1f} per review); "
                   f"{r['evidence_verbatim']:.1%} of evidence quotes appear verbatim in the review. "
                   f"{r['high_star_with_negative_aspect']:.0%} of 4–5★ reviews contain at least one negative aspect, and "
                   f"{r['low_star_with_positive_aspect']:.0%} of 1–2★ reviews at least one positive aspect.")

    st.markdown("#### B. Aspect extraction on SemEval-2014 Laptops test (external benchmark)")
    sb = PROCESSED / "eval_semeval.json"
    if sb.exists():
        r = json.loads(sb.read_text())
        st.dataframe(pd.DataFrame([
            {"Match": "exact", **{k.capitalize(): round(v, 3) for k, v in r["exact"].items() if k != "tp"}},
            {"Match": "partial (shared word)", **{k.capitalize(): round(v, 3) for k, v in r["partial"].items() if k != "tp"}},
        ]), hide_index=True, width="stretch")
        st.caption(f"800 sentences, {r['n_gold']} gold aspect terms, {r['n_pred']} predicted. Aspect polarity accuracy "
                   f"on exact matches: {r['polarity_accuracy']:.3f} (n = {r['polarity_n']}, excluding 'conflict').")
    else:
        st.info("Not computed yet.")

    st.markdown("#### C. The comparison: ungrounded vs. grounded vs. grounded + verified summaries")
    sc = PROCESSED / "eval_summaries.csv"
    if sc.exists():
        e = pd.read_csv(sc)
        names = {"ungrounded": "Ungrounded", "grounded": "Grounded", "grounded_verified": "Grounded + verified"}
        fig = go.Figure()
        for v, color in [("supported", COLORS["positive"]), ("partially_supported", COLORS["neutral"]),
                         ("unsupported", COLORS["negative"])]:
            fig.add_bar(y=[names[c] for c in e["condition"]], x=e[v], name=v.replace("_", " "), orientation="h",
                        marker=dict(color=color, line=dict(width=1, color="rgba(255,255,255,0.9)")),
                        hovertemplate="%{y}: %{x:.1%} " + v.replace("_", " ") + "<extra></extra>")
        fig.update_layout(barmode="stack", height=220, margin=dict(l=10, r=10, t=10, b=10), bargap=0.35,
                          xaxis=dict(tickformat=".0%", range=[0, 1], gridcolor="rgba(128,128,128,0.15)"),
                          yaxis=dict(autorange="reversed"), legend=dict(orientation="h", y=1.2, x=0))
        st.plotly_chart(fig, width="stretch")
        t = e.rename(columns={"condition": "Condition", "claims": "Claims", "supported": "Supported",
                              "partially_supported": "Partially", "unsupported": "Unsupported",
                              "citation_precision": "Citation precision", "aspect_coverage": "Aspect coverage"})
        t["Condition"] = t["Condition"].map(names)
        st.dataframe(t.drop(columns=["claims_per_product"]).round(3), hide_index=True, width="stretch")
        st.caption("Judge: gpt-4.1 (evaluation only), 12 evidence reviews per claim gathered the same way for every "
                   "condition. Citation precision: share of cited reviews the judge says support the claim. "
                   "Aspect coverage: share of each laptop's top aspects with at least one supported claim.")
        kp = PROCESSED / "eval_judge_kappa.json"
        if kp.exists():
            k = json.loads(kp.read_text())
            st.markdown(f"**Is the judge reliable?** On {k['n']} claims I judged blind: agreement {k['agreement']:.0%}, "
                        f"Cohen's κ {k['kappa_3way']:.2f} (3 labels), {k['kappa_supported_vs_not']:.2f} (supported vs. not).")
            hp = PROCESSED / "eval_summaries_human.csv"
            if hp.exists():
                h = pd.read_csv(hp)
                h["condition"] = h["condition"].map(names)
                st.dataframe(h.round(3), hide_index=True, width="stretch")
    else:
        st.info("Not computed yet.")

    st.markdown("#### D. Cost and latency")
    from pipeline.llm import usage_table

    u = usage_table()
    if not u.empty:
        st.dataframe(u.round(4), hide_index=True, width="stretch")
        st.caption("From the call cache. Batch calls cost 50% less and have no per-call latency.")
