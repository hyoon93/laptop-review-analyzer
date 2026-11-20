"""Stage 4: product summaries, in the three conditions of the one comparison.

Both generation runs get the same task: the product name, the same list of top aspects
(by mention count from stage 2), the same number of review snippets, and the same target
number of claims. Only the evidence differs:

1. ungrounded:         random review snippets, no review IDs, no citations
2. grounded:           snippets retrieved per aspect from ChromaDB, labeled with review IDs;
                       every claim must cite the IDs that support it
3. grounded_verified:  (2) after the claim checker (pipeline/verify.py): supported claims
                       are kept, partially supported ones are rewritten to match their
                       evidence and re-checked, and anything not fully supported is dropped.

Each summary is a list of short, single-fact claims, so claims can be checked one by one.
"""

import random

import pandas as pd

from . import index, verify
from .aspects import CATEGORIES
from .config import PROCESSED
from .llm import chat_json

SNIPPETS_PER_ASPECT = 6
N_ASPECTS = 6
ASPECT_ENUM = [c for c in CATEGORIES if c != "other"] + ["overall"]

TASK = """You write a short summary of what Amazon customers say about one laptop, as a list of claims.

Rules:
- Each claim is one sentence stating one fact about what reviewers report (for example "Several reviewers say the battery lasts about 6 hours.").
- Cover each listed aspect with 1-3 claims, then add 1-2 "overall" claims.
- Describe what reviewers say, not the product specification.
- Say when reviewers disagree, and do not overstate how many reviewers say something."""

GROUNDED_RULES = """
- Use only the review snippets below. Do not add facts from outside them.
- Each claim must cite the review IDs (for example "L01-023") of the snippets that support it, 1-4 IDs. Cite only IDs that appear below.
- If the snippets do not give enough information about an aspect, write fewer claims for it."""


def _schema(with_citations):
    claim = {"aspect": {"type": "string", "enum": ASPECT_ENUM}, "claim": {"type": "string"}}
    if with_citations:
        claim["citations"] = {"type": "array", "items": {"type": "string"}}
    return {
        "name": "review_summary",
        "schema": {
            "type": "object",
            "properties": {"claims": {"type": "array", "items": {
                "type": "object", "properties": claim,
                "required": list(claim), "additionalProperties": False}}},
            "required": ["claims"],
            "additionalProperties": False,
        },
    }


def load_tables():
    products = pd.read_csv(PROCESSED / "products.csv").set_index("product_id")
    asp = pd.read_csv(PROCESSED / "aspects.csv", keep_default_na=False)
    return products, asp


def top_aspects(asp, product_id, n=N_ASPECTS):
    counts = asp[(asp["product_id"] == product_id) & (asp["category"] != "other")]["category"].value_counts()
    return counts.index[:n].tolist()


def grounded_evidence(product_id, aspects, k=SNIPPETS_PER_ASPECT):
    """Per aspect, the k reviews whose best chunk is closest to the aspect description."""
    return {a: index.search(f"{a.replace('_', ' ')}: {CATEGORIES[a]}", product_id=product_id, k=k)
            for a in aspects}


def random_evidence(product_id, n, seed=661):
    """n random chunks from the product's reviews (same snippet budget as grounded)."""
    col = index.collection()
    got = col.get(where={"product_id": product_id}, include=["documents"])
    docs = sorted(zip(got["ids"], got["documents"]))
    return [d for _, d in random.Random(seed).sample(docs, min(n, len(docs)))]


def _header(products, product_id, aspects):
    p = products.loc[product_id]
    return f"Laptop: {p['short_name']} ({p['title'][:150]})\nAspects to cover: {', '.join(aspects)}"


def ungrounded(product_id, products, aspects, n_snippets):
    snippets = random_evidence(product_id, n_snippets)
    body = "\n".join(f"- {s}" for s in snippets)
    messages = [{"role": "system", "content": TASK},
                {"role": "user", "content": f"{_header(products, product_id, aspects)}\n\nReview snippets:\n{body}"}]
    out = chat_json(messages, _schema(False), stage="summary_ungrounded")
    return [{**c, "citations": []} for c in out["claims"]]


def grounded(product_id, products, aspects, evidence):
    blocks = []
    for a, hits in evidence.items():
        blocks.append(f"## {a}\n" + "\n".join(f"[{h['review_id']}] {h['chunk']}" for h in hits))
    messages = [{"role": "system", "content": TASK + GROUNDED_RULES},
                {"role": "user", "content": f"{_header(products, product_id, aspects)}\n\nReview snippets:\n\n"
                                            + "\n\n".join(blocks)}]
    return chat_json(messages, _schema(True), stage="summary_grounded")["claims"]


def summarize_product(product_id, products=None, asp=None, reviews=None):
    """All three conditions for one product, as a list of claim dicts (one row per claim).

    ungrounded:         the claims of the ungrounded run
    grounded:           the claims of the grounded run, with the claim checker's verdict
    grounded_verified:  every grounded claim the checker found supported, plus every
                        partially supported claim that was rewritten to match its evidence
                        (verify.repair) and then passed a second check; the rest is dropped
                        (grounded rows carry kept = True/False)
    """
    if products is None or asp is None:
        products, asp = load_tables()
    aspects = top_aspects(asp, product_id)
    evidence = grounded_evidence(product_id, aspects)
    allowed = {h["review_id"] for hits in evidence.values() for h in hits}
    n_snippets = sum(len(h) for h in evidence.values())

    rows = [{**c, "condition": "ungrounded"} for c in ungrounded(product_id, products, aspects, n_snippets)]
    g = grounded(product_id, products, aspects, evidence)
    checks = verify.check_claims(g, product_id, allowed, reviews=reviews) if g else []
    partial = [i for i, chk in enumerate(checks) if chk["verdict"] == "partially_supported"]
    fixes = verify.repair_claims([g[i] for i in partial], [checks[i] for i in partial], reviews) if partial else []
    rechecks = verify.check_claims(fixes, product_id, allowed, reviews=reviews) if fixes else []
    fixed = {i: (f, chk) for i, f, chk in zip(partial, fixes, rechecks)}

    for i, (c, chk) in enumerate(zip(g, checks)):
        final = None
        if chk["verdict"] == "supported":
            final = ({**c, **chk}, False)
        elif i in fixed and fixed[i][1]["verdict"] == "supported":
            final = ({"aspect": c["aspect"], **fixed[i][0], **fixed[i][1]}, True)
        rows.append({**c, "condition": "grounded", **chk, "kept": final is not None,
                     "repaired": bool(final and final[1])})
        if final:
            rows.append({**final[0], "condition": "grounded_verified", "kept": True, "repaired": final[1],
                         "original_claim": c["claim"] if final[1] else ""})
    for i, r in enumerate(rows):
        r.update(product_id=product_id, claim_id=f"{product_id}-{r['condition'][0].upper()}"
                 f"{'V' if r['condition'] == 'grounded_verified' else ''}{i:02d}",
                 top_aspects=",".join(aspects))
    return rows
