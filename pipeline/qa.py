"""Live feature: "ask the reviews". A question about one laptop is answered from its reviews.

1. Retrieve the K reviews whose chunks best match the question (local embeddings).
2. gpt-4.1-mini answers from those snippets only, as claims that cite review IDs, or says
   the reviews do not answer the question.
3. The claim checker (verify.py) checks every claim against the full cited reviews and
   the question; unsupported or off-topic claims are dropped and shown separately.

Edge case this is built for: a question the reviews never discuss ("does it have a
fingerprint reader?"). Retrieval still returns the nearest snippets, so the model must
notice that they do not answer the question instead of guessing from product knowledge.
"""

from . import index, verify
from .llm import chat_json

K = 8

SYSTEM = """You answer a shopper's question about one laptop using only the customer review snippets provided.

- Answer as a list of short claims, each citing the review IDs (for example "L01-023") of the snippets that support it.
- Every claim must answer the question asked. Do not add claims about other features.
- Use only the snippets. Do not use outside knowledge about the product, even if you know it.
- If no snippet speaks to the question, set answerable to false and return no claims. A feature not being mentioned is not evidence that the laptop lacks it.
- Mention disagreement between reviewers when there is some."""

SCHEMA = {
    "name": "review_answer",
    "schema": {
        "type": "object",
        "properties": {
            "answerable": {"type": "boolean"},
            "claims": {"type": "array", "items": {
                "type": "object",
                "properties": {"claim": {"type": "string"},
                               "citations": {"type": "array", "items": {"type": "string"}}},
                "required": ["claim", "citations"],
                "additionalProperties": False}},
        },
        "required": ["answerable", "claims"],
        "additionalProperties": False,
    },
}


def answer(question, product_id, product_name, k=K):
    """Returns {"answerable", "kept": [...], "removed": [...], "hits": [...]}."""
    hits = index.search(question, product_id=product_id, k=k)
    body = "\n".join(f"[{h['review_id']}] {h['chunk']}" for h in hits)
    messages = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": f"Laptop: {product_name}\nQuestion: {question}\n\nReview snippets:\n{body}"}]
    out = chat_json(messages, SCHEMA, stage="qa")
    claims = out["claims"] if out["answerable"] else []
    allowed = {h["review_id"] for h in hits}
    checks = verify.check_claims(claims, product_id, allowed, workers=4, question=question) if claims else []
    rows = [{**c, **chk} for c, chk in zip(claims, checks)]
    kept = [r for r in rows if r["verdict"] != "unsupported"]
    return {"answerable": bool(kept), "kept": kept,
            "removed": [r for r in rows if r["verdict"] == "unsupported"], "hits": hits}
