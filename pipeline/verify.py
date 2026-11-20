"""Stage 5: claim checker. Is each generated claim backed by the reviews it cites?

Used inside the app (gpt-4.1-mini) to filter grounded summaries and Q&A answers. It is NOT
the evaluation judge: evaluation uses a different model (gpt-4.1, eval/judge.py) and a
human-checked subset, so the grounded_verified condition is not graded by its own filter.

Verdicts
    supported            the cited reviews state the claim (a "several/many" claim needs 2+ reviews)
    partially_supported  part of the claim is backed, but it is overstated or adds details
    unsupported          not stated in the cited reviews, contradicted, or no valid citation
"""

from functools import lru_cache

import pandas as pd

from .config import PROCESSED
from .llm import chat_json, map_parallel

VERDICTS = ["supported", "partially_supported", "unsupported"]
MAX_WORDS = 400

SYSTEM = """You check whether a claim about a laptop is supported by the customer reviews it cites.

Verdicts:
- supported: the cited reviews clearly state what the claim says. A claim about "several", "many" or "most" reviewers needs at least two cited reviews that say it.
- partially_supported: some of the claim is backed, but it overstates how many reviewers say it, or adds details the reviews do not give.
- unsupported: the cited reviews do not say it, or contradict it.

Judge only from the review text shown, not from what you know about the product.
Also list which cited review IDs actually support the claim."""

SCHEMA = {
    "name": "claim_check",
    "schema": {
        "type": "object",
        "properties": {
            "verdict": {"type": "string", "enum": VERDICTS},
            "reason": {"type": "string"},
            "supporting_ids": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["verdict", "reason", "supporting_ids"],
        "additionalProperties": False,
    },
}


@lru_cache(maxsize=1)
def _reviews():
    df = pd.read_csv(PROCESSED / "reviews.csv", keep_default_na=False)
    return df.set_index("review_id")


def review_text(review_id, reviews=None, max_words=MAX_WORDS):
    r = (reviews if reviews is not None else _reviews()).loc[review_id]
    words = f"{r['title']}. {r['text']}".split()
    return " ".join(words[:max_words]) + (" [...]" if len(words) > max_words else "")


QA_RULES = """

This claim is part of an answer to a shopper's question (given below). First decide answers_question: true only if the claim is about what the question asks (a claim about a different feature is false), and the claim does not conclude something from what the reviews do NOT mention (absence of a mention is not evidence)."""

QA_SCHEMA = {
    "name": "qa_claim_check",
    "schema": {
        "type": "object",
        "properties": {"answers_question": {"type": "boolean"}, **SCHEMA["schema"]["properties"]},
        "required": ["answers_question", *SCHEMA["schema"]["required"]],
        "additionalProperties": False,
    },
}


def build_messages(claim, cited_ids, reviews=None, question=None):
    body = "\n\n".join(f"[{rid}] {review_text(rid, reviews)}" for rid in cited_ids)
    system = SYSTEM + (QA_RULES if question else "")
    q = f"Question: {question}\n" if question else ""
    return [{"role": "system", "content": system},
            {"role": "user", "content": f"{q}Claim: {claim}\n\nCited reviews:\n\n{body}"}]


def check(claim, citations, product_id, allowed=None, reviews=None, question=None, **kw):
    """Check one claim. allowed: review IDs that were in the evidence shown to the generator.
    question: set for Q&A answers, so off-topic claims are rejected too."""
    known = (reviews if reviews is not None else _reviews())
    valid = [c for c in dict.fromkeys(citations)
             if c in known.index and known.loc[c, "product_id"] == product_id
             and (allowed is None or c in allowed)]
    invalid = [c for c in citations if c not in valid]
    if not valid:
        return {"verdict": "unsupported", "reason": "No valid citation.",
                "supporting_ids": [], "invalid_citations": invalid}
    out = chat_json(build_messages(claim, valid, reviews, question), QA_SCHEMA if question else SCHEMA,
                    stage="qa_verify" if question else "verify", **kw)
    out = dict(out)
    if question and not out.pop("answers_question", True):
        out["verdict"] = "unsupported"
        out["reason"] = "Does not answer the question. " + out["reason"]
    out["supporting_ids"] = [i for i in out["supporting_ids"] if i in valid]
    return {**out, "invalid_citations": invalid}


def check_claims(claims, product_id, allowed=None, reviews=None, workers=6, question=None):
    """claims: dicts with 'claim' and 'citations'. Returns one check dict per claim, in order."""
    return map_parallel(lambda c: check(c["claim"], c["citations"], product_id, allowed, reviews, question),
                        claims, workers=workers, desc="verify")


REPAIR_SYSTEM = """A claim in a summary of laptop reviews was found to be only partially supported by the reviews it cites. Rewrite it so that it says only what those reviews support.

- Keep it one sentence about the same aspect.
- Match how many reviewers say it to the evidence ("One reviewer", "Two reviewers", "Several reviewers" only for 3 or more).
- Remove details the reviews do not give.
- Cite only the review IDs that support the rewritten claim."""

REPAIR_SCHEMA = {
    "name": "claim_repair",
    "schema": {
        "type": "object",
        "properties": {"claim": {"type": "string"},
                       "citations": {"type": "array", "items": {"type": "string"}}},
        "required": ["claim", "citations"],
        "additionalProperties": False,
    },
}


def repair(claim, citations, check_result, reviews=None, **kw):
    """Rewrite a partially supported claim to match its evidence (then re-check it)."""
    known = reviews if reviews is not None else _reviews()
    valid = [c for c in dict.fromkeys(citations) if c in known.index]
    body = "\n\n".join(f"[{rid}] {review_text(rid, reviews)}" for rid in valid)
    messages = [{"role": "system", "content": REPAIR_SYSTEM},
                {"role": "user", "content": f"Claim: {claim}\nWhy it is only partially supported: "
                                            f"{check_result['reason']}\n\nCited reviews:\n\n{body}"}]
    return chat_json(messages, REPAIR_SCHEMA, stage="repair", **kw)


def repair_claims(claims, checks, reviews=None, workers=6):
    return map_parallel(lambda cc: repair(cc[0]["claim"], cc[0]["citations"], cc[1], reviews),
                        list(zip(claims, checks)), workers=workers, desc="repair")
