"""Evaluation B: aspect-term extraction on the SemEval-2014 Task 4 Laptops test set.

This is the external benchmark, reported separately from the Amazon application data.
The app's own extractor (pipeline/aspects.py, same prompt) runs on each test sentence;
only aspects with an explicit term are scored, since SemEval annotates explicit terms only.

Metrics
- exact:   predicted term == gold term (lowercased, whitespace-normalized)
- partial: predicted and gold terms share at least one token
  Matching is one-to-one within a sentence; precision/recall/F1 are micro-averaged.
- polarity accuracy on exactly matched terms, excluding gold "conflict" terms
  (the extractor has no conflict label).
"""

import xml.etree.ElementTree as ET

from pipeline.config import SEMEVAL

TEST_FILE = SEMEVAL / "Laptops_Test_Gold.xml"
TRAIN_FILE = SEMEVAL / "Laptop_Train_v2.xml"


def load(path=TEST_FILE):
    sentences = []
    for s in ET.parse(path).getroot().findall("sentence"):
        gold = [{"term": t.get("term"), "polarity": t.get("polarity")}
                for t in s.findall("aspectTerms/aspectTerm")]
        sentences.append({"id": s.get("id"), "text": s.find("text").text, "gold": gold})
    return sentences


def _norm(term):
    return " ".join(term.lower().split())


def _match(preds, golds, same):
    """Greedy one-to-one matching; returns list of (pred, gold) pairs."""
    used, pairs = set(), []
    for p in preds:
        for j, g in enumerate(golds):
            if j not in used and same(p, g):
                used.add(j)
                pairs.append((p, g))
                break
    return pairs


def exact(p, g):
    return _norm(p["term"]) == _norm(g["term"])


def partial(p, g):
    return bool(set(_norm(p["term"]).split()) & set(_norm(g["term"]).split()))


def score(sentences, predictions):
    """sentences: from load(); predictions: list (same order) of aspect lists from pipeline.aspects."""
    out = {}
    n_gold = sum(len(s["gold"]) for s in sentences)
    explicit = [[a for a in preds if a.get("term")] for preds in predictions]
    n_pred = sum(len(p) for p in explicit)
    for name, same in (("exact", exact), ("partial", partial)):
        tp = sum(len(_match(p, s["gold"], same)) for s, p in zip(sentences, explicit))
        prec = tp / n_pred if n_pred else 0.0
        rec = tp / n_gold if n_gold else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        out[name] = {"precision": prec, "recall": rec, "f1": f1, "tp": tp}
    pairs = [pg for s, p in zip(sentences, explicit) for pg in _match(p, s["gold"], exact)]
    pairs = [(p, g) for p, g in pairs if g["polarity"] != "conflict"]
    out["polarity_accuracy"] = (sum(p["polarity"] == g["polarity"] for p, g in pairs) / len(pairs)
                                if pairs else 0.0)
    out["polarity_n"] = len(pairs)
    out["n_gold"], out["n_pred"] = n_gold, n_pred
    return out


def run(limit=None, workers=8):
    """Run the app's aspect extractor on the test set and score it (needs OpenAI credit)."""
    from pipeline.aspects import extract
    from pipeline.llm import map_parallel

    sentences = load()[:limit]
    preds = map_parallel(lambda s: extract(s["text"], stage="semeval"), sentences,
                         workers=workers, desc="semeval")
    return sentences, preds, score(sentences, preds)


def main():
    """python -m eval.semeval [--limit N]  (all calls go through the Batch API)."""
    import argparse
    import json

    from pipeline import batch
    from pipeline.config import PROCESSED

    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()
    sentences, preds, result = batch.run_rounds(lambda: run(args.limit, workers=16), stage="semeval")
    PROCESSED.mkdir(parents=True, exist_ok=True)
    with open(PROCESSED / "semeval_predictions.jsonl", "w", encoding="utf-8") as f:
        for s, p in zip(sentences, preds):
            f.write(json.dumps({"id": s["id"], "text": s["text"], "gold": s["gold"], "pred": p},
                               ensure_ascii=False) + "\n")
    (PROCESSED / "eval_semeval.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
