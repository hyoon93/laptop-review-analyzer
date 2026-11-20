# Product Review Analyzer: revised plan (v2, after proposal feedback)

Started 2026-10-29. Final presentations are in Week 14 of the syllabus (late Nov / early Dec; confirm the exact date).

## 1. What changed and why

| Professor's feedback | Change in v2 |
|---|---|
| 5,000 reviews + several LLMs + RAG + verification + baselines + many experiments is too much | Smaller core: **one LLM (gpt-4.1-mini)**, **one product category (laptops)**, **8 products / 1,538 English reviews**. gpt-4.1 and Llama comparisons are dropped from the core plan. |
| Build the end-to-end application before model comparisons | Phases 1–2 are the full pipeline and the app. No experiment runs until the app works on all products. |
| Then add **one** meaningful comparison | **Grounded vs. ungrounded summarization** (it tests the project's central claim that retrieval + verification reduce unsupported claims). Zero-shot vs. few-shot is a stretch goal only. |
| SemEval laptops may not match all electronics categories; keep benchmark and application evaluation apart | The application category is now **laptops**, the same domain as SemEval-2014 Laptops. Results are reported in **two separate tables**: (A) application data (Amazon laptop reviews, my own labels) and (B) external benchmark (SemEval, gold labels). They are never averaged together. |
| Star ratings are weak labels; a 5-star review can contain negative aspects | Stars are **not ground truth**. Ground truth = 300 **hand-labeled** reviews. "Stars → sentiment" becomes one more **baseline** to beat, and star/text disagreement becomes a finding: "% of 4–5★ reviews with at least one negative aspect". |
| An LLM judge is not ground truth; verify it on a manual subset | I hand-judge a blind, stratified subset of claims and report **judge–human agreement (Cohen's κ)**. Judge-only numbers are reported only if κ is acceptable; otherwise the human numbers are primary. The judge is also a different model (gpt-4.1) from the pipeline's claim checker (gpt-4.1-mini), so the grounded condition is not graded by the same model that filtered it. |
| Offline batch + caching is good; the UI must still be interactive | The app adds filters, drill-down from charts to the cited reviews, product-vs-product comparison, a local semantic search box, and an optional live "ask the reviews" box and small CSV upload (see §5). |

## 2. Scope v2

- **Data:** Amazon Reviews 2023, Electronics → Computers & Tablets → **Laptops** (traditional and 2-in-1). 8 products chosen for variety (budget, ultra-budget, Chromebook, 2-in-1, gaming, MacBook, premium, off-brand), with 200 reviews sampled from each. The 62 Spanish reviews (3.9%) are dropped because the baselines and SemEval are English-only, which leaves 1,538 reviews (`data/processed/products.csv`).
- **External benchmark:** SemEval-2014 Task 4, Laptops. Test: 800 sentences, 654 gold aspect terms. Train (3,045 sentences) is used only for few-shot examples.
- **LLM:** gpt-4.1-mini for every pipeline stage, temperature 0, JSON-schema structured outputs, every call cached on disk with tokens and latency logged.
- **Local, free components:** all-MiniLM-L6-v2 embeddings, ChromaDB, VADER, DistilBERT SST-2.

## 3. Phases

| Phase | Goal | Done when |
|---|---|---|
| 0 | Data subset, environment, project skeleton | Laptop subset extracted; products chosen; cleaned review table |
| 1 | End-to-end pipeline on **one** product | Sentiment → aspects → index → cited summary → claim check → JSON results, for one product |
| 2 | All products + interactive Streamlit app | App runs on cached results for all ~8 products |
| 3 | Evaluation A (application data) and B (SemEval benchmark) | Sentiment and aspect tables, error analysis |
| 4 | The one comparison: grounded vs. ungrounded summaries | Supported-claim rates, κ for the judge |
| 5 | Report, slides, demo video | Submitted |
| Stretch | Only if time remains | Zero- vs. few-shot sentiment; consistency (100 reviews × 3 runs); gpt-4.1 cost/quality point |

## 4. Evaluation v2

### A. Application dataset (Amazon laptop reviews)

| What | Test set | Metrics | Compared with |
|---|---|---|---|
| Overall sentiment | **300 reviews hand-labeled by me**, 60 per star level. Label = overall verdict (pos / neu / neg) plus a **"mixed"** checkbox | Accuracy, macro-F1, confusion matrix; accuracy on mixed vs. non-mixed reviews | VADER, DistilBERT SST-2, star-derived label |
| Aspect extraction | 100 of the 300 reviews: I judge each extracted aspect (real? right polarity?) and note missed aspects | Aspect precision, polarity accuracy, approximate recall | — |
| Evidence spans | All reviews (automatic) | % of quoted evidence spans found verbatim in the review | — |
| Star/text disagreement | All reviews | % of 4–5★ reviews with ≥1 negative aspect, checked against the hand-labeled subset | — |

### B. External benchmark (SemEval-2014 Laptops test, 800 sentences)

Aspect-term precision, recall, F1 (exact and partial match) and aspect polarity accuracy. Prompts are not tuned on the test set. Reported separately from A.

### C. The one comparison: grounded vs. ungrounded summaries

Two generation runs give three conditions:

1. **Ungrounded:** the summarizer sees a random sample of reviews (same token budget), with no retrieval, citations or verification.
2. **Grounded:** aspect-wise retrieval from ChromaDB, and every claim cites review IDs.
3. **Grounded + verified:** (2) after the claim checker removes unsupported claims. This costs nothing extra to generate.

Each summary is split into atomic claims. An LLM judge (gpt-4.1, evaluation only) marks each claim supported or unsupported against evidence retrieved the same way for every condition, plus the cited reviews for 2 and 3. I hand-judge **120 claims (40 per condition)**, blind to condition and to the judge's verdict.

- Metrics: supported-claim rate, citation precision (conditions 2–3), coverage of each product's top aspects, judge–human κ.

### D. Cost and latency

Logged on every call at no extra cost: tokens, USD and seconds per 100 reviews, per stage.

### Error analysis

Misclassified reviews and unsupported claims are grouped by cause (sarcasm, mixed opinion, implicit aspect, off-topic or shipping text, retrieval miss), with the share of errors in each group.

### My manual work (≈5–6 hours total, with a labeling page in the app)

- 300 sentiment labels (~1.5–2 h)
- 100 aspect judgments (~1.5 h)
- 120 claim judgments (~1.5–2 h)

## 5. App (Streamlit)

- Product picker, plus side-by-side comparison of two products.
- Sentiment distribution, plus an aspect × polarity chart; clicking an aspect filters the review list.
- Filters: stars, verified purchase, date range, aspect, polarity, "stars disagree with the text".
- Review explorer with the evidence span highlighted.
- Cited summary: every claim links to the reviews it cites; a "verification" tab shows the claims that were removed and why.
- Semantic search over reviews (local embeddings, no API cost).
- Optional live features (small gpt-4.1-mini calls): "ask the reviews" Q&A with citations; CSV upload of up to ~200 reviews analyzed on the spot.
- Every generated text is labeled "AI-generated".

## 6. Timeline (revised 2026-11-20)

Deadlines (`../Final_Project_Demo_Showcase_With_Schedule.pdf`): **live demo Mon Nov 30** (Session 12, 18 min) from a **deployed public URL** (Streamlit Community Cloud), not a local app. **GitHub repo** due Dec 7, with the six required README sections. **Report** (4–6 pages) due Dec 15.

The showcase grades a live "user input → processing → output" workflow, so the live features in §5 ("analyze a review", "ask the reviews") are now core, not optional. Demo scenarios: one normal (product overview, a question with citations) and one edge case (a question no review answers, which should get "not enough evidence" instead of a made-up answer).

| Dates | Work |
|---|---|
| Oct 29 – Nov 17 | Done: data, environment, pipeline code, baselines, search index |
| Nov 20 | Done: 300 hand labels; baseline sentiment evaluation |
| Nov 20 – 22 | App shell on non-LLM data; GitHub repo. Once credit is added: full pipeline, LLM sentiment eval, SemEval |
| Nov 22 – 24 | Summaries (3 conditions) + judge; my aspect (100) and claim (120) judgments |
| Nov 24 – 25 | Deploy to Streamlit Community Cloud; test the public URL and API limits |
| Nov 26 – 29 | 3–5 slides, backup screen recording, rehearsal; buffer |
| Nov 30 | **Live demo** |
| Dec 7 / Dec 15 | GitHub repo submission / final report |

## 7. Cost estimate

At gpt-4.1-mini list prices (≈$0.40 / $1.60 per million input / output tokens), the whole pipeline on ~1,540 reviews plus SemEval and the summaries comes to roughly $2–4. The gpt-4.1 judge adds ~$2–3. **$10 of OpenAI credit should cover everything, including reruns.**
