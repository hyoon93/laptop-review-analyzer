# Laptop Review Analyzer

COSC 661 final project (Hee Young Yoon). Turns hundreds of Amazon laptop reviews into aspect-level sentiment, a summary in which every claim cites the reviews behind it, and a question box that answers from the reviews or says it can't. Scope and evaluation plan: [PLAN.md](PLAN.md).

**Deployed app:** _link added after deployment_

## 1. What it does

- **Overview** of each laptop: text sentiment by star rating (stars and text often disagree), and a chart of the aspects reviewers discuss (battery, display, keyboard, …), positive vs. negative. Click an aspect to read those reviews.
- **Summary** with citations: every claim links to the reviews it is based on. Switch between the three versions compared in the evaluation (ungrounded, grounded, grounded + verified) to see what verification changes.
- **Review explorer**: semantic search (local embeddings) plus filters for stars, sentiment, aspect, polarity, verified purchase, date, mixed opinions and "stars disagree with text"; the evidence for each aspect is highlighted in the review.
- **Ask the reviews** (live): a question about one laptop is answered only from its reviews, with citations, and every claim is checked before it is shown.
- **Analyze a review** (live): paste any review to get its sentiment and aspects, next to the VADER baseline.
- **Compare** two laptops aspect by aspect, and an **Evaluation** tab with all results.

## 2. GenAI components and architecture

```
Amazon reviews (8 laptops, 1,538 English reviews)
  │
  ├─ (1) Structured-output LLM extraction   gpt-4.1-mini, JSON schema, temperature 0
  │        review → overall sentiment + mixed flag
  │        review → aspects [term, category, polarity, verbatim evidence quote]
  │
  ├─ (2) Embeddings + vector store          all-MiniLM-L6-v2 (ONNX) → ChromaDB, review chunks ≤ 80 words
  │
  ├─ (3) Retrieval-augmented summary        top aspects → retrieve 6 reviews per aspect →
  │        gpt-4.1-mini writes one-fact claims, each citing review IDs
  │
  └─ (4) Verification                       each claim vs. the full cited reviews:
           supported → kept; partially supported → rewritten to match the evidence and re-checked;
           anything still not supported → removed (shown in the app with the reason)

Live: question → (2) retrieve 8 reviews → cited answer or "not answered by the reviews" → (4) check
Evaluation only: gpt-4.1 as an independent judge, validated against my own blind judgments
```

- Every LLM call goes through one wrapper (`pipeline/llm.py`) with a disk cache (no call is paid twice) and logs tokens and latency. Offline stages run through the OpenAI Batch API (`pipeline/batch.py`, 50% cheaper); the app reads the stored results, so browsing costs nothing.
- Prompts: `pipeline/sentiment.py`, `aspects.py`, `summarize.py`, `verify.py`, `qa.py`.

## 3. Install and run locally

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt            # the app
streamlit run app/app.py
```

Set `OPENAI_API_KEY` in the environment (or a `.env` file) for the two live tabs; everything else works without a key. On Streamlit Community Cloud the key goes in the app's **Secrets**, never in the repository.

To rebuild the data and the evaluation (needs `requirements-offline.txt` and CPU PyTorch):

| Step | Command | Output |
|---|---|---|
| Download laptop reviews | `python scripts/01_fetch_laptops.py` | `data/raw/` (streams 28 GB of Amazon Reviews 2023, keeps laptops) |
| Clean and sample | `python scripts/02_prepare_dataset.py --per-product 200` | `data/processed/reviews.csv`, `products.csv` |
| Hand-label sample | `python scripts/03_make_label_sample.py` | `data/labels/label_sample.csv` |
| Baselines | `python scripts/04_run_baselines.py` | stars, VADER, DistilBERT |
| LLM sentiment + aspects | `python scripts/05_run_pipeline.py --batch` | `sentiment_llm.csv`, `aspects.csv` |
| Search index | `python -c "import pandas as pd; from pipeline import index; index.build(pd.read_csv('data/processed/reviews.csv', keep_default_na=False))"` | `data/chroma/` |
| Summaries + verification | `python scripts/06_summaries.py` | `summaries.csv` |
| Evaluations | `python -m eval.sentiment_eval`, `python -m eval.aspect_eval`, `python -m eval.semeval`, `python -m eval.judge` | `data/processed/eval_*` |

SemEval-2014 files are not redistributed here: put `Laptops_Test_Gold.xml` and `Laptop_Train_v2.xml` from the [`alexcadillon/SemEval2014Task4`](https://huggingface.co/datasets/alexcadillon/SemEval2014Task4) mirror in `data/semeval/` before running `eval.semeval`.

Labeling pages (my ground truth): `streamlit run app/label_sentiment.py`, `app/label_aspects.py`, `app/label_claims.py`.

## 4. Sample inputs

- **Ask the reviews** → pick *Acer Aspire 3*, ask "How long does the battery last?" (normal case), then "Does it have a fingerprint reader?" (edge case: the reviews don't say, so the app declines instead of guessing).
- **Analyze a review** → the built-in *Mixed review*, *Sarcasm* and *Shipping only* examples, or paste your own.
- **Summary** → switch from *Grounded + verified* to *Ungrounded* and compare.

## 5. Evaluation (summary; details in the app's Evaluation tab)

Application data and the external benchmark are reported separately.

- **A. Sentiment, 300 hand-labeled reviews** (60 per star level, labeled without seeing the stars): gpt-4.1-mini accuracy 0.847, macro-F1 0.660; baselines DistilBERT 0.817 / 0.618, star rating 0.753 / 0.626, VADER 0.663 / 0.498.
- **B. SemEval-2014 Laptops test** (800 sentences, zero-shot, prompts not tuned on it): aspect-term F1 0.594 exact / 0.678 partial match; aspect polarity accuracy 0.809.
- **C. Ungrounded vs. grounded vs. grounded + verified summaries** (8 laptops, 101–107 claims each; gpt-4.1 judge, 12 evidence reviews per claim gathered the same way for every version): supported 82% / 80% / 88%, unsupported 0.9% / 0.9% / 0%; citation precision 83% → 90% after verification. Grounding alone did not raise the supported rate (the ungrounded model also read real reviews); what it adds is traceability, and that is what makes verification possible. The judge is being checked against my blind judgments of 111 claims (Cohen's κ).
- **D. Cost**: the whole offline pipeline cost under $1.

## 6. Known limitations

- One annotator (me) for all hand labels; no inter-annotator agreement. Only 17 of the 300 reviews are neutral, so neutral-class scores are noisy, and every system is weak on that class.
- An Amazon listing groups several hardware versions (the Acer listing's title says Ryzen 5 7520U, older reviews describe a Ryzen 3). Summaries describe the listing's reviews, not one exact configuration.
- The aspect extractor targets opinions, while SemEval also annotates neutral mentions, which lowers its benchmark recall.
- The in-app claim checker uses the same model as the generator; the independent judge (gpt-4.1) is used only for evaluation and is itself checked against human judgments.
- English reviews only (62 Spanish reviews removed), 8 laptops, 200 sampled reviews each.
- About 5% of evidence quotes are not verbatim (the model paraphrased or shortened them); the app highlights only exact matches.
- "Ask the reviews" sometimes answers a question the reviews don't cover with a claim about a different feature. The claim checker now also asks "does this claim answer the question?" and removes most of these, but in testing one such claim still got through on 1 of 8 laptops.
- The live tabs depend on the OpenAI API and are capped per session and per day. The example buttons are pre-computed, so they work even when the API is slow or rate-limited.

## Data

- Amazon Reviews 2023 (Hou et al., 2024; McAuley Lab), Electronics → Laptops. Reviewer IDs are dropped at extraction.
- English only: 62 Spanish reviews are dropped after sampling (`data/processed/excluded_non_english.csv`), leaving 1,538. The 10 that were in the labeling sample were replaced by English reviews of the same product and star level (`data/labels/sample_replacements.csv`).
- SemEval-2014 Task 4 Laptops (Pontiki et al., 2014), gold XML from the `alexcadillon/SemEval2014Task4` Hugging Face mirror.
