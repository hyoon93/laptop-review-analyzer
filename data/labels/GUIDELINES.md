# Labeling guidelines

These are the same definitions the LLM prompt uses (`pipeline/sentiment.py`), so labels and predictions measure the same thing.

## Overall sentiment (300 reviews)

Read the title and text. The star rating is hidden on purpose: label what the **text** says.

| Label | Use when |
|---|---|
| positive | The reviewer is satisfied overall (would keep or recommend it), even with minor complaints. |
| negative | The reviewer is dissatisfied overall (returned it, regrets it, warns others), even with some good points. |
| neutral | No clear overall verdict: purely factual, a question, or balanced pros and cons with no conclusion. |

**Mixed** checkbox: tick it if the review has at least one clearly positive **and** at least one clearly negative opinion about the product or purchase, whatever the overall label. Example: "Great screen, but the battery dies in two hours. Still keeping it." is positive and mixed.

Rules of thumb:
- Shipping or seller complaints count. A review that is only about a late delivery is negative if the reviewer is unhappy.
- Sarcasm: label the intended meaning.
- If you hesitate for more than ~20 seconds, pick the best label and write a short note; notes feed the error analysis.
- Non-English reviews: the project is English-only (the baselines and SemEval are English), so the 62 Spanish reviews were removed from the dataset and the 10 that were in this sample were swapped for English reviews of the same product and star level (`sample_replacements.csv`). If you still see one, pick any label and write `not English` in the note; it will be excluded from scoring.

## Aspect judgments (100 reviews): `streamlit run app/label_aspects.py`

For each aspect the model extracted: is it really discussed (yes/no), is the polarity right for that aspect (yes/no)? Then list any aspect the model missed, one per line (for example `keyboard: negative`), or leave it empty. These are judgments of the model's output, so the model's answer is shown; judge it strictly.

## Claim judgments (~111 claims): `streamlit run app/label_claims.py`

40 random claims were drawn from each of the three summary versions; a claim that appears unchanged in two versions is shown once, so there are fewer than 120. For each claim, read the 12 evidence reviews shown and mark it:

| Verdict | Use when |
|---|---|
| supported | The reviews clearly say it. "Several / many / most reviewers" needs at least two reviews that say it. |
| partially supported | Part of it is backed, but it overstates how many reviewers say it, or adds details the reviews don't give. |
| unsupported | No review says it, or the reviews contradict it. |

Judge only from the reviews shown, not from what you know about the laptop. The page never shows which version produced a claim, its citations, or the LLM judge's verdict. Please don't open `claim_sample.csv` or `summaries.csv` before you finish: the claim IDs in them reveal the version.
