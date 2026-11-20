# How to do the aspect review and the claim review

Two hand-review tasks are left. Both are ground truth for the evaluation, so the answers must be **my own judgment**. Don't ask an AI to do them.

| Task | Page | Items | Time | What it measures |
|---|---|---|---|---|
| 1. Aspect review | `app/label_aspects.py` | 100 reviews | ~1.5 h | How accurate the model's aspect extraction is (precision, polarity accuracy, recall) |
| 2. Claim review | `app/label_claims.py` | 111 claims | ~1.5–2 h | Whether the gpt-4.1 judge can be trusted (Cohen's κ), and my own supported-claim rates for the three summary versions |

Short definitions are also in [GUIDELINES.md](GUIDELINES.md).

---

## Starting and stopping a page

In PowerShell:

```powershell
cd "C:\Users\hyoon\OneDrive - Towson University\COSC 661 Gen AI\Project\review-analyzer"
C:\Users\hyoon\venvs\review-analyzer\Scripts\python.exe -m streamlit run app/label_aspects.py
```

- The page opens in the browser. If it doesn't, open the "Local URL" shown in the PowerShell window (for example http://localhost:8501).
- For the claim review, use `app/label_claims.py` instead.
- To stop: press **Ctrl+C** in that PowerShell window.

**Saving:** every click on **Save and next** is written to disk at once (`aspect_judgments.csv` + `aspect_missed.csv`, or `claim_judgments.csv`). You can close the browser at any time. Next time, the page opens at the first item you haven't done yet. Use **◀ Previous** to fix an answer; saving again overwrites it.

---

## Task 1: Aspect review

### What the page shows

The review (title and text) and, below it, every aspect the model extracted:

```
2. performance · term: Intel Celeron N4020 Processor · positive
   Evidence: "Intel Celeron N4020 Processor is lightweight and fast"
   Real aspect?     ○ yes  ○ no
   Polarity right?  ○ yes  ○ no
```

At the bottom is a box for **missed aspects**, then **Save and next**.

### The two questions for each aspect

| Question | Answer **yes** when | Answer **no** when |
|---|---|---|
| **Real aspect?** | The review really talks about this part of the laptop or the purchase (with or without an opinion). | The review never discusses it, it's a misreading, or the model turned a product-name phrase into an opinion. |
| **Polarity right?** | positive / negative / neutral matches what the reviewer thinks of **this aspect**. | The direction is wrong, for example "negative" for something the reviewer is fine with. If the aspect isn't real, skip this question. |

**Don't judge** the exact term wording, the category name, or whether the evidence quote is word for word. Only judge: is the aspect discussed, and is the polarity right?

### Missed aspects box

Read the review once more and list any aspect with an opinion that the model **didn't** extract, one per line:

```
keyboard: negative
seller_support: negative
```

Leave it empty if nothing was missed. Missing a minor or purely factual mention ("it has 8 GB RAM") doesn't count, unless the reviewer gives an opinion on it.

### Worked example (not one of my 100)

> **Convenient and Lightweight / Keyboard Not Backlighted**
> The ASUS Laptop L210 Ultra Thin Laptop, 11.6" HD Display, Intel Celeron N4020 Processor is lightweight and fast, including a fast connection to the existing WiFi connection. However, this model was advertised on Amazon with a keyboard that is supposed to be backlighted. The model I received does not include that feature which is quite disappointing. ...

| Extracted | Real? | Polarity right? | Reason |
|---|---|---|---|
| portability · *lightweight* · positive | yes | yes | Praises the light weight. |
| performance · *Intel Celeron N4020 Processor* · positive | yes | yes | "fast". The term is clumsy, but that isn't judged. |
| ports_connectivity · *WiFi connection* · positive | yes | yes | "fast connection to the existing WiFi". |
| keyboard_touchpad · *keyboard* · negative | yes | yes | The backlight is missing and the reviewer is disappointed. |

Missed: `seller_support: negative` *(the listing advertised a feature it didn't have)*. This is a judgment call: list it if you see it as a separate complaint about the listing, leave the box empty if you see it as the same keyboard complaint. Then do the same in similar cases.

### Tricky cases

- **"Screen is rather dim but this is not a problem"** labeled *negative*: the reviewer says it doesn't bother them, so neutral is a fair reading. Answer **Polarity right? → no** if that's how you read it.
- **The same aspect listed twice** ("screen is dim", "screen colors are bad"): judge each one on its own. Both can be real.
- **Implied aspects** (no term, for example "it is so slow" → performance): these are real if the meaning is clearly there.
- **Shipping, packaging, returns, customer service** are real aspects (category seller_support).
- **Unsure?** Go with your first reading. Don't spend more than ~20 seconds per aspect.

---

## Task 2: Claim review

### What the page shows

One sentence from a product summary, a verdict choice, and **12 evidence reviews** below it:

```
"Several reviewers say the battery lasts about 6 hours."
○ supported   ○ partially supported   ○ unsupported
Note (optional): ________
[Save and next]
```

The page is **blind**: it never shows which summary version wrote the claim, which reviews it cited, or what the LLM judge said. The 12 reviews are in random order.

### The three verdicts

| Verdict | Use when |
|---|---|
| **supported** | The reviews shown clearly say it. *"Several / many / most reviewers"* needs **at least two** reviews that say it. |
| **partially supported** | Part of it is backed, but it **overstates** how many reviewers say it, or adds **details** the reviews don't give. |
| **unsupported** | No review shown says it, or the reviews **contradict** it. |

Judge **only from the 12 reviews shown**, not from what you know about the laptop. Even if a claim is true in real life, it's *unsupported* if none of these reviews say it.

### How to read efficiently

1. Read the claim and pick out its key words (for example *battery*, *6 hours*, *several*).
2. Press **Ctrl+F** in the browser and search a key word (`battery`, `hours`) to jump to the relevant reviews.
3. Count how many reviews actually say it, and check the details (numbers, "most", "all", "after upgrading RAM", ...).
4. Choose the verdict. Use the note for anything odd, such as "true but only for gaming" or "one review says the opposite".

### Worked examples (made up, not from the sample)

| Claim | What the 12 reviews say | Verdict |
|---|---|---|
| "One reviewer says the hinge broke after three months." | One review: "the hinge cracked at month 3". | **supported** |
| "Several reviewers say the battery lasts about 6 hours." | One review says 6 hours; another says "battery is fine"; nothing else. | **partially supported** (only one gives 6 hours, so "several ... about 6 hours" overstates it) |
| "Many reviewers praise the speakers." | Two reviews say the speakers are loud and clear. | **supported** (two or more is enough for "many" here; be consistent) |
| "Reviewers praise the fingerprint reader." | No review mentions a fingerprint reader. | **unsupported** |
| "Most reviewers find the keyboard comfortable." | One says comfortable, two say the keys are mushy. | **unsupported** (the reviews contradict "most") |
| "The laptop is great for students and gamers." | Three say it's great for school; nobody mentions gaming. | **partially supported** (the gaming part has no backing) |

### Rules of thumb

- **Every part of the claim counts.** If one half is backed and the other half isn't, choose *partially supported*.
- **Count words matter:** *one* = 1 review, *several / some / many* = at least 2, *most / all* = clearly the majority of the reviews that discuss it.
- **Disagreement claims** ("reviewers are split on battery life") are *supported* if the reviews really show both sides.
- **Don't open** `claim_sample.csv`, `summaries.csv` or `judge.csv` until you're done: the IDs in them reveal the version and the judge's verdict.

---

## When both are done

Tell Claude, or run these yourself:

```powershell
C:\Users\hyoon\venvs\review-analyzer\Scripts\python.exe -m eval.aspect_eval   # aspect precision, polarity accuracy, recall
C:\Users\hyoon\venvs\review-analyzer\Scripts\python.exe -m eval.judge         # judge vs. me: agreement and Cohen's kappa
```

The results also appear in the app's **Evaluation** tab.
