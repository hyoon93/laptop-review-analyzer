"""Paths, model names and prices shared by the pipeline, the app and the evaluation."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RAW = DATA / "raw"
PROCESSED = DATA / "processed"
CACHE = DATA / "cache"
LABELS = DATA / "labels"
SEMEVAL = DATA / "semeval"
CHROMA_DIR = DATA / "chroma"

# API keys live in the course-folder .env (shared with the weekly notebooks);
# a .env next to this project overrides it.
ENV_FILES = [ROOT.parents[1] / ".env", ROOT / ".env"]

MODEL = "gpt-4.1-mini"      # every pipeline stage
JUDGE_MODEL = "gpt-4.1"     # evaluation only: grades claims, never used inside the app
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# USD per 1M tokens (input, output); check https://openai.com/api/pricing before reporting costs.
PRICES = {
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-4.1": (2.00, 8.00),
}

SENTIMENTS = ["positive", "neutral", "negative"]
