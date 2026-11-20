"""OpenAI wrapper used by every LLM stage: JSON-schema outputs, a disk cache, usage logging.

Each call is keyed by (model, messages, schema, temperature). A repeated call is answered
from data/cache/llm_cache.jsonl, so batch runs can be stopped and restarted and the app
never pays twice. Each cache entry keeps the original token counts and latency, which is
what the cost/latency evaluation reads.
"""

import hashlib
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager

from dotenv import dotenv_values
from openai import OpenAI
from tqdm import tqdm

from .config import CACHE, ENV_FILES, MODEL, PRICES

CACHE_FILE = CACHE / "llm_cache.jsonl"
BATCH_DISCOUNT = 0.5  # Batch API price relative to synchronous calls
_lock = threading.Lock()
_cache = None
_client = None


def _api_key():
    key = os.environ.get("OPENAI_API_KEY")
    for env in ENV_FILES:
        if env.exists():
            key = dotenv_values(env).get("OPENAI_API_KEY") or key
    if not key:
        raise RuntimeError("OPENAI_API_KEY not found in the environment or the course .env")
    return key


def client():
    global _client
    if _client is None:
        _client = OpenAI(api_key=_api_key(), max_retries=5)
    return _client


def _load_cache():
    global _cache
    if _cache is None:
        _cache = {}
        if CACHE_FILE.exists():
            with open(CACHE_FILE, encoding="utf-8") as f:
                for line in f:
                    try:
                        entry = json.loads(line)
                    except json.JSONDecodeError:  # a line still being written by another process
                        continue
                    _cache[entry["key"]] = entry
    return _cache


def _key(model, messages, schema, temperature):
    blob = json.dumps([model, messages, schema, temperature], sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def cost_usd(model, input_tokens, output_tokens):
    price_in, price_out = PRICES[model]
    return (input_tokens * price_in + output_tokens * price_out) / 1e6


_collect = None  # a list while collecting() is active


@contextmanager
def collecting():
    """Record uncached calls instead of making them (see batch.run_rounds).

    Inside this block chat_json returns a minimal schema-valid placeholder for every cache
    miss, so the calling code keeps running and later stages that depend on it are skipped.
    """
    global _collect
    _collect = []
    try:
        yield _collect
    finally:
        _collect = None


def placeholder(schema):
    """Smallest value that satisfies a JSON schema (empty arrays, first enum value)."""
    kind = schema.get("type")
    if isinstance(kind, list):
        return None if "null" in kind else placeholder({**schema, "type": kind[0]})
    if kind == "object":
        return {k: placeholder(v) for k, v in schema.get("properties", {}).items()}
    if kind == "array":
        return []
    if "enum" in schema:
        return schema["enum"][0]
    return {"string": "", "boolean": False, "integer": 0, "number": 0}.get(kind)


def chat_json(messages, schema, *, stage, model=MODEL, temperature=0.0, use_cache=True):
    """Call the model with a strict JSON schema and return the parsed object.

    schema is {"name": ..., "schema": {...}} in OpenAI's json_schema format.
    stage is a short label ("sentiment", "aspects", ...) stored with the usage record.
    """
    cache = _load_cache()
    key = _key(model, messages, schema, temperature)
    if use_cache and key in cache:
        return cache[key]["output"]
    if _collect is not None:
        _collect.append({"messages": messages, "schema": schema, "model": model,
                         "temperature": temperature, "stage": stage})
        return placeholder(schema["schema"])

    t0 = time.time()
    resp = client().chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        response_format={"type": "json_schema", "json_schema": {**schema, "strict": True}},
    )
    latency = time.time() - t0
    msg = resp.choices[0].message
    if getattr(msg, "refusal", None):
        raise RuntimeError(f"model refused: {msg.refusal}")
    output = json.loads(msg.content)

    entry = {
        "key": key,
        "stage": stage,
        "model": model,
        "temperature": temperature,
        "input_tokens": resp.usage.prompt_tokens,
        "output_tokens": resp.usage.completion_tokens,
        "latency_s": round(latency, 3),
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "output": output,
    }
    with _lock:
        cache[key] = entry
        CACHE.mkdir(parents=True, exist_ok=True)
        with open(CACHE_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return output


def map_parallel(fn, items, workers=8, desc=None):
    """Run fn over items with a thread pool, keeping input order."""
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(tqdm(pool.map(fn, items), total=len(items), desc=desc))


def usage_table():
    """Per-stage token, cost and latency totals from the cache (pandas DataFrame)."""
    import pandas as pd

    rows = list(_load_cache().values())
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["batch"] = df["batch"].fillna(False).astype(bool) if "batch" in df else False
    df["cost_usd"] = [cost_usd(m, i, o) * (BATCH_DISCOUNT if b else 1)
                      for m, i, o, b in zip(df.model, df.input_tokens, df.output_tokens, df.batch)]
    return (df.groupby(["stage", "model", "batch"])
              .agg(calls=("key", "size"), input_tokens=("input_tokens", "sum"),
                   output_tokens=("output_tokens", "sum"), cost_usd=("cost_usd", "sum"),
                   mean_latency_s=("latency_s", "mean"))
              .reset_index())
