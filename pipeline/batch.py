"""Run many chat_json calls through the OpenAI Batch API and store the results in the cache.

Why: the account's synchronous limit is low (requests per day), while the Batch API has
its own limits and costs 50% less. Results are written to the same disk cache that
chat_json reads, under the same keys, so afterwards the normal pipeline functions
(sentiment.classify, aspects.extract, ...) answer every call from the cache.

Batch entries are marked "batch": true; they have token counts but no per-call latency.
"""

import io
import json
import time

from .llm import CACHE, CACHE_FILE, _key, _load_cache, _lock, client

POLL_S = 15


def _request_line(key, model, messages, schema, temperature):
    body = {"model": model, "temperature": temperature, "messages": messages,
            "response_format": {"type": "json_schema", "json_schema": {**schema, "strict": True}}}
    return json.dumps({"custom_id": key, "method": "POST", "url": "/v1/chat/completions", "body": body},
                      ensure_ascii=False)


def _submit(lines, stage):
    c = client()
    f = c.files.create(file=(f"{stage}.jsonl", io.BytesIO("\n".join(lines).encode("utf-8"))), purpose="batch")
    return c.batches.create(input_file_id=f.id, endpoint="/v1/chat/completions",
                            completion_window="24h", metadata={"stage": stage})


def _wait(batch_id):
    c = client()
    while True:
        b = c.batches.retrieve(batch_id)
        n = b.request_counts
        print(f"  {batch_id[-8:]}: {b.status} {n.completed if n else 0}/{n.total if n else 0}", flush=True)
        if b.status in ("completed", "failed", "expired", "cancelled"):
            return b
        time.sleep(POLL_S)


def _store(batch, meta):
    """Write successful results to the cache; returns the number stored."""
    c = client()
    cache, stored, entries = _load_cache(), 0, []
    if batch.output_file_id:
        for line in c.files.content(batch.output_file_id).text.splitlines():
            res = json.loads(line)
            key, resp = res["custom_id"], res.get("response") or {}
            if resp.get("status_code") != 200:
                continue
            body = resp["body"]
            msg = body["choices"][0]["message"]
            if msg.get("refusal") or not msg.get("content"):
                continue
            stage, model, temperature = meta[key]
            entries.append({
                "key": key, "stage": stage, "model": model, "temperature": temperature,
                "input_tokens": body["usage"]["prompt_tokens"],
                "output_tokens": body["usage"]["completion_tokens"],
                "latency_s": None, "batch": True,
                "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "output": json.loads(msg["content"]),
            })
    with _lock:
        CACHE.mkdir(parents=True, exist_ok=True)
        with open(CACHE_FILE, "a", encoding="utf-8") as f:
            for e in entries:
                cache[e["key"]] = e
                f.write(json.dumps(e, ensure_ascii=False) + "\n")
                stored += 1
    return stored


def run(calls, stage, max_requests=500):
    """calls: list of dicts with messages, schema, and optional model/temperature.

    Calls already in the cache are skipped. The rest are split into batches of at most
    max_requests, all submitted at once (the queue limit is 20M tokens/day, far above
    this project's ~3M), then cached as each finishes.
    Returns the number of calls that are still missing afterwards.
    """
    from .config import MODEL

    cache = _load_cache()
    todo, meta = {}, {}
    for call in calls:
        model, temperature = call.get("model", MODEL), call.get("temperature", 0.0)
        key = _key(model, call["messages"], call["schema"], temperature)
        if key in cache or key in todo:
            continue
        todo[key] = _request_line(key, model, call["messages"], call["schema"], temperature)
        meta[key] = (call.get("stage", stage), model, temperature)
    print(f"{stage}: {len(calls)} calls, {len(calls) - len(todo)} cached, {len(todo)} to batch", flush=True)

    keys = list(todo)
    parts = [keys[i:i + max_requests] for i in range(0, len(keys), max_requests)]
    submitted = [(_submit([todo[k] for k in part], stage).id, part) for part in parts]
    for batch_id, part in submitted:
        b = _wait(batch_id)
        n = _store(b, meta)
        print(f"  stored {n}/{len(part)}" + (f" ({b.errors})" if b.errors else ""), flush=True)
    cache = _load_cache()
    return sum(k not in cache for k in keys)


def run_rounds(fn, stage, max_rounds=6):
    """Run fn() with every LLM call answered by the Batch API, then return fn()'s result.

    Each round runs fn in collecting mode, batches the calls it could not answer from the
    cache, and repeats; a stage whose inputs depend on an earlier stage (for example the
    claim checker reading generated claims) is picked up in the next round.
    """
    from .llm import collecting

    for r in range(max_rounds):
        with collecting() as calls:
            fn()
        if not calls:
            break
        print(f"round {r + 1}: {len(calls)} uncached calls", flush=True)
        run(calls, stage)
    return fn()
