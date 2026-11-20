"""Stage 2: aspect-based sentiment extraction with structured JSON output.

For each review the model lists the product aspects it discusses. Each aspect has:
    term      the exact words naming the aspect ("battery life"), or null if only implied
    category  one of CATEGORIES, so aspects can be counted across reviews
    polarity  positive / negative / neutral, for this aspect only
    evidence  a short verbatim quote that supports the polarity

The same prompt is used on the Amazon reviews (application) and on SemEval-2014 sentences
(external benchmark); there, only aspects with an explicit term are scored.
"""

from .llm import chat_json, map_parallel

CATEGORIES = {
    "performance": "speed, processor, graphics, multitasking, gaming, boot time",
    "battery": "battery life, charging, charger",
    "display": "screen, resolution, brightness, colors, touchscreen",
    "keyboard_touchpad": "keyboard, keys, backlight, touchpad, trackpad",
    "build_design": "build quality, materials, hinge, looks, size",
    "portability": "weight, thickness, carrying it around",
    "audio_camera": "speakers, sound, webcam, microphone",
    "ports_connectivity": "USB, HDMI, ports, Wi-Fi, Bluetooth",
    "storage_memory": "RAM, SSD, eMMC, storage space",
    "software_os": "Windows, ChromeOS, macOS, S mode, preinstalled software, updates, drivers",
    "thermals_noise": "heat, fan, noise",
    "reliability": "defects, failures, dead on arrival, durability over time",
    "price_value": "price, value for money",
    "seller_support": "shipping, packaging, condition on arrival, returns, warranty, customer service",
    "other": "anything else about the product",
}
POLARITIES = ["positive", "negative", "neutral"]

SYSTEM = """You extract aspect-level opinions from customer reviews of laptops.

List every aspect of the laptop (or of the purchase) that the review talks about. For each aspect give:
- term: the exact words from the review that name the aspect, copied character for character (for example "battery life", "keyboard", "screen"). Do not include opinion words in the term. Use null if the aspect is only implied (for example "it is so slow" implies performance but names no term). Do not use words for the laptop as a whole ("laptop", "computer", "it") as a term.
- category: the best matching category from the list below.
- polarity: the reviewer's opinion of that aspect: positive, negative, or neutral (mentioned without an opinion).
- evidence: a short quote copied exactly from the review (at most 25 words) that shows the opinion.

Categories:
""" + "\n".join(f"- {k}: {v}" for k, v in CATEGORIES.items()) + """

If the review discusses no specific aspect (for example "Great laptop!"), return an empty list."""

SCHEMA = {
    "name": "review_aspects",
    "schema": {
        "type": "object",
        "properties": {
            "aspects": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "term": {"type": ["string", "null"]},
                        "category": {"type": "string", "enum": list(CATEGORIES)},
                        "polarity": {"type": "string", "enum": POLARITIES},
                        "evidence": {"type": "string"},
                    },
                    "required": ["term", "category", "polarity", "evidence"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["aspects"],
        "additionalProperties": False,
    },
}


def build_messages(text, title=None):
    content = f"Review title: {title}\nReview text: {text}" if title else f"Review text: {text}"
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": content}]


def extract(text, title=None, stage="aspects", **kw):
    return chat_json(build_messages(text, title), SCHEMA, stage=stage, **kw)["aspects"]


def evidence_found(evidence, title, text):
    """True if the quoted evidence appears verbatim (case/space-insensitive) in the review."""
    norm = lambda s: " ".join((s or "").lower().split())
    return norm(evidence) in norm(f"{title} {text}")


def extract_frame(df, workers=8, **kw):
    """Extract aspects for every row of a DataFrame with title/text columns."""
    rows = list(zip(df["title"], df["text"]))
    return map_parallel(lambda r: extract(r[1], r[0], **kw), rows, workers=workers, desc="aspects")
