"""Example inputs shown as buttons in the app's live tabs (pre-cached by scripts/08)."""

EXAMPLE_QUESTIONS = [
    "How long does the battery last?",
    "Is it good for gaming?",
    "Does it have a fingerprint reader?",  # edge case: reviews rarely say, so the app should decline
]

EXAMPLE_REVIEWS = {
    "Mixed review": ("Good value, but...",
                     "The screen is bright and sharp and it boots fast. But the battery barely lasts 3 hours "
                     "and the trackpad is jumpy. For the price I'm keeping it."),
    "Sarcasm": ("Wow, just wow",
                "Oh great, another laptop that dies after 40 minutes unplugged. Love waiting for Windows "
                "updates every single morning. Five stars for the box it came in."),
    "Shipping only": ("Arrived late",
                      "Package showed up a week after the promised date and the box was crushed. "
                      "Haven't even turned it on yet."),
}
