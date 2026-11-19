"""Synthetic support-log generator with programmatic ground truth (PRD 5, 9.1).

Labels (category, sentiment, refund) are expressed only through phrasing; the category
words themselves never appear in ticket text, so a model has to infer them.
"""
from __future__ import annotations

import json
import random
from datetime import datetime, timedelta, timezone

from mosaic.tokens import count_tokens
from mosaic.types import Ticket

PRODUCTS = {
    "Router X2": 129.99, "Mesh Pro": 249.00, "Cloud Plan": 59.99, "Switch 8": 89.50,
    "Cam Mini": 44.99, "Hub Max": 179.00, "VPN Plus": 39.99, "Repeater S": 34.95,
}
CATEGORIES = ["billing", "shipping", "defect", "account", "other"]
CATEGORY_WEIGHTS = [0.24, 0.22, 0.26, 0.16, 0.12]
REFUND_RATE = {"billing": 0.45, "shipping": 0.30, "defect": 0.35, "account": 0.05, "other": 0.02}
START = datetime(2026, 1, 1, tzinfo=timezone.utc)
SPAN_SECONDS = int((datetime(2026, 7, 1, tzinfo=timezone.utc) - START).total_seconds())
ZIPF_S = 1.1
CANONICAL_REFUND_SHARE = 0.4  # so at least 30% of refund tickets avoid the word "refund"

ISSUE = {
    "billing": [
        "I was charged twice for my {p} this month.",
        "My invoice shows a fee for {p} that I never agreed to.",
        "The amount taken from my card does not match the price I was quoted for {p}.",
        "There is a duplicate charge on my bank statement for {p}.",
        "Your payment system took money for a {p} plan I cancelled weeks ago.",
        "The renewal price for {p} went up without any notice to me.",
        "I was billed in the wrong currency for {p}.",
    ],
    "shipping": [
        "My {p} still has not arrived after two weeks.",
        "The courier left the {p} package at the wrong address.",
        "Tracking for my {p} has said in transit for ten days.",
        "The box with my {p} arrived crushed and two days late.",
        "Delivery of the {p} was promised for Monday and nothing came.",
        "I received someone else's parcel instead of my {p}.",
        "The {p} was sent to my old address even though I updated it.",
    ],
    "defect": [
        "My {p} stopped working after three days.",
        "The {p} keeps rebooting every few minutes.",
        "The power light on the {p} blinks red and it will not connect.",
        "The {p} died after a week of normal use.",
        "The {p} overheats and shuts itself off.",
        "The {p} drops the connection whenever two devices are on it.",
        "One of the ports on the {p} is physically loose and does nothing.",
    ],
    "account": [
        "I cannot log in to my {p} profile anymore.",
        "The password reset email for {p} never arrives.",
        "Someone changed the email on my {p} login without my permission.",
        "Two-factor codes for {p} are rejected every time.",
        "My {p} user profile got locked after the latest update.",
        "I cannot add a second user to our {p} workspace.",
    ],
    "other": [
        "Do you have a guide for setting up guest Wi-Fi on the {p}?",
        "Is there a printed manual for the {p}?",
        "Can I use the {p} with a third-party app?",
        "Does the {p} work in Canada as well?",
        "I would love a dark mode in the {p} app.",
        "What colours does the {p} come in?",
    ],
}
REFUND_CANONICAL = [
    "I would like a refund of ${a}.", "Please refund the ${a}.",
    "Your agent approved a refund of ${a}.", "I am requesting a refund for ${a}.",
]
REFUND_PARAPHRASE = [
    "I want my ${a} back.", "Please reimburse me the ${a}.", "Can you reverse the ${a} charge?",
    "Your agent confirmed ${a} would be credited back to my card.", "Send the ${a} back to me.",
    "I expect the ${a} returned to my card this week.", "Put the ${a} back on my card, please.",
    "I was told the ${a} would be paid back to me.",
]
# Mentions of money or refunds that are NOT refunds: they break naive keyword filters.
DISTRACTOR = [
    "I paid ${a} for this and expected better.", "The ${a} price is fair otherwise.",
    "I do not need a refund, I just want it fixed.", "I am not asking for my money back yet.",
    "A friend got a refund for a different issue, but I just want help.",
]
SENTIMENT = {
    "positive": ["Thanks so much, your team has always been great.", "Really happy with you overall.",
                 "Appreciate the quick help last time.", "Love the product apart from this."],
    "neutral": ["Let me know the next steps.", "Please advise.", "Logging this for your records.",
                "I can share more details if needed."],
    "negative": ["This is completely unacceptable.", "I am furious and thinking of switching providers.",
                 "Worst experience I have had with a vendor.", "I am losing patience with you."],
}
SENTIMENT_WEIGHTS = {"positive": 0.25, "neutral": 0.4, "negative": 0.35}
FILLER = [
    "I have been a customer since 2023.", "We use it in a small office with six people.",
    "I already tried restarting it twice.", "My order number should be on file.",
    "I contacted chat support yesterday but the session dropped.", "This is my second message about it.",
    "I set it up exactly as the quick start guide says.", "My partner uses the same model without trouble.",
    "I work from home so this matters a lot.", "I am writing from my phone, sorry for typos.",
    "The firmware is the latest version.", "I checked the FAQ before writing.",
    "I bought it from your online store in the spring.", "Our building has thick concrete walls.",
    "I followed the steps your colleague sent me last week.", "I have attached a photo to my previous email.",
    "The issue started right after I moved house.", "My neighbour recommended your brand to me.",
    "I need this sorted before the weekend.", "I tried a different cable and nothing changed.",
    "Your website status page shows everything as normal.", "I have kept the original box and receipt.",
    "I run a small design studio and we rely on it daily.", "I can be reached on this email any time.",
]
GREETING = ["Hi,", "Hello team,", "Good morning,", "Hey support,", "Hi there,", "Dear support,"]
SIGNOFF = ["Thanks.", "Regards.", "Cheers.", "Best.", "Thank you."]
AVG_TICKET_TOKENS = 120


def _amount(rng: random.Random, product: str) -> float:
    price = PRODUCTS[product]
    return round(price if rng.random() < 0.6 else price * rng.choice([0.25, 0.5, 0.2, 0.75]), 2)


def _ticket(rng: random.Random, idx: int, ts: datetime, customer: str) -> Ticket:
    product = rng.choice(list(PRODUCTS))
    category = rng.choices(CATEGORIES, CATEGORY_WEIGHTS)[0]
    sentiment = rng.choices(list(SENTIMENT_WEIGHTS), list(SENTIMENT_WEIGHTS.values()))[0]
    refund = _amount(rng, product) if rng.random() < REFUND_RATE[category] else None
    parts = [rng.choice(GREETING), rng.choice(ISSUE[category]).format(p=product)]
    parts += rng.sample(FILLER, rng.randint(4, 8))
    if refund is not None:
        bank = REFUND_CANONICAL if rng.random() < CANONICAL_REFUND_SHARE else REFUND_PARAPHRASE
        parts.append(rng.choice(bank).format(a=f"{refund:.2f}"))
    elif rng.random() < 0.3:
        parts.append(rng.choice(DISTRACTOR).format(a=f"{_amount(rng, product):.2f}"))
    parts += [rng.choice(SENTIMENT[sentiment]), rng.choice(SIGNOFF)]
    head, body, tail = parts[:2], parts[2:-2], parts[-2:]
    rng.shuffle(body)
    return Ticket(
        ticket_id=f"T-{idx:05d}", ts=ts.strftime("%Y-%m-%dT%H:%M:%SZ"), customer_id=customer,
        product=product, category=category, refund_amount=refund, sentiment=sentiment,
        text=" ".join(head + body + tail),
    )


def generate(n_tokens: int, seed: int) -> list[Ticket]:
    """Generate tickets until the corpus reaches n_tokens (proxy tokenizer), sorted by time."""
    rng = random.Random(f"mosaic-gen-{seed}-{n_tokens}")
    n_est = max(10, n_tokens // AVG_TICKET_TOKENS)
    pool = max(20, n_est // 3)
    weights = [1.0 / (r ** ZIPF_S) for r in range(1, pool + 1)]
    customers = [f"C-{r:04d}" for r in range(1, pool + 1)]
    drafts, total = [], 0
    while total < n_tokens:
        ts = START + timedelta(seconds=rng.randrange(SPAN_SECONDS))
        t = _ticket(rng, 0, ts, rng.choices(customers, weights)[0])
        if total + count_tokens(t.line()) + 1 > n_tokens * 1.02 and drafts:
            break
        drafts.append(t)
        total += count_tokens(t.line()) + 1
    drafts.sort(key=lambda t: t.ts)
    first = rng.randrange(1000, 9000)
    return [t.model_copy(update={"ticket_id": f"T-{first + i:05d}"}) for i, t in enumerate(drafts)]


def corpus_text(tickets: list[Ticket]) -> str:
    return "\n".join(t.line() for t in tickets)


def dumps(tickets: list[Ticket]) -> str:
    return "\n".join(json.dumps(t.model_dump(), sort_keys=True) for t in tickets) + "\n"


def loads(text: str) -> list[Ticket]:
    return [Ticket(**json.loads(line)) for line in text.splitlines() if line.strip()]


SIZES = {"8k": 8_000, "32k": 32_000, "128k": 128_000, "512k": 512_000}
