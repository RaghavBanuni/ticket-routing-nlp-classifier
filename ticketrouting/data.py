"""Synthetic support ticket stream.

Four properties are built in on purpose, because each one is a real failure mode
that a clean dataset would hide.

**Class imbalance.**  The largest queue is about four times the smallest.  Any
metric that is not per-class will be dominated by password resets.

**Confusable intent pairs.**  ``billing_charge`` / ``refund_request`` and
``order_cancel`` / ``refund_request`` deliberately share vocabulary ("charge",
"money back", "order").  Aggregate accuracy cannot show you this; the confusion-pair
report can.

**Typos.**  Characters are swapped, dropped and doubled at a configurable rate, so
character n-grams have something to earn their place on.

**Vocabulary drift over time.**  Tickets carry a timestamp, and in the final months a
new product line and new phrasings appear that occur nowhere earlier.  This is what
makes a temporal split score lower than a random one - and the temporal number is
the honest one.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta

import numpy as np
import pandas as pd

QUEUES: tuple[str, ...] = (
    "password_reset",
    "billing_charge",
    "refund_request",
    "shipping_delay",
    "order_cancel",
    "product_defect",
    "account_deletion",
    "api_error",
)

#: Queue mix, roughly what a consumer support desk sees.  Deliberately uneven.
QUEUE_WEIGHTS: dict[str, float] = {
    "password_reset": 0.22,
    "billing_charge": 0.18,
    "shipping_delay": 0.14,
    "refund_request": 0.12,
    "api_error": 0.11,
    "product_defect": 0.10,
    "order_cancel": 0.08,
    "account_deletion": 0.05,
}

#: Phrase pools per queue.  The overlapping phrases between billing / refund /
#: cancel are intentional - that is where the classifier is supposed to struggle.
PHRASES: dict[str, tuple[str, ...]] = {
    "password_reset": (
        "i cannot log in and the reset link never arrives",
        "forgot my password and the reset email goes to spam",
        "two factor code is not accepted when i sign in",
        "locked out of my account after too many attempts",
        "password reset page keeps saying token expired",
        "need to change my password but the old one is not recognised",
    ),
    "billing_charge": (
        "i was charged twice this month for the same subscription",
        "there is an unexpected charge on my card from you",
        "my invoice shows a higher amount than the plan price",
        "you billed me after i downgraded the plan",
        "the vat on my invoice looks wrong for my country",
        "a duplicate payment was taken from my card",
    ),
    "refund_request": (
        "please refund the charge i did not authorise",
        "i want my money back for the order that arrived broken",
        "how long does a refund take to reach my card",
        "i was promised a refund two weeks ago and nothing arrived",
        "requesting a refund because the item was not as described",
        "refund the duplicate payment you took from my card",
    ),
    "shipping_delay": (
        "my parcel has not moved in the tracking for six days",
        "delivery was due last week and still has not arrived",
        "the courier keeps rescheduling my delivery",
        "tracking says delivered but nothing came to my address",
        "shipment is stuck at the sorting centre",
        "when will my order actually ship",
    ),
    "order_cancel": (
        "please cancel my order before it ships",
        "i want to cancel the order i placed by mistake",
        "cancel this order and do not dispatch it",
        "i ordered the wrong size and need to cancel",
        "can i still cancel if it has not been picked yet",
        "cancel the order and put the money back on my card",
    ),
    "product_defect": (
        "the device stopped charging after two days of use",
        "the screen has a dead pixel straight out of the box",
        "the item arrived with a cracked casing",
        "it rattles when shaken and the button does not click",
        "the unit overheats and shuts itself down",
        "the seal was broken and parts are missing",
    ),
    "account_deletion": (
        "please delete my account and all my personal data",
        "i want to exercise my right to erasure under gdpr",
        "close my account permanently and remove my details",
        "remove my data from your systems and confirm in writing",
        "i no longer consent to you storing my information",
        "delete my profile and unsubscribe me from everything",
    ),
    "api_error": (
        "the api returns 500 on every post to the orders endpoint",
        "webhook deliveries stopped arriving this morning",
        "i get a 401 with a token that worked yesterday",
        "rate limit errors even though we send fewer calls",
        "the sdk throws a timeout on the sandbox environment",
        "pagination cursor returns duplicate records",
    ),
}

#: Boilerplate that appears across every queue, so a model cannot key on it.
OPENERS: tuple[str, ...] = (
    "hi team",
    "hello support",
    "good morning",
    "hi there",
    "dear support team",
    "",
)
CLOSERS: tuple[str, ...] = (
    "thanks in advance",
    "please advise",
    "any help appreciated",
    "regards",
    "looking forward to your reply",
    "",
)

#: Vocabulary that only exists in the drift period.  A temporal split has never
#: seen these tokens at training time; a random split has.
DRIFT_PRODUCT: str = "nimbus"
DRIFT_PHRASES: dict[str, tuple[str, ...]] = {
    "api_error": (
        "the nimbus gateway rejects our idempotency key",
        "nimbus streaming endpoint closes the connection midway",
    ),
    "product_defect": (
        "the nimbus hub loses pairing with the remote",
        "my nimbus speaker crackles at low volume",
    ),
    "billing_charge": (
        "i was migrated to nimbus pricing without being told",
        "nimbus plan invoice shows a proration i do not understand",
    ),
}


@dataclass(frozen=True)
class StreamConfig:
    """Shape of the generated ticket stream.

    n_tickets:
        Total tickets across the whole window.
    months:
        Length of the window; tickets are spread uniformly across it.
    drift_start_fraction:
        Point in the window where the new product vocabulary starts appearing.
    drift_share:
        Within the drift period, share of eligible tickets that use it.
    typo_rate:
        Per-character probability of a typo.
    """

    n_tickets: int = 9_000
    months: int = 12
    drift_start_fraction: float = 0.75
    drift_share: float = 0.45
    typo_rate: float = 0.015
    seed: int = 29
    weights: dict[str, float] = field(default_factory=lambda: dict(QUEUE_WEIGHTS))

    def validate(self) -> None:
        if self.n_tickets < len(QUEUES) * 20:
            raise ValueError("need at least twenty tickets per queue")
        if self.months < 3:
            raise ValueError("the window must span at least three months")
        if not 0.0 < self.drift_start_fraction < 1.0:
            raise ValueError("drift_start_fraction must lie in (0, 1)")
        if not 0.0 <= self.drift_share <= 1.0:
            raise ValueError("drift_share must lie in [0, 1]")
        if not 0.0 <= self.typo_rate <= 0.2:
            raise ValueError("typo_rate must lie in [0, 0.2]")
        if set(self.weights) != set(QUEUES):
            raise ValueError("weights must cover exactly the known queues")


def _typo(text: str, rate: float, rng: np.random.Generator) -> str:
    """Swap, drop or double characters at ``rate`` per character."""
    if rate <= 0.0:
        return text
    characters = list(text)
    output: list[str] = []
    index = 0
    while index < len(characters):
        character = characters[index]
        if character != " " and rng.random() < rate:
            choice = rng.integers(0, 3)
            if choice == 0 and index + 1 < len(characters):  # swap
                output.append(characters[index + 1])
                output.append(character)
                index += 2
                continue
            if choice == 1:  # drop
                index += 1
                continue
            output.append(character)  # double
        output.append(character)
        index += 1
    return "".join(output)


def _order_reference(rng: np.random.Generator) -> str:
    """Ticket-specific noise: order ids, amounts, dates.

    High-cardinality tokens like these are pure noise for classification, and a
    model that starts using them is overfitting - which the explanation report will
    show.
    """
    style = rng.integers(0, 4)
    if style == 0:
        return f"order {rng.integers(10_000, 99_999)}"
    if style == 1:
        return f"invoice inv-{rng.integers(1_000, 9_999)}"
    if style == 2:
        return f"amount {rng.integers(5, 400)}.{rng.integers(10, 99)} eur"
    return f"reference {rng.integers(100_000, 999_999)}"


def generate_tickets(config: StreamConfig | None = None) -> pd.DataFrame:
    """Generate a labelled, timestamped ticket stream."""
    settings = config or StreamConfig()
    settings.validate()
    rng = np.random.default_rng(settings.seed)

    queues = list(settings.weights)
    probabilities = np.array([settings.weights[queue] for queue in queues], dtype=float)
    probabilities = probabilities / probabilities.sum()
    labels = rng.choice(queues, size=settings.n_tickets, p=probabilities)

    start = date(2024, 1, 1)
    window_days = settings.months * 30
    offsets = np.sort(rng.integers(0, window_days, size=settings.n_tickets))
    drift_begins = int(window_days * settings.drift_start_fraction)

    rows: list[dict[str, object]] = []
    for index, (label, offset) in enumerate(zip(labels, offsets, strict=True)):
        in_drift = offset >= drift_begins
        drifted = (
            in_drift
            and label in DRIFT_PHRASES
            and rng.random() < settings.drift_share
        )
        pool = DRIFT_PHRASES[label] if drifted else PHRASES[label]
        body = str(rng.choice(pool))

        parts = [str(rng.choice(OPENERS)), body]
        if rng.random() < 0.55:
            parts.append(_order_reference(rng))
        parts.append(str(rng.choice(CLOSERS)))
        text = " ".join(part for part in parts if part).strip()

        rows.append(
            {
                "ticket_id": f"T{index:06d}",
                "created_at": pd.Timestamp(start + timedelta(days=int(offset))),
                "text": _typo(text, settings.typo_rate, rng),
                "queue": label,
                "drifted": bool(drifted),
            }
        )

    frame = pd.DataFrame(rows)
    return frame.sort_values("created_at").reset_index(drop=True)


def class_balance(frame: pd.DataFrame, label: str = "queue") -> pd.DataFrame:
    """Ticket count and share per queue, largest first."""
    counts = frame[label].value_counts()
    table = pd.DataFrame(
        {
            "queue": counts.index,
            "tickets": counts.to_numpy(),
            "share": (counts / len(frame)).round(4).to_numpy(),
        }
    )
    table["imbalance_vs_largest"] = (table["tickets"].max() / table["tickets"]).round(2)
    return table.reset_index(drop=True)


def monthly_volume(frame: pd.DataFrame) -> pd.DataFrame:
    """Tickets per month per queue, plus how many use the drift vocabulary."""
    monthly = frame.assign(month=frame["created_at"].dt.to_period("M").astype(str))
    return (
        monthly.groupby("month")
        .agg(tickets=("ticket_id", "count"), drifted=("drifted", "sum"))
        .assign(drift_share=lambda table: (table["drifted"] / table["tickets"]).round(4))
        .reset_index()
    )
