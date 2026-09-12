"""Tests for the ticket stream and the feature space.

The generator's deliberate defects - imbalance, confusable pairs, typos, vocabulary
drift - are asserted here.  If a refactor removes the drift, the temporal-split
lesson silently becomes a tautology, so the drift itself is tested.
"""

from __future__ import annotations

import pandas as pd
import pytest

from ticketrouting.data import (
    DRIFT_PRODUCT,
    PHRASES,
    QUEUES,
    StreamConfig,
    class_balance,
    generate_tickets,
    monthly_volume,
)
from ticketrouting.features import (
    NUMBER_TOKEN,
    build_vectorizer,
    normalise_text,
    oov_rate,
    tokens,
    unseen_terms,
)
from ticketrouting.model import TEXT_COLUMN, random_split, temporal_split


def test_stream_is_reproducible():
    first = generate_tickets(StreamConfig(n_tickets=600, seed=3))
    second = generate_tickets(StreamConfig(n_tickets=600, seed=3))
    pd.testing.assert_frame_equal(first, second)


def test_every_queue_is_represented(stream):
    assert set(stream["queue"]) == set(QUEUES)


def test_classes_are_imbalanced(stream):
    balance = class_balance(stream)
    assert balance["imbalance_vs_largest"].max() > 2.5
    assert balance["tickets"].sum() == len(stream)


def test_tickets_are_ordered_in_time(stream):
    assert stream["created_at"].is_monotonic_increasing


def test_drift_vocabulary_only_appears_late(stream):
    """The new product name must be absent from the early period."""
    drifted = stream[stream["text"].str.contains(DRIFT_PRODUCT, case=False)]
    assert not drifted.empty
    first_use = drifted["created_at"].min()
    early = stream[stream["created_at"] < first_use]
    assert len(early) > len(stream) * 0.5
    assert not early["text"].str.contains(DRIFT_PRODUCT, case=False).any()


def test_confusable_queues_share_vocabulary():
    """Billing, refund and cancel overlap on purpose - that is the hard part."""
    billing = " ".join(PHRASES["billing_charge"])
    refund = " ".join(PHRASES["refund_request"])
    cancel = " ".join(PHRASES["order_cancel"])
    assert set(tokens(billing)) & set(tokens(refund))
    assert set(tokens(refund)) & set(tokens(cancel))


def test_typos_change_the_text_but_not_the_labels():
    clean = generate_tickets(StreamConfig(n_tickets=600, typo_rate=0.0, seed=11))
    noisy = generate_tickets(StreamConfig(n_tickets=600, typo_rate=0.06, seed=11))
    assert (clean["queue"] == noisy["queue"]).all()
    assert (clean["text"] != noisy["text"]).mean() > 0.3


def test_monthly_volume_accounts_for_every_ticket(stream):
    monthly = monthly_volume(stream)
    assert monthly["tickets"].sum() == len(stream)
    assert monthly["drift_share"].iloc[-1] > monthly["drift_share"].iloc[0]


def test_invalid_stream_configuration_is_rejected():
    with pytest.raises(ValueError):
        generate_tickets(StreamConfig(n_tickets=10))
    with pytest.raises(ValueError):
        generate_tickets(StreamConfig(typo_rate=0.5))
    with pytest.raises(ValueError):
        generate_tickets(StreamConfig(drift_start_fraction=1.0))


# ---------------------------------------------------------------- feature space
def test_identifiers_are_masked_by_default():
    assert normalise_text("Order 88213 refunded 45.20 EUR!") == (
        f"order {NUMBER_TOKEN} refunded {NUMBER_TOKEN}.{NUMBER_TOKEN} eur".replace(".", " ")
    )


def test_masking_can_be_turned_off():
    assert "88213" in normalise_text("order 88213", mask_numbers=False)


def test_normalisation_is_idempotent():
    once = normalise_text("Hi!! I was CHARGED  twice, order 12")
    assert normalise_text(once) == once


def test_feature_kinds_produce_different_spaces(stream):
    texts = stream[TEXT_COLUMN].head(400)
    sizes = {
        kind: build_vectorizer(kind).fit_transform(texts).shape[1]
        for kind in ("word", "char", "union")
    }
    assert sizes["union"] == sizes["word"] + sizes["char"]
    assert sizes["char"] > 0


def test_unknown_feature_kind_is_rejected():
    with pytest.raises(ValueError, match="kind must be one of"):
        build_vectorizer("embeddings")


def test_temporal_split_has_more_unseen_vocabulary_than_a_random_one(stream):
    """This is the mechanism behind the score gap, measured rather than asserted."""
    train, test = temporal_split(stream, 0.25)
    random_train, random_test = random_split(stream, 0.25, seed=5)
    temporal_oov = oov_rate(train[TEXT_COLUMN], test[TEXT_COLUMN])
    random_oov = oov_rate(random_train[TEXT_COLUMN], random_test[TEXT_COLUMN])
    assert temporal_oov["token_oov_rate"] > random_oov["token_oov_rate"]


def test_the_new_product_name_is_reported_as_unseen(stream):
    train, test = temporal_split(stream, 0.25)
    terms = unseen_terms(train[TEXT_COLUMN], test[TEXT_COLUMN], top_n=25)
    assert DRIFT_PRODUCT in set(terms["term"])
