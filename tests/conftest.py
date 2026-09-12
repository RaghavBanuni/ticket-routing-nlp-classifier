"""Shared fixtures.

The fitted-model fixtures use the word feature space and a small stream so the
suite stays fast; the union space is exercised once, in the feature tests.
"""

from __future__ import annotations

import pytest

from ticketrouting.data import StreamConfig, generate_tickets
from ticketrouting.model import (
    LABEL_COLUMN,
    TEXT_COLUMN,
    calibrate,
    fit,
    predict_with_confidence,
    temporal_split,
)


@pytest.fixture(scope="session")
def stream():
    return generate_tickets(StreamConfig(n_tickets=2_400, seed=5))


@pytest.fixture(scope="session")
def splits(stream):
    return temporal_split(stream, test_fraction=0.25)


@pytest.fixture(scope="session")
def fitted(splits):
    train, _test = splits
    return fit(train, "logistic", "word", seed=5)


@pytest.fixture(scope="session")
def calibrated(splits):
    train, _test = splits
    return calibrate(train, "linear_svm", "word", folds=3, seed=5)


@pytest.fixture(scope="session")
def scored(calibrated, splits):
    _train, test = splits
    frame = predict_with_confidence(calibrated, test[TEXT_COLUMN])
    return test[LABEL_COLUMN].to_numpy(), frame
