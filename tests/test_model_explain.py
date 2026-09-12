"""Tests for the splits, the models and the calibrated confidence contract."""

from __future__ import annotations

import numpy as np
import pytest

from ticketrouting.evaluate import classification_metrics
from ticketrouting.explain import boilerplate_alarms, boilerplate_terms, top_features_per_class
from ticketrouting.features import normalise_series
from ticketrouting.model import (
    LABEL_COLUMN,
    TEXT_COLUMN,
    build_model,
    fit,
    predict_with_confidence,
    temporal_split,
)


def test_temporal_split_never_lets_the_future_into_training(splits):
    train, test = splits
    assert train["created_at"].max() <= test["created_at"].min()


def test_temporal_split_keeps_every_queue_in_training(splits):
    train, test = splits
    assert set(test[LABEL_COLUMN]) <= set(train[LABEL_COLUMN])


def test_temporal_split_validates_its_fraction(stream):
    with pytest.raises(ValueError):
        temporal_split(stream, test_fraction=0.95)


def test_unknown_model_kind_is_rejected():
    with pytest.raises(ValueError, match="kind must be one of"):
        build_model("transformer")


def test_model_beats_the_majority_baseline(splits):
    train, test = splits
    truth = test[LABEL_COLUMN]
    prepared = normalise_series(test[TEXT_COLUMN])

    baseline = fit(train, "baseline", "word", seed=5).predict(prepared)
    logistic = fit(train, "logistic", "word", seed=5).predict(prepared)

    baseline_score = classification_metrics(truth, baseline)
    model_score = classification_metrics(truth, logistic)
    assert model_score["macro_f1"] > baseline_score["macro_f1"] * 2
    assert model_score["balanced_accuracy"] > 0.5


def test_baseline_only_ever_predicts_one_queue(splits):
    train, test = splits
    predictions = fit(train, "baseline", "word", seed=5).predict(normalise_series(test[TEXT_COLUMN]))
    assert len(set(predictions)) == 1


def test_confidence_requires_a_calibrated_model(splits):
    """A linear SVM margin is not a probability, so thresholding it is refused."""
    train, test = splits
    raw_svm = fit(train, "linear_svm", "word", seed=5)
    with pytest.raises(TypeError, match="calibrate"):
        predict_with_confidence(raw_svm, test[TEXT_COLUMN])


def test_calibrated_confidence_is_a_probability(scored):
    _truth, frame = scored
    assert frame["confidence"].between(0.0, 1.0).all()
    assert (frame["confidence"] >= frame["runner_up_confidence"]).all()


def test_runner_up_is_never_the_prediction(scored):
    _truth, frame = scored
    assert (frame["predicted"] != frame["runner_up"]).all()


def test_calibrated_model_is_usefully_accurate(scored):
    truth, frame = scored
    metrics = classification_metrics(truth, frame["predicted"].to_numpy())
    assert metrics["macro_f1"] > 0.5
    assert metrics["classes_predicted"] >= 6


def test_a_clear_billing_ticket_is_routed_to_billing(calibrated):
    frame = predict_with_confidence(
        calibrated, ["i was charged twice this month for the same subscription"]
    )
    assert frame["predicted"].iloc[0] in {"billing_charge", "refund_request"}


def test_a_clear_password_ticket_is_routed_to_password_reset(calibrated):
    frame = predict_with_confidence(
        calibrated, ["i cannot log in and the password reset link never arrives"]
    )
    assert frame["predicted"].iloc[0] == "password_reset"


# --------------------------------------------------------------- explanations
def test_top_features_cover_every_queue(fitted):
    table = top_features_per_class(fitted, top_n=8)
    assert table["queue"].nunique() == len(np.unique(fitted.named_steps["model"].classes_))
    assert (table.groupby("queue").size() == 8).all()
    assert set(table["kind"]) <= {"word", "char"}


def test_top_features_are_refused_for_naive_bayes_free_models(splits):
    train, _test = splits
    model = fit(train, "baseline", "word", seed=5)
    with pytest.raises(TypeError):
        top_features_per_class(model)


def test_boilerplate_alarms_only_flag_boilerplate(fitted):
    suspicious = boilerplate_terms()
    alarms = boilerplate_alarms(fitted, top_n=10)
    for feature in alarms["feature"]:
        assert all(part in suspicious for part in feature.split())


def test_masked_identifiers_are_treated_as_boilerplate():
    assert "<num>" in boilerplate_terms()
