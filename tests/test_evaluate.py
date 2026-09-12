"""Tests for the metrics, the deferral layer and calibration.

The deferral arithmetic is checked on a four-ticket example small enough to verify by
hand, because coverage and selective accuracy are the numbers an operations team will
negotiate over - an off-by-one in the mask would be invisible in aggregate reporting.
"""

from __future__ import annotations

import numpy as np
import pytest

from ticketrouting.evaluate import (
    classification_metrics,
    confidence_calibration,
    confusion_pairs,
    deferral_curve,
    deferral_value,
    expected_calibration_error,
    overconfidence,
    per_class_report,
    threshold_for_target_accuracy,
)

TRUTH = np.array(["a", "a", "b", "b"])
PREDICTED = np.array(["a", "b", "b", "b"])
CONFIDENCE = np.array([0.9, 0.8, 0.7, 0.4])
GRID = (0.0, 0.5, 0.75, 0.85)


def test_metrics_are_hand_computable():
    metrics = classification_metrics(TRUTH, PREDICTED)
    assert metrics["tickets"] == 4
    assert metrics["accuracy"] == pytest.approx(0.75)
    # class a: precision 1.0, recall 0.5 -> f1 2/3 ; class b: precision 2/3, recall 1.0 -> f1 0.8
    assert metrics["macro_f1"] == pytest.approx((2 / 3 + 0.8) / 2, abs=1e-5)
    assert metrics["balanced_accuracy"] == pytest.approx(0.75)


def test_accuracy_and_macro_f1_disagree_under_imbalance():
    """Ninety correct majority tickets and a dead minority class: 0.9 accuracy, poor F1."""
    truth = np.array(["big"] * 90 + ["small"] * 10)
    predicted = np.array(["big"] * 100)
    metrics = classification_metrics(truth, predicted)
    assert metrics["accuracy"] == pytest.approx(0.9)
    assert metrics["macro_f1"] < 0.5
    assert metrics["classes_predicted"] == 1


def test_per_class_report_sorts_the_weakest_queue_first():
    report = per_class_report(TRUTH, PREDICTED)
    assert list(report["queue"]) == ["a", "b"]
    assert report["support"].sum() == 4


def test_confusion_pairs_name_the_mistake():
    pairs = confusion_pairs(TRUTH, PREDICTED)
    assert len(pairs) == 1
    row = pairs.iloc[0]
    assert (row["true"], row["predicted"]) == ("a", "b")
    assert row["errors"] == 1
    assert row["share_of_true_class"] == pytest.approx(0.5)


def test_confusion_pairs_are_empty_for_a_perfect_model():
    assert confusion_pairs(TRUTH, TRUTH).empty


def test_deferral_curve_is_exact():
    curve = deferral_curve(TRUTH, PREDICTED, CONFIDENCE, thresholds=GRID).set_index("threshold")

    assert curve.loc[0.0, "coverage"] == pytest.approx(1.0)
    assert curve.loc[0.0, "selective_accuracy"] == pytest.approx(0.75)

    # threshold 0.75 routes the two most confident tickets, one of which is wrong
    assert curve.loc[0.75, "routed"] == 2
    assert curve.loc[0.75, "coverage"] == pytest.approx(0.5)
    assert curve.loc[0.75, "selective_accuracy"] == pytest.approx(0.5)
    assert curve.loc[0.75, "deferred"] == 2

    # threshold 0.85 routes only the single correct top ticket
    assert curve.loc[0.85, "routed"] == 1
    assert curve.loc[0.85, "selective_accuracy"] == pytest.approx(1.0)
    assert curve.loc[0.85, "errors_routed"] == 0


def test_coverage_falls_as_the_threshold_rises(scored):
    truth, frame = scored
    curve = deferral_curve(truth, frame["predicted"], frame["confidence"])
    assert curve["coverage"].is_monotonic_decreasing
    assert curve["coverage"].iloc[0] == pytest.approx(1.0)


def test_deferring_the_least_confident_tickets_raises_accuracy(scored):
    """If it did not, the confidence score would carry no usable information."""
    truth, frame = scored
    curve = deferral_curve(truth, frame["predicted"], frame["confidence"])
    full_coverage = curve.iloc[0]["selective_accuracy"]
    tightest = curve.iloc[-1]["selective_accuracy"]
    assert tightest >= full_coverage


def test_target_accuracy_picks_the_widest_feasible_coverage():
    curve = deferral_curve(TRUTH, PREDICTED, CONFIDENCE, thresholds=GRID)
    chosen = threshold_for_target_accuracy(curve, target=1.0)
    assert chosen["threshold"] == pytest.approx(0.85)
    assert chosen["selective_accuracy"] == pytest.approx(1.0)


def test_an_unreachable_target_is_an_error_not_a_silent_miss():
    curve = deferral_curve(TRUTH, PREDICTED, np.array([0.5, 0.5, 0.5, 0.5]), thresholds=(0.0, 0.5))
    with pytest.raises(ValueError, match="never reaches"):
        threshold_for_target_accuracy(curve, target=0.99)


def test_deferral_costs_add_up():
    curve = deferral_curve(TRUTH, PREDICTED, CONFIDENCE, thresholds=GRID)
    costed = deferral_value(curve, cost_per_manual_ticket=4.0, cost_per_misroute=14.0)
    row = costed[costed["threshold"] == 0.75].iloc[0]
    assert row["manual_cost"] == pytest.approx(8.0)     # 2 deferred x 4
    assert row["misroute_cost"] == pytest.approx(14.0)  # 1 misroute x 14
    assert row["total_cost"] == pytest.approx(22.0)
    assert costed["total_cost"].is_monotonic_increasing


def test_misaligned_confidence_is_rejected():
    with pytest.raises(ValueError, match="align"):
        deferral_curve(TRUTH, PREDICTED, np.array([0.5, 0.5]))


def test_confidence_outside_zero_one_is_rejected():
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        deferral_curve(TRUTH, PREDICTED, np.array([1.4, 0.5, 0.5, 0.5]))


# --------------------------------------------------------------- calibration
def test_perfect_calibration_has_no_error():
    confidence = np.array([1.0] * 20)
    correct = np.array([True] * 20)
    assert expected_calibration_error(confidence, correct) == pytest.approx(0.0)
    assert overconfidence(confidence, correct) == pytest.approx(0.0)


def test_overconfidence_is_detected():
    confidence = np.array([0.95] * 100)
    correct = np.array([True] * 60 + [False] * 40)
    assert overconfidence(confidence, correct) == pytest.approx(0.35, abs=1e-6)
    assert expected_calibration_error(confidence, correct) > 0.3


def test_calibration_table_accounts_for_every_ticket(scored):
    truth, frame = scored
    correct = truth == frame["predicted"].to_numpy()
    table = confidence_calibration(frame["confidence"], correct)
    assert table["tickets"].sum() == len(truth)


def test_calibrated_confidence_is_not_wildly_off(scored):
    truth, frame = scored
    correct = truth == frame["predicted"].to_numpy()
    assert expected_calibration_error(frame["confidence"], correct) < 0.15
