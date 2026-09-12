"""Evaluation for an imbalanced router that is allowed to say "I don't know".

Accuracy is computed and then largely ignored: on this queue mix, always predicting
``password_reset`` scores about 22%, and a model that quietly abandons the two
smallest queues can still look respectable.  The decisions are made on macro-F1,
per-class recall, the confusion pairs, and the deferral curve.

The deferral curve is the operational contract:

* **coverage** - share of tickets routed automatically;
* **selective accuracy** - accuracy on that routed slice only.

Raising the confidence threshold always trades coverage for selective accuracy, and
:func:`threshold_for_target_accuracy` inverts the relationship: given "routed tickets
must be 95% correct", it returns the threshold and the coverage that buys.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_recall_fscore_support,
)


def _checked(y_true, y_pred) -> tuple[np.ndarray, np.ndarray]:
    truth = np.asarray(y_true).ravel()
    predicted = np.asarray(y_pred).ravel()
    if truth.size != predicted.size:
        raise ValueError("labels and predictions must have the same length")
    if truth.size == 0:
        raise ValueError("nothing to evaluate")
    return truth, predicted


def classification_metrics(y_true, y_pred) -> dict[str, float]:
    """Headline metrics, with the imbalance-aware ones first."""
    truth, predicted = _checked(y_true, y_pred)
    return {
        "tickets": int(truth.size),
        "macro_f1": round(float(f1_score(truth, predicted, average="macro", zero_division=0)), 5),
        "balanced_accuracy": round(float(balanced_accuracy_score(truth, predicted)), 5),
        "weighted_f1": round(
            float(f1_score(truth, predicted, average="weighted", zero_division=0)), 5
        ),
        "accuracy": round(float(accuracy_score(truth, predicted)), 5),
        "classes_predicted": int(len(np.unique(predicted))),
        "classes_present": int(len(np.unique(truth))),
    }


def per_class_report(y_true, y_pred) -> pd.DataFrame:
    """Precision, recall, F1 and support per queue, weakest recall first."""
    truth, predicted = _checked(y_true, y_pred)
    labels = sorted(set(truth) | set(predicted))
    precision, recall, f1, support = precision_recall_fscore_support(
        truth, predicted, labels=labels, zero_division=0
    )
    table = pd.DataFrame(
        {
            "queue": labels,
            "precision": precision.round(5),
            "recall": recall.round(5),
            "f1": f1.round(5),
            "support": support,
        }
    )
    return table.sort_values("recall").reset_index(drop=True)


def confusion_pairs(y_true, y_pred, top_n: int = 10) -> pd.DataFrame:
    """The mistakes, ranked - which queue is being mistaken for which.

    ``share_of_true_class`` matters more than the raw count: fifteen errors out of a
    hundred small-queue tickets is a broken queue, while fifteen out of two thousand
    is noise.
    """
    truth, predicted = _checked(y_true, y_pred)
    frame = pd.DataFrame({"true": truth, "predicted": predicted})
    mistakes = frame[frame["true"] != frame["predicted"]]
    if mistakes.empty:
        return pd.DataFrame(
            columns=["true", "predicted", "errors", "share_of_true_class"]
        )
    counts = (
        mistakes.groupby(["true", "predicted"]).size().reset_index(name="errors")
    )
    class_totals = frame["true"].value_counts()
    counts["share_of_true_class"] = (
        counts["errors"] / counts["true"].map(class_totals)
    ).round(5)
    return (
        counts.sort_values(["errors", "share_of_true_class"], ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )


# ------------------------------------------------------------------- deferral
def deferral_curve(
    y_true,
    y_pred,
    confidence,
    thresholds: tuple[float, ...] | None = None,
) -> pd.DataFrame:
    """Coverage against selective accuracy across confidence thresholds."""
    truth, predicted = _checked(y_true, y_pred)
    scores = np.asarray(confidence, dtype=float).ravel()
    if scores.size != truth.size:
        raise ValueError("confidence must align with the predictions")
    if scores.min() < 0.0 or scores.max() > 1.0:
        raise ValueError("confidence must lie in [0, 1]")

    grid = thresholds or tuple(round(value, 3) for value in np.arange(0.0, 0.96, 0.05))
    correct = truth == predicted

    rows: list[dict[str, float]] = []
    for threshold in grid:
        routed = scores >= threshold
        if not routed.any():
            continue
        rows.append(
            {
                "threshold": float(threshold),
                "coverage": round(float(routed.mean()), 5),
                "routed": int(routed.sum()),
                "selective_accuracy": round(float(correct[routed].mean()), 5),
                "routed_macro_f1": round(
                    float(
                        f1_score(
                            truth[routed], predicted[routed], average="macro", zero_division=0
                        )
                    ),
                    5,
                ),
                "errors_routed": int((~correct[routed]).sum()),
                "deferred": int((~routed).sum()),
                "deferred_that_were_wrong": int((~correct[~routed]).sum()),
            }
        )
    if not rows:
        raise ValueError("no threshold routed any ticket")
    return pd.DataFrame(rows)


def threshold_for_target_accuracy(curve: pd.DataFrame, target: float = 0.95) -> dict[str, float]:
    """Lowest threshold meeting a selective-accuracy target - i.e. best coverage.

    Raises when the target is unreachable at any threshold, rather than returning
    the closest row: silently missing an agreed quality bar is worse than failing.
    """
    if not 0.0 < target <= 1.0:
        raise ValueError("target must lie in (0, 1]")
    if curve.empty:
        raise ValueError("the deferral curve is empty")
    feasible = curve[curve["selective_accuracy"] >= target]
    if feasible.empty:
        best = float(curve["selective_accuracy"].max())
        raise ValueError(
            f"selective accuracy never reaches {target:.2%}; the best achievable is {best:.2%}"
        )
    chosen = feasible.loc[feasible["coverage"].idxmax()]
    return {key: float(value) for key, value in chosen.to_dict().items()}


def deferral_value(
    curve: pd.DataFrame,
    cost_per_manual_ticket: float = 4.0,
    cost_per_misroute: float = 14.0,
) -> pd.DataFrame:
    """Cost of each operating point: manual handling plus misroute rework.

    A misroute is more expensive than a deliberate hand-off, because the ticket is
    worked by the wrong team first and only then reassigned - the customer waits
    twice.  That asymmetry is what makes deferral worth having at all.
    """
    if cost_per_manual_ticket < 0 or cost_per_misroute < 0:
        raise ValueError("costs must be non-negative")
    table = curve.copy()
    table["manual_cost"] = (table["deferred"] * cost_per_manual_ticket).round(2)
    table["misroute_cost"] = (table["errors_routed"] * cost_per_misroute).round(2)
    table["total_cost"] = (table["manual_cost"] + table["misroute_cost"]).round(2)
    return table.sort_values("total_cost").reset_index(drop=True)


# ---------------------------------------------------------------- calibration
def confidence_calibration(confidence, correct, n_bins: int = 10) -> pd.DataFrame:
    """Reliability of top-1 confidence: predicted vs observed correctness."""
    scores = np.asarray(confidence, dtype=float).ravel()
    hits = np.asarray(correct).astype(bool).ravel()
    if scores.size != hits.size:
        raise ValueError("confidence and correctness must align")
    if n_bins < 2:
        raise ValueError("n_bins must be at least 2")

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    assignment = np.clip(np.digitize(scores, edges[1:-1], right=True), 0, n_bins - 1)
    rows: list[dict[str, float]] = []
    for index in range(n_bins):
        mask = assignment == index
        if not mask.any():
            continue
        rows.append(
            {
                "bin": index + 1,
                "lower": round(float(edges[index]), 3),
                "upper": round(float(edges[index + 1]), 3),
                "tickets": int(mask.sum()),
                "mean_confidence": round(float(scores[mask].mean()), 5),
                "observed_accuracy": round(float(hits[mask].mean()), 5),
                "gap": round(float(scores[mask].mean() - hits[mask].mean()), 5),
            }
        )
    return pd.DataFrame(rows)


def expected_calibration_error(confidence, correct, n_bins: int = 10) -> float:
    """Volume-weighted mean absolute gap between confidence and accuracy."""
    table = confidence_calibration(confidence, correct, n_bins)
    weights = table["tickets"].to_numpy(dtype=float)
    gaps = table["gap"].abs().to_numpy(dtype=float)
    return round(float(np.average(gaps, weights=weights)), 5)


def overconfidence(confidence, correct) -> float:
    """Mean confidence minus accuracy; positive means the model oversells itself."""
    scores = np.asarray(confidence, dtype=float).ravel()
    hits = np.asarray(correct).astype(bool).ravel()
    return round(float(scores.mean() - hits.mean()), 5)
