"""Candidate classifiers, the two splits, and calibrated confidence.

The split is the important part of this module.  :func:`temporal_split` trains on
earlier tickets and tests on later ones, which is the only split that answers the
question "will this work next month".  :func:`random_split` is kept so the two can
be reported side by side: the gap between them is the cost of pretending ticket
language is stationary.

A linear SVM's decision margin is not a probability, so any deferral threshold set
on it is meaningless across retrains.  :func:`calibrate` therefore wraps a candidate
in cross-fitted isotonic regression, and :func:`predict_with_confidence` refuses to
guess a confidence from a model that cannot produce one.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from .features import build_vectorizer, normalise_series

MODEL_KINDS: tuple[str, ...] = ("baseline", "complement_nb", "logistic", "linear_svm")
TEXT_COLUMN: str = "text"
LABEL_COLUMN: str = "queue"
TIME_COLUMN: str = "created_at"


def build_model(
    kind: str = "linear_svm",
    feature_kind: str = "union",
    seed: int = 29,
    min_df: int = 2,
) -> Pipeline:
    """Vectoriser plus classifier as one fittable object.

    ``class_weight="balanced"`` is on for the discriminative models: without it the
    small queues (account deletion, order cancel) are quietly sacrificed to overall
    accuracy.  Complement Naive Bayes handles imbalance by construction, which is
    why it is a fair "is the bigger model earning its keep" check.
    """
    if kind not in MODEL_KINDS:
        raise ValueError(f"kind must be one of {MODEL_KINDS}, got {kind!r}")

    if kind == "baseline":
        estimator = DummyClassifier(strategy="most_frequent")
    elif kind == "complement_nb":
        estimator = ComplementNB(alpha=0.35)
    elif kind == "logistic":
        estimator = LogisticRegression(
            C=6.0, max_iter=3_000, class_weight="balanced", random_state=seed
        )
    else:
        estimator = LinearSVC(C=0.7, class_weight="balanced", random_state=seed)

    return Pipeline(
        steps=[
            ("features", build_vectorizer(feature_kind, min_df=min_df)),
            ("model", estimator),
        ]
    )


def temporal_split(
    frame: pd.DataFrame, test_fraction: float = 0.25
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train on the earliest tickets, test on the most recent ones.

    The cut is placed on a *date* rather than a row index, so a ticket and its
    same-day neighbours never straddle the boundary.
    """
    if not 0.0 < test_fraction < 0.9:
        raise ValueError("test_fraction must lie in (0, 0.9)")
    if TIME_COLUMN not in frame.columns:
        raise ValueError(f"frame must contain '{TIME_COLUMN}'")

    ordered = frame.sort_values(TIME_COLUMN).reset_index(drop=True)
    cutoff = ordered[TIME_COLUMN].quantile(1.0 - test_fraction)
    train = ordered[ordered[TIME_COLUMN] < cutoff].reset_index(drop=True)
    test = ordered[ordered[TIME_COLUMN] >= cutoff].reset_index(drop=True)
    if train.empty or test.empty:
        raise ValueError("the temporal cut produced an empty side")
    missing = set(test[LABEL_COLUMN]) - set(train[LABEL_COLUMN])
    if missing:
        raise ValueError(f"queues appear only after the cut: {sorted(missing)}")
    return train, test


def random_split(
    frame: pd.DataFrame, test_fraction: float = 0.25, seed: int = 29
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stratified random split - reported only as a contrast to the temporal one."""
    from sklearn.model_selection import train_test_split

    train, test = train_test_split(
        frame, test_size=test_fraction, stratify=frame[LABEL_COLUMN], random_state=seed
    )
    return train.reset_index(drop=True), test.reset_index(drop=True)


def fit(
    train: pd.DataFrame,
    kind: str = "linear_svm",
    feature_kind: str = "union",
    mask_numbers: bool = True,
    seed: int = 29,
) -> Pipeline:
    """Fit one candidate on normalised ticket text."""
    model = build_model(kind, feature_kind, seed=seed)
    model.fit(
        normalise_series(train[TEXT_COLUMN], mask_numbers), train[LABEL_COLUMN].to_numpy()
    )
    return model


def calibrate(
    train: pd.DataFrame,
    kind: str = "linear_svm",
    feature_kind: str = "union",
    method: str = "isotonic",
    folds: int = 3,
    mask_numbers: bool = True,
    seed: int = 29,
) -> CalibratedClassifierCV:
    """Cross-fitted calibration, fitted strictly inside the training window.

    Calibrating on the evaluation slice would leak the future into the confidence
    scores - and confidence is exactly what the deferral decision is made on.
    """
    if method not in {"isotonic", "sigmoid"}:
        raise ValueError("method must be 'isotonic' or 'sigmoid'")
    calibrated = CalibratedClassifierCV(
        build_model(kind, feature_kind, seed=seed), method=method, cv=folds
    )
    calibrated.fit(
        normalise_series(train[TEXT_COLUMN], mask_numbers), train[LABEL_COLUMN].to_numpy()
    )
    return calibrated


def predict_with_confidence(
    model, texts: pd.Series, mask_numbers: bool = True
) -> pd.DataFrame:
    """Predicted queue, calibrated confidence, and the runner-up.

    The runner-up is returned because "billing vs refund, 0.51 to 0.44" is the
    information a human triager needs, and a bare label throws it away.
    """
    if not hasattr(model, "predict_proba"):
        raise TypeError(
            "this model cannot produce probabilities; wrap it with calibrate() before "
            "thresholding on confidence"
        )
    prepared = normalise_series(pd.Series(texts).astype(str), mask_numbers)
    probabilities = model.predict_proba(prepared)
    classes = np.asarray(model.classes_)

    order = np.argsort(-probabilities, axis=1)
    top = order[:, 0]
    second = order[:, 1] if probabilities.shape[1] > 1 else top
    rows = np.arange(len(prepared))
    return pd.DataFrame(
        {
            "text": pd.Series(texts).astype(str).to_numpy(),
            "predicted": classes[top],
            "confidence": probabilities[rows, top].round(6),
            "runner_up": classes[second],
            "runner_up_confidence": probabilities[rows, second].round(6),
        }
    )
