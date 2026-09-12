"""What the linear models actually learned.

This exists to catch a failure metrics cannot see: a classifier that scores well by
keying on a template artefact rather than the intent.  Support text is full of them -
a greeting used mostly by one help-centre form, an identifier format specific to one
product, a signature block.  Such a model looks excellent in evaluation and collapses
the day the template changes.

:func:`boilerplate_alarms` checks the top word features of every class against the
corpus's own greetings, sign-offs and masked identifiers, and reports any that made
the cut.  A clean run finds none.  Character n-grams are excluded from that check on
purpose: a 3-gram inside "thanks" is not evidence of anything.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from .data import CLOSERS, OPENERS
from .features import NUMBER_TOKEN, tokens

#: FeatureUnion prefixes feature names with the transformer that produced them.
WORD_PREFIX: str = "word__"
CHAR_PREFIX: str = "char__"


def _feature_names(pipeline: Pipeline) -> np.ndarray:
    return np.asarray(pipeline.named_steps["features"].get_feature_names_out())


def is_word_feature(name: str) -> bool:
    """True for word n-grams; False for character n-grams."""
    return not name.startswith(CHAR_PREFIX)


def strip_prefix(name: str) -> str:
    for prefix in (WORD_PREFIX, CHAR_PREFIX):
        if name.startswith(prefix):
            return name[len(prefix) :]
    return name


def top_features_per_class(pipeline: Pipeline, top_n: int = 12) -> pd.DataFrame:
    """Highest-weight features per queue for a linear model."""
    estimator = pipeline.named_steps["model"]
    if not hasattr(estimator, "coef_"):
        raise TypeError("top_features_per_class requires a linear model")
    if top_n < 1:
        raise ValueError("top_n must be at least 1")

    names = _feature_names(pipeline)
    coefficients = np.atleast_2d(estimator.coef_)
    classes = np.asarray(estimator.classes_)
    if coefficients.shape[0] == 1 and classes.size == 2:  # binary: one vector, two classes
        coefficients = np.vstack([-coefficients[0], coefficients[0]])

    rows: list[dict[str, object]] = []
    for index, queue in enumerate(classes):
        weights = coefficients[index]
        for rank, position in enumerate(np.argsort(-weights)[:top_n], start=1):
            name = str(names[position])
            rows.append(
                {
                    "queue": str(queue),
                    "rank": rank,
                    "feature": strip_prefix(name),
                    "kind": "word" if is_word_feature(name) else "char",
                    "weight": round(float(weights[position]), 5),
                }
            )
    return pd.DataFrame(rows)


def boilerplate_terms() -> set[str]:
    """Words that carry no intent: greetings, sign-offs, masked identifiers."""
    terms = {NUMBER_TOKEN}
    for phrase in (*OPENERS, *CLOSERS):
        terms.update(tokens(phrase))
    return {term for term in terms if term}


def boilerplate_alarms(pipeline: Pipeline, top_n: int = 12) -> pd.DataFrame:
    """Top word features made entirely of boilerplate - a template leak."""
    suspicious = boilerplate_terms()
    table = top_features_per_class(pipeline, top_n)
    words = table[table["kind"] == "word"]
    flagged = words[
        words["feature"].map(
            lambda feature: bool(feature.split())
            and all(part in suspicious for part in feature.split())
        )
    ]
    return flagged.reset_index(drop=True)


def feature_space_size(pipeline: Pipeline) -> dict[str, int]:
    """How wide the fitted feature space is - a sanity check on ``min_df``."""
    names = _feature_names(pipeline)
    word_features = sum(1 for name in names if is_word_feature(name))
    return {
        "features": int(names.size),
        "word_features": int(word_features),
        "char_features": int(names.size - word_features),
    }
