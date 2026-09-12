"""Text normalisation, the feature space, and drift diagnostics.

Two decisions are worth stating.

**Digit runs are masked.**  Order ids, invoice numbers and amounts are
high-cardinality noise: they cannot generalise, and a model that starts leaning on
them is memorising tickets.  ``mask_numbers=False`` is available so the effect can
be measured rather than taken on trust.

**Words and characters are used together.**  Word 1-2 grams carry the intent
("double charged", "never arrived"); character 3-5 grams survive the typos real
tickets are full of ("chagred", "refudn") and the spelling variants no stemmer
handles ("canceled" / "cancelled").  Each space is available on its own so the
ablation in the harness is a real comparison.
"""

from __future__ import annotations

import re

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion

FEATURE_KINDS: tuple[str, ...] = ("word", "char", "union")

_DIGIT_RUN = re.compile(r"\d+")
_NON_TEXT = re.compile(r"[^a-z0-9<>\s]+")
_WHITESPACE = re.compile(r"\s+")
_WORD = re.compile(r"[a-z<][a-z0-9<>]*")

NUMBER_TOKEN: str = "<num>"


def normalise_text(text: str, mask_numbers: bool = True) -> str:
    """Lowercase, strip punctuation, optionally mask digit runs."""
    lowered = str(text).lower()
    if mask_numbers:
        lowered = _DIGIT_RUN.sub(NUMBER_TOKEN, lowered)
    cleaned = _NON_TEXT.sub(" ", lowered)
    return _WHITESPACE.sub(" ", cleaned).strip()


def normalise_series(texts: pd.Series, mask_numbers: bool = True) -> pd.Series:
    return texts.astype(str).map(lambda value: normalise_text(value, mask_numbers))


def tokens(text: str) -> list[str]:
    """Word tokens of already-normalised text; used for OOV diagnostics."""
    return _WORD.findall(normalise_text(text))


def build_vectorizer(kind: str = "union", min_df: int = 2):
    """Word, character, or the union of both feature spaces.

    ``sublinear_tf`` is on because a repeated word in a two-line ticket says little,
    and ``min_df`` drops hapax noise (typo'd one-offs, order ids that escaped masking).
    """
    if kind not in FEATURE_KINDS:
        raise ValueError(f"kind must be one of {FEATURE_KINDS}, got {kind!r}")
    if min_df < 1:
        raise ValueError("min_df must be at least 1")

    word = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        min_df=min_df,
        sublinear_tf=True,
        strip_accents="unicode",
    )
    char = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=min_df,
        sublinear_tf=True,
        strip_accents="unicode",
    )
    if kind == "word":
        return word
    if kind == "char":
        return char
    return FeatureUnion([("word", word), ("char", char)])


def vocabulary(texts: pd.Series) -> set[str]:
    """Word vocabulary of a set of tickets."""
    return {token for text in texts.astype(str) for token in tokens(text)}


def oov_rate(train_texts: pd.Series, test_texts: pd.Series) -> dict[str, float]:
    """How much of the evaluation text the training data never saw.

    ``token_oov_rate`` weights by occurrence (what the model actually meets at
    inference); ``type_oov_rate`` counts distinct unseen words.  A temporal split
    shows a materially higher rate than a random split, which is the mechanism
    behind the score gap rather than a vague appeal to "drift".
    """
    known = vocabulary(train_texts)
    seen = 0
    unseen = 0
    unseen_types: set[str] = set()
    for text in test_texts.astype(str):
        for token in tokens(text):
            if token in known:
                seen += 1
            else:
                unseen += 1
                unseen_types.add(token)
    total = seen + unseen
    distinct = len(vocabulary(test_texts))
    return {
        "train_vocabulary": len(known),
        "test_vocabulary": distinct,
        "token_oov_rate": round(unseen / total, 6) if total else 0.0,
        "type_oov_rate": round(len(unseen_types) / distinct, 6) if distinct else 0.0,
        "unseen_types": len(unseen_types),
    }


def unseen_terms(train_texts: pd.Series, test_texts: pd.Series, top_n: int = 15) -> pd.DataFrame:
    """Most frequent evaluation words absent from training - drift, named."""
    known = vocabulary(train_texts)
    counts: dict[str, int] = {}
    for text in test_texts.astype(str):
        for token in tokens(text):
            if token not in known:
                counts[token] = counts.get(token, 0) + 1
    frame = pd.DataFrame(
        sorted(counts.items(), key=lambda item: -item[1])[:top_n],
        columns=["term", "occurrences"],
    )
    return frame
