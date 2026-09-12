# Support Ticket Routing - Imbalanced Text Classification That Survives Contact With Production

Routes free-text support tickets to one of eight queues. The classifier itself is the
easy part; this repository is about the four things that decide whether an automated
router is safe to switch on:

1. **Temporal validation, not a random split.** Tickets arrive over time and the
   language changes - a new product ships, a new failure mode appears. A random split
   lets the model train on next month's vocabulary and reports a number you will never
   see again in production.
2. **Imbalance handled where it matters.** The largest queue is roughly four times the
   smallest. Accuracy is reported but never used for a decision; macro-F1 and per-class
   recall are, because the small queues are the ones a business notices when they break.
3. **Confidence you can act on.** Predictions are calibrated and low-confidence tickets
   are **deferred to a human** rather than routed at random. The trade-off is reported as
   a coverage curve: for a target accuracy on the routed slice, what fraction of tickets
   can be automated?
4. **Error analysis that names names.** Aggregate scores hide the actual problem, which
   is almost always a specific pair of classes. The harness reports the top confusion
   pairs, per-class support-weighted cost, and the tickets each error came from.

---

## Why temporal validation changes the answer

The synthetic ticket stream has deliberate **vocabulary drift**: in the final months a
new product line and new phrasings appear that occur nowhere in the earlier period. The
harness quantifies it directly:

- out-of-vocabulary rate of the later period against the training vocabulary;
- per-class macro-F1 under a **temporal** split (train early, test late) next to the same
  model under a **random** split of the same data.

Run `evaluate` and compare the two columns. The random split is the number that gets put
in a slide deck; the temporal split is the number the on-call engineer lives with.

## Why word features alone are not enough

Real tickets contain typos, so the vectoriser is a union of:

- **word 1-2 grams** - carries intent ("double charged", "never arrived");
- **character 3-5 grams** - survives "chagred", "refudn", "canceled/cancelled".

The generator injects a configurable typo rate, so the benefit is measurable rather than
assumed: `evaluate --ablation` trains word-only, char-only and union feature spaces and
prints them side by side.

## Deferral: the decision layer

A router that must answer every ticket is judged on overall accuracy. A router allowed to
say *"I am not sure, send this to a human"* is judged on two numbers that trade off:

```
coverage  = share of tickets routed automatically
selective accuracy = accuracy on the routed slice only
```

`deferral_curve` sweeps the confidence threshold and reports both, plus the queue-level
recall inside the routed slice. `threshold_for_target_accuracy` inverts it: given "routed
tickets must be 95% correct", it returns the threshold and the coverage that buys. This
is the number an operations manager actually negotiates over.

Calibration is checked before any of this is trusted: an uncalibrated linear SVM's decision
margin is not a probability, and a threshold on it means nothing across releases. The
champion is selected on temporal macro-F1, then calibrated with cross-fitted isotonic
regression and re-scored for Brier and expected calibration error.

## Candidate models

| Model | Why it is here |
| --- | --- |
| Linear SVM (hinge, class-weighted) | Strong on sparse high-dimensional text; the usual production baseline |
| Logistic regression (class-weighted) | Native probabilities, directly interpretable coefficients |
| Complement Naive Bayes | Designed for imbalanced text; trains instantly, a fair "is the fancy model earning its keep" check |
| Majority-class baseline | The number every other model must beat before anyone discusses architecture |

Each is scored on the same temporal split. `explain.py` extracts the top positive features
per class from the linear models - which is how you catch a classifier that learned to key
on a template artefact (a signature line, a boilerplate greeting) instead of the intent.

## Quickstart

```bash
pip install -r requirements.txt

python -m ticketrouting.cli data                    # generate tickets, show class balance and drift
python -m ticketrouting.cli train                   # candidates on the temporal split
python -m ticketrouting.cli evaluate --out reports  # full report: confusion pairs, deferral, ablation
python -m ticketrouting.cli route --text "i was charged twice for order 88213, refund please"

pytest
```

`route` prints the predicted queue, the calibrated confidence, whether the ticket would be
deferred at the configured threshold, and the runner-up class - because "billing vs refund"
being a close call is exactly what a human triager needs to know.

## Layout

```
ticketrouting/
  data.py      synthetic ticket stream: imbalance, confusable intent pairs, typos, vocabulary drift
  features.py  word + character n-gram union, normalisation, OOV diagnostics
  model.py     candidate classifiers, calibration, temporal and random splits
  evaluate.py  macro/micro F1, per-class report, confusion pairs, deferral curves, calibration
  explain.py   top features per class from the linear models
  cli.py       data | train | evaluate | route
tests/         hand-computed F1 and deferral cases, drift assertions, imbalance invariants
```

## Notes and limits

- The corpus is synthetic and template-generated, so absolute scores are optimistic
  compared with real support text; the *comparisons* (temporal vs random, word vs char,
  deferral trade-offs) are the transferable part.
- Linear models on n-grams were chosen deliberately over a transformer: on eight queues
  of short text they are within a few points, train in seconds, and can be explained to a
  support lead. A transformer belongs here only once the deferral rate stops improving.
- Class definitions are assumed stable. Genuine new intents need an out-of-distribution
  detector, not a threshold on a softmax over the old classes; the deferral hook is where
  that would attach.

## Licence

MIT
