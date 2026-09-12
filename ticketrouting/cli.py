"""Command line interface: ``python -m ticketrouting.cli <command>``."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .data import StreamConfig, class_balance, generate_tickets, monthly_volume
from .evaluate import (
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
from .explain import boilerplate_alarms, feature_space_size, top_features_per_class
from .features import FEATURE_KINDS, normalise_series, oov_rate, unseen_terms
from .model import (
    LABEL_COLUMN,
    MODEL_KINDS,
    TEXT_COLUMN,
    calibrate,
    fit,
    predict_with_confidence,
    random_split,
    temporal_split,
)


def _stream(args: argparse.Namespace) -> pd.DataFrame:
    return generate_tickets(
        StreamConfig(
            n_tickets=args.tickets,
            months=args.months,
            typo_rate=args.typo_rate,
            seed=args.seed,
        )
    )


def _write(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    print(f"wrote {path}")


def _fit_and_predict(
    train: pd.DataFrame, test: pd.DataFrame, kind: str, feature_kind: str, seed: int
):
    """Fit a candidate and return it with its predictions on the evaluation slice."""
    model = fit(train, kind, feature_kind, seed=seed)
    predictions = model.predict(normalise_series(test[TEXT_COLUMN]))
    return model, predictions


def cmd_data(args: argparse.Namespace) -> int:
    """Generate the ticket stream and describe imbalance and drift."""
    stream = _stream(args)
    print(
        f"{len(stream)} tickets from {stream['created_at'].min().date()} "
        f"to {stream['created_at'].max().date()}\n"
    )
    print("queue mix")
    print(class_balance(stream).to_string(index=False), "\n")

    print("monthly volume and share of tickets using the new product vocabulary")
    print(monthly_volume(stream).to_string(index=False), "\n")

    train, test = temporal_split(stream, args.test_fraction)
    random_train, random_test = random_split(stream, args.test_fraction, seed=args.seed)
    print("out-of-vocabulary rate of the evaluation slice")
    print(
        pd.DataFrame(
            [
                {"split": "temporal", **oov_rate(train[TEXT_COLUMN], test[TEXT_COLUMN])},
                {
                    "split": "random",
                    **oov_rate(random_train[TEXT_COLUMN], random_test[TEXT_COLUMN]),
                },
            ]
        ).to_string(index=False),
        "\n",
    )
    print("most frequent words the temporal training window never saw")
    print(unseen_terms(train[TEXT_COLUMN], test[TEXT_COLUMN]).to_string(index=False), "\n")

    destination = Path(args.out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    stream.to_csv(destination, index=False)
    print(f"wrote {destination}")
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    """Compare candidates on the temporal split, then temporal against random."""
    stream = _stream(args)
    train, test = temporal_split(stream, args.test_fraction)
    print(
        f"temporal split: train={len(train)} test={len(test)} "
        f"(cut at {test['created_at'].min().date()})\n"
    )

    scored: dict[str, object] = {}
    rows = []
    for kind in MODEL_KINDS:
        _model, predictions = _fit_and_predict(train, test, kind, args.features, args.seed)
        scored[kind] = predictions
        rows.append({"model": kind, **classification_metrics(test[LABEL_COLUMN], predictions)})
    table = pd.DataFrame(rows).sort_values("macro_f1", ascending=False).reset_index(drop=True)
    print("candidates on the temporal split")
    print(table.to_string(index=False), "\n")

    champion = str(table.iloc[0]["model"])
    random_train, random_test = random_split(stream, args.test_fraction, seed=args.seed)
    _model, random_predictions = _fit_and_predict(
        random_train, random_test, champion, args.features, args.seed
    )
    print(f"{champion}: what the choice of split does to the reported score")
    print(
        pd.DataFrame(
            [
                {
                    "split": "temporal (honest)",
                    **classification_metrics(test[LABEL_COLUMN], scored[champion]),
                },
                {
                    "split": "random (optimistic)",
                    **classification_metrics(random_test[LABEL_COLUMN], random_predictions),
                },
            ]
        ).to_string(index=False),
        "\n",
    )

    print(f"{champion}: feature space ablation on the temporal split")
    ablation = []
    for feature_kind in FEATURE_KINDS:
        _model, predictions = _fit_and_predict(train, test, champion, feature_kind, args.seed)
        ablation.append(
            {"features": feature_kind, **classification_metrics(test[LABEL_COLUMN], predictions)}
        )
    print(pd.DataFrame(ablation).to_string(index=False))
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    """Full report for the calibrated champion: errors, deferral, explanations."""
    out_dir = Path(args.out)
    stream = _stream(args)
    train, test = temporal_split(stream, args.test_fraction)

    calibrated = calibrate(train, args.model, args.features, seed=args.seed)
    scored = predict_with_confidence(calibrated, test[TEXT_COLUMN])
    truth = test[LABEL_COLUMN].to_numpy()
    predictions = scored["predicted"].to_numpy()
    correct = truth == predictions

    print(f"{args.model} with {args.features} features, calibrated, temporal split")
    for key, value in classification_metrics(truth, predictions).items():
        print(f"  {key:<22}{value}")
    print(
        f"  {'expected_cal_error':<22}"
        f"{expected_calibration_error(scored['confidence'], correct)}"
    )
    print(f"  {'overconfidence':<22}{overconfidence(scored['confidence'], correct)}\n")

    report = per_class_report(truth, predictions)
    print("per queue (weakest recall first)")
    print(report.to_string(index=False), "\n")

    pairs = confusion_pairs(truth, predictions)
    print("where the errors actually are")
    print(pairs.to_string(index=False) if not pairs.empty else "  no errors", "\n")

    curve = deferral_curve(truth, predictions, scored["confidence"])
    print("deferral: coverage against accuracy on the routed slice")
    print(curve.to_string(index=False), "\n")

    operating_point: dict[str, float] = {}
    try:
        operating_point = threshold_for_target_accuracy(curve, args.target_accuracy)
        print(f"to keep routed tickets at least {args.target_accuracy:.0%} correct")
        for key in ("threshold", "coverage", "selective_accuracy", "routed", "deferred"):
            print(f"  {key:<22}{operating_point[key]}")
    except ValueError as error:
        print(f"no threshold meets the target: {error}")

    costed = deferral_value(curve)
    print("\ncheapest operating points (manual handling vs misroute rework)")
    print(
        costed[
            [
                "threshold",
                "coverage",
                "selective_accuracy",
                "manual_cost",
                "misroute_cost",
                "total_cost",
            ]
        ]
        .head(5)
        .to_string(index=False),
        "\n",
    )

    calibration = confidence_calibration(scored["confidence"], correct)
    print("confidence reliability")
    print(calibration.to_string(index=False), "\n")

    explainable = "logistic" if args.model in {"baseline", "complement_nb"} else args.model
    linear = fit(train, explainable, args.features, seed=args.seed)
    features = top_features_per_class(linear, top_n=args.top_features)
    print(f"{explainable} feature space: {feature_space_size(linear)}")
    print("\ntop word features per queue")
    print(features[features["kind"] == "word"].head(24).to_string(index=False), "\n")
    alarms = boilerplate_alarms(linear, top_n=args.top_features)
    print(f"boilerplate features among the top weights: {len(alarms)}")
    if not alarms.empty:
        print(alarms.to_string(index=False))

    _write(features, out_dir / "top_features.csv")
    _write(report, out_dir / "per_class.csv")
    _write(pairs, out_dir / "confusion_pairs.csv")
    _write(curve, out_dir / "deferral_curve.csv")
    _write(calibration, out_dir / "calibration.csv")
    _write(scored.assign(actual=truth, correct=correct), out_dir / "predictions.csv")

    summary = {
        "model": args.model,
        "features": args.features,
        "split": "temporal",
        "metrics": classification_metrics(truth, predictions),
        "expected_calibration_error": expected_calibration_error(scored["confidence"], correct),
        "operating_point": operating_point,
        "oov": oov_rate(train[TEXT_COLUMN], test[TEXT_COLUMN]),
    }
    path = out_dir / "evaluation.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"wrote {path}")
    return 0


def cmd_route(args: argparse.Namespace) -> int:
    """Route one ticket, showing confidence, runner-up and the deferral decision."""
    stream = _stream(args)
    train, _test = temporal_split(stream, args.test_fraction)
    calibrated = calibrate(train, args.model, args.features, seed=args.seed)

    scored = predict_with_confidence(calibrated, pd.Series([args.text])).iloc[0]
    deferred = float(scored["confidence"]) < args.threshold
    print(f"ticket: {args.text}\n")
    print(f"  queue          {scored['predicted']}")
    print(f"  confidence     {float(scored['confidence']):.4f}")
    print(f"  runner up      {scored['runner_up']} ({float(scored['runner_up_confidence']):.4f})")
    print(f"  threshold      {args.threshold:.2f}")
    print(f"  decision       {'defer to a human' if deferred else 'route automatically'}")
    return 0


def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--tickets", type=int, default=9_000)
    parser.add_argument("--months", type=int, default=12)
    parser.add_argument("--typo-rate", dest="typo_rate", type=float, default=0.015)
    parser.add_argument("--test-fraction", dest="test_fraction", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=29)


def _model_args(parser: argparse.ArgumentParser) -> None:
    _common(parser)
    parser.add_argument("--model", default="linear_svm", choices=list(MODEL_KINDS))
    parser.add_argument("--features", default="union", choices=list(FEATURE_KINDS))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ticketrouting",
        description="Imbalanced support ticket routing with temporal validation and deferral.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    data_parser = subparsers.add_parser("data", help="generate and describe the ticket stream")
    _common(data_parser)
    data_parser.add_argument("--out", default="data/tickets.csv")
    data_parser.set_defaults(func=cmd_data)

    train_parser = subparsers.add_parser("train", help="compare candidates and splits")
    _model_args(train_parser)
    train_parser.set_defaults(func=cmd_train)

    evaluate_parser = subparsers.add_parser("evaluate", help="full report for the champion")
    _model_args(evaluate_parser)
    evaluate_parser.add_argument(
        "--target-accuracy", dest="target_accuracy", type=float, default=0.95
    )
    evaluate_parser.add_argument("--top-features", dest="top_features", type=int, default=10)
    evaluate_parser.add_argument("--out", default="reports")
    evaluate_parser.set_defaults(func=cmd_evaluate)

    route_parser = subparsers.add_parser("route", help="route a single ticket")
    _model_args(route_parser)
    route_parser.add_argument("--text", required=True)
    route_parser.add_argument("--threshold", type=float, default=0.60)
    route_parser.set_defaults(func=cmd_route)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
