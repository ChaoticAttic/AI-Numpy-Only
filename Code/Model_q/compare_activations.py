"""Compare activations across several explicit Fashion-MNIST architectures."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from fashion_mnist import ACTIVATIONS, _train_one, load_fashion_mnist


def _parse_architecture(value: str) -> tuple[int, ...]:
    try:
        sizes = tuple(int(size) for size in value.split("-"))
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            f"Invalid architecture {value!r}; use a format such as 128-64"
        ) from error
    if not sizes or any(size < 1 for size in sizes):
        raise argparse.ArgumentTypeError("Architecture sizes must be positive")
    return sizes


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--activations", nargs="+", choices=sorted(ACTIVATIONS),
        default=["relu", "sin", "tanh"],
    )
    parser.add_argument(
        "--architectures", nargs="+", type=_parse_architecture,
        default=[(64,), (128,), (128, 64), (256, 128), (256, 128, 64)],
        metavar="WIDTHS",
        help="Hidden-layer widths, for example 64 128-64 256-128-64",
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--train-limit", type=int, default=10000)
    parser.add_argument("--test-limit", type=int, default=2000)
    parser.add_argument("--data-dir", default="data/fashion-mnist")
    parser.add_argument("--output", default="activation_results.csv")
    parser.add_argument(
        "--summary", default=None,
        help="JSON summary path; defaults to the CSV path with a .json suffix",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser


def _format_table(results: list[dict[str, float | str]]) -> str:
    ranked = sorted(results, key=lambda result: result["test_accuracy"], reverse=True)
    headers = ("Rank", "Activation", "Architecture", "Params", "Train", "Test", "Time")
    rows = [
        (
            str(index),
            str(result["activation"]),
            str(result["architecture"]),
            f"{int(result['parameters']):,}",
            f"{result['train_accuracy']:.3f}",
            f"{result['test_accuracy']:.3f}",
            f"{result['seconds']:.1f}s",
        )
        for index, result in enumerate(ranked, start=1)
    ]
    widths = [max(len(header), *(len(row[index]) for row in rows)) for index, header in enumerate(headers)]
    line = " | ".join(header.ljust(widths[index]) for index, header in enumerate(headers))
    separator = "-+-".join("-" * width for width in widths)
    body = [" | ".join(value.ljust(widths[index]) for index, value in enumerate(row)) for row in rows]
    return "\n".join([line, separator, *body])


def compare(args: argparse.Namespace) -> None:
    x_train, y_train, x_test, y_test = load_fashion_mnist(
        args.data_dir, args.train_limit, args.test_limit
    )
    results = []
    total = len(args.activations) * len(args.architectures)
    run = 0
    for activation in args.activations:
        for architecture in args.architectures:
            run += 1
            architecture_name = "-".join(str(size) for size in architecture)
            print(f"\n[{run}/{total}] {activation} | architecture {architecture_name}")
            results.append(_train_one(
                x_train, y_train, x_test, y_test, activation,
                len(architecture), architecture[0], args.epochs,
                args.learning_rate, args.batch_size, args.seed,
                hidden_sizes=architecture,
            ))

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    ranked = sorted(results, key=lambda result: result["test_accuracy"], reverse=True)
    summary_path = Path(args.summary) if args.summary else output.with_suffix(".json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps({
        "experiments": len(results),
        "best": ranked[0],
        "results": ranked,
    }, indent=2) + "\n")
    print(f"\nResults saved to {output}")
    print(f"Summary saved to {summary_path}")
    print("\nRanked results (highest test accuracy first):")
    print(_format_table(results))


if __name__ == "__main__":
    compare(build_parser().parse_args())
