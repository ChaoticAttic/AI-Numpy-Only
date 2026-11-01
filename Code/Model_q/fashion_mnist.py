"""Train a small NumPy-only multilayer perceptron on Fashion-MNIST.

The dataset is downloaded from a public Fashion-MNIST mirror on the first run
and cached locally. Images are returned as rows with values in
[0, 1], which keeps this module independent of sklearn and deep-learning
frameworks.
"""

from __future__ import annotations

import argparse
import gzip
import struct
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Callable

import numpy as np


FASHION_MNIST_URL = "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/"
DATA_FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}


def _download_dataset(data_dir: Path) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    for filename in DATA_FILES.values():
        target = data_dir / filename
        if target.exists():
            continue
        print(f"Downloading {filename}...")
        try:
            urllib.request.urlretrieve(FASHION_MNIST_URL + filename, target)
        except (OSError, urllib.error.URLError) as error:
            target.unlink(missing_ok=True)
            raise RuntimeError(
                f"Could not download Fashion-MNIST to {target}. "
                "Check your network connection or download the files manually."
            ) from error


def _read_images(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as file:
        magic, count, rows, columns = struct.unpack(">IIII", file.read(16))
        if magic != 2051:
            raise ValueError(f"Invalid image file: {path}")
        values = np.frombuffer(file.read(), dtype=np.uint8)
    expected = count * rows * columns
    if values.size != expected:
        raise ValueError(f"Invalid image count in {path}")
    return values.reshape(count, rows * columns).astype(np.float64) / 255.0


def _read_labels(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as file:
        magic, count = struct.unpack(">II", file.read(8))
        if magic != 2049:
            raise ValueError(f"Invalid label file: {path}")
        labels = np.frombuffer(file.read(), dtype=np.uint8)
    if labels.size != count:
        raise ValueError(f"Invalid label count in {path}")
    return labels.astype(np.int64)


def load_fashion_mnist(
    data_dir: str | Path = "data/fashion-mnist",
    train_limit: int | None = None,
    test_limit: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load cached Fashion-MNIST as ``(x_train, y_train, x_test, y_test)``."""
    data_path = Path(data_dir)
    _download_dataset(data_path)
    x_train = _read_images(data_path / DATA_FILES["train_images"])
    y_train = _read_labels(data_path / DATA_FILES["train_labels"])
    x_test = _read_images(data_path / DATA_FILES["test_images"])
    y_test = _read_labels(data_path / DATA_FILES["test_labels"])
    if train_limit is not None:
        x_train, y_train = x_train[:train_limit], y_train[:train_limit]
    if test_limit is not None:
        x_test, y_test = x_test[:test_limit], y_test[:test_limit]
    return x_train, y_train, x_test, y_test


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0)


def relu_derivative(x: np.ndarray) -> np.ndarray:
    return (x > 0.0).astype(np.float64)


def tanh(x: np.ndarray) -> np.ndarray:
    return np.tanh(x)


def tanh_derivative(x: np.ndarray) -> np.ndarray:
    values = np.tanh(x)
    return 1.0 - values * values


def sin(x: np.ndarray) -> np.ndarray:
    return np.sin(x)


def sin_derivative(x: np.ndarray) -> np.ndarray:
    return np.cos(x)


def cos(x: np.ndarray) -> np.ndarray:
    return np.cos(x)


def cos_derivative(x: np.ndarray) -> np.ndarray:
    return -np.sin(x)


ACTIVATIONS: dict[str, tuple[Callable[[np.ndarray], np.ndarray], Callable[[np.ndarray], np.ndarray]]] = {
    "cos": (cos, cos_derivative),
    "relu": (relu, relu_derivative),
    "sin": (sin, sin_derivative),
    "tanh": (tanh, tanh_derivative),
}


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp_values = np.exp(shifted)
    return exp_values / np.sum(exp_values, axis=1, keepdims=True)


def _one_hot(labels: np.ndarray, classes: int = 10) -> np.ndarray:
    encoded = np.zeros((labels.size, classes), dtype=np.float64)
    encoded[np.arange(labels.size), labels] = 1.0
    return encoded


def _train_one(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    activation: str,
    hidden_layers: int,
    neurons: int,
    epochs: int,
    learning_rate: float,
    batch_size: int,
    seed: int,
    hidden_sizes: tuple[int, ...] | None = None,
) -> dict[str, float | str]:
    if activation not in ACTIVATIONS:
        supported = ", ".join(sorted(ACTIVATIONS))
        raise ValueError(
            f"Unsupported activation {activation!r}; choose one of {supported}"
        )
    if hidden_sizes is None:
        hidden_sizes = (neurons,) * hidden_layers
    if not hidden_sizes or any(size < 1 for size in hidden_sizes):
        raise ValueError("hidden layer sizes must be positive")

    activation_fn, activation_derivative = ACTIVATIONS[activation]
    rng = np.random.default_rng(seed)
    dimensions = [x_train.shape[1], *hidden_sizes, 10]
    started = time.perf_counter()
    weights = [
        rng.normal(0.0, np.sqrt(2.0 / dimensions[i]), (dimensions[i], dimensions[i + 1]))
        for i in range(len(dimensions) - 1)
    ]
    biases = [np.zeros((1, dimensions[i + 1])) for i in range(len(dimensions) - 1)]
    targets = _one_hot(y_train)

    for epoch in range(epochs):
        order = rng.permutation(x_train.shape[0])
        for start in range(0, x_train.shape[0], batch_size):
            indices = order[start : start + batch_size]
            activations = [x_train[indices]]
            pre_activations = []
            current = activations[0]
            for layer in range(hidden_layers):
                linear = current @ weights[layer] + biases[layer]
                pre_activations.append(linear)
                current = activation_fn(linear)
                activations.append(current)
            logits = current @ weights[-1] + biases[-1]
            probabilities = _softmax(logits)

            gradient = (probabilities - targets[indices]) / indices.size
            weight_gradients = [None] * len(weights)
            bias_gradients = [None] * len(biases)
            weight_gradients[-1] = activations[-1].T @ gradient
            bias_gradients[-1] = np.sum(gradient, axis=0, keepdims=True)
            for layer in range(hidden_layers - 1, -1, -1):
                gradient = (gradient @ weights[layer + 1].T) * activation_derivative(
                    pre_activations[layer]
                )
                weight_gradients[layer] = activations[layer].T @ gradient
                bias_gradients[layer] = np.sum(gradient, axis=0, keepdims=True)
            for layer in range(len(weights)):
                weights[layer] -= learning_rate * weight_gradients[layer]
                biases[layer] -= learning_rate * bias_gradients[layer]

        if epoch == 0 or epoch == epochs - 1:
            train_accuracy = _accuracy(x_train, y_train, weights, biases, activation_fn)
            print(f"{activation} epoch {epoch + 1}/{epochs}: train accuracy {train_accuracy:.4f}")

    train_accuracy = _accuracy(x_train, y_train, weights, biases, activation_fn)
    test_accuracy = _accuracy(x_test, y_test, weights, biases, activation_fn)
    parameter_count = sum(
        weight.size + bias.size for weight, bias in zip(weights, biases)
    )
    return {
        "activation": activation,
        "architecture": "-".join(str(size) for size in hidden_sizes),
        "hidden_layers": len(hidden_sizes),
        "neurons": "-".join(str(size) for size in hidden_sizes),
        "parameters": parameter_count,
        "train_accuracy": train_accuracy,
        "test_accuracy": test_accuracy,
        "seconds": time.perf_counter() - started,
    }


def _accuracy(
    features: np.ndarray,
    labels: np.ndarray,
    weights: list[np.ndarray],
    biases: list[np.ndarray],
    activation_fn: Callable[[np.ndarray], np.ndarray],
) -> float:
    current = features
    for weight, bias in zip(weights[:-1], biases[:-1]):
        current = activation_fn(current @ weight + bias)
    predictions = np.argmax(_softmax(current @ weights[-1] + biases[-1]), axis=1)
    return float(np.mean(predictions == labels))


def train_model(args: argparse.Namespace) -> None:
    x_train, y_train, x_test, y_test = load_fashion_mnist(
        args.data_dir, args.train_limit, args.test_limit
    )
    result = _train_one(
        x_train, y_train, x_test, y_test, args.activation, args.hidden_layers,
        args.neurons, args.epochs, args.learning_rate, args.batch_size, args.seed,
    )
    print(f"Test accuracy: {result['test_accuracy']:.4f}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--activation", choices=sorted(ACTIVATIONS), default="relu")
    parser.add_argument("--hidden-layers", type=int, default=2)
    parser.add_argument("--neurons", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--test-limit", type=int, default=None)
    parser.add_argument("--data-dir", default="data/fashion-mnist")
    parser.add_argument("--seed", type=int, default=42)
    return parser


if __name__ == "__main__":
    train_model(build_parser().parse_args())
