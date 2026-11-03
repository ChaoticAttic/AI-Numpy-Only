import gzip
import struct
import urllib.request
from pathlib import Path

import numpy as np


MNIST_URL = "https://storage.googleapis.com/cvdf-datasets/mnist/"
FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}


def download_mnist(data_dir="data/mnist"):
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)
    for filename in FILES.values():
        target = data_path / filename
        if not target.exists():
            print(f"Downloading {filename}...")
            urllib.request.urlretrieve(MNIST_URL + filename, target)
    return data_path


def _read_images(path):
    with gzip.open(path, "rb") as file:
        magic, count, rows, columns = struct.unpack(">IIII", file.read(16))
        if magic != 2051:
            raise ValueError(f"Invalid MNIST image file: {path}")
        images = np.frombuffer(file.read(), dtype=np.uint8)
    return images.reshape(count, rows * columns).astype(np.float64) / 255.0


def _read_labels(path):
    with gzip.open(path, "rb") as file:
        magic, count = struct.unpack(">II", file.read(8))
        if magic != 2049:
            raise ValueError(f"Invalid MNIST label file: {path}")
        labels = np.frombuffer(file.read(), dtype=np.uint8)
    if labels.size != count:
        raise ValueError(f"Invalid MNIST label count: {path}")
    return labels


def one_hot(labels, classes=10):
    encoded = np.zeros((classes, labels.size), dtype=np.float64)
    encoded[labels, np.arange(labels.size)] = 1.0
    return encoded


def load_mnist(data_dir="data/mnist"):
    data_path = download_mnist(data_dir)
    train_images = _read_images(data_path / FILES["train_images"]).T
    train_labels = _read_labels(data_path / FILES["train_labels"])
    test_images = _read_images(data_path / FILES["test_images"]).T
    test_labels = _read_labels(data_path / FILES["test_labels"])
    return train_images, one_hot(train_labels), test_images, one_hot(test_labels)


def load_mnist_8x8(test_fraction=0.2, seed=2):
    """Load sklearn's 8x8 handwritten-digit dataset in the project format."""
    try:
        from sklearn.datasets import load_digits
    except ImportError as error:
        raise ImportError(
            "The 8x8 dataset requires scikit-learn. Install it with "
            "`python -m pip install scikit-learn`."
        ) from error

    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be between 0 and 1")

    digits = load_digits()
    images = digits.data.astype(np.float64) / 16.0
    labels = one_hot(digits.target)
    rng = np.random.default_rng(seed)
    indices = rng.permutation(images.shape[0])
    test_size = int(images.shape[0] * test_fraction)
    test_indices = indices[:test_size]
    train_indices = indices[test_size:]
    return (
        images[train_indices].T,
        labels[:, train_indices],
        images[test_indices].T,
        labels[:, test_indices],
    )
