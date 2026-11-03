import json
from pathlib import Path

import numpy as np

from main import (
    full_forward_propagation,
    get_multiclass_accuracy_value,
    train_multiclass,
)
from mnist import load_mnist_8x8


with open(Path(__file__).with_name("architecture_mnist.json"), "r") as file:
    architecture = json.load(file)

X_train, Y_train, X_test, Y_test = load_mnist_8x8()
params, costs, accuracies = train_multiclass(
    X_train, Y_train, architecture, epochs=5000,learning_rate=0.01
)

Y_test_hat, _ = full_forward_propagation(X_test, params, architecture)
test_accuracy = get_multiclass_accuracy_value(Y_test_hat, Y_test)
print("Test accuracy:", test_accuracy)
np.save("mnist_params.npy", params)
