from main import get_accuracy_value
from main import full_forward_propagation
from main import get_cost_value
from data_gen import gen_mixed_points

import json

import numpy as np

with open("architecture.json", "r") as f:
    nn_architecture = json.load(f)

X, Y = gen_mixed_points(1000000, 0)

X = X.T
Y = Y.reshape(1, -1)

params_values = np.load(
    "params.npy",
    allow_pickle=True
).item()


Y_hat, _ = full_forward_propagation(
    X,
    params_values,
    nn_architecture
)


accuracy = get_accuracy_value(Y_hat, Y)
cost = get_cost_value(Y_hat, Y)


with open("log.txt", "a") as f:
    print("\n===== MODELLTEST =====", file=f)
    print("Accuracy :", accuracy, file=f)
    print("Cost     :", cost, file=f)

    print("\n===== OUTPUTS =====", file=f)
    print("Yhat min :", Y_hat.min(), file=f)
    print("Yhat max :", Y_hat.max(), file=f)
    print("Yhat mean:", Y_hat.mean(), file=f)
