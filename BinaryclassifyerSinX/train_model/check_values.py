from main import get_accuracy_value
from main import full_forward_propagation
from main import get_cost_value
from sin_gen import gen_sin

import json

import numpy as np

with open("architecturesin.json", "r") as f:
    nn_architecture = json.load(f)

X, Y = gen_sin(100000)

X = X.T
Y = Y.reshape(1, -1)

params_values = np.load(
    "sinmodel.npy",
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
