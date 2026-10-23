from main import tune, tune_with_synth_data
from main import init_layers
from main import full_forward_propagation
from main import get_accuracy_value
from data_gen import gen_mixed_points
import numpy as np
import json
from datetime import datetime

with open("architecture.json", "r") as f:
    nn_architecture = json.load(f)



model = np.load(
    "params.npy",
    allow_pickle=True
).item()


X, Y = (gen_mixed_points(1000, 0))

X = X.T
Y = Y.reshape(1, -1)



Y_hat, _ = full_forward_propagation(X, model, nn_architecture)

print(
    "Accuracy normal vor Tune:",
    get_accuracy_value(Y_hat, Y)
)

Xhard, Yhard = (gen_mixed_points(1000, 1))

Xhard = Xhard.T
Yhard = Yhard.reshape(1, -1)



Y_hat_hard, _ = full_forward_propagation(Xhard, model, nn_architecture)

print(
    "Accuracy hard vor Tune:",
    get_accuracy_value(Y_hat_hard, Yhard)
)


#params, costs, accuracity = tune(X, Y, nn_architecture, epochs=100, learning_rate=0.003, params_values=model)
params, costs, accuracity = tune_with_synth_data(nn_architecture=nn_architecture, epochs=10000, data_size=1000, bias=0, learning_rate=0.003, params_values=model)

np.save("params.npy", params)

with open("Code/Data/stats.txt", "a") as f:
    f.write(f"{datetime.now()}Genauigkeit: {max(accuracity)}\n,\n")


Y_hat, _ = full_forward_propagation(X, model, nn_architecture)

print(
    "Accuracy normal after Tune:",
    get_accuracy_value(Y_hat, Y)
)

Y_hat_hard, _ = full_forward_propagation(Xhard, model, nn_architecture)

print(
    "Accuracy hard after Tune:",
    get_accuracy_value(Y_hat_hard, Yhard)
)