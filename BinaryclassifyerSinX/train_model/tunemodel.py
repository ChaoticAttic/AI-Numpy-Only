from main import tune
from main import init_layers
from main import full_forward_propagation
from main import get_accuracy_value
from sin_gen import gen_sin
import numpy as np
import json
from datetime import datetime

with open("architecturesin.json", "r") as f:
    nn_architecture = json.load(f)



model = np.load(
    "sinmodel.npy",
    allow_pickle=True
).item()


X, Y = (gen_sin(1000))

X = X.T
Y = Y.reshape(1, -1)



Y_hat, _ = full_forward_propagation(X, model, nn_architecture)

print(
    "Accuracy normal vor Tune:",
    get_accuracy_value(Y_hat, Y)
)







#params, costs, accuracity = tune(X, Y, nn_architecture, epochs=100, learning_rate=0.003, params_values=model)
params, costs, accuracity = tune(nn_architecture=nn_architecture, epochs=10000, data_size=1000, learning_rate=0.005, params_values=model)

np.save("params.npy", params)

with open("Code/Data/stats.txt", "a") as f:
    f.write(f"{datetime.now()}Genauigkeit: {max(accuracity)}\n,\n")


Y_hat, _ = full_forward_propagation(X, model, nn_architecture)

print(
    "Accuracy normal after Tune:",
    get_accuracy_value(Y_hat, Y)
)

