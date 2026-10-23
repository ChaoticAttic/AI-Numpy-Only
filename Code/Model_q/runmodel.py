from main import full_forward_propagation
from main import convert_prob_into_class
import numpy as np
import json

with open("architecture.json", "r") as f:
    nn_architecture = json.load(f)



params_values = np.load(
    "params.npy",
    allow_pickle=True
).item()

input_number = np.array([
    [2],
    [0.1]
])

Y_hat, memory = full_forward_propagation(input_number,params_values,nn_architecture)

print(Y_hat)
print(convert_prob_into_class(Y_hat))