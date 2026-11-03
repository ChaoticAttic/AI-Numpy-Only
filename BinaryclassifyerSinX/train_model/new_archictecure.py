import json


nn_architecture = [
    {"input_dim": 2, "output_dim": 64, "activation": "sin"},
    {"input_dim": 64, "output_dim": 16, "activation": "sin"},
    {"input_dim": 16, "output_dim": 1, "activation": "sigmoid"},
]


with open("architecturesin.json", "w") as f:
    json.dump(nn_architecture, f, indent=4)