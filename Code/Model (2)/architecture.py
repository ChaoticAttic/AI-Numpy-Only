import json

#nn_architecture = [
#    {"input_dim": 2, "output_dim": 4, "activation": "relu"},
#    {"input_dim": 4, "output_dim": 6, "activation": "relu"},
#    {"input_dim": 6, "output_dim": 6, "activation": "relu"},
 #   {"input_dim": 6, "output_dim": 4, "activation": "relu"},
#    {"input_dim": 4, "output_dim": 1, "activation": "sigmoid"}
#]

nn_architecture = [
    {"input_dim": 2, "output_dim": 4, "activation": "relu"},
    {"input_dim": 4, "output_dim": 1, "activation": "sigmoid"},
]


with open("architecture.json", "w") as f:
    json.dump(nn_architecture, f, indent=4)