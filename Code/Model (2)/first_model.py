from main import train
from main import init_layers
from data_gen import gen_points
nn_architecture = [
    {"input_dim": 2, "output_dim": 4, "activation": "relu"},
    {"input_dim": 4, "output_dim": 6, "activation": "relu"},
    {"input_dim": 6, "output_dim": 6, "activation": "relu"},
    {"input_dim": 6, "output_dim": 4, "activation": "relu"},
    {"input_dim": 4, "output_dim": 1, "activation": "sigmoid"},
]

pfad = "Code/Data/trainingswerte.txt"


model = init_layers(nn_architecture)


with open(pfad, 'r') as fp:
    lines = sum(1 for line in fp)
    for i in range lines:
        X = 