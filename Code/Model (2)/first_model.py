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




model = init_layers(nn_architecture)



X, Y = (gen_points(500))

print(X.shape)
print(Y.shape)

X = X.T
Y = Y.reshape(1, -1)
print(X.shape)
print(Y.shape)

params, costs, accuracity = train(X, Y, nn_architecture, epochs=1, learning_rate=0.01)

with open("Code/Data/model.txt", "w") as f:
    f.write(params)

print("Die Genauigkeit ist:", accuracity)
