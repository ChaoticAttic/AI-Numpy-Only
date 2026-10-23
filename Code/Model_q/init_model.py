from main import train
from main import init_layers
from data_gen import gen_mixed_points
import numpy as np
import json
from datetime import datetime

with open("architecture.json", "r") as f:
    nn_architecture = json.load(f)



model = init_layers(nn_architecture)



X, Y = (gen_mixed_points(2000, 1))



X = X.T
Y = Y.reshape(1, -1)

print(np.max(X))
print(np.sum(Y) / len(Y))

params, costs, accuracity = train(X, Y, nn_architecture, epochs=20000, learning_rate=0.01)

#with open("Code/Data/model.txt", "w") as f:
 #   f.write(str(params))

np.save("params.npy", params)

with open("Code/Data/stats.txt", "a") as f:
    f.write(f"{datetime.now()}Genauigkeit: {max(accuracity)}\n,\n")


#print("Die Genauigkeit ist:", accuracity)
#print(costs)
