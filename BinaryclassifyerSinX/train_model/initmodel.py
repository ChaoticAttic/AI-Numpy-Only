from main import train, init_layers

from sin_gen import gen_sin
import numpy as np
import json
from datetime import datetime

with open("/workspaces/AI-Numpy-Only/BinaryclassifyerSinX/saved_data/architecturesin.json", "r") as f:
    nn_architecture = json.load(f)



model = init_layers(nn_architecture)



X, Y = (gen_sin(2000))



X = X.T
Y = Y.reshape(1, -1)

print(np.max(X))
print(np.sum(Y) / len(Y))

params, costs, accuracity = train(X, Y, nn_architecture, epochs=20000, learning_rate=0.01)

#with open("Code/Data/model.txt", "w") as f:
 #   f.write(str(params))

np.save("sinmodel.npy", params)

with open("Code/Data/stats.txt", "a") as f:
    f.write(f"{datetime.now()}Genauigkeit: {max(accuracity)}\n,\n")


#print("Die Genauigkeit ist:", accuracity)
#print(costs)
