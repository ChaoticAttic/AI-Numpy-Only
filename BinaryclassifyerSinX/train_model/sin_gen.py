import numpy as np
import random


def used_function(x):
    return np.sin(x)

def gen_sin(amount):
    points_X = []
    points_Y = []

    for _ in range(amount):
        x = round(random.uniform(0, 10), 2)
      
        y = round(random.uniform(-1, 1), 2)

        points_X.append([x, y])
        points_Y.append(0 if y < used_function(x) else 1)


    return np.array(points_X), np.array(points_Y)


print(gen_sin(10))
    
