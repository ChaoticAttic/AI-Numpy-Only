import numpy as np
import random

#random_number = round(random.uniform(0, 10), 2)

def used_function(x):
    f = x^2
    return f

def gen_points(amount):
    points_X = np.array[]
    points_Y = np.array[]
def used_function(x):
    return x ** 2

def gen_points(amount):
    points_X = []
    points_Y = []

    for _ in range(amount):
        x = round(random.uniform(0, 10), 2)
        y = round(random.uniform(0, 10), 2)

        points_X.append([x, y])
        points_Y.append(0 if y > used_function(x) else 1)

    return np.array(points_X), np.array(points_Y)

print(gen_points(2))

        


X = np.array([
    [1.2, 3.4],
    [4.5, 2.1],
    [0.8, 0.9]
])

y = np.array([0, 1, 0])

#np.savez("dataset.npz", X=X, y=y)
