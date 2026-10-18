import numpy as np
import random

#random_number = round(random.uniform(0, 10), 2)


def used_function(x):
    return x ** 2

def gen_points(amount):
    points_X = []
    points_Y = []

    for _ in range(amount):
        x = round(random.uniform(0, 10), 2)
      
        y = round(random.triangular(0, 100, 0), 2)

        points_X.append([x, y])
        points_Y.append(0 if y < used_function(x) else 1)


    return np.array(points_X), np.array(points_Y)

def gen_close_points(amount):
    points_X = []
    points_Y = []

    for _ in range(amount):

        x = round(random.uniform(0, 10), 2)
    
        y = max(0, round(used_function(x) + random.uniform(-0.5, 0.5), 2))
       

        points_X.append([x, y])
        points_Y.append(0 if y < used_function(x) else 1)

    return np.array(points_X), np.array(points_Y)


def gen_mixed_points(amount, bias):

    normal_amount = int(amount * (1 - bias))
    close_amount = amount - normal_amount

    X_parts = []
    Y_parts = []

    if normal_amount > 0:
        X1, Y1 = gen_points(normal_amount)
        X_parts.append(X1)
        Y_parts.append(Y1)

    if close_amount > 0:
        X2, Y2 = gen_close_points(close_amount)
        X_parts.append(X2)
        Y_parts.append(Y2)

    X = np.vstack(X_parts)
    Y = np.concatenate(Y_parts)

    indices = np.random.permutation(len(X))

    return X[indices], Y[indices]



