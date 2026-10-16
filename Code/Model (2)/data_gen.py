import numpy as np
import random

#random_number = round(random.uniform(0, 10), 2)

def used_function(x):
    f = x^2
    return f


def used_function(x):
    return x ** 2

def gen_points(amount):
    points_X = []
    points_Y = []

    for _ in range(amount):
        x = round(random.uniform(0, 10), 2)
      
        y = round(used_function(x) + random.uniform(-1, 1), 2)

        points_X.append([x, y])
        points_Y.append(0 if y < used_function(x) else 1)


        x = round(random.uniform(0, 10), 2)
        y = round(random.uniform(0, 59), 2)
       

        points_X.append([x, y])
        points_Y.append(0 if y < used_function(x) else 1)

    return np.array(points_X), np.array(points_Y)

def gen_close_points(amaount):
    points_X = []
    points_Y = []

    for _ in range(amount):

        x = round(random.uniform(0, 10), 2)
        y = round(used_function(x) + random.uniform(-0.5, 0.5), 2)
       

        points_X.append([x, y])
        points_Y.append(0 if y < used_function(x) else 1)

    return np.array(points_X), np.array(points_Y)


with open("Code/Data/trainingswerte.txt", "a") as f:
    X, Y = gen_points(5000)
    for (x, y), label in zip(X, Y):
        f.write(f"{x},{y},{label}\n")
    print(np.mean(Y))