'''
Train a single perceptron and SVM to learn an AND gate with two inputs x1 and x2. 
Assume that all the weights of the perceptron are initialized as 0. 
Show the calculation for each step and also 
draw the decision boundary for each updation.


x1  x2  y
0   0   0
0   1   0
1   0   0
1   1   1
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from cvxopt import solvers, matrix


#step 1: 
x1 = [[0, 0, -1],
     [0, 1, -1],
     [1, 0, -1],
     [1, 1, 1]
     ]
x1 = np.array(x1)
#step 2 iteration
#initailising a random weight vector
a = np.array([0,0,0])
learning_rate = 0.5

def is_all_true(weight_vector):
    all = True
    for i in range(4):
        if np.dot(x1[i, :], weight_vector) <= 0:
            all = False
    return all


def perceptron(weight_vector):
    a = weight_vector.copy()  # Copy the initial weight vector
    while not is_all_true(a):
        for i in range(4):
            if np.dot(x1[i, :], a) > 0:
                continue
            elif np.dot(x1[i, :], a) <= 0:
                a = a + learning_rate * x1[i, :]
    return a

b = perceptron(np.array([0, 0, 0]))
plt.scatter(x1[:, 0], x1[:, 1], c=x1[:, 2], cmap='viridis')
plt.xlabel('x1')
plt.ylabel('x2')
slope = -b[0] / b[1]
intercept = -b[2] / b[1]
x_decision = np.linspace(0, 1.5, 100)
y_decision = slope * x_decision + intercept
plt.plot(x_decision, y_decision, '-r', label='Decision Boundary')
plt.title("Using Single Perceptron")
plt.show()

## svm



def plot(w, b):
    x_min = -5
    x_max = 5
    y_min = -5
    y_max = 5
    xx = np.linspace(x_min, x_max)
    a = -w[0] / w[1]
    yy = a * xx - (b) / w[1]
    margin = 1 / np.sqrt(np.sum(w**2))
    yy_neg = yy - np.sqrt(1 + a**2) * margin
    yy_pos = yy + np.sqrt(1 + a**2) * margin

    plt.plot(xx, yy, "b-")
    plt.plot(xx, yy_neg, "m--")
    plt.plot(xx, yy_pos, "m--")
    plt.scatter(x[:, 0], x[:, 1], c=y.ravel())
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.show()


x = np.array([[0,0], [0,1], [1,0], [1,1]])
y = np.array([-1,-1,-1,1]).reshape(-1, 1)

n = x.shape[0]
H = matrix(np.dot(y * x, (y * x).T)) * 1.0
q = matrix(np.repeat([-1.0], n)[..., None])
A = matrix(y.reshape(1, -1)) * 1.0
b = matrix(0.0)
G = matrix(np.negative(np.eye(n)))
h = matrix(np.zeros(n))

sol = solvers.qp(H, q, G, h, A, b)
alphas = np.array(sol["x"])
w = np.dot((y * alphas).T, x)[0]
S = (alphas > 1e-5).flatten()
b = np.mean(y[S] - np.dot(x[S], w.reshape(-1, 1)))
print(f"Weight = {w}")
print(f"Bias   = {b}")
plot(w, b)
