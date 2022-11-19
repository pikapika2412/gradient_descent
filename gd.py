import numpy as np

n = 0.1
w = np.array([0, 1, 5])
y = np.array([1, 6, 1]).T
x = np.array([[1, 1, 1], [1, -2, 0], [2, 5, 1]])

def loss(x, y, w):
    return 0.5 * np.sum((y - w @ x)**2)

def grad(x, y, w):
    return (y - w @ x) @ x

def update_w(w, n, x, y):
    return w + n * grad(x, y, w).T

print(loss(x, y, w))

for i in range(60):
    w = update_w(w, n, x, y)
    print(loss(x, y, w))

print('\n', w)