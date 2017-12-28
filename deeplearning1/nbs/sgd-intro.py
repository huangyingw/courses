import numpy as np
from numpy.random import random
from matplotlib import animation, pyplot as plt
np.set_printoptions(precision=4, linewidth=100)


def lin(a, b, x): return a * x + b


a = 3.
b = 8.
n = 30
x = random(n)
y = lin(a, b, x)
a_guess = -1.
b_guess = 1.
lr = 0.01
# d[(y-(a*x+b))**2,b] = 2 (b + a x - y)      = 2 (y_pred - y)
# d[(y-(a*x+b))**2,a] = 2 x (b + a x - y)    = x * dy/db


def upd():
    global a_guess, b_guess
    y_pred = lin(a_guess, b_guess, x)
    dydb = 2 * (y_pred - y)
    dyda = x * dydb
    a_guess -= lr * dyda.mean()
    b_guess -= lr * dydb.mean()


fig = plt.figure(dpi=100, figsize=(5, 4))
plt.scatter(x, y)
line, = plt.plot(x, lin(a_guess, b_guess, x))


def animate(i):
    line.set_ydata(lin(a_guess, b_guess, x))
    for i in range(10):
        upd()
    return line,


ani = animation.FuncAnimation(fig, animate, np.arange(0, 40), interval=100)
plt.show()
