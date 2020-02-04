
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

import math
import numpy as np
from sklearn.linear_model import LinearRegression


def f1(x):
    return -4 * x**2 + 10 * x - 0.5 * math.cos(7 * math.pi * x)

def grad_f1(x):
    return -8 * x + 10 + 3.5 * math.pi * math.sin(7 * math.pi * x)

def f2(x):
    return 4 * math.cos(10 * math.pi * x) +  x + 5

n = 100

x_grid = np.zeros((n + 1, 1))
for i in range(n + 1):
    x = i / n
    x_grid[i] = x

def gen(f):
    
    y_grid = np.zeros((n + 1, 1))
    for i in range(n + 1):
        y_grid[i] = f(x_grid[i])
        
    reg = LinearRegression().fit(x_grid, y_grid)

    pred_grid = reg.predict(x_grid)

    residuals_grid = y_grid - pred_grid
    
    
    return y_grid, pred_grid, residuals_grid


fig, axes = plt.subplots(nrows=2, ncols=2)

y_grid, pred_grid, residuals_grid = gen(f1)
axes[0,0].plot(x_grid, y_grid)
axes[0,0].plot(x_grid, pred_grid)
axes[1,0].plot(x_grid, residuals_grid)

y_grid, pred_grid, residuals_grid = gen(f2)
axes[0,1].plot(x_grid, y_grid)
axes[0,1].plot(x_grid, pred_grid)
axes[1,1].plot(x_grid, residuals_grid)

fig.tight_layout()
plt.savefig("Theory-variance.png")
plt.close()

def gen_taylor(f, f_grad, x):
    
    def pred(x_new):
        return f(x) + f_grad(x) * (x_new - x)
    
    y_grid = np.zeros((n + 1, 1))
    for i in range(n + 1):
        y_grid[i] = pred(x_grid[i])
        
    return y_grid
    
    

y_grid, pred_grid, residuals_grid  = gen(f1)
plt.plot(x_grid, y_grid)
plt.plot(x_grid, pred_grid)

taylor_1 = gen_taylor(f1, grad_f1, 0.4)

plt.plot(x_grid, taylor_1)

taylor_2 = gen_taylor(f1, grad_f1, 0.5)

plt.plot(x_grid, taylor_2)

plt.legend(["Function", "Local Explanation", "Taylor approximation at x=0.4", "Taylor approximation at x=0.5"])

plt.savefig("Theory-taylor.png")
plt.close()
    

