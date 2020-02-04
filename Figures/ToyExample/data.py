
import numpy as np
import pandas as pd

def f(x):
    return 10 * (x - 0.2)*(x - 0.5)*(x - 0.9)

def generate(n = 50, sigma = 0.1):

    x = np.random.uniform(size = (n))

    noise = np.random.normal(scale = sigma, size = (n))

    y = f(x) + noise

    return x,y

x,y = generate()

df = pd.DataFrame(y, x)

df.to_csv("data.csv", header = False, index = True)
