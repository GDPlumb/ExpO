
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

import itertools
import json
import numpy as np
import os
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import sys

sys.path.insert(0, os.path.join(os.getcwd(), "../Code/"))
from eval import eval
from run_search import run_search, args2name
from Data import DataManager
from ExplanationMetrics import metrics_maple, metrics_lime

from multiprocessing import Pool

# The networks are small enough that training is faster on CPU
os.environ["CUDA_VISIBLE_DEVICES"]="-1"

# Location of Datasets
DATASET_PATH = os.path.join(os.getcwd(), "../Datasets/")

# Search Space
dataset = "housing"
depth = 5
size = 200
rate = 0.001
regs = [0.0, 0.0001, 0.001, 0.01, 0.1, 0.25]
stddevs_eval = [0.1, 0.25, 0.5]
stddevs_reg = [0.1, 0.25, 0.5]
trials = list(range(10))

configs = itertools.product(trials, regs, stddevs_eval, stddevs_reg)

flag_run = True
flag_agg = True

# Run function
def run_fn(args):

    np.random.seed()

    trial = args[0]
    reg = args[1]
    stddev_eval = args[2]
    stddev_reg = args[3]

    name = "TF/" + str(stddev_eval) + "/" + str(stddev_reg) + "/" + str(reg) + "/trial" + str(trial) + "/"

    cwd = os.getcwd()

    os.makedirs(name)
    os.chdir(name)

    manager = "regression"
    source =  DATASET_PATH + dataset + ".csv"
    shape = [size] * depth
    out = eval(manager, source,
           hidden_layer_sizes = shape, learning_rate = rate,
           regularizer = "Causal", c = reg, stddev_reg = stddev_reg,
           stop_on_loss = True,
           evaluate_explanation = True, stddev_eval = stddev_eval)

    with open("out.json", "w") as f:
        json.dump(out, f)

    os.chdir(cwd)

if flag_run:
    p = Pool(4)
    p.map(run_fn, configs)
    p.close()
    p.join()

if flag_agg:

    out = []

    for stddev_eval in stddevs_eval:
        for stddev_reg in stddevs_reg:
            for reg in regs:
                mean_acc = 0.0
                mean_lime = 0.0

                for trial in trials:
                    name = "TF/" + str(stddev_eval) + "/" + str(stddev_reg) + "/" + str(reg) + "/trial" + str(trial) + "/"

                    with open(name + "out.json") as f:
                        results = json.load(f)
                    mean_acc += results["test_acc"]
                    mean_lime += results["lime_causal_metric"][0]
                mean_acc /= len(trials)
                mean_lime /= len(trials)

                out.append((stddev_eval, stddev_reg, reg, mean_acc, mean_lime))

    df = pd.DataFrame(out, columns = ["stddev_eval", "stddev_reg", "weight_reg", "mse", "lime"])
    df.to_csv("results.csv")
