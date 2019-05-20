
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

import itertools
import json
import numpy as np
import os
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
regs = [0.0, 0.0001, 0.001, 0.01, 0.1, 1.0]
trials = list(range(10))

configs = itertools.product(trials, regs)

flag_run = False
flag_agg = False
flag_run_baselines = False
flag_plot = True

# Run function
def run_fn(args):

    np.random.seed()

    trial = args[0]
    reg = args[1]

    name = args2name(dataset, trial, depth, size, rate, reg)

    cwd = os.getcwd()

    os.makedirs(name)
    os.chdir(name)

    manager = "regression"
    source =  DATASET_PATH + dataset + ".csv"
    shape = [size] * depth
    out = eval(manager, source,
           hidden_layer_sizes = shape,
           learning_rate = rate,
           regularizer = "Causal1D", c = reg,
           evaluate_explanation = True,
           stop_on_loss = True)

    with open("out.json", "w") as f:
        json.dump(out, f)

    os.chdir(cwd)

if flag_run:
    p = Pool(6)
    p.map(run_fn, configs)
    p.close()
    p.join()

if flag_agg:

    agg = {}
    config = {}

    list_means = {}

    for reg in regs:
        mean_acc = 0.0
        mean_lime = 0.0
        for trial in trials:
            name = args2name(dataset, trial, depth, size, rate, reg)
            with open(name + "/out.json") as f:
                results = json.load(f)
            mean_acc += results["test_acc"]
            mean_lime += results["lime_causal_metric"][0]
        mean_acc /= len(trials)
        mean_lime /= len(trials)
        list_means[str(reg)] = [mean_acc, mean_lime]

    with open("results.json", "w") as f:
        json.dump(list_means, f)

if flag_run_baselines:

    def baseline(name):

        data = DataManager(DATASET_PATH + dataset + ".csv")

        # Wrapper for SKlearn models so that their interface matches metrics_maple and metrics_lime
        class Wrapper():
            def __init__(self, pred):
                self.pred = pred
                self.index = 0

            def predict(self, x):
                return self.pred(x)

            def set_index(self, i):
                self.index = i

            def predict_index(self, x):
                return np.squeeze(self.pred(x)[:, self.index])

        if name == "lr":
            model = LinearRegression()
        elif name == "dt":
            model = DecisionTreeRegressor()
        elif name == "rf":
            model = RandomForestRegressor()


        model.fit(data.X_train, data.y_train)

        predictions = model.predict(data.X_test)
        acc = mean_squared_error(data.y_test, predictions)

        if name == "lr":
            predict_fn = model.predict
        elif name == "dt" or name == "rf":
            def predict_fn(x):
                return np.expand_dims(model.predict(x), 1)
        wrapper = Wrapper(predict_fn)


        _, lime_causal_metric , _= metrics_lime(wrapper, data.X_train, data.X_test)

        return acc, lime_causal_metric[0]

    out = {}
    for name in ["lr", "rf", "dt"]:
        mean_acc = 0.0
        mean_lime = 0.0
        for trial in trials:
            t_a, t_l = baseline(name)
            mean_acc += t_a
            mean_lime += t_l
        mean_acc /= len(trials)
        mean_lime /= len(trials)
        out[name] = [mean_acc, mean_lime]

    with open("baselines.json", "w") as f:
        json.dump(out, f)

if flag_plot:

    with open("results.json", "r") as f:
        networks = json.load(f)

    with open("baselines.json", "r") as f:
        baselines = json.load(f)

    names = []
    acc = []
    lime = []

    for key in networks.keys():
        names.append("c="+key)
        acc.append(networks[key][0])
        lime.append(networks[key][1])

    for key in baselines.keys():
        if key == "lr":
            names.append("linear regression")
        elif key == "dt":
            names.append("decision tree")
        elif key == "rf":
            names.append("random forest")
        acc.append(baselines[key][0])
        lime.append(baselines[key][1])

    plt.scatter(acc, lime)
    for i, label in enumerate(names):
        text = plt.annotate(label, (acc[i], lime[i]))
        text.set_alpha(0.8)
    plt.xlabel("Predictive MSE")
    plt.ylabel("Lime Neighborhood Fidelity Metric")
    plt.savefig("plot.pdf")
    plt.close()
