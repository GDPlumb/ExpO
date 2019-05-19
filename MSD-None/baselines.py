
import json
import numpy as np
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import sys

sys.path.insert(0, os.path.join(os.getcwd(), "../Code/"))
from MillionSongData import YearPredictionMSDDataManager as DataManager
from ExplanationMetrics import metrics_maple, metrics_lime

from multiprocessing import Pool

# The networks are small enough that training is faster on CPU
os.environ["CUDA_VISIBLE_DEVICES"]="-1"

dataset =  "msd"
trials = list(range(5))

def baseline(name):

    data = DataManager("../Datasets/YearPredictionMSD/")

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

    #_, lime_causal_metric , _= metrics_lime(wrapper, data.X_train, data.X_test)

    return acc, 0.0 #lime_causal_metric[0]

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
