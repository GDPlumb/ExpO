
import numpy as np
import os
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import sys

sys.path.insert(0, os.path.join(os.getcwd(), "../Code/"))
from Data import DataManager
from ExplanationMetrics import metrics_maple, metrics_lime

# Fit a Linear Model and a Random forest to the datset and evaluate their performance
def eval(source):

    data = DataManager(source)

    out = {}

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

    for name in ["lr", "dt"]:
        if name == "lr":
            model = LinearRegression()
        elif name == "rf":
            model = RandomForestRegressor(n_estimators = 100)
        elif name == "dt":
            model = DecisionTreeRegressor(min_samples_split = 10)

        model.fit(data.X_train, data.y_train)

        predictions = model.predict(data.X_test)
        out[name + "_mse"] = mean_squared_error(data.y_test, predictions)

        if name == "lr":
            predict_fn = model.predict
        elif name == "rf":
            def predict_fn(x):
                return np.expand_dims(model.predict(x), 1)
        wrapper = Wrapper(predict_fn)

        maple_standard_metric, maple_causal_metric, maple_stability_metric = metrics_maple(wrapper, data.X_train, data.X_val, data.X_test)

        out[name + "_maple_standard_metric"] = maple_standard_metric.tolist()
        out[name + "_maple_causal_metric"] = maple_causal_metric.tolist()
        out[name + "_maple_stability_metric"] = maple_stability_metric.tolist()

        lime_standard_metric, lime_causal_metric, lime_stability_metric = metrics_lime(wrapper, data.X_train, data.X_test)

        out[name + "_lime_standard_metric"] = lime_standard_metric.tolist()
        out[name + "_lime_causal_metric"] = lime_causal_metric.tolist()
        out[name + "_lime_stability_metric"] = lime_stability_metric.tolist()

    return out

datasets = ["autompgs", "communities", "day", "happiness", "housing", "music", "winequality-red"]
n_trials = 10
trials = list(range(n_trials))

data = eval("../Datasets/autompgs.csv")

columns = list(data.keys())
df = pd.DataFrame(0, index = datasets, columns = columns)
df = df.astype("object")

for dataset in datasets:

    for i in range(n_trials):

        data = eval("../Datasets/" + dataset + ".csv")

        for name in columns:
            df.ix[dataset, name] += np.asarray(data[name]) / n_trials

df.to_csv("results_baselines.csv")
