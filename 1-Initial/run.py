import itertools
import json
import pandas as pd
import sys
sys.path.insert(0, "../Code/")
from load import load_normalize_data
from eval import eval

datasets = ["autompgs", "communities", "crimes", "day", "happiness", "housing", "music", "winequality-red"]
trials = []
for i in range(10):
    trials.append(i + 1)
args = itertools.product(datasets, trials)

###
# Run Experiments
###

def run(args):
    dataset = args[0]
    trial = args[1]

    X_train, y_train, X_valid, y_valid, X_test, y_test, mu, sigma = load_normalize_data("../Datasets/" + dataset  + ".csv")
    out = eval(X_train, y_train, X_valid, y_valid, X_test, y_test, name = "TB/" + dataset + str(trial))

    file = open("Trials/" + dataset + "_" + str(trial) + ".json", "w")
    json.dump(out, file)
    file.close()

for i in args:
    run(i)

###
# Merge Results
###

with open("Trials/" + datasets[0] + "_" + str(trials[0]) + ".json") as f:
    data = json.load(f)

columns = list(data.keys())
df = pd.DataFrame(0, index = datasets, columns = columns)

for dataset in datasets:
    for trial in trials:
        with open("Trials/" + dataset + "_" + str(trial) + ".json") as f:
            data = json.load(f)
        for name in columns:
            df.ix[dataset, name] += data[name] / len(trials)

df.to_csv("results.csv")
