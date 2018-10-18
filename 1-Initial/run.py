import itertools
import json
import numpy as np
import pandas as pd
import tensorflow as tf
import sys
sys.path.insert(0, "../Code/")
from eval import eval

# Fix Random Seeds
np.random.seed(1)
tf.set_random_seed(1)

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

    out = eval("../Datasets/" + dataset  + ".csv", name = "TB/" + dataset + str(trial))

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
