import itertools
import json
import numpy as np
import pandas as pd
import tensorflow as tf
import sys
sys.path.insert(0, "../Code/")
from eval_classification import eval

# Fix Random Seeds
np.random.seed(1)
tf.set_random_seed(1)

datasets = ["hospital_readmission"]
trials = []
for i in range(10):
    trials.append(i + 1)
args = itertools.product(datasets, trials)

DATASET_PATHS = {
    "hospital_readmission": "../Datasets/hospital_readmission/diabetic_data.csv",
}

###
# Run Experiments
###


def run(args):
    dataset = args[0]
    trial = args[1]

    out = eval(DATASET_PATHS[dataset], regularizer=None, name="TB/" + dataset + str(trial))

    file = open("Trials/" + dataset + "_" + str(trial) + ".json", "w")
    json.dump(out, file)
    file.close()


for a in args:
    run(a)

###
# Merge Results
###

with open("Trials/" + datasets[0] + "_" + str(trials[0]) + ".json") as f:
    data = json.load(f)

columns = list(data.keys())
df = pd.DataFrame(0, index=datasets, columns=columns)

for dataset in datasets:
    for trial in trials:
        with open("Trials/" + dataset + "_" + str(trial) + ".json") as f:
            data = json.load(f)
        for name in columns:
            df.ix[dataset, name] += data[name] / len(trials)

df.to_csv("results.csv")

