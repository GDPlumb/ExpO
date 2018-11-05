
import itertools
import json
import numpy as np
import tensorflow as tf
import sys
sys.path.insert(0, "../Code/")
from eval import eval
from Merge import merge

# Fix Random Seeds
np.random.seed(1)
tf.set_random_seed(1)

datasets = ["hospital_readmission"]
trials = []
for i in range(1):
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

    out = eval(DATASET_PATHS[dataset], dataset, regularizer = "Causal", name = "TB/" + dataset + str(trial))

    file = open("Trials/" + dataset + "_" + str(trial) + ".json", "w")
    json.dump(out, file)
    file.close()

for a in args:
    run(a)

###
# Merge Results
###

merge(datasets, trials)
