import os
import itertools
import json
import numpy as np
import tensorflow as tf
import sys
sys.path.insert(0, os.path.join(os.getcwd(), "../Code/"))
from eval import eval
from Merge import merge

# Fix Random Seeds
np.random.seed(1)
tf.set_random_seed(1)

datasets = ["support2"] # ["hospital_readmission"]
trials = []
for i in range(1):
    trials.append(i + 1)
args = itertools.product(datasets, trials)

DATASET_PATHS = {
    "hospital_readmission": os.path.join(
        os.getcwd(), "../Datasets/hospital_readmission/diabetic_data.csv"),
    "support2": os.path.join(os.getcwd(), "../Datasets/support2.csv"),
}

###
# Run Experiments
###

def run(args):
    dataset = args[0]
    trial = args[1]

    os.makedirs("TB/support2/temp")
    os.chdir("TB/support2/temp")

    out = eval(dataset, DATASET_PATHS[dataset], [32, 32], 0.01,
               regularizer=None, c=1)

    file = open("Trials/" + dataset + "_" + str(trial) + ".json", "w")
    json.dump(out, file)
    file.close()

for a in args:
    run(a)

###
# Merge Results
###

merge(datasets, trials)
