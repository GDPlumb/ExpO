
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

datasets = ["communities"] #["autompgs", "communities", "day", "happiness", "housing", "music", "winequality-red"]
trials = []
for i in range(1):
    trials.append(i + 1)
args = itertools.product(datasets, trials)

###
# Run Experiments
###

def run(args):
    dataset = args[0]
    trial = args[1]

    out = eval("regression","../Datasets/" + dataset  + ".csv", [100,100], 0.01, name = "TF/" + dataset + str(trial))

    file = open("Trials/" + dataset + "_" + str(trial) + ".json", "w")
    json.dump(out, file)
    file.close()

for i in args:
    run(i)

###
# Merge Results
###

merge(datasets, trials)
