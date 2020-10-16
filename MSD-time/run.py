
import json
import numpy as np
import os
import sys

sys.path.insert(0, os.path.join(os.getcwd(), "../Code/"))
from eval import eval
from run_search import run_search, args2name

# The networks are small enough that training is faster on CPU
os.environ["CUDA_VISIBLE_DEVICES"]="-1"

# Location of Datasets
DATASET_PATH = os.path.join(os.getcwd(), "../Datasets/YearPredictionMSD/")

import time

# Run function
def run_fn(trial, type, evaluate_explanation = True):

    np.random.seed()

    dataset = "msd"
    depth = 5
    size = 100
    rate = 0.001
    reg = 0.1

    name = args2name(dataset, trial, depth, size, rate, reg)

    cwd = os.getcwd()

    os.makedirs(name)
    os.chdir(name)

    manager = "msd"
    source =  DATASET_PATH
    shape = [size] * depth
    out, epoch = eval(manager, source,
           hidden_layer_sizes = shape,
           learning_rate = rate,
           regularizer = type, c = reg, stddev_reg = 0.5,
           evaluate_explanation = False,
           stop_on_loss = True,
           min_epochs = 10, stopping_epochs = 10)

    with open("out.json", "w") as f:
        json.dump(out, f)

    os.chdir(cwd)
    
    return epoch


t = 10

epoch_full = 0
start = time.time()
for i in range(t):
    os.system("rm -rf TF")
    epoch_full += run_fn(0, "Causal")
stop = time.time()

time_full = (stop - start) / t
epoch_full /= t

epoch_rand = 0
start = time.time()
for i in range(t):
    os.system("rm -rf TF")
    epoch_rand += run_fn(0, "Causal1D")
stop = time.time()

time_rand = (stop - start) / t
epoch_rand /= t

print("ExpO-Fidelity: ", time_full / epoch_full)
print("ExpO-Fidelity-1D: ", time_rand / epoch_rand)
