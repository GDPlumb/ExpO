
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
DATASET_PATH = os.path.join(os.getcwd(), "../Datasets/")

# Search Space
datasets = ["autompgs", "communities", "day", "housing", "music", "winequality-red"]
depths = [1, 2, 3]
sizes = [100, 150, 200, 250, 300]
rates = [0.001]
regs = [0.5, 1.0, 2.0, 4.0]

# Run function
def run_fn(args, evaluate_explanation = True):

    np.random.seed()

    dataset = args[0]
    trial = args[1]
    depth = args[2]
    size = args[3]
    rate = args[4]
    reg = args[5]

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
           regularizer = "Causal", c = reg,
           evaluate_explanation = evaluate_explanation,
           stop_on_loss = True)

    with open("out.json", "w") as f:
        json.dump(out, f)

    os.chdir(cwd)

def run_fn_search(*args):
    return run_fn(*args, evaluate_explanation = False)

run_search(run_fn_search = run_fn_search, n_search = 2, lower_is_better = True,
            run_search = True, process_search = True,
            run_fn_final = run_fn, n_final = 10,
            run_final = True, process_final = True,
            datasets = datasets, depths = depths, sizes = sizes, rates = rates,
            regularized = True, regs = regs,
            num_processes = 1)
