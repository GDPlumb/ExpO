
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
datasets = ["cancer"]

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

    manager = "cancer"
    source =  DATASET_PATH + dataset + ".csv"
    shape = [size] * depth
    out = eval(manager, source,
           hidden_layer_sizes = shape,
           learning_rate = rate,
           regularizer = "Causal", c = reg, stddev_reg = 0.5,
           evaluate_explanation = evaluate_explanation, apply_sigmoid = True,
           stop_on_loss = True, stopping_epochs = 10)

    with open("out.json", "w") as f:
        json.dump(out, f)

    os.chdir(cwd)

def run_fn_search(*args):
    return run_fn(*args, evaluate_explanation = False)

run_search(run_fn_search = run_fn_search, n_search = 1, lower_is_better = False,
            run_search = False, process_search = False,
            run_fn_final = run_fn, n_final = 10,
            run_final = True, process_final = True,
            datasets = datasets, depths = None, sizes = None, rates = None,
            regularized = True, regs = None,
            num_processes = 5)
