
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

# Search Space
datasets = ["msd"]
depths = [1,2,3,4,5]
sizes = [100,200,300,400,500]
rates = [0.001]

# Run function
def run_fn(args, evaluate_explanation = True):

    np.random.seed()

    dataset = args[0]
    trial = args[1]
    depth = args[2]
    size = args[3]
    rate = args[4]

    name = args2name(dataset, trial, depth, size, rate)

    cwd = os.getcwd()

    os.makedirs(name)
    os.chdir(name)

    manager = "msd"
    source =  DATASET_PATH
    shape = [size] * depth
    out = eval(manager, source,
               hidden_layer_sizes = shape,
               learning_rate = rate,
               evaluate_explanation = evaluate_explanation,
               stop_on_loss = True)

    with open("out.json", "w") as f:
        json.dump(out, f)

    os.chdir(cwd)

def run_fn_search(*args):
    return run_fn(*args, evaluate_explanation = False)

run_search(run_fn_search = run_fn_search, n_search = 10, lower_is_better = True,
            run_search = True, process_search = True,
            run_fn_final = run_fn, n_final = 3,
            run_final = False, process_final = False,
            datasets = datasets, depths = depths, sizes = sizes, rates = rates,
            regularized = False, regs = None,
            num_processes = 4)
