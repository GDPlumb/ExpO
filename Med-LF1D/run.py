
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
DATASET_PATHS = {
    "hospital_readmission": os.path.join(os.getcwd(), "../Datasets/hospital_readmission/diabetic_data.csv"),
    "support2": os.path.join(os.getcwd(), "../Datasets/support2.csv"),
}

# Search Space
datasets = ["support2"] #, "hospital_readmission"]
regs = [0.1, 0.05, 0.025, 0.01, 0.005, 0.0025, 0.001]

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

    manager = dataset
    source = DATASET_PATHS[dataset]
    shape = [size] * depth
    out = eval(manager, source,
           hidden_layer_sizes = shape,
           learning_rate = rate,
           regularizer = "Causal1D", c = reg, stddev_reg = 0.5,
           evaluate_explanation = evaluate_explanation,
           stop_on_loss = True, tol = 5e-4,
           min_epochs = 10, stopping_epochs = 10)

    with open("out.json", "w") as f:
        json.dump(out, f)

    os.chdir(cwd)

def run_fn_search(*args):
    return run_fn(*args, evaluate_explanation = False)

run_search(run_fn_search = run_fn_search, n_search = 5, lower_is_better = False, gamma = 0.0,
            run_search = False, process_search = False,
            run_fn_final = run_fn, n_final = 10,
            run_final = True, process_final = True,
            datasets = datasets, source = "../Med-None/config.json",
            regularized = True, regs = regs,
            num_processes = 5)
