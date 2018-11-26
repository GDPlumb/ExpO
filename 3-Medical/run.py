
import json
import os
import sys

sys.path.insert(0, os.path.join(os.getcwd(), "../Code/"))
from eval import eval
from run_search import run_search, args2name

# The networks are small enough that training is faster on CPU
os.environ["CUDA_VISIBLE_DEVICES"]="0"

# Location of Datasets
DATASET_PATHS = {
    "hospital_readmission": os.path.join(os.getcwd(), "../Datasets/hospital_readmission/diabetic_data.csv"),
    "support2": os.path.join(os.getcwd(), "../Datasets/support2.csv"),
}

# Search Space
datasets = ["hospital_readmission", "support2"]
depths = [1, 2]
sizes = [32, 64, 128]
rates = [0.0001, 0.001, 0.01]
regs = []

# Run function
def run(args):
    dataset = args[0]
    trial = args[1]
    depth = args[2]
    size = args[3]
    rate = args[4]

    name = args2name(dataset, trial, depth, size, rate)

    cwd = os.getcwd()

    os.makedirs(name)
    os.chdir(name)

    manager = dataset
    source = DATASET_PATHS[dataset]
    shape = [size] * depth
    out = eval(manager, source,
               hidden_layer_sizes=shape,
               learning_rate=rate,
               stopping_epochs=100)

    with open("out.json", "w") as f:
        json.dump(out, f)

    os.chdir(cwd)


run_search(run_fn=run,
           num_processes=12,
           run_search=True,
           process_search=True,
           run_final=True,
           process_final=True,
           n_search=1,
           n_final=1,
           datasets=datasets,
           depths=depths,
           sizes=sizes,
           rates=rates,
           regularized=False,
           regs=regs)
