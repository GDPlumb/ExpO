
# Flags to control what parts of the experiment run
run_search = False
process_search = False
run_final = True

# Initial Search Space
datasets = ["autompgs", "communities", "day", "happiness", "housing", "music", "winequality-red"]
trials = list(range(3))
depths = [2,3,4]
sizes = [50, 100, 150, 200] # This should probably be [150, 200, 250, 300] because almost all datasets chose a size of 200
rates = [0.0001, 0.001]

project_dir = "/home/gregory/Desktop/Regularization"

import itertools
import json
from multiprocessing import Pool
import numpy as np
import operator
import os
import sys
sys.path.insert(0, project_dir + "/Code/")
from eval import eval

# The networks are small enough that training is faster on CPU
os.environ["CUDA_VISIBLE_DEVICES"]="-1"

def args2name(dataset, depth, size, rate, trial):
    return "TF/" + dataset + "/" + str([size] * depth) + "/" + str(rate) + "/trial" + str(trial)

def name2args(name):
    chunks = name.split("/")
    dataset = chunks[1]
    shape = chunks[2]
    shape = json.loads(shape)
    depth = len(shape)
    size = shape[0]
    rate = np.float(chunks[3])
    return dataset, depth, size, rate

def run(args):
    dataset = args[0]
    trial = args[1]
    depth = args[2]
    size = args[3]
    rate = args[4]
    
    name = args2name(dataset, depth, size, rate, trial)

    cwd = os.getcwd()
    
    os.makedirs(name)
    os.chdir(name)
    
    manager = "regression"
    source =  project_dir + "/Datasets/" + dataset  + ".csv"
    shape = [size] * depth
    out = eval(manager, source, shape, rate)

    with open("out.json", "w") as f:
        json.dump(out, f)

    os.chdir(cwd)

if run_search:

    list = itertools.product(datasets, trials, depths, sizes, rates)

    p = Pool(5)
    if execute:
        p.map(run, list)
    
if process_search:

    config = {}
    for dataset in datasets:

        list_means = {}

        for depth in depths:
            for size in sizes:
                for rate in rates:

                    mean = 0.0

                    for trial in trials:
                        name = args2name(dataset, depth, size, rate, trial)

                        with open(name + "/out.json") as f:
                            results = json.load(f)

                        mean += results["test_acc"]

                    mean /= len(trials)
                    name = args2name(dataset, depth, size, rate, "_avg")
                    list_means[name] = mean


        sorted_means = sorted(list_means.items(), key = operator.itemgetter(1))
        
        # Get the configuration with the best average performance
        args = name2args(sorted_means[0][0])

        config[args[0]] = [args[1], args[2], args[3]]

    with open("config.json", "w") as f:
        json.dump(config, f)

    os.rename("TF", "TF-initial")

else:
    with open("config.json") as f:
        config = json.load(f)

    list = []
    for dataset in datasets:
        c = config[dataset]
        for trial in range(10):
            args = [dataset, trial, c[0], c[1], c[2]]
            list.append(args)

    p = Pool(5)
    p.map(run, list)
