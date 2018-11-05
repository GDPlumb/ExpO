
project_dir = "/home/gregory/Desktop/Regularization"

import itertools
import json
from multiprocessing import Pool
import numpy as np
import operator
import os
import pandas as pd
import sys
sys.path.insert(0, project_dir + "/Code/")
from eval import eval

# The networks are small enough that training is faster on CPU
os.environ["CUDA_VISIBLE_DEVICES"]="-1"

def args2name(dataset, trial, depth, size, rate):
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
    
    name = args2name(dataset, trial, depth, size, rate)

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

# Flags to control what parts of the experiment run
run_search = True
process_search = True
run_final = True
process_final = True

n_search = 5
n_final = 20

# Initial Search Space
datasets = ["autompgs", "communities", "day", "happiness", "housing", "music", "winequality-red"]
trials = list(range(n_search))
depths = [1,2,3,4,5]
sizes = [100, 150, 200, 250]
rates = [0.0001, 0.001]

if run_search:

    configs = itertools.product(datasets, trials, depths, sizes, rates)

    p = Pool(5)
    p.map(run, configs)
    
if process_search:

    config = {}
    for dataset in datasets:

        list_means = {}

        for depth in depths:
            for size in sizes:
                for rate in rates:

                    mean = 0.0

                    for trial in trials:
                        name = args2name(dataset, trial, depth, size, rate)

                        with open(name + "/out.json") as f:
                            results = json.load(f)

                        mean += results["test_acc"]

                    mean /= n_search
                    name = args2name(dataset, "_avg", depth, size, rate)
                    list_means[name] = mean

        sorted_means = sorted(list_means.items(), key = operator.itemgetter(1))
        
        # Get the configuration with the best average performance
        args = name2args(sorted_means[0][0])

        config[args[0]] = [args[1], args[2], args[3]]

    with open("config.json", "w") as f:
        json.dump(config, f)

    os.rename("TF", "TF-initial")

if run_final or aggreggate:
    with open("config.json") as f:
        config = json.load(f)
    configs = []
    for dataset in datasets:
        c = config[dataset]
        for trial in range(n_final):
            args = [dataset, trial, c[0], c[1], c[2]]
            configs.append(args)

if run_final:
    p = Pool(5)
    p.map(run, configs)

if process_final:

    args = configs[0]
    with open(args2name(args[0], args[1], args[2], args[3], args[4]) + "/out.json") as f:
        data = json.load(f)

    columns = list(data.keys())
    df = pd.DataFrame(0, index = datasets, columns = columns)
    df = df.astype("object")

    for args in configs:

        with open(args2name(args[0], args[1], args[2], args[3], args[4]) + "/out.json") as f:
            data = json.load(f)
        
        for name in columns:
            df.ix[args[0], name] += np.asarray(data[name]) / n_final

    df.to_csv("results.csv")
