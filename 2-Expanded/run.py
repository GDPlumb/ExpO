
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

def args2name(dataset, trial, depth, size, rate, reg):
    return "TF/" + dataset + "/" + str([size] * depth) + "/" + str(rate) + "/" + str(reg) + "/trial" + str(trial)

def name2args(name):
    chunks = name.split("/")
    dataset = chunks[1]
    shape = chunks[2]
    shape = json.loads(shape)
    depth = len(shape)
    size = shape[0]
    rate = np.float(chunks[3])
    reg = np.float(chunks[4])
    return dataset, depth, size, rate, reg

def run(args):
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
    source =  project_dir + "/Datasets/" + dataset  + ".csv"
    shape = [size] * depth
    out = eval(manager, source, shape, rate, regularizer = "Causal", c = reg)

    with open("out.json", "w") as f:
        json.dump(out, f)

    os.chdir(cwd)

# Flags to control what parts of the experiment run
run_search = True
process_search = True
run_final = True
process_final = True

n_search = 3
n_final = 10

# Initial Search Space
datasets = ["autompgs", "communities", "day", "happiness", "housing", "music", "winequality-red"]
trials = list(range(n_search))
depths = [1, 2,3,4,5]
sizes = [100, 150, 200, 250, 300]
rates = [0.0001, 0.001]
regs = [2.5e3, 5e3, 7.5e3, 1e4, 2.5e4, 5e4, 7.5e4]

if run_search:

    configs = itertools.product(datasets, trials, depths, sizes, rates, regs)

    p = Pool(5)
    p.map(run, configs)

if process_search:

    agg = {}
    config = {}
    for dataset in datasets:

        list_means = {}
        
        for depth in depths:
            for size in sizes:
                for rate in rates:
                    for reg in regs:

                        mean = 0.0

                        for trial in trials:
                            name = args2name(dataset, trial, depth, size, rate, reg)

                            with open(name + "/out.json") as f:
                                results = json.load(f)

                            mean += results["test_acc"]

                        mean /= len(trials)
                        name = args2name(dataset, "_avg", depth, size, rate, reg)
                        list_means[name] = mean
                        
        agg[dataset] = list_means

        sorted_means = sorted(list_means.items(), key = operator.itemgetter(1))
        
        # Get the configuration with the best average performance
        args = name2args(sorted_means[0][0])

        config[args[0]] = [args[1], args[2], args[3], args[4]]

    with open("search.json", "w") as f:
        json.dump(agg, f)

    with open("config.json", "w") as f:
        json.dump(config, f)

    os.rename("TF", "TF-initial")

if run_final or process_final:
    with open("config.json") as f:
        config = json.load(f)
    configs = []
    for dataset in datasets:
        c = config[dataset]
        for trial in range(n_final):
            args = [dataset, trial, c[0], c[1], c[2], c[3]]
            configs.append(args)

if run_final:
    p = Pool(5)
    p.map(run, configs)

if process_final:

    args = configs[0]
    with open(args2name(args[0], args[1], args[2], args[3], args[4], args[5]) + "/out.json") as f:
        data = json.load(f)

    columns = list(data.keys())
    df = pd.DataFrame(0, index = datasets, columns = columns)
    df = df.astype("object")

    for args in configs:

        with open(args2name(args[0], args[1], args[2], args[3], args[4], args[5]) + "/out.json") as f:
            data = json.load(f)
        
        for name in columns:
            df.ix[args[0], name] += np.asarray(data[name]) / n_final

    df.to_csv("results.csv")

