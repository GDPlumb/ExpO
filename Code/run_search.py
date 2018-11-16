
import itertools
import json
from multiprocessing import Pool
import numpy as np
import operator
import os
import pandas as pd

def args2name(dataset, trial, depth, size, rate, reg = None):
    if reg == None:
        return "TF/" + dataset + "/" + str([size] * depth) + "/" + str(rate) + "/trial" + str(trial)
    else:
        return "TF/" + dataset + "/" + str([size] * depth) + "/" + str(rate) + "/" + str(reg) + "/trial" + str(trial)

def name2args(name):
    chunks = name.split("/")
    dataset = chunks[1]
    shape = chunks[2]
    shape = json.loads(shape)
    depth = len(shape)
    size = shape[0]
    rate = np.float(chunks[3])
    if len(chunks) > 5:
        reg = np.float(chunks[4])
        return dataset, depth, size, rate, reg
    else:
        return dataset, depth, size, rate

def run_search(run_fn = None, num_processes = 5,
                    run_search = True, process_search = True, run_final = True, process_final = True,
                    n_search = 1, n_final = 1,
                    datasets = None, depths = None, sizes = None, rates = None,
                    regularized = False, regs = None):

    if run_search or process_search:
        trials = list(range(n_search))

    if run_search:

        if regularized:
            configs = itertools.product(datasets, trials, depths, sizes, rates, regs)
        else:
            configs = itertools.product(datasets, trials, depths, sizes, rates)

        p = Pool(num_processes)
        p.map(run_fn, configs)

    if process_search:

        agg = {}
        config = {}
        for dataset in datasets:

            list_means = {}

            for depth in depths:
                for size in sizes:
                    for rate in rates:

                        if regularized:

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

                        else:

                            mean = 0.0
                            for trial in trials:
                                name = args2name(dataset, trial, depth, size, rate)
                                with open(name + "/out.json") as f:
                                    results = json.load(f)
                                mean += results["test_acc"]
                            mean /= len(trials)
                            name = args2name(dataset, "_avg", depth, size, rate)
                            list_means[name] = mean

            agg[dataset] = list_means

            sorted_means = sorted(list_means.items(), key = operator.itemgetter(1))

            # Get the configuration with the best average performance
            args = name2args(sorted_means[0][0])

            if regularized:
                config[args[0]] = [args[1], args[2], args[3], args[4]]
            else:
                config[args[0]] = [args[1], args[2], args[3]]

        with open("search.json", "w") as f:
            json.dump(agg, f)

        with open("config.json", "w") as f:
            json.dump(config, f)

        os.rename("TF", "TF-initial")

    if run_final or process_final:

        with open("config.json") as f:
            config_list = json.load(f)

        configs = []
        for dataset in datasets:
            c = config_list[dataset]

            for trial in range(n_final):
                if regularized:
                    args = [dataset, trial, c[0], c[1], c[2], c[3]]
                else:
                    args = [dataset, trial, c[0], c[1], c[2]]
                configs.append(args)

    if run_final:
        p = Pool(num_processes)
        p.map(run_fn, configs)

    if process_final:

        args = configs[0]
        if regularized:
            with open(args2name(args[0], args[1], args[2], args[3], args[4], args[5]) + "/out.json") as f:
                data = json.load(f)
        else:
            with open(args2name(args[0], args[1], args[2], args[3], args[4]) + "/out.json") as f:
                data = json.load(f)

        columns = list(data.keys())
        df = pd.DataFrame(0, index = datasets, columns = columns)
        df = df.astype("object")

        for args in configs:

            if regularized:
                with open(args2name(args[0], args[1], args[2], args[3], args[4], args[5]) + "/out.json") as f:
                    data = json.load(f)
            else:
                with open(args2name(args[0], args[1], args[2], args[3], args[4]) + "/out.json") as f:
                    data = json.load(f)

            for name in columns:
                df.ix[args[0], name] += np.asarray(data[name]) / n_final

        df.to_csv("results.csv")
