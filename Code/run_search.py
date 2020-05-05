
from collections import OrderedDict
import itertools
import json
import numpy as np
import operator
import os
import pandas as pd

from multiprocessing import Pool

def args2name(dataset, trial, depth, size, rate, reg=None):
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

def run_search(run_fn_search = None, n_search = 1, lower_is_better = True, gamma = 0.1,
                run_search = True, process_search = True,
                run_fn_final = None,  n_final = 1,
                run_final = True, process_final = True,
                datasets = None, depths = None, sizes = None, rates = None, source = None,
                regularized = False, regs = None,
                num_processes = 1):
    
    # gamma is the discount factor applied to slightly encourage more regularized models
    if lower_is_better:
        gamma *= -1.0

    if run_search or process_search:
        trials = list(range(n_search))
    
        if source is not None:
            with open(source, "r") as f:
                config_list = json.load(f)

    if run_search:

        if source is not None:
            configs = []
            for dataset in datasets:
                c = config_list[dataset]
                for reg in regs:
                    for trial in trials:
                        args = [dataset, trial, c[0], c[1], c[2], reg]
                        configs.append(args)
        elif regularized:
            configs = itertools.product(datasets, trials, depths, sizes, rates, regs)
        else:
            configs = itertools.product(datasets, trials, depths, sizes, rates)

        p = Pool(num_processes)
        p.map(run_fn_search, configs)
        p.close()
        p.join()

    if process_search:

        agg = OrderedDict()
        config = OrderedDict()
        for dataset in datasets:

            list_means = {}

            if source is not None:
                c = config_list[dataset]

                for reg in regs:
                    mean = 0.0
                    for trial in trials:
                        name = args2name(dataset, trial, c[0], c[1], c[2], reg)
                        with open(name + "/out.json") as f:
                            results = json.load(f)
                        mean += results["test_acc"]
                    mean /= len(trials)
                    name = args2name(dataset, "_avg", c[0], c[1], c[2], reg)
                    list_means[name] = mean + gamma * reg
            else:

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
                                    list_means[name] = mean + gamma * reg

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

            if lower_is_better:
                sorted_means = sorted(list_means.items(), key = operator.itemgetter(1))
            else:
                sorted_means = sorted(list_means.items(), key = operator.itemgetter(1), reverse = True)
                
            agg[dataset] = sorted_means

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
        p.map(run_fn_final, configs)

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

        df.to_csv("results_mean.csv")

        df_sd = pd.DataFrame(0, index = datasets, columns = columns)
        df_sd = df_sd.astype("object")

        for args in configs:

            if regularized:
                with open(args2name(args[0], args[1], args[2], args[3], args[4], args[5]) + "/out.json") as f:
                    data = json.load(f)
            else:
                with open(args2name(args[0], args[1], args[2], args[3], args[4]) + "/out.json") as f:
                    data = json.load(f)

            for name in columns:
                delta = np.asarray(data[name]) - df.ix[args[0], name]
                df_sd.ix[args[0], name] += delta**2 / (n_final - 1)
                
        for dataset in datasets:
            for column in columns:
                df_sd.ix[dataset, column] = np.sqrt(df_sd.ix[dataset, column])

        df_sd.to_csv("results_sd.csv")
