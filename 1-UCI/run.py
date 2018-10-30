
# Flag to switch between running the hyperparameter search and the final chosen architectures
final = True
# Flag to swtich between displaying results and running the experiments
execute = True

project_dir = "/home/gregory/Desktop/Regularization"

import itertools
import json
from multiprocessing import Pool
import operator
import os
import sys
sys.path.insert(0, project_dir + "/Code/")
from eval import eval

# The networks are small enough that training is faster on CPU
os.environ["CUDA_VISIBLE_DEVICES"]="-1"

def args2name(dataset, shape, rate, trial):
    return "TF/" + dataset + "/" + str(shape) + "/" + str(rate) + "/trial" + str(trial)

def run(args):
    dataset = args[0]
    trial = args[1]
    depth = args[2]
    size = args[3]
    rate = args[4]
    
    manager = "regression"
    source =  project_dir + "/Datasets/" + dataset  + ".csv"
    
    shape = [size] * depth
    
    name = args2name(dataset, shape, rate, trial)
    
    cwd = os.getcwd()
    
    os.makedirs(name)
    os.chdir(name)
    
    with open("config.json", "w") as f:
        json.dump(args, f)

    out = eval(manager, source, shape, rate)

    with open("out.json", "w") as f:
        json.dump(out, f)

    os.chdir(cwd)

datasets = ["autompgs", "communities", "day", "happiness", "housing", "music", "winequality-red"]

if not final:
    trials = list(range(3))
    depths = [2,3,4]
    # This should probably be [150, 200, 250, 300] because almost all datasets chose a size of 200
    sizes = [50, 100, 150, 200]
    rates = [0.0001, 0.001]
    list = itertools.product(datasets, trials, depths, sizes, rates)

    p = Pool(5)
    if execute:
        p.map(run, list)

    for dataset in datasets:

        list_trials = {}
        list_means = {}

        for depth in depths:
            for size in sizes:
                shape = [size] * depth
                for rate in rates:

                    mean = 0.0

                    for trial in trials:
                        name = args2name(dataset, shape, rate, trial)

                        with open(name + "/out.json") as f:
                            results = json.load(f)

                        acc = results["test_acc"]
                        list_trials[name] = acc
                        mean += acc

                    mean /= len(trials)
                    name = args2name(dataset, shape, rate, "_avg")
                    list_means[name] = mean

        print("")
        print(dataset)
        print("")
        print("10 Best Individual Trials")
        sorted_trials = sorted(list_trials.items(), key = operator.itemgetter(1))
        for i in range(10):
            print(sorted_trials[i])
        print("")
        print("5 Best Average Configurations")
        sorted_means = sorted(list_means.items(), key = operator.itemgetter(1))
        for i in range(5):
            print(sorted_means[i])
        print("")

else:
    config = {}
    config["autompgs"] = [2, 200, 0.0001]
    config["communities"] = [3, 200, 0.001]
    config["day"] = [3, 200, 0.001] #Linear dataset, the difference between all of these was small
    config["happiness"] = [2, 200, 0.001] #Same as above
    config["housing"] = [3, 150, 0.001]
    config["music"] = [4, 200, 0.001]
    config["winequality-red"] = [2, 200, 0.0001]

    list = []
    for dataset in datasets:
        for trial in range(10):
            c = config[dataset]
            args = [dataset, trial, c[0], c[1], c[2]]
            list.append(args)

    p = Pool(5)
    if execute:
        p.map(run, list)




