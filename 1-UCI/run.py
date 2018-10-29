
project_dir = "/home/gregory/Desktop/Regularization"

import itertools
import json
from multiprocessing import Pool
import os
import sys
sys.path.insert(0, project_dir + "/Code/")
from eval import eval

# The networks are small enough that training is faster on CPU
os.environ["CUDA_VISIBLE_DEVICES"]="-1"

# Define the experimental parameter search space
datasets = ["autompgs", "communities", "day", "happiness", "housing", "music", "winequality-red"]

trials = list(range(3))

hidden_layers = [2,3,4]

layer_size = [50, 100, 150, 200]

learning_rate = [0.0001, 0.001]

args = itertools.product(datasets, trials, hidden_layers, layer_size, learning_rate)

###
# Run Experiments
###

def run(args):
    dataset = args[0]
    trial = args[1]
    num_layers = args[2]
    layer_size = args[3]
    learning_rate = args[4]
    
    manager = "regression"
    source =  project_dir + "/Datasets/" + dataset  + ".csv"
    
    shape = [layer_size] * num_layers
    
    cwd = os.getcwd()
    
    name = "TF/" + dataset + "/" + str(shape) + "/" + str(learning_rate) + "/trial" + str(trial)
    os.makedirs(name)
    os.chdir(name)
    
    with open("config.json", "w") as outfile:
        json.dump(args, outfile)

    out = eval(manager, source, shape, learning_rate)

    with open("out.json", "w") as outfile:
        json.dump(out, outfile)

    os.chdir(cwd)

p = Pool(5)
p.map(run, args)
