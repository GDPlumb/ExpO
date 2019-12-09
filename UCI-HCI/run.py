
t = 20
seed = 5

import csv
import json
from lime import lime_tabular
import numpy as np
import os
import sys
import tensorflow as tf

sys.path.insert(0, os.path.join(os.getcwd(), "../Code/"))
from Data import DataManager
from ExplanationMetrics import Wrapper
from Models import MLP

# The networks are small enough that training is faster on CPU
os.environ["CUDA_VISIBLE_DEVICES"]="-1"

# BUG: 1st feature is a dataset index for the UCI datasets
# It doesn't appear to be used by the model, so for now we are simply ignoring it
# As a result, anything indexing based on 'n_input' has been messed with

def unpack_coefs(explainer, x, predict_fn, num_features, x_train, num_samples = 1000):
    d = x_train.shape[1]
    coefs = np.zeros((d))

    u = np.mean(x_train, axis = 0)
    sd = np.sqrt(np.var(x_train, axis = 0))

    exp = explainer.explain_instance(x, predict_fn, num_features = num_features, num_samples = num_samples)

    coef_pairs = exp.local_exp[1]
    for pair in coef_pairs:
        coefs[pair[0]] = pair[1]

    coefs = coefs / sd

    intercept = exp.intercept[1] - np.sum(coefs * u)

    return np.insert(coefs, 0, intercept)

def iterate(x):
    print("")
    print("Predicted value: ")
    print(np.round(sess.run(pred, {X: np.expand_dims(x, 0)}), 1)[0][0])
    coefs = unpack_coefs(exp, x, wrapper.predict_index, n_input, X_train)
    coefs = np.round(coefs, 1)
    print("Explanation: ")
    for i in range(coefs.shape[0]):
        print(i - 1, coefs[i])
    print("")
    
def mod(x, i, d):
    x_new = np.copy(x)
    x_new[i] += d
    return x_new
    
# Load the data
np.random.seed(seed) # Get the same data everytime
data = DataManager("../Datasets/housing.csv")

X_train = data.X_train
n_input = X_train.shape[1]

# Load the network shape
with open("../UCI-None/config.json", "r") as f:
    config_list = json.load(f)

config = config_list["housing"]

shape = [n_input]
for i in range(config[0]):
    shape.append(config[1])
shape.append(1)

# Create the model
tf.reset_default_graph()

X = tf.placeholder("float", [None, n_input], name = "X_in")

network = MLP(shape)
with tf.variable_scope("model", reuse = tf.AUTO_REUSE):
    pred = network.model(X)

saver = tf.train.Saver(max_to_keep = 1)

# Create tf session
sess = tf.Session()

# Wrap it for LIME
wrapper = Wrapper(sess, pred, X)
wrapper.set_index(0)

# Configure LIME
exp = lime_tabular.LimeTabularExplainer(X_train, discretize_continuous = False, mode = "regression")

###
# Run Experiment
###

print("")
print("What is the 'Explanation'?")
print("The left column is the feature index")
print("The right column is the expected change of the model's prediction if we increase that feature by 1.")
print("The index of -1 is the intercept term for the explanation and should not be changed.")
print("")
print("Suppose that we see the Explanation:")
print("-1 0.0")
print("0 0.3")
print("1 -0.1")
print("")
print("If you enter '0 +', we would expect the model's prediction to increase by 0.3")
print("If you enter '1 +', we would expect the model's prediction to decrease by 0.1")
print("If you enter '1 -', we would expect the model's prediction to increase by 0.1")
print("")
print("The goal is to increase the model's precition by 0.5 with a tolerance of plus or minus 0.1")
print("")

results = np.zeros((t, 2))

for i in range(t):

    x = X_train[i]
    
    # Randomly choose a model
    if np.random.uniform() < 0.5:
        results[i, 0] = 0.0
        saver.restore(sess, "./None/model.cpkt")
    else:
        results[i, 0] = 1.0
        saver.restore(sess, "./ExpO/model.cpkt")
        
    target = sess.run(pred, {X: np.expand_dims(x, 0)})
    
    c = 0
    while True:
    
        iterate(x)
        
        value = sess.run(pred, {X: np.expand_dims(x, 0)})
        delta = value - target
        print("Current difference: ", np.round(delta[0][0], 2))
        if delta < 0.6 and delta > 0.4:
            results[i, 1] = c
            print("")
            print("Success!")
            print("")
            break
        
        io = input("index direction: ")
        io = str.split(io, " ")
        index = int(io[0])
        if io[1] == "+":
            dir = 1.0
        elif io[1] == "-":
            dir = -1.0
        else:
            print("Bad direction")
            sys.exit(0)
            
        x = mod(x, index, dir)
        c += 1
        
np.savetxt("results.csv", results, delimiter=",")
