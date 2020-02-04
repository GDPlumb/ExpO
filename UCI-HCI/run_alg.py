
seed = 32
t = 2000
goal_change = 1.0
goal_range = 0.1

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
    
def mod(x, i, d):
    x_new = np.copy(x)
    x_new[i] += d
    return x_new
    
# Load the data
np.random.seed(seed) # Get the same data everytime
data = DataManager("../Datasets/housing.csv")

X_train = data.X_train
n_input = X_train.shape[1]
n_samples = X_train.shape[0]

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

options = [0.0, 0.5, 1.0, 2.0, 5.0, 10.0, 15.0, 20.0]
out = np.zeros((len(options), 3))
config = 0
for agent_scale in options:

    results = np.zeros((t, 2))
    
    for i in range(t):

        x = X_train[i % n_samples]
        
        # Randomly choose a model
        if np.random.uniform() < 0.5:
            results[i, 0] = 0.0
            saver.restore(sess, "./None/model.cpkt")
        else:
            results[i, 0] = 1.0
            saver.restore(sess, "./ExpO/model.cpkt")
            
        original_prediction = sess.run(pred, {X: np.expand_dims(x, 0)})
        
        target = original_prediction + goal_change
        
        c = 0
        while True:
        
            coefs = unpack_coefs(exp, x, wrapper.predict_index, n_input, X_train)
            coefs = coefs[1:] #Ignore the intercept term of the explanation
                
            value = sess.run(pred, {X: np.expand_dims(x, 0)})
                    
            delta = (value - target)[0][0]
                    
            if np.abs(delta) < goal_range:
                results[i, 1] = c
                break
                
            # Greedy agent - gets stuck in loops
            # idx = (np.abs(np.abs(coefs) - np.abs(delta))).argmin()
            
            # Random agent from scaled softmax
            scores = -1.0 * agent_scale * np.abs(np.abs(coefs) - np.abs(delta))
            dist = np.exp(scores) / sum(np.exp(scores))
            idx = np.random.choice(dist.shape[0], 1, p = dist)
            
            if delta < 0:
                if coefs[idx] < 0:
                    d = -1.0
                else:
                    d = 1.0
            else:
                if coefs[idx] < 0:
                    d = 1.0
                else:
                    d = -1.0
            
            x = mod(x, idx, d)
            c += 1
            
    none = np.where(results[:, 0] == 0)[0]
    none_vals = results[none, 1]

    expo = np.where(results[:, 0] == 1)[0]
    expo_vals = results[expo, 1]

    out[config, 0] = agent_scale
    out[config, 1] = np.mean(none_vals)
    out[config, 2] = np.mean(expo_vals)
    config += 1
    
np.savetxt("agent.csv", out, delimiter=",")
