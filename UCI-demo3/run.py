
import csv
import json
from lime import lime_tabular #pip install lime
import numpy as np
import os
import sys
import tensorflow as tf

sys.path.insert(0, os.path.join(os.getcwd(), "../Code/"))
from Data import DataManager
from ExplanationMetrics import Wrapper, generate_neighbor
from Models import MLP

# The networks are small enough that training is faster on CPU
os.environ["CUDA_VISIBLE_DEVICES"]="-1"

# BUG: 1st feature is a dataset index for the UCI datasets
# It doesn't appear to be used by the model, so for now we are simply ignoring it
# As a result, anything indexing based on 'n_input' has been messed with


name = "housing"

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


# Load the data
data = DataManager("../Datasets/" + name + ".csv")

X_train = data.X_train
n_input = X_train.shape[1]

# Load the network shape

with open("../UCI-None/config.json", "r") as f:
    config_list = json.load(f)

config = config_list[name]

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


# Run the evaluation
t = 5

out = np.zeros((t * 3, n_input))

for example in range(t):
    
    x = X_train[example]
    out[example * 3, :n_input-1] = np.round(x[1:], 2)
    

    # Evaluate LIME on the unregularized model (copy of UCI-None/TF/housing/..../trial0)

    with tf.Session() as sess:
        # Restore model
        saver.restore(sess, "./None/model.cpkt")

        # Wrap it for LIME
        wrapper = Wrapper(sess, pred, X)
        wrapper.set_index(0)

        # Configure LIME
        exp = lime_tabular.LimeTabularExplainer(X_train, discretize_continuous = False, mode = "regression")

        # Get the explanation for the chosen point
        coefs = unpack_coefs(exp, x, wrapper.predict_index, n_input, X_train)
        
        out[example * 3 + 1, :n_input-1] =  np.round(coefs, 2)[2:] #First value is the intercept term
        # Evaluate the fidelity metric
        fidelity = 0.0
        for i in range(10):
            x_pert = generate_neighbor(x)

            model_pred = wrapper.predict_index(x_pert.reshape(1, n_input))
            lime_pred = np.dot(np.insert(x_pert, 0, 1), coefs)
            fidelity += (lime_pred - model_pred)**2
        fidelity /= 10

        out[example * 3 + 1, n_input-1] =  fidelity
    
    # Load the ExpO regularized model (copy of UCI-LF1D/TF/housing/..../trial0)
    with tf.Session() as sess:
        saver.restore(sess, "./ExpO/model.cpkt")

        # Wrap it for LIME
        wrapper = Wrapper(sess, pred, X)
        wrapper.set_index(0)

        # Configure LIME
        exp = lime_tabular.LimeTabularExplainer(X_train, discretize_continuous = False, mode = "regression")

        # Get the explanation for the chosen point
        coefs = unpack_coefs(exp, x, wrapper.predict_index, n_input, X_train)
        
        out[example * 3 + 2, :n_input-1] =  np.round(coefs, 2)[2:] #First value is the intercept term

        # Evaluate the fidelity metric
        fidelity = 0.0
        for i in range(10):
            x_pert = generate_neighbor(x)

            model_pred = wrapper.predict_index(x_pert.reshape(1, n_input))
            lime_pred = np.dot(np.insert(x_pert, 0, 1), coefs)
            fidelity += (lime_pred - model_pred)**2
        fidelity /= 10

        out[example * 3 + 2, n_input-1] =  fidelity

with open("out.csv", mode='w') as f:
    writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    writer.writerows(out)
