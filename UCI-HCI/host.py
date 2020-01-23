
seed = 0

from flask import Flask, request, send_from_directory
from flask_cors import CORS
import json
from lime import lime_tabular
import numpy as np
import os
import pandas as pd
import sys
import tensorflow as tf

sys.path.insert(0, os.path.join(os.getcwd(), "../Code/"))
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

app = Flask(__name__)
CORS(app)


# Path for our main Svelte page
@app.route("/")
def base():
    return send_from_directory('expo-experiment/public', 'index.html')
# Path for all the static files (compiled JS/CSS, etc.)
@app.route("/<path:path>")
def home(path):
    return send_from_directory('expo-experiment/public', path)

'''
@app.route('/')
def home_endpoint():
    return 'Host is currently running'
'''
    
@app.route('/point',  methods=['POST'])
def get_point():
    if request.method == 'POST':
        global index
        point = data[index, :]
        index = (index + 1) % n_obs
        return json.dumps(point.tolist())
        
@app.route('/predict', methods=['POST'])
def get_prediction():
    if request.method == 'POST':
        input = request.get_json()  # Get data posted as a json
        x = np.array(input["x"])
        # Load the model
        saver.restore(sess, "./" + input["mode"] + "/model.cpkt")
        out = {}
        value = sess.run(pred, {X: np.expand_dims(x, 0)})[0][0]
        out["prediction"] = value.tolist()
        coefs = unpack_coefs(exp, x, wrapper.predict_index, n_input, data)
        out["explanation"] = coefs.tolist()
        return json.dumps(out)

if __name__ == '__main__':
    np.random.seed(seed)

    # Load the full dataset
    df = pd.read_csv("../Datasets/housing.csv", header = None).dropna()
    
    stddev = df[df.columns[:]].std()
    drop_small = np.where(stddev < .0000000001)
    if stddev[df.shape[1] - 1] < .0000000001:
        print("ERROR: Near constant predicted value")
    df = df.drop(drop_small[0], axis = 1)

    stddev = df[df.columns[:]].std()
    mean = df[df.columns[:]].mean()
    
    df = (df - mean) / stddev
    
    data = df[df.columns[:-1]].values
    np.random.shuffle(data)
    #y = np.expand_dims(df[df.columns[-1]].values, 1)

    index = 0
    n_obs, n_input = data.shape

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
    exp = lime_tabular.LimeTabularExplainer(data, discretize_continuous = False, mode = "regression")
    
    #app.run(host="0.0.0.0", port=5000, ssl_context=('cert.pem', 'key.pem'))
    app.run(host="0.0.0.0", port=5000)
