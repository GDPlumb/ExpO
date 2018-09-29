
import numpy as np
import tensorflow as tf

import sys
sys.path.insert(0, "../../Code/")
from load import load_normalize_data
from eval_test import eval

dataset = "autompgs"

X_train, y_train, X_valid, y_valid, X_test, y_test, mu, sigma = load_normalize_data("../../Datasets/" + dataset  + ".csv")

out = eval(X_train, y_train, X_valid, y_valid, X_test, y_test, regularizer = "Causal")

