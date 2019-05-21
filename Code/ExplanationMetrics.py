
from lime import lime_tabular #pip install lime
import numpy as np

from SLIM import SLIM as MAPLE

# Configure the Local Neighborhood
num_perturbations = 5

def generate_neighbor(x, stddev = 0.1):
    return x + stddev * np.random.normal(loc=0.0, scale=1.0, size = x.shape)

# Wrapper for TF models to make prediction easy
class Wrapper():
    def __init__(self, sess, pred, X):
        self.sess = sess
        self.pred = pred
        self.X = X
        self.index = 0

    def predict(self, x):
        return self.sess.run(self.pred, feed_dict = {self.X: x})

    def set_index(self, i):
        self.index = i

    def predict_index(self, x):
        return np.squeeze(self.sess.run(self.pred, feed_dict = {self.X: x})[:, self.index])

# Evaluate MAPLE as a black-box explainer for this model
def metrics_maple(model, X_train, X_val, X_test, stddev = 0.1):

    # Get the model predictions on data
    train_pred = model.predict(X_train)
    val_pred = model.predict(X_val)
    test_pred = model.predict(X_test)

    # Get the necessary sizes
    n_test = X_test.shape[0]
    d_in = X_test.shape[1]
    d_out = train_pred.shape[1]

    # Fit MAPLE to each dimension of the output
    exp = [None] * d_out
    for i in range(d_out):
        exp[i] = MAPLE(X_train, train_pred[:, i], X_val, val_pred[:, i])

    # Compute the standard, causal, and stability metrics
    standard_metric = np.zeros(d_out)
    causal_metric = np.zeros(d_out)
    stability_metric = np.zeros(d_out)
    for i in range(d_out):
        model.set_index(i)

        for j in range(n_test):

            x = X_test[j, :]

            # Get MAPLE's Explanation
            e = exp[i].explain(x)
            coefs = e["coefs"]

            # Standard Metric
            standard_metric[i] += (e["pred"][0] - test_pred[j,i])**2

            for k in range(num_perturbations):
                x_pert = generate_neighbor(x, stddev = stddev)

                # Causal Metric
                model_pred = model.predict_index(x_pert.reshape(1, d_in))
                maple_pred = np.dot(np.insert(x_pert, 0, 1), coefs)
                causal_metric[i] += (maple_pred - model_pred)**2

                # Stability Metric
                e_pert = exp[i].explain(x_pert)
                stability_metric[i] += np.sum((e_pert["coefs"] - coefs)**2)

    standard_metric /= n_test
    causal_metric /= num_perturbations * n_test
    stability_metric /= num_perturbations * n_test

    return standard_metric, causal_metric, stability_metric

# Evaluate LIME as a black-box explainer for this model
def metrics_lime(model, X_train, X_test, stddev = 0.1):

    # Get the model predictions on the test data
    test_pred = model.predict(X_test)

    # Get the necessary sizes
    n_test = X_test.shape[0]
    d_in = X_test.shape[1]
    d_out = test_pred.shape[1]

    # Configure LIME
    exp = lime_tabular.LimeTabularExplainer(X_train, discretize_continuous = False, mode = "regression")

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

    # Compute the standard, causal, and stability metrics
    standard_metric = np.zeros(d_out)
    causal_metric = np.zeros(d_out)
    stability_metric = np.zeros(d_out)
    for i in range(d_out):
        model.set_index(i)

        for j in range(n_test):
            x = X_test[j, :]

            # Get LIME's Explanation
            coefs = unpack_coefs(exp, x, model.predict_index, d_in, X_train)

            # Standard Metric
            standard_metric[i] += (np.dot(np.insert(x, 0, 1), coefs) - test_pred[j,i])**2

            for k in range(num_perturbations):
                x_pert = generate_neighbor(x, stddev = stddev)

                # Causal Metric
                model_pred = model.predict_index(x_pert.reshape(1, d_in))
                lime_pred = np.dot(np.insert(x_pert, 0, 1), coefs)
                causal_metric[i] += (lime_pred - model_pred)**2

                # Stability Metric
                coefs_pert = unpack_coefs(exp, x, model.predict_index, d_in, X_train)
                stability_metric[i] += np.sum((coefs_pert - coefs)**2)

    standard_metric /= n_test
    causal_metric /= num_perturbations * n_test
    stability_metric /= num_perturbations * n_test

    return standard_metric, causal_metric, stability_metric

'''
The observed values for the Support datasets for the previous two metrics seem unreasonable.
This function is intended to check the variance of a learned model across the neighborhood to see if the results make sense.

It also may help provide context to the other metrics
'''
def metrics_variance(model, X, stddev = 0.1):

    n_pert = 20

    n = X.shape[0]
    d = X.shape[1]

    n_out = model.predict(X[0, :].reshape((1,d))).shape[1]

    var = np.zeros((n_out))

    for i in range(n):
        x = X[i, :]

        x_pert = np.zeros((n_pert, d))
        for j in range(n_pert):
            x_pert[j, :] = generate_neighbor(x, stddev = stddev)

        pred_pert = model.predict(x_pert)

        var += np.var(pred_pert, axis = 0)

    var /= n

    return var
