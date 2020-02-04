
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
import os
from sklearn.model_selection import train_test_split
import sys
import tensorflow as tf

from data import f

sys.path.insert(0, os.path.join(os.getcwd(), "../../Code/"))
from Data import DataManager
from Models import MLP
from Regularizers import Regularizer

from SLIM import SLIM as MAPLE

class ToyDataManager(DataManager):


    def load_normalize_data(self):
        df_train = self.load_data()
            
        # Split train, test, valid - Change up train valid test every iteration
        df_train, df_test = train_test_split(df_train, test_size = 0.5)
        df_valid, df_test = train_test_split(df_test, test_size = 0.5)
        
        # Convert to np arrays
        X_train = df_train[df_train.columns[:-1]].values
        y_train = np.expand_dims(df_train[df_train.columns[-1]].values, 1)
        
        X_valid = df_valid[df_valid.columns[:-1]].values
        y_valid = np.expand_dims(df_valid[df_valid.columns[-1]].values, 1)
        
        X_test = df_test[df_test.columns[:-1]].values
        y_test = np.expand_dims(df_test[df_test.columns[-1]].values, 1)
        
        return X_train, y_train, X_valid, y_valid, X_test, y_test, 0,0

def eval(source,
         # Network parameters
         hidden_layer_sizes = [100,100,100], learning_rate = 0.001,
         # Regularizer parameters
         regularizer = None, c = 1.0,
         # Training parameters
         batch_size = 2, reg_batch_size = 1, stopping_epochs = 200, min_epochs = 500, stop_on_loss = True,
         evaluate_explanation = True):

    # Allow multiple sessions on a single GPU.
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True

    # Reset TF graph (avoids issues with repeat experiments)
    tf.reset_default_graph()

    # Get Data
    if regularizer is not None:
        data = ToyDataManager(source,
                           train_batch_size = batch_size,
                           reg_batch_size = reg_batch_size)
    else:
        data = ToyDataManager(source,
                           train_batch_size = batch_size)

    n = data.X_train.shape[0]
    n_input = data.X_train.shape[1]
    n_out = data.y_train.shape[1]

    # Regularizer Parameters
    if regularizer is not None:
        # Weight of the regularization term in the loss function
        c = tf.constant(c)
        # Number of neighbors to hallucinate per point
        num_samples = np.max((20, np.int(2 * n_input)))

    # Network Parameters
    shape = [n_input]
    for size in hidden_layer_sizes:
        shape.append(size)
    shape.append(n_out)

    # Graph inputs
    X = tf.placeholder("float", [None, n_input], name = "X_in")
    Y = tf.placeholder("float", [None, n_out], name = "Y_in")
    if regularizer is not None:
        X_reg = tf.placeholder("float", [None, n_input], name="X_reg")

    # Link the graph inputs into the DataManager so its train_feed() and eval_feed() functions work
    if regularizer is not None:
        data.link_graph(X, Y, X_reg = X_reg)
    else:
        data.link_graph(X, Y)

    # Build the model
    network = MLP(shape)
    with tf.variable_scope("model", reuse = tf.AUTO_REUSE):
        pred = network.model(X)

    # Build the regularizer
    if regularizer == "Causal":
        regularizer = Regularizer(network.model, n_input, num_samples)
        reg = regularizer.causal(X_reg)

    # Define the loss and optimization process
    model_loss = tf.losses.mean_squared_error(labels = Y, predictions = pred)
    tf.summary.scalar("MSE", model_loss)

    perf_op = model_loss
    smaller_is_better = True


    if regularizer is None:
        loss_op = model_loss
    else:
        loss_op = model_loss + c * reg
        tf.summary.scalar("Regularizer", reg)
    tf.summary.scalar("Loss", loss_op)

    summary_op = tf.summary.merge_all()

    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
    train_op = optimizer.minimize(loss_op)

    # Train the model
    init = [tf.global_variables_initializer(), tf.local_variables_initializer()]

    saver = tf.train.Saver(max_to_keep = 1)

    if stop_on_loss: # Stop on validation loss function
        best_loss = np.inf
    else: # Stop on the validation performance
        if smaller_is_better:
            best_perf = np.inf
        else:
            best_perf = 0.0

    best_epoch = 0

    with tf.Session(config = tf_config) as sess:
        train_writer = tf.summary.FileWriter("train", sess.graph)
        val_writer = tf.summary.FileWriter("val")

        sess.run(init)

        epoch = 0
        while True:

            # Stopping condition
            if epoch - best_epoch > stopping_epochs and epoch > min_epochs:
                break

            # Run a training epoch
            total_batch = int(n / batch_size)
            for i in range(total_batch):
                dict = data.train_feed()
                sess.run(train_op, feed_dict = dict)

            # Run model metrics
            if epoch % 10 == 0:
                dict = data.eval_feed()
                summary = sess.run(summary_op, feed_dict = dict)
                train_writer.add_summary(summary, epoch)

                dict = data.eval_feed(val = True)
                
                if stop_on_loss:
                    summary, val_loss = sess.run([summary_op, loss_op], feed_dict = dict)
                    
                    if val_loss < best_loss:
                        print(os.getcwd(), " ", epoch, " ", val_loss)
                        best_loss = val_loss
                        best_epoch = epoch
                        saver.save(sess, "./model.cpkt")
                
                else:
                    summary, val_perf = sess.run([summary_op, perf_op], feed_dict = dict)
                    
                    if smaller_is_better and val_perf < best_perf:
                        progress = True
                    elif not smaller_is_better and val_perf > best_perf:
                        progress = True
                    else:
                        progress = False

                    if progress:
                        print(os.getcwd(), " ", epoch, " ", val_perf)
                        best_perf = val_perf
                        best_epoch = epoch
                        saver.save(sess, "./model.cpkt")
        
                val_writer.add_summary(summary, epoch)

            epoch += 1

        train_writer.close()
        val_writer.close()

        ###
        # Evaluate the chosen model
        ###

        saver.restore(sess, "./model.cpkt")

        pred_grid = sess.run(pred, {X: x_grid})
        plt.plot(x_grid, pred_grid)

        ###
        # Make explanations
        ####

        def predict_fn(x):
            return np.squeeze(sess.run(pred, {X: x}))
                
        exp = MAPLE(data.X_train, predict_fn(data.X_train), data.X_val, predict_fn(data.X_val))
        
        x = np.zeros((1,1))
        x[0] = 0.1
        e = exp.explain(x)
        c1 = e["coefs"]
        
        x[0] = 0.4
        e = exp.explain(x)
        c2 = e["coefs"]
        
        return c1, c2


# Plot the target function
x_grid = np.expand_dims(np.array(range(200))/200, axis = 1)
plt.plot(x_grid, f(x_grid))

# Plot the training set
np.random.seed(0)
data = ToyDataManager("data.csv", train_batch_size = None, reg_batch_size = None)
plt.plot(data.X_train, data.y_train, 'bo')

plt.legend(['True Function', 'Training Data'])

plt.savefig("Functions-1.pdf")

# Fit and plot the unregularized model
np.random.seed(0)
c1, c2 = eval("data.csv")

y_1_unreg = []
y_2_unreg = []
for x in x_grid:
    y_1_unreg.append(np.dot(np.insert(x, 0, 1), c1))
    y_2_unreg.append(np.dot(np.insert(x, 0, 1), c2))
 
plt.legend(['True Function', 'Training Data', 'Unregularized Model'])
 
plt.savefig("Functions-2.pdf")

# Fit and plot a regularized model
np.random.seed(0)
c1, c2 = eval("data.csv", regularizer = "Causal", c = 5.0)

y_1_reg = []
y_2_reg = []
for x in x_grid:
    y_1_reg.append(np.dot(np.insert(x, 0, 1), c1))
    y_2_reg.append(np.dot(np.insert(x, 0, 1), c2))
    
plt.legend(['True Function', 'Training Data', 'Unregularized Model', 'Regularized Model'])
    
plt.savefig("Functions-3.pdf")

# Fit and plot an overregularized model
np.random.seed(0)
c1, c2 = eval("data.csv", regularizer = "Causal", c = 1000.0)

plt.legend(['True Function', 'Training Data', 'Unregularized Model', 'Regularized Model', 'Over-regularized Model'])

# Save plots
plt.savefig("Functions-4.pdf")
plt.close()

# Get explanations at points
plt.plot(x_grid, f(x_grid))

plt.plot(x_grid, y_1_unreg)
plt.plot(x_grid, y_2_unreg)
plt.plot(x_grid, y_1_reg)
plt.plot(x_grid, y_2_reg)
plt.legend(['True Function','Unregularized Explanation at x = 0.1', 'Regularized Explanation at x = 0.1','Unregularized Explanation at x = 0.4', 'Regularized Explanation at x = 0.4'])
plt.savefig("Explanations.pdf")
plt.close()


