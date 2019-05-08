
import numpy as np
import os
import tensorflow as tf

from ExplanationMetrics import Wrapper, metrics_maple, metrics_lime, metrics_variance
from Models import MLP
from Regularizers import Regularizer, Regularizer_1D

def eval(manager, source,
         # Network parameters
         hidden_layer_sizes = [32], learning_rate = 0.0001,
         # Regularizer parameters
         regularizer = None, c = 1.0,
         # Training parameters
         batch_size = 128, reg_batch_size = 16, stopping_epochs = 100, min_epochs = 500, stop_on_loss = False,
         # Explanation evaluation metrics
         evaluate_explanation = True):

    # Allow multiple sessions on a single GPU.
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True

    # Reset TF graph (avoids issues with repeat experiments)
    tf.reset_default_graph()

    # Load the Data Manager
    if manager == "regression":
        from Data import DataManager
    elif manager == "binary_classification":
        from BinaryData import BinaryClassificationDataManager as DataManager
    elif manager == "hospital_readmission":
        from MedicalData import HospitalReadmissionDataManager as DataManager
    elif manager == "support2":
        from MedicalData import Support2DataManager as DataManager
    else:
        raise ValueError("Unknown manager: %s" % manager)

    # Get Data
    if regularizer is not None:
        data = DataManager(source,
                           train_batch_size = batch_size,
                           reg_batch_size = reg_batch_size)
    else:
        data = DataManager(source,
                           train_batch_size = batch_size)

    n = data.X_train.shape[0]
    n_input = data.X_train.shape[1]
    n_out = data.y_train.shape[1]

    # Regularizer Parameters
    if regularizer == "Causal":
        # Weight of the regularization term in the loss function
        c = tf.constant(c)
        # Number of neighbors to hallucinate per point
        num_samples = np.max((20, np.int(2 * n_input)))
    if regularizer == "Causal1D":
        # Weight of the regularization term in the loss function
        c = tf.constant(c)
        # Number of neighbors to hallucinate per point
        num_samples = 10


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
    elif regularizer == "Causal1D":
        regularizer = Regularizer_1D(network.model, n_input, num_samples)
        reg = regularizer.causal(X_reg)

    # Define the loss and optimization process
    if manager == "regression":
    
        model_loss = tf.losses.mean_squared_error(labels = Y, predictions = pred)
        tf.summary.scalar("MSE", model_loss)
        
        perf_op = model_loss
        smaller_is_better = True

    elif manager == "binary_classification":
    
        model_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels = Y, logits = pred))
        tf.summary.scalar("Cross-entropy:", model_loss)
        
        _, perf_op = tf.metrics.accuracy(labels = Y, predictions = tf.round(tf.nn.sigmoid(pred)))
        tf.summary.scalar("Accuracy:", perf_op)
        smaller_is_better = False

    elif manager in {"hospital_readmission", "support2"}:
    
        model_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = Y, logits = pred))
        tf.summary.scalar("Cross-entropy", model_loss)
        
        _, perf_op = tf.metrics.accuracy(labels = tf.argmax(Y, 1), predictions = tf.argmax(pred, 1))
        tf.summary.scalar("Accuracy", perf_op)
        smaller_is_better = False

    if regularizer is None:
        loss_op = model_loss
    else:
        loss_op = model_loss + c * reg
        tf.summary.scalar("Regularizer", c * reg)
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
            if epoch % 1 == 0:
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
        out = {}

        # Evaluate Accuracy
        test_acc, test_pred = sess.run([perf_op, pred], feed_dict = {X: data.X_test, Y: data.y_test})

        out["test_acc"] = np.float64(test_acc)

        if evaluate_explanation:
            wrapper = Wrapper(sess, pred, X)

            out["variance"] = metrics_variance(wrapper, data.X_test).tolist()

            print(os.getcwd(), " MAPLE")
            maple_standard_metric, maple_causal_metric, maple_stability_metric = metrics_maple(wrapper, data.X_train, data.X_val, data.X_test)

            out["maple_standard_metric"] = maple_standard_metric.tolist()
            out["maple_causal_metric"] = maple_causal_metric.tolist()
            out["maple_stability_metric"] = maple_stability_metric.tolist()

            print(os.getcwd(), " LIME")
            lime_standard_metric, lime_causal_metric, lime_stability_metric = metrics_lime(wrapper, data.X_train, data.X_test)

            out["lime_standard_metric"] = lime_standard_metric.tolist()
            out["lime_causal_metric"] = lime_causal_metric.tolist()
            out["lime_stability_metric"] = lime_stability_metric.tolist()

        return out
