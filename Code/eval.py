
import numpy as np
import os
import tensorflow as tf

from ExplanationMetrics import Wrapper, metrics_maple, metrics_lime
from Models import MLP
from Regularizers import Regularizer

def eval(manager, source, #Data magnager and data source
        hidden_layer_sizes, learning_rate, stopping_epochs = 1000, #Network parameters
        regularizer = None, c = 1, #Regularizer parameters
        evaluate_maple = True, evaluate_lime = True): #Explanation evaluation metrics

    if manager == "regression":
        from Data import DataManager
    elif manager == "binary_classification":
        from BinaryData import BinaryClassificationDataManager as DataManager
    elif manager == "hospital_readmission":
        from MedicalData import HospitalReadmissionDataManager as DataManager
    elif manager == "support2":
        from MedicalData import Support2DataManager as DataManager
    else:
        raise ValueError("Unknown dataset: %s" % manager)

    # Reset TF graph (avoids issues with repeat experiments)
    tf.reset_default_graph()

    # Parameters
    batch_size = 32
    reg_batch_size = 4

    # Get Data
    if regularizer != None:
        data = DataManager(source, train_batch_size = batch_size, reg_batch_size = reg_batch_size)
    else:
        data = DataManager(source, train_batch_size = batch_size)

    n = data.X_train.shape[0]
    n_input = data.X_train.shape[1]
    n_out = data.y_train.shape[1]

    # Regularizer Parameters
    if regularizer != None:
        # Weight of the regularization term in the loss function
        c = tf.constant(c)
        #Number of neighbors to hallucinate per point
        num_samples = np.max((20, np.int(2 * n_input)))

    # Network Parameters
    shape = [n_input]
    for size in hidden_layer_sizes:
        shape.append(size)
    shape.append(n_out)

    # Graph inputs
    X = tf.placeholder("float", [None, n_input], name = "X_in")
    Y = tf.placeholder("float", [None, n_out], name = "Y_in")
    if regularizer != None:
        X_reg = tf.placeholder("float", [None, n_input], name="X_reg")

    # Link the graph inputs into the DataManager so its train_feed() and eval_feed() functions work
    if regularizer != None:
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
    if manager == "regression":
        model_loss = tf.losses.mean_squared_error(labels = Y, predictions = pred)
        tf.summary.scalar("MSE", model_loss)
    elif manager == "binary_classification":
        model_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels = Y, logits = pred))
        _, acc_op = tf.metrics.accuracy(labels = Y, predictions = tf.round(tf.nn.sigmoid(pred)))
        tf.summary.scalar("Cross-entropy:", model_loss)
        tf.summary.scalar("Accuracy:", acc_op)
    elif manager in {"hospital_readmission", "support2"}:
        model_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = Y, logits = pred))
        _, acc_op = tf.metrics.accuracy(labels = tf.argmax(Y, 1), predictions = tf.argmax(pred, 1))
        tf.summary.scalar("Cross-entropy:", model_loss)
        tf.summary.scalar("Accuracy:", acc_op)

    if regularizer == None:
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

    saver = tf.train.Saver(max_to_keep=1) #We are going to keep the model with the best loss
    best_loss = np.inf
    best_epoch = 0

    with tf.Session() as sess:
        train_writer = tf.summary.FileWriter("train", sess.graph)
        val_writer = tf.summary.FileWriter("val")

        sess.run(init)

        #print("Training NN")
        epoch = 0
        while True:

            # Early stopping condition
            if epoch - best_epoch > stopping_epochs:
                break

            # Run a training epoch
            total_batch = int(n / batch_size)
            for i in range(total_batch):
                dict = data.train_feed()
                sess.run(train_op, feed_dict=dict)

            # Run model metrics
            if epoch % 20 == 0:
                dict = data.eval_feed()
                summary = sess.run(summary_op, feed_dict = dict)
                train_writer.add_summary(summary, epoch)

                dict = data.eval_feed(val = True)
                summary, val_loss = sess.run([summary_op, loss_op], feed_dict = dict)
                val_writer.add_summary(summary, epoch)

                #Save best model
                if val_loss < best_loss:
                    print(os.getcwd(), " ", epoch, " ", val_loss)
                    best_loss = val_loss
                    best_epoch = epoch
                    saver.save(sess, "./model.cpkt")

            epoch += 1

        train_writer.close()
        val_writer.close()

        ###
        # Evaluate the chosen model
        ###

        saver.restore(sess, "./model.cpkt")
        out = {}

        # Evaluate Accuracy
        if manager == "regression":
            test_acc, test_pred = sess.run([model_loss, pred], feed_dict = {X: data.X_test, Y: data.y_test})
        elif manager in {"binary_classification", "hospital_readmission", "support2"}:
            test_acc, test_pred = sess.run([acc_op, pred], feed_dict={X: data.X_test, Y: data.y_test})

        out["test_acc"] = np.float64(test_acc)

        wrapper = Wrapper(sess, pred, X)

        if evaluate_maple:
            print(os.getcwd(), " MAPLE")
            maple_standard_metric, maple_causal_metric, maple_stability_metric = metrics_maple(wrapper, data.X_train, data.X_val, data.X_test)

            out["maple_standard_metric"] = maple_standard_metric.tolist()
            out["maple_causal_metric"] = maple_causal_metric.tolist()
            out["maple_stability_metric"] = maple_stability_metric.tolist()

        if evaluate_lime:
            print(os.getcwd(), " LIME")
            lime_standard_metric, lime_causal_metric, lime_stability_metric = metrics_lime(wrapper, data.X_train, data.X_test)

            out["lime_standard_metric"] = lime_standard_metric.tolist()
            out["lime_causal_metric"] = lime_causal_metric.tolist()
            out["lime_stability_metric"] = lime_stability_metric.tolist()

        return out
