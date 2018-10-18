
import numpy as np
import tensorflow as tf

from MedicalData import HospitalReadmissionDataManager as DataManager
from Models import MLP
from Regularizers import Regularizer
from SLIM import SLIM


def eval(source, regularizer=None, name="tb"):
    if regularizer is not None:
        raise NotImplementedError("Regularizers do not support classification.")
        # TODO (GDPlumb): Make regularizers work with classification.

    # Reset TF graph (avoids issues with repeat experiments)
    tf.reset_default_graph()
    
    # Dataset Parameters
    batch_size = 20
    reg_batch_size = 2
    
    if regularizer != None:
        data = DataManager(
            source,
            train_batch_size=batch_size,
            reg_batch_size=reg_batch_size)
    else:
        data = DataManager(
            source,
            train_batch_size=batch_size)

    n = data.X_train.shape[0]
    n_input = data.X_train.shape[1]
    n_classes = data.y_train.shape[1]
    
    # Regularizer Parameters
    if regularizer != None:
        # Weight of the regularization term in the loss function
        c = tf.constant(100.0)
        #Number of neighbors to hallucinate per point
        num_samples = np.max((20, np.int(1.2 * n_input)))

    # Network Parameters
    shape = [n_input, 100, 100, n_classes]

    # Training Parameters
    learning_rate = 0.01
    training_epochs = 1000

    # Graph inputs - these names are tied into the DataManager class
    X = tf.placeholder("float", [None, n_input], name="X_in")
    Y = tf.placeholder("float", [None, n_classes], name="Y_in")
    if regularizer != None:
        X_reg = tf.placeholder("float", [None, n_input], name="X_reg")

    # Link the graph inputs into the DataManager so its train_feed() and eval_feed() functions work
    if regularizer != None:
        data.link_graph(X, Y, X_reg=X_reg)
    else:
        data.link_graph(X, Y)
    
    # Build the model
    network = MLP(shape)
    with tf.variable_scope("model", reuse=tf.AUTO_REUSE):
        logits = network.model(X)

    # Build the regularizer
    if regularizer == "Causal":
        regularizer = Regularizer(network.model, n_input, num_samples)
        reg = regularizer.causal(X_reg)

    # Define the loss and optimization process
    xent_loss = tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=logits)
    xent_loss = tf.reduce_mean(xent_loss)
    _, acc_op = tf.metrics.accuracy(
        labels=tf.argmax(Y, 1),
        predictions=tf.argmax(logits, 1))
    tf.summary.scalar("Cross-entropy:", xent_loss)
    tf.summary.scalar("Accuracy:", acc_op)

    if regularizer == None:
        loss_op = xent_loss
    else:
        loss_op = xent_loss + c * reg
        tf.summary.scalar("Regularizer", reg)
    tf.summary.scalar("Loss", loss_op)

    summary_op = tf.summary.merge_all()

    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
    train_op = optimizer.minimize(loss_op)

    # Train the model
    init = [tf.global_variables_initializer(), tf.local_variables_initializer()]

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        train_writer = tf.summary.FileWriter(name + "/train", sess.graph)
        val_writer = tf.summary.FileWriter(name + "/val")
        
        sess.run(init)
        
        print("")
        print("Training NN")
        for epoch in range(training_epochs):
            total_batch = int(n / batch_size)
            for i in range(total_batch):
                dict = data.train_feed()
                sess.run(train_op, feed_dict=dict)
            
            if epoch % 20 == 0:
                dict = data.eval_feed()
                summary = sess.run(summary_op, feed_dict = dict)
                train_writer.add_summary(summary, epoch)
                
                dict = data.eval_feed(val = True)
                summary = sess.run(summary_op, feed_dict = dict)
                val_writer.add_summary(summary, epoch)

        train_writer.close()
        val_writer.close()

        # Evaluate the model
        test_acc, test_logits = sess.run(
            [acc_op, logits],
            feed_dict={X: data.X_test, Y: data.y_test})

        out = {}
        print("Test acc: ", test_acc)
        out["test_acc"] = np.float64(test_acc)

        print("Not computing any other metrics since they don't work with classification yet.")
        # TODO (GDPlumb): Make metrics work with classification.

        if False:
            print("Fitting SLIM")
            exp_slim = SLIM(data.X_train, pred.eval({X: data.X_train}), data.X_val, pred.eval({X: data.X_val}))

            print("Computing Metrics")
            num_perturbations = 5

            # Define the 'local neighborhood': used for evaluation but is coded equivalently for training
            def generate_neighbor(x):
                return x + 0.1 * np.random.normal(loc=0.0, scale=1.0, size=n_input)

            n = data.X_test.shape[0]
            standard_metric = 0.0
            causal_metric = 0.0
            stability_metric = 0.0
            for i in range(n):
                x = data.X_test[i, :]

                e_slim = exp_slim.explain(x)
                coefs_slim = e_slim["coefs"]

                standard_metric += (e_slim["pred"][0] - test_pred[i])**2

                for j in range(num_perturbations):
                    x_pert = generate_neighbor(x)

                    model_pred = pred.eval({X: x_pert.reshape((1, n_input))})
                    slim_pred = np.dot(np.insert(x_pert, 0, 1), coefs_slim)
                    causal_metric += (slim_pred - model_pred)**2

                    e_slim_pert = exp_slim.explain(x_pert)
                    stability_metric += np.sum((e_slim_pert["coefs"] - coefs_slim)**2)

            standard_metric /= n
            causal_metric /= num_perturbations * n
            stability_metric /= num_perturbations * n

            print("Standard Metric: ", standard_metric)
            print("Causal Metric: ", causal_metric)
            print("Stability Metric: ", stability_metric)

            out["standard_metric"] = np.float64(standard_metric)
            out["causal_metric"] = np.float64(causal_metric)
            out["stability_metric"] = np.float64(stability_metric)

        return out
