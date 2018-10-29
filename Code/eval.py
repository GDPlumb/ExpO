
import numpy as np
import tensorflow as tf

from Models import MLP
from Regularizers import Regularizer
from SLIM import SLIM

def eval(manager, source, hidden_layer_sizes, learning_rate, regularizer = None, c = 1):

    if manager == "regression":
        from Data import DataManager
    elif manager == "hospital_readmission":
        from MedicalData import HospitalReadmissionDataManager as DataManager

    # Reset TF graph (avoids issues with repeat experiments)
    tf.reset_default_graph()

    # Parameters
    batch_size = 32
    reg_batch_size = 2

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
    elif manager == "hospital_readmission":
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
            if epoch - best_epoch > 500:
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
                    #print(epoch, " ", val_loss)
                    best_loss = val_loss
                    best_epoch = epoch
                    saver.save(sess, "./model.cpkt")
    
            epoch += 1

        train_writer.close()
        val_writer.close()

        # Evaluate the chosen model
        saver.restore(sess, "./model.cpkt")

        if manager == "regression":
            test_acc, test_pred = sess.run([model_loss, pred], feed_dict = {X: data.X_test, Y: data.y_test})
        elif manager == "hospital_readmission":
            test_acc, test_pred = sess.run([acc_op, pred], feed_dict={X: data.X_test, Y: data.y_test})

        out = {}
        #print("Test Acc: ", test_acc)
        out["test_acc"] = np.float64(test_acc)

        #print("Fitting SLIM")
        train_pred = sess.run(pred, feed_dict = {X: data.X_train})
        val_pred = sess.run(pred, feed_dict = {X: data.X_val})

        exp_slim = [None] * n_out
        for i in range(n_out):
            exp_slim[i] = SLIM(data.X_train, train_pred[:, i], data.X_val, val_pred[:, i])

        #print("Computing Metrics")
        num_perturbations = 5

        # Define the 'local neighborhood': used for evaluation but is coded equivalently for training
        def generate_neighbor(x):
            return x + 0.1 * np.random.normal(loc=0.0, scale=1.0, size=n_input)

        n = data.X_test.shape[0]
        standard_metric = np.zeros(n_out)
        causal_metric = np.zeros(n_out)
        stability_metric = np.zeros(n_out)
        for i in range(n_out):
            for j in range(n):
                x = data.X_test[j, :]

                e_slim = exp_slim[i].explain(x)
                coefs_slim = e_slim["coefs"]

                standard_metric[i] += (e_slim["pred"][0] - test_pred[j,i])**2

                for j in range(num_perturbations):
                    x_pert = generate_neighbor(x)

                    model_pred = pred.eval({X: x_pert.reshape((1, n_input))})[0,i]
                    slim_pred = np.dot(np.insert(x_pert, 0, 1), coefs_slim)
                    causal_metric[i] += (slim_pred - model_pred)**2

                    e_slim_pert = exp_slim[i].explain(x_pert)
                    stability_metric[i] += np.sum((e_slim_pert["coefs"] - coefs_slim)**2)

        standard_metric /= n
        causal_metric /= num_perturbations * n
        stability_metric /= num_perturbations * n

        standard_metric = standard_metric
        causal_metric = causal_metric
        standard_metric = stability_metric

        #print("Standard Metric: ", standard_metric)
        #print("Causal Metric: ", causal_metric)
        #print("Stability Metric: ", stability_metric)

        out["standard_metric"] = standard_metric.tolist()
        out["causal_metric"] = causal_metric.tolist()
        out["stability_metric"] = stability_metric.tolist()

        return out
