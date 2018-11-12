
import numpy as np
import tensorflow as tf

from Models import MLP
from Regularizers import Regularizer
from SLIM import SLIM as MAPLE

from lime import lime_tabular

def eval(manager, source, hidden_layer_sizes, learning_rate, regularizer = None, c = 1):

    if manager == "regression":
        from Data import DataManager
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
            if epoch - best_epoch > 1000:
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

        ###
        # Evaluate the chosen model
        ###
        
        saver.restore(sess, "./model.cpkt")
        out = {}

        # Evaluate Accuracy
        if manager == "regression":
            test_acc, test_pred = sess.run([model_loss, pred], feed_dict = {X: data.X_test, Y: data.y_test})
        elif manager == "hospital_readmission":
            test_acc, test_pred = sess.run([acc_op, pred], feed_dict={X: data.X_test, Y: data.y_test})

        out["test_acc"] = np.float64(test_acc)

        # Training and Validation Predictions
        train_pred = sess.run(pred, feed_dict = {X: data.X_train})
        val_pred = sess.run(pred, feed_dict = {X: data.X_val})

        # Configure MAPLE
        exp_maple = [None] * n_out
        for i in range(n_out):
            exp_maple[i] = MAPLE(data.X_train, train_pred[:, i], data.X_val, val_pred[:, i])

        # Configure LIME
        if manager == "regression":
            exp_lime = lime_tabular.LimeTabularExplainer(data.X_train, discretize_continuous=False, mode="regression")
        elif manager == "hospital_readmission":
            raise NotImplementedError

        class Wrapper():
            def __init__(self):
                self.index = 0

            def set_index(self, i):
                self.index = i

            def predict(self, x):
                return np.squeeze(sess.run(pred, feed_dict = {X: x})[:, self.index])
        wrapper = Wrapper()
            
        def unpack_coefs(explainer, x, predict_fn, num_features, x_train, num_samples = 5000):
            d = x_train.shape[1]
            coefs = np.zeros((d))
            
            u = np.mean(x_train, axis = 0)
            sd = np.sqrt(np.var(x_train, axis = 0))
            
            exp = explainer.explain_instance(x, predict_fn, num_features=num_features, num_samples = num_samples)
            
            coef_pairs = exp.local_exp[1]
            for pair in coef_pairs:
                coefs[pair[0]] = pair[1]
            
            coefs = coefs / sd

            intercept = exp.intercept[1] - np.sum(coefs * u)

            return np.insert(coefs, 0, intercept)

        # Configure the Local Neighborhood
        num_perturbations = 5

        def generate_neighbor(x):
            return x + 0.1 * np.random.normal(loc=0.0, scale=1.0, size=n_input)

        # Explanation metrics
        n = data.X_test.shape[0]
        d = data.X_test.shape[1]
        
        maple_standard_metric = np.zeros(n_out)
        maple_causal_metric = np.zeros(n_out)
        maple_stability_metric = np.zeros(n_out)
        
        lime_standard_metric = np.zeros(n_out)
        lime_causal_metric = np.zeros(n_out)
        lime_stability_metric = np.zeros(n_out)
        
        for i in range(n_out):
            wrapper.set_index(i)
            for j in range(n):
                x = data.X_test[j, :]

                # Get MAPLE's Explanation
                e_maple = exp_maple[i].explain(x)
                coefs_maple = e_maple["coefs"]
                
                # Get LIME's Explanation
                coefs_lime = unpack_coefs(exp_lime, x, wrapper.predict, d, data.X_train)

                # Standard Metric
                maple_standard_metric[i] += (e_maple["pred"][0] - test_pred[j,i])**2
                lime_standard_metric[i] += (np.dot(np.insert(x, 0, 1), coefs_lime) - test_pred[j,i])**2

                for j in range(num_perturbations):
                    x_pert = generate_neighbor(x)

                    # Causal Metric
                    model_pred = pred.eval({X: x_pert.reshape((1, n_input))})[0,i]
                    
                    maple_pred = np.dot(np.insert(x_pert, 0, 1), coefs_maple)
                    maple_causal_metric[i] += (maple_pred - model_pred)**2

                    lime_pred = np.dot(np.insert(x_pert, 0, 1), coefs_lime)
                    lime_causal_metric[i] += (lime_pred - model_pred)**2

                    # Stability Metric
                    e_maple_pert = exp_maple[i].explain(x_pert)
                    maple_stability_metric[i] += np.sum((e_maple_pert["coefs"] - coefs_maple)**2)

                    coefs_lime_pert = unpack_coefs(exp_lime, x, wrapper.predict, d, data.X_train)
                    lime_stability_metric[i] += np.sum((coefs_lime_pert - coefs_lime)**2)

        maple_standard_metric /= n
        maple_causal_metric /= num_perturbations * n
        maple_stability_metric /= num_perturbations * n

        lime_standard_metric /= n
        lime_causal_metric /= num_perturbations * n
        lime_stability_metric /= num_perturbations * n

        out["maple_standard_metric"] = maple_standard_metric.tolist()
        out["maple_causal_metric"] = maple_causal_metric.tolist()
        out["maple_stability_metric"] = maple_stability_metric.tolist()

        out["lime_standard_metric"] = lime_standard_metric.tolist()
        out["lime_causal_metric"] = lime_causal_metric.tolist()
        out["lime_stability_metric"] = lime_stability_metric.tolist()

        return out
