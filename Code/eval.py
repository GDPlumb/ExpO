
import numpy as np
import tensorflow as tf

from dataset import Dataset
from SLIM import SLIM

def eval(X_train, y_train, X_val, y_val, X_test, y_test, regularizer = "None", name = "tb"):

    # Reset TF graph (avoids issues with repeat exeriments)
    tf.reset_default_graph()

    # Gives access to minibatches of the dataset
    data = Dataset(X_train, y_train)
    
    # Basic data stats
    n = X_train.shape[0]
    n_input = X_train.shape[1]

    # Optimization Parameters
    learning_rate = 0.01
    training_epochs = 2000
    batch_size = 20

    # Network Parameters
    n_hidden_1 = 100
    n_hidden_2 = 100
    #n_hidden_3 = 100
    
    # Graph inputs
    X = tf.placeholder("float", [None, n_input], name = "X_in")
    Y = tf.placeholder("float", [None], name = "Y_in")
    
    # Configure Regularizer
    if regularizer != "None":
        # Parameters
        c = tf.constant(100.0)
        num_points = 2 #Number to use to approximate the regularizer per minibatch
        num_samples = np.max((20, np.int(1.2 * n_input))) #Number of neighbors to hallucinate per point
        # Use a second copy of the dataset to get regularizer minibatches
        data_reg = Dataset(X_train, y_train)
        # Placeholder to pass the regularizer minibatches in
        X_reg = tf.placeholder("float", [None, n_input], name = "X_reg")
    
    # Define the data feeding process
    def training_feed():
        batch_x, batch_y = data.next_batch(batch_size)
        if regularizer == "None":
            return {X: batch_x, Y: batch_y}
        else:
            batch_x_reg, batch_y_reg = data_reg.next_batch(num_points)
            return {X: batch_x, Y: batch_y, X_reg: batch_x_reg}
            
    def eval_feed(test):
        if test:
            X_eval = X_test
            Y_eval = y_test
        else:
            X_eval = X_train
            Y_eval = y_train
        indices = np.random.choice(X_eval.shape[0], 20, replace = False)
        batch_x = X_eval[indices]
        batch_y = Y_eval[indices]
        if regularizer == "None":
            return {X: batch_x, Y: batch_y}
        else:
            indices_reg = np.random.choice(X_eval.shape[0], 5, replace = False)
            batch_x_reg = X_eval[indices_reg]
            return {X: batch_x, Y: batch_y, X_reg: batch_x_reg}

    # Store layers weight & bias
    with tf.name_scope("Weights") as scope:
        weights = {
            'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1]), name = "l1"),
            'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2]), name = "l2"),
            #'h3': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_3]), name = "l3"),
            'out': tf.Variable(tf.random_normal([n_hidden_2, 1]), name = "out")
        }
    with tf.name_scope("Biases") as scope:
        biases = {
            'b1': tf.Variable(tf.random_normal([n_hidden_1]), name = "l1"),
            'b2': tf.Variable(tf.random_normal([n_hidden_2]), name = "l2"),
            #'b3': tf.Variable(tf.random_normal([n_hidden_2]), name = "l3"),
            'out': tf.Variable(tf.random_normal([1]), name = "out")
        }

    # Define model
    def multilayer_perceptron(x):
        with tf.name_scope("Layer1") as scope:
            layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
            rep_1 = tf.nn.relu(layer_1)
        with tf.name_scope("Layer2") as scope:
            layer_2 = tf.add(tf.matmul(rep_1, weights['h2']), biases['b2'])
            rep_2 = tf.nn.relu(layer_2)
        with tf.name_scope("Output") as scope:
            out_layer = tf.matmul(rep_2, weights['out']) + biases['out']
            return tf.squeeze(out_layer)
    pred = multilayer_perceptron(X)

    if regularizer == "Causal":
        # Approximates the Causal metric with the average MSE of the LLMs
        def regularizer(x):
            with tf.name_scope("CausalRegularizer") as scope:
                def compute_mse(x):
                    with tf.name_scope("GenerateNeighborhood") as scope:
                        x_expanded = tf.reshape(tf.tile(x, [num_samples]), [num_samples, n_input])
                        noise = tf.random_normal([num_samples, n_input], stddev = 0.1)
                        constant_term = tf.ones([num_samples, 1])
                        x_local = tf.stop_gradient(tf.concat([x_expanded + noise, constant_term], 1))
            
                    with tf.name_scope("ComputeProjectionMatrix") as scope:
                        P = tf.stop_gradient(tf.matmul(tf.matrix_inverse(tf.matmul(tf.transpose(x_local), x_local)), tf.transpose(x_local)))
            
                    with tf.name_scope("ModelPredictLocal") as scope:
                        y = multilayer_perceptron(x_local[:, :-1])
            
                    with tf.name_scope("ComputeCoefficients") as scope:
                        B = tf.einsum('ij,j->i', P, y)
                        
                    with tf.name_scope("LinearPredictLocal") as scope:
                        y_lin = tf.einsum('ij,j->i', x_local, B)
                    
                    with tf.name_scope("LocalLinearMSE") as scope:
                        return tf.losses.mean_squared_error(labels = y, predictions = y_lin)
                return tf.reduce_mean(tf.map_fn(compute_mse, x))
        reg = regularizer(X_reg)

    # Define the loss and optimization process
    accuracy_loss = tf.losses.mean_squared_error(labels = Y, predictions = pred)
    tf.summary.scalar("MSE", accuracy_loss)
    
    if regularizer == "None":
        loss_op = accuracy_loss
    else:
        loss_op = accuracy_loss + c * reg
        tf.summary.scalar("Regularizer", reg)
    tf.summary.scalar("Loss", loss_op)

    summary_op = tf.summary.merge_all()

    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
    train_op = optimizer.minimize(loss_op)

    # Start the experiment
    out = {}
    init = tf.global_variables_initializer()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    with tf.Session(config=config) as sess:
        train_writer = tf.summary.FileWriter(name + "/train", sess.graph)
        test_writer = tf.summary.FileWriter(name + "/test")
        
        sess.run(init)
        
        print("")
        print("Training NN")
        for epoch in range(training_epochs):
            total_batch = int(n / batch_size)
            for i in range(total_batch):
                dict = training_feed()
                sess.run(train_op, feed_dict=dict)
            if epoch % 10 == 0:
                dict = eval_feed(False)
                summary = sess.run(summary_op, feed_dict = dict)
                train_writer.add_summary(summary, epoch)
                dict = eval_feed(True)
                summary = sess.run(summary_op, feed_dict = dict)
                test_writer.add_summary(summary, epoch)

        train_writer.close()
        test_writer.close()

        test_acc, test_pred = sess.run([accuracy_loss, pred], feed_dict = {X: X_test, Y: y_test})

        print("Test MSE: ", test_acc)
        out["test_acc"] = np.float64(test_acc)

        print("Fitting SLIM")
        exp_slim = SLIM(X_train, pred.eval({X: X_train}), X_val, pred.eval({X: X_val}))

        print("Computing Metrics")
        num_perturbations = 5

        # Define the 'local neighborhood': used for evaluation but is coded equivalently for training
        def generate_neighbor(x):
            return x + 0.1 * np.random.normal(loc = 0.0, scale = 1.0, size = n_input)
            
        n = X_test.shape[0]
        standard_metric = 0.0
        causal_metric = 0.0
        stability_metric = 0.0
        for i in range(n):
            x = X_test[i, :]

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

