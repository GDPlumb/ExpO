
import numpy as np
import tensorflow as tf

from dataset import Dataset
from SLIM import SLIM

def eval(X_train, y_train, X_val, y_val, X_test, y_test, regularizer = "None"):

    data = Dataset(X_train, y_train)
    n = X_train.shape[0]
    n_input = X_train.shape[1]
    
    # Define the 'local neighborhood': this definition is used for both regularization and evaluation
    def generate_neighbor(x):
        return x + 0.1 * np.random.normal(loc = 0.0, scale = 1.0, size = n_input)
    
    # Regularization parameters
    c = tf.constant(1.0)
    num_points = np.int(1.0 * n) #Number of points from the training set sampled to compute the regularizer
    num_samples =  np.max((20, np.int(1.5 * n_input))) #Number of neighbors to hallucinate (via 'generate_neighbor') per sampled training point
    
    # Evaluation parameters
    num_perturbations = 5

    # Optimization Parameters
    learning_rate = 0.01
    training_epochs = 2000
    batch_size = 20

    # Network Parameters
    n_hidden_1 = 100
    n_hidden_2 = 100

    # Graph inputs
    X = tf.placeholder("float", [None, n_input], name = "X_in")
    Y = tf.placeholder("float", [None], name = "Y_in")

    # Store layers weight & bias
    with tf.name_scope("Weights") as scope:
        weights = {
            'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1]), name = "l1"),
            'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2]), name = "l2"),
            'out': tf.Variable(tf.random_normal([n_hidden_2, 1]), name = "out")
        }
    with tf.name_scope("Biases") as scope:
        biases = {
            'b1': tf.Variable(tf.random_normal([n_hidden_1]), name = "l1"),
            'b2': tf.Variable(tf.random_normal([n_hidden_2]), name = "l2"),
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

    # Construct model
    pred = multilayer_perceptron(X)

    if regularizer != "None":
        # Construct the sample and projection tensors
        X_sampled = X_train[np.random.choice(n, num_points, replace = False)]

        X_tensor = np.zeros((num_points, num_samples, n_input + 1))
        for i in range(num_points):
            for j in range(num_samples):
                X_tensor[i,j,:] = np.append(generate_neighbor(X_sampled[i]), 1)

        P_tensor = np.zeros((num_points, n_input + 1, num_samples))
        for i in range(num_points):
            x_h = X_tensor[i]
            P_tensor[i] = np.matmul(np.linalg.inv(np.matmul(x_h.transpose(), x_h)), x_h.transpose())
            
        X_tensor = tf.constant(X_tensor, dtype = "float")
        P_tensor = tf.constant(P_tensor, dtype = "float")
        Indices = tf.constant(np.arange(num_points), dtype = tf.int32)
        
        # Define the causal evaluation metric (LLM MSE version)
        def reg_causal():
            with tf.name_scope("CausalRegularizer") as scope:
                def compute_mse(args):
                    x = args[0]
                    P = args[1]
                    y = multilayer_perceptron(x[:, :-1])
                    B = tf.einsum('ij,j->i', P, y)
                    pred = tf.einsum('ij,j->i' , x, B)
                    return tf.losses.mean_squared_error(labels = y, predictions = pred), 0
                input = (X_tensor, P_tensor)
                mse, _ = tf.map_fn(compute_mse, input)
                return tf.reduce_mean(mse)
        
        if regularizer == "Causal":
            print("")
            print("Causal Regularizer")
            print("")
            reg = reg_causal()

    # Define loss and optimizer
    accuracy_loss = tf.losses.mean_squared_error(labels = Y, predictions = pred)
    tf.summary.scalar("MSE", accuracy_loss)
    
    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
    
    if regularizer == "None":
        loss_op = accuracy_loss
    else:
        loss_op = accuracy_loss + c * reg
        tf.summary.scalar("Regularizer", reg)
    tf.summary.scalar("Loss", loss_op)
    train_op = optimizer.minimize(loss_op)

    summary_op = tf.summary.merge_all()

    # Initializing the variables
    out = {}
    init = tf.global_variables_initializer()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    with tf.Session(config=config) as sess:
        train_writer = tf.summary.FileWriter("tb/train", sess.graph)
        test_writer = tf.summary.FileWriter("tb/test")
        
        sess.run(init)
        
        print("")
        print("Training NN")
        for epoch in range(training_epochs):
            total_batch = int(n / batch_size)
            for i in range(total_batch):
                batch_x, batch_y = data.next_batch(batch_size)
                sess.run(train_op, feed_dict={X: batch_x, Y: batch_y})
            if epoch % 10 == 0:
                summary = sess.run(summary_op, feed_dict = {X: X_train, Y: y_train})
                train_writer.add_summary(summary, epoch)
                summary = sess.run(summary_op, feed_dict = {X: X_test, Y: y_test})
                test_writer.add_summary(summary, epoch)

        train_writer.close()
        test_writer.close()

        test_acc, test_pred = sess.run([accuracy_loss, pred], feed_dict = {X: X_test, Y: y_test})

        print("Test MSE: ", test_acc)
        out["test_acc"] = np.float64(test_acc)

        print("Fitting SLIM")
        exp_slim = SLIM(X_train, pred.eval({X: X_train}), X_val, pred.eval({X: X_val}))

        print("Computing Metrics")
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

