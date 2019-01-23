
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

import numpy as np

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

import os
import sys
sys.path.insert(0, os.path.join(os.getcwd(), "../Code/"))
from Models import CNN

from tqdm import tqdm

out = open("results.txt", "w")

def eval(
        # Network Parameters
        shape = [1, 6, 12, 24, 200, 10],
        filters = [6, 5, 4],
        strides = [1, 2, 2],
        final_im_size = 7,
        # Optimization Parameters
        learning_rate = 0.001,
        epochs = 100,
        batch_size = 50,
        # Regularizer parameters
        regularize = True,
        weight = 1.0,
        pert_range = 0.05,
        # Experiment parameters
        name = None):

    if name == None:
        name = "summary"

    # Allow multiple sessions on a single GPU.
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True

    tf.reset_default_graph()

    # Load the dataset
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

    # Graph inputs
    X = tf.placeholder(tf.float32, [None, 784], name = "X_in")
    X_shaped = tf.reshape(X, [-1, 28, 28, 1])
    Y = tf.placeholder(tf.float32, [None, 10], name = "Y_in")

    # Build the model
    network = CNN(shape, filters, strides, final_im_size)
    with tf.variable_scope("model", reuse = tf.AUTO_REUSE):
        pred = network.model(X_shaped)

    # Build the regularizer
    X_perturbed = X_shaped + tf.random_uniform(shape = tf.shape(X_shaped), minval = -1.0 * pert_range, maxval = 1.0 * pert_range)
    with tf.variable_scope("model", reuse = tf.AUTO_REUSE):
        pred_perturbed = network.model(X_perturbed)

    # Model loss
    model_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = Y, logits = pred))
    tf.summary.scalar("Loss_model", model_loss)
    _, accuracy = tf.metrics.accuracy(labels = tf.argmax(Y, 1), predictions = tf.argmax(pred, 1))
    tf.summary.scalar("Accuracy", accuracy)

    # Regularizer loss
    if regularize:
        reg_loss = tf.losses.mean_squared_error(labels = pred, predictions = pred_perturbed)
        tf.summary.scalar("Loss_reg", reg_loss)

    # Optimization process
    if regularize:
        loss_op = model_loss + weight * reg_loss
    else:
        loss_op = model_loss
    tf.summary.scalar("Loss", loss_op)

    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
    train_op = optimizer.minimize(loss_op)

    # Saliency Explanation Evaluation
    def saliency_map(predictions):
        indices = tf.cast(tf.argmax(predictions, axis = 1), tf.int32)
        row_indices = tf.cast(tf.range(tf.shape(indices)[0]), tf.int32)
        full_indices = tf.stack([row_indices, indices], axis = 1)
        targets = tf.gather_nd(predictions, full_indices)
        map = tf.gradients(targets, X_shaped)
        return map

    sm_reg = saliency_map(pred)
    sm_per = saliency_map(pred_perturbed)

    explanation_stability = tf.losses.mean_squared_error(labels = sm_reg, predictions = sm_per)
    tf.summary.scalar("Explanation Stability", explanation_stability)

    # Train the model
    summary_op = tf.summary.merge_all()
    init = [tf.global_variables_initializer(), tf.local_variables_initializer()]
    with tf.Session(config = tf_config) as sess:

        writer = tf.summary.FileWriter(name, sess.graph)

        sess.run(init)

        total_batch = int(len(mnist.train.labels) / batch_size)
        for epoch in tqdm(range(epochs)):
            for i in range(total_batch):
                batch_x, batch_y = mnist.train.next_batch(batch_size = batch_size)
                sess.run(train_op, feed_dict = {X: batch_x, Y: batch_y})

                if i % 100 == 0:
                    summary = sess.run(summary_op, feed_dict = {X: mnist.test.images, Y: mnist.test.labels})
                    writer.add_summary(summary, epoch * len(mnist.train.labels) + i * batch_size)

        print("Final Test Accuracy", file = out)
        print(sess.run(accuracy, feed_dict={X: mnist.test.images, Y: mnist.test.labels}), file = out)
        out.flush()

        print("Explanation MSE", file = out)
        print(sess.run(explanation_stability, {X: mnist.test.images}), file = out)
        out.flush()

        # Sanity check that the explanations look reasonable
        n = 5
        num_pert = 100
        x = mnist.test.images[:n]

        plt.imshow(np.reshape(x, (n * 28,28)), cmap = "gray")
        plt.savefig(name + "_original.jpeg")
        plt.close()
        
 
        def scale(input):
            input -= np.mean(input)
            input /= (np.std(input) + 1e-5)
            input *= 0.2
            input += 0.5
            input = np.clip(input, 0, 1)
            return input
            
        m = sess.run(sm_reg, {X: x})
        m = scale(m)

        m = np.reshape(m, (n * 28, 28))
        
        plt.imshow(m, cmap = "jet")
        plt.savefig(name + "_explanation.jpeg")
        plt.close()

        m = np.zeros((n, 784))
        for i in range(n):
            x_pert = np.zeros((num_pert, 784))
            for j in range(num_pert):
                x_pert[j, :] = x[i] + np.random.uniform(low = -1.0 * pert_range, high = pert_range, size = (1,784))

            m_avg = sess.run(sm_reg, {X: x_pert})
            m_avg = np.squeeze(m_avg) #Remove the gradient first dim and the channel dim
            m_avg = np.mean(m_avg, axis = 0)

            m_avg = scale(m_avg)
            
            m[i, :] = np.reshape(m_avg, (1, 784))

        plt.imshow(np.reshape(m, (n * 28,28)), cmap = "jet")
        plt.savefig(name + "_averaged_explanation.jpeg")
        plt.close()

print("Unregularized Model", file = out)
eval(regularize = False, name = "unregularized")
print("Regularized Model", file = out)
eval(regularize = True, name = "regularized")
