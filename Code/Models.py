
import tensorflow as tf

class MLP():
    # shape[0] = input dimension
    # shape[end] = output dimension
    def __init__(self, shape):
        self.shape = shape
        self.weight_init = tf.random_normal_initializer()
        self.bias_init = tf.constant_initializer(0.0)
    
    def layer(self, input, input_size, output_size):
        weights = tf.get_variable("weights", [input_size, output_size], initializer = self.weight_init)
        biases = tf.get_variable("biases", output_size, initializer = self.bias_init)
        return tf.nn.leaky_relu(tf.matmul(input, weights) + biases)

    def model(self, input):
        shape = self.shape
        n = len(shape)
        x = input
        for i in range(n - 2):
            with tf.variable_scope("hidden_" + str(i + 1)):
                out = self.layer(x, shape[i], shape[i + 1])
                x = out
        with tf.variable_scope("output"):
            weights = tf.get_variable("weights", [shape[n - 2], shape[n - 1]], initializer = self.weight_init)
            biases = tf.get_variable("biases", shape[n - 1], initializer = self.bias_init)
            #return tf.squeeze(tf.matmul(x, weights) + biases)
            return tf.matmul(x, weights) + biases
