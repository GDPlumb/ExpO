
import tensorflow as tf

class MLP():
    # shape[0] = input dimension
    # shape[end] = output dimension
    def __init__(self, shape):
        self.shape = shape
        self.weight_init = tf.contrib.layers.xavier_initializer()
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

class CNN():

    # shape = [1, 6, 12, 24, 200, 10]
    # filters = [6, 5, 4]
    # strides = [1, 2, 2]
    # final_im_size = 7

    # shape[0] = number of input channels
    def __init__(self, shape, filters, strides, final_im_size):
        self.shape = shape
        self.filters = filters
        self.strides = strides
        self.final_im_size = final_im_size
        
        self.weight_init = tf.initializers.truncated_normal(stddev = 0.1)
        self.bias_init = tf.constant_initializer(0.1)

    def conv_layer(self, input, filter_size, channels_in, channels_out, stride):
        weights = tf.get_variable("weights", [filter_size, filter_size, channels_in, channels_out], initializer = self.weight_init)
        biases = tf.get_variable("biases", [channels_out], initializer = self.bias_init)
        return tf.nn.relu(tf.nn.conv2d(input, weights, strides = [1, stride, stride, 1], padding = "SAME") + biases)

    def model(self, input):
        shape = self.shape
        n = len(shape)
        filters = self.filters
        strides = self.strides
        final_im_size = self.final_im_size

        x = input
        for i in range(n - 3):
            with tf.variable_scope("conv_" + str(i + 1)):
                x = self.conv_layer(x, filters[i], shape[i], shape[i + 1], strides[i])
    
        x = tf.reshape(x, shape = [-1, final_im_size * final_im_size * shape[n - 3]])
        
        with tf.variable_scope("fully_connected"):
            weights = tf.get_variable("weights", [final_im_size * final_im_size * shape[n - 3], shape[n - 2]], initializer = self.weight_init)
            biases = tf.get_variable("biases", [shape[n - 2]], initializer = self.bias_init)
            x = tf.nn.relu(tf.matmul(x, weights) + biases)

        #with tf.variable_scope("dropout"):
        #    x = tf.nn.dropout(x, 0.5)

        with tf.variable_scope("output"):
            weights = tf.get_variable("weights", [shape[n - 2], shape[n - 1]], initializer = self.weight_init)
            biases = tf.get_variable("biases", [shape[n - 1]], initializer = self.bias_init)
            return tf.matmul(x, weights) + biases

