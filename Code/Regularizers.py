
import tensorflow as tf

class Regularizer():
    def __init__(self, model, n_input, num_samples, stddev = 0.5):
        self.model = model
        self.n_input = n_input
        self.num_samples = num_samples
        self.stddev = stddev
    
    def neighborhood(self, x):
        num_samples = self.num_samples
        n_input = self.n_input
        with tf.name_scope("GenerateNeighborhood") as scope:
            x_expanded = tf.reshape(tf.tile(x, [num_samples]), [num_samples, n_input])
            noise = tf.random_normal([num_samples, n_input], stddev = self.stddev)
            constant_term = tf.ones([num_samples, 1])
            return tf.stop_gradient(tf.concat([x_expanded + noise, constant_term], 1))

    def projection_matrix(self, x_local):
        with tf.name_scope("ComputeProjectionMatrix") as scope:
            return tf.stop_gradient(tf.matmul(tf.matrix_inverse(tf.matmul(tf.transpose(x_local), x_local)), tf.transpose(x_local)))

    # P is [n_input, num_samples], y is [num_samples, n_classes]
    # Output is [num_samples, n_classes]
    def coefficients(self, P, y):
        with tf.name_scope("ComputeCoefficients") as scope:
            #return tf.einsum('ij,j->i', P, y)
            return tf.einsum('ij,jk->ik', P, y)

    # It may be possible to make this computation more efficient by using broadcasting rather than map_fn
    def causal(self, x):
        with tf.name_scope("CausalRegularizer") as scope:
            def compute_mse(x):
                x_local = self.neighborhood(x)
                P = self.projection_matrix(x_local)
                with tf.variable_scope("model", reuse = tf.AUTO_REUSE):
                    y = self.model(x_local[:, :-1])
                B = self.coefficients(P, y)
                with tf.name_scope("LinearPredictLocal") as scope:
                    #y_lin = tf.einsum('ij,j->i', x_local, B)
                    y_lin = tf.einsum('ij,jk->ik', x_local, B)
                with tf.name_scope("LocalLinearMSE") as scope:
                    return tf.losses.mean_squared_error(labels = y, predictions = y_lin)
            return tf.reduce_mean(tf.map_fn(compute_mse, x))

class Regularizer_1D():
    def __init__(self, model, n_input, num_samples, stddev = 0.5):
        self.model = model
        self.n_input = n_input
        self.num_samples = num_samples
        self.stddev = stddev
    
    def neighborhood(self, x):
        num_samples = self.num_samples
        n_input = self.n_input
        with tf.name_scope("GenerateNeighborhood") as scope:
            x_expanded = tf.reshape(tf.tile(x, [num_samples]), [num_samples, n_input]) # Create 'num_samples' copies of 'x' in one batch
            dim = tf.random_uniform(shape = [1], minval = 0, maxval = n_input, dtype = tf.int32)[0] # Choose a random dimension to regularize
            noise = tf.random_normal(shape = [num_samples, 1], stddev = self.stddev) # Generate the perturbations
            noise_padded = tf.pad(noise, [[0,0], [dim, n_input - dim - 1]]) # Pad the perturbations such that they correspond to the chosen random dimension
            return x_expanded + noise_padded, dim

    def coefficients(self, x_chosen, y):
        with tf.name_scope("ComputeCoefficients") as scope:
            x_bar = tf.reduce_mean(x_chosen)
            y_bar = tf.reduce_mean(y, axis = 0)
            
            x_delta = x_chosen - x_bar
            y_delta = y - y_bar
            
            betas = tf.reduce_sum(x_delta * y_delta, axis = 0) / tf.reduce_sum(x_delta * x_delta)
            
            ints = y_bar - betas * x_bar
            return betas, ints

    # It may be possible to make this computation more efficient by using broadcasting rather than map_fn
    def causal(self, x):
        with tf.name_scope("CausalRegularizer") as scope:
            def compute_mse(x):
                x_local, dim = self.neighborhood(x)
                x_chosen = tf.expand_dims(x_local[:, dim], axis = 1) #shape = [num_samples, 1]
                x_chosen = tf.stop_gradient(x_chosen)
                with tf.variable_scope("model", reuse = tf.AUTO_REUSE):
                    y = self.model(x_local)
                betas, ints = self.coefficients(x_chosen, y)
                with tf.name_scope("LinearPredictLocal") as scope:
                    y_lin = x_chosen * betas + ints
                with tf.name_scope("LocalLinearMSE") as scope:
                    return tf.losses.mean_squared_error(labels = y, predictions = y_lin)
            return tf.reduce_mean(tf.map_fn(compute_mse, x))

