
import tensorflow as tf

class Regularizer():
    def __init__(self, model, n_input, num_samples):
        self.model = model
        self.n_input = n_input
        self.num_samples = num_samples
    
    def neighborhood(self, x):
        num_samples = self.num_samples
        n_input = self.n_input
        with tf.name_scope("GenerateNeighborhood") as scope:
            x_expanded = tf.reshape(tf.tile(x, [num_samples]), [num_samples, n_input])
            noise = tf.random_normal([num_samples, n_input], stddev = 0.1)
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

