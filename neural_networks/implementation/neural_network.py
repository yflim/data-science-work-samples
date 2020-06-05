import numpy as np
from scipy.special import softmax
from sklearn.metrics import log_loss
from sklearn.metrics import mean_squared_error
import utils

class NeuralNetwork():

    class Layer():
        @staticmethod
        def _identity(a):
            return a

        @staticmethod
        def _identity_deriv(a):
            return np.ones(a.shape)

        @staticmethod
        def _relu(a):
            return np.where(a > 0, a, 0)

        @staticmethod
        def _relu_deriv(a):
            return np.where(a > 0, 1, 0)

        @staticmethod
        def _logistic(a):
            return softmax(a, axis=0) # assumes each column represents 1 observation

        def _logistic_deriv(self, weighted_input_sums):   # unused argument for interface consistent with other activation derivative fns
            return self.vals * (1 - self.vals)  # see pg 209 (multiclass logistic regression) of PRML

        def _tanh_deriv(self, weighted_input_sums):
            return 1 - np.square(self.vals)

        def init_weights(self, inputs_dim):
            self.weights = np.random.rand(self.nodes_dim, inputs_dim) - 0.5
            self.last_weight_delta = np.zeros(self.weights.shape)

        def __init__(self, nodes_dim, inputs_dim, activation_str):
            self.nodes_dim = nodes_dim
            if inputs_dim:
                self.init_weights(inputs_dim)
            self.activation_str = activation_str

        def _activation(self):
            return { 'identity': self._identity, 'relu': self._relu, 'logistic': self._logistic, 'tanh': np.tanh }[self.activation_str](np.copy(self.weighted_input_sums))

        def _activation_deriv(self):
            return { 'identity': self._identity_deriv, 'relu': self._relu_deriv,
                'logistic': self._logistic_deriv, 'tanh': self._tanh_deriv }[self.activation_str](self.weighted_input_sums)

        def feed(self, z):
            self.weighted_input_sums = np.matmul(self.weights, z)
            self.vals = self._activation()
            return self.vals

        # * for element-wise product
        def compute_errors(self, next_layer=None, t=None):
            if next_layer:
                self.errors = np.matmul(next_layer.weights.T, next_layer.errors) * self._activation_deriv()
            else:
                self.errors = self.vals - t   # Assuming canonical link as activation (see backpropagation, pg. 243, in PRML)

        def adjust_weights(self, momentum, change):
            self.last_weight_delta = momentum * self.last_weight_delta - change
            self.weights = self.weights + self.last_weight_delta


    def __init__(self, hidden_layer_dims=None, hidden_layer_activations=None, fit_intercept=True, samples_per_cycle=1,
          output_activation=None, loss_fn=None, learning_rate=1e-6, momentum=0.9, tol=1e-5, n_epochs_no_change=5, max_epochs=10000):
        self.fit_intercept = fit_intercept
        self.samples_per_cycle = samples_per_cycle
        self.output_activation = output_activation
        self.loss_fn = (self.output_activation and self.set_loss_fn(loss_fn)) or loss_fn
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.tol = tol
        self.n_epochs_no_change = n_epochs_no_change
        self.max_epochs = max_epochs
        self.hidden_layers = []
        prev_dim = None
        if hidden_layer_dims:
            layers_count = len(hidden_layer_dims)
            hidden_layer_activations = hidden_layer_activations or (['relu'] * layers_count)
            for i in range(layers_count):
                self.hidden_layers.append(self.Layer(hidden_layer_dims[i], prev_dim, hidden_layer_activations[i]))
                prev_dim = hidden_layer_dims[i]

    def add_hidden_layer(self, dim, activation='relu'):
        prev_dim = ((len(self.hidden_layers) > 0) and len(self.hidden_layers[-1].weights)) or None
        self.hidden_layers.append(self.Layer(dim, prev_dim, activation))

    def add_output_layer(self, dim, activation):
        self.output_activation = activation
        input_dim = (hasattr(self.hidden_layers[-1], 'weights') and len(self.hidden_layers[-1].weights)) or None
        self.output_layer = self.Layer(dim, input_dim, self.output_activation)

    def set_loss_fn(self, loss_fn=None):
        if loss_fn and (self.output_activation == 'logistic') and (loss_fn != 'log_loss'):
            raise ValueError('Output activation function "logistic" compatible only with loss function "log_loss"')
        self.loss_fn = (self.output_activation == 'logistic' and 'log_loss') or loss_fn or 'mse'
        return self.loss_fn

    def _shuffle_obs(self):
        # Apparently the most performant option.
        # I'm nervous about counting on np.take(vals, permuted_idx, axis=0, out=vals) not to mess vals up
        obs = np.hstack((self.input_vals, self.target_vals))
        np.random.shuffle(obs)
        self.input_vals = obs[:, :self.input_vals.shape[1]]
        self.target_vals = obs[:, self.input_vals.shape[1]:]

    def _feed_forward(self, input_vals):
        z = input_vals.T
        for layer in self.hidden_layers:
            z = layer.feed(z)
        self.output_layer.feed(z)

    def _back_propagate(self, target_vals):
        self.output_layer.compute_errors(t=target_vals.T)
        next_layer = self.output_layer
        for layer in self.hidden_layers[::-1]:
            layer.compute_errors(next_layer)
            next_layer = layer

    @staticmethod
    def _compute_gradient(errors, vals):
        # where errors is vector with # elements of # nodes in layer, vals is vector of dim # nodes in previous layer
        return np.matmul(errors.reshape(len(errors), -1), [vals])

    def _update_weights(self, input_vals):
        prev_vals = input_vals
        for layer in self.hidden_layers:
            layer_shape = layer.errors.shape
            if len(layer_shape) > 1:
                for obs in range(layer_shape[1]):
                    gradient = self._compute_gradient(layer.errors[:, obs], prev_vals[obs])
                    layer.adjust_weights(self.momentum, self.learning_rate * gradient)
            else:
                gradient = self._compute_gradient(layer.errors, prev_vals)
                layer.adjust_weights(self.momentum, self.learning_rate * gradient)
            prev_vals = layer.vals.T

    def _compute_loss(self):
        return { 'mse': mean_squared_error, 'log_loss': log_loss }[self.loss_fn](self.target_vals, self.output_layer.vals.T)

    def fit(self, x, y, store_losses=False, verbose=False):
        self.input_vals, self.target_vals = x, y
        if self.fit_intercept:
            self.input_vals = np.hstack((np.ones((len(x),1)), x))
        self.hidden_layers[0].init_weights(self.input_vals.shape[1])
        self.output_activation = self.output_activation or ((utils.is_numeric(y, int_as_numeric=False) and 'identity') or 'logistic')
        if hasattr(self, 'output_layer'):
            self.output_layer.init_weights(len(self.hidden_layers[-1].weights))
        else:
            self.add_output_layer(y.shape[1], self.output_activation)
        self.loss_fn = self.loss_fn or self.set_loss_fn()
        if store_losses:
            self.losses = []
        self._feed_forward(self.input_vals)
        loss = self._compute_loss()
        epochs, n_epochs_no_change, last_diff, n = 0, 0, None, self.input_vals.shape[0]
        while (n_epochs_no_change < self.n_epochs_no_change) and (epochs < self.max_epochs):
            self._shuffle_obs()
            i = 0
            while i < n:
                end = min(i + self.samples_per_cycle, n)
                self._feed_forward(self.input_vals[i:end])
                self._back_propagate(self.target_vals[i:end])
                self._update_weights(self.input_vals[i:end])
                i = end
            self._feed_forward(self.input_vals)
            next_loss = self._compute_loss()
            if abs(loss - next_loss) < self.tol:
                n_epochs_no_change += 1
            loss = next_loss
            if store_losses:
                self.losses.append(loss)
            if verbose:
                print('epoch %d: loss = %f' % (epoch, loss))
            epochs += 1
        del self.input_vals, self.target_vals
        return self

    def predict(self, x):
        self._feed_forward(x)
        return self.output_layer.vals.T

