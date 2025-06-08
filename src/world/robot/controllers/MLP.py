from typing import List

import numpy as np

import numpy as np

class InputNormalizer:
    """
    Tracks running mean/variance for each input dimension and provides
    a method to normalize new inputs.
    """
    def __init__(self, dim, eps=1e-8):
        self.n = 0
        self.mean = np.zeros(dim)
        self.mean_sq = np.zeros(dim)
        self.eps = eps

    def observe(self, x: np.ndarray):
        """Update running statistics with a new sample x."""
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self.mean_sq += delta * delta2

    def normalize(self, x: np.ndarray) -> np.ndarray:
        """Return the normalized version of x."""
        if self.n < 2:
            return x  # avoid division by zero early on
        var = (self.mean_sq / (self.n - 1)).clip(min=self.eps)
        std = np.sqrt(var)
        return (x - self.mean) / std

class NumpyNetwork:
    """
    One-hidden-layer network with:
    - Softplus hidden activations
    - Tanh output
    - Xavier initialization
    - Bias terms
    - Built-in input normalization
    """
    def __init__(self, n_input: int, n_hidden: int, n_output: int):
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_output = n_output

        # Xavier/Glorot initialization for weights
        lim1 = np.sqrt(6 / (n_input + n_hidden))
        self.lin = np.random.uniform(-lim1, lim1, (n_hidden, n_input))
        self.b1  = np.zeros(n_hidden)

        lim2 = np.sqrt(6 / (n_hidden + n_output))
        self.output = np.random.uniform(-lim2, lim2, (n_output, n_hidden))
        self.b2     = np.zeros(n_output)

        # Parameter counts for genotype vector
        self.n_con1 = n_input * n_hidden
        self.n_b1   = n_hidden
        self.n_con2 = n_hidden * n_output
        self.n_b2   = n_output
        self.n_params = self.n_con1 + self.n_b1 + self.n_con2 + self.n_b2

        # Input normalizer
        self.normalizer = InputNormalizer(n_input)

    def set_weights(self, weights: np.ndarray):
        """Unpack flat weight vector into weight matrices and biases."""
        assert len(weights) == self.n_params
        idx = 0
        self.lin = weights[idx:idx + self.n_con1].reshape(self.lin.shape)
        idx += self.n_con1
        self.b1 = weights[idx:idx + self.n_b1]
        idx += self.n_b1
        self.output = weights[idx:idx + self.n_con2].reshape(self.output.shape)
        idx += self.n_con2
        self.b2 = weights[idx:idx + self.n_b2]

    def forward(self, state: np.ndarray) -> np.ndarray:
        """
        Normalize inputs, compute hidden layer with Softplus,
        and produce final actions with Tanh.
        """
        # 1) update normalization stats
        self.normalizer.observe(state)

        # 2) normalize
        x = self.normalizer.normalize(state)

        # 3) hidden layer: Softplus
        z1 = self.lin.dot(x) + self.b1
        hid = np.log1p(np.exp(z1))  # Softplus: log(1 + e^z)

        # 4) output layer: Tanh
        z2 = self.output.dot(hid) + self.b2
        out = np.tanh(z2)
        return out

# Example usage in your controller:
class NNController:
    def __init__(self, n_states: int, n_actions: int):
        self.model = NumpyNetwork(n_states, n_hidden=int(1.3 * n_states), n_output=n_actions)
        self.n_params = self.model.n_params

    def geno2pheno(self, genotype: np.ndarray):
        self.model.set_weights(genotype)

    def get_action(self, state: np.ndarray) -> np.ndarray:
        return self.model.forward(state)

# Now CMA-ES will optimize a vector of length `controller.n_params`:
# controller = NNController(n_states=10, n_actions=4)
# print("Total parameters to optimize:", controller.n_params)


# class NumpyNetwork:
#     def __init__(self, n_input: int, n_hidden: int, n_output: int):
#         """
#         A minimalistic Neural Network, using numpy.
#         - One hidden layer: SoftReLU [0, inf]
#         - Output layer: sigmoid (0, 1)

#         :param int n_input: Size of input vector
#         :param int n_hidden: Size of hidden layer
#         :param int n_output: Size of output vector
#         """
#         n_hidden = n_hidden
#         self.n_con1 = n_input * n_hidden
#         self.n_con2 = n_hidden * n_output
#         self.lin = np.random.uniform(-1, 1, (n_hidden, n_input))
#         self.output = np.random.uniform(-1, 1, (n_output, n_hidden))

#     def set_weights(self, weights: np.array):
#         """
#         Set weights of NN.

#         :param np.array weights: Vector of weights
#         """
#         assert len(weights) == self.n_con1 + self.n_con2, f"Got {len(weights)} but expected {self.n_con1 + self.n_con2}"
#         weight_matrix1 = weights[:self.n_con1].reshape(self.lin.shape)
#         weight_matrix2 = weights[-self.n_con2:].reshape(self.output.shape)
#         self.lin = weight_matrix1
#         self.output = weight_matrix2

#     def forward(self, state: np.array):
#         hid_l = np.tanh(np.dot(self.lin, state))
#         output_l = np.tanh(np.dot(self.output, hid_l))
#         return output_l


# class NumpyNetwork_najaro:
#     def __init__(self, n_input: int, n_hidden: int, n_output: int):
#         """
#         A minimalistic Neural Network, using numpy.
#         - One hidden layer: SoftReLU [0, inf]
#         - Output layer: sigmoid (0, 1)

#         :param int n_input: Size of input vector
#         :param int n_hidden: Size of hidden layer
#         :param int n_output: Size of output vector
#         """
#         self.n_con1 = n_input * n_input
#         self.n_con2 = n_input * n_input
#         self.n_con3 = n_input * n_output
#         self.lin1 = np.random.uniform(-1, 1, (n_input, n_input))
#         self.lin2 = np.random.uniform(-1, 1, (n_input, n_input))
#         self.output = np.random.uniform(-1, 1, (n_output, n_input))

#     def set_weights(self, weights: np.array):
#         """
#         Set weights of NN.

#         :param np.array weights: Vector of weights
#         """
#         assert len(weights) == self.n_con1 + self.n_con2 + self.n_con3, f"Got {len(weights)} but expected {self.n_con1 + self.n_con2 + self.n_con3}"
#         weight_matrix1 = weights[:self.n_con1].reshape(self.lin1.shape)
#         weight_matrix2 = weights[self.n_con1:-self.n_con3].reshape(self.lin2.shape)
#         weight_matrix3 = weights[-self.n_con3:].reshape(self.output.shape)
#         self.lin1 = weight_matrix1
#         self.lin2 = weight_matrix2
#         self.output = weight_matrix3

#     def forward(self, state: np.array):
#         hid_l = np.tanh(np.dot(self.lin1, state))
#         hid_l = np.tanh(np.dot(self.lin2, hid_l))
#         output_l = np.tanh(np.dot(self.output, hid_l))
#         return output_l

# class NNController():
#     def __init__(self, n_states, n_actions):
#         self.controller_type = "NN"
#         self.n_input = n_states
#         self.n_output = n_actions
#         # print(n_states)
#         self.model = NumpyNetwork(n_states, int(1.3*n_states), n_actions)
#         # self.model = NumpyNetwork(n_states, 64, n_actions)
#         self.n_params = self.model.n_con1 + self.model.n_con2

#     def geno2pheno(self, genotype: np.array):
#         self.model.set_weights(genotype)

#     def get_action(self, state: np.ndarray) -> np.ndarray:
#         """
#         Given a state, give an appropriate action

#         :param <np.array> state: A single observation of the current state, dimension is (state_dim)
#         :return: np.ndarray action: A vector of motor inputs
#         """

#         assert (state.shape[0] == self.n_input), "State does not correspond with expected input size"
#         raw_action = self.model.forward(state)
#         action = np.clip(raw_action, -1.0, 1.0)
#         return action



class NN_najaroController():
    def __init__(self, n_states, n_actions):
        self.controller_type = "NN"
        self.n_input = n_states
        self.n_output = n_actions
        self.model = NumpyNetwork_najaro(n_states, n_states, n_actions)
        self.n_params = self.model.n_con1 + self.model.n_con2 + self.model.n_con3

    def geno2pheno(self, genotype: np.array):
        self.model.set_weights(genotype)

    def get_action(self, state: np.ndarray) -> np.ndarray:
        """
        Given a state, give an appropriate action

        :param <np.array> state: A single observation of the current state, dimension is (state_dim)
        :return: np.ndarray action: A vector of motor inputs
        """

        assert (len(state) == self.n_input), "State does not correspond with expected input size"
        action = self.model.forward(state)
        return action

