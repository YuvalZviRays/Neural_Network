import numpy as np # type: ignore
import logging
from Hidden_Layer_Functions.classical_hidden_layer import Classical_Hidden_Layer
from neural_networks.standard_neural_network import Standard_Neural_Network
from output_functions.ObjectiveFunction import ObjectiveFunction
import output_functions.Softmax as Softmax


class Neural_Network(ObjectiveFunction):
    def __init__(self, input_dim, hidden_layer_dim_list, activation_function, output_dim):

        hidden_layer_functions = []

        curr_dim = input_dim
        
        # Initialize neural network
        for i in range(len(hidden_layer_dim_list)):
            weight_matrix = np.random.randn(hidden_layer_dim_list[i], curr_dim) * np.sqrt(2 / curr_dim)  # He initialization
            bias_vector = np.random.randn(hidden_layer_dim_list[i], 1) * np.sqrt(2 / curr_dim)  # He initialization
            hidden_layer_functions.append(Classical_Hidden_Layer(weight_matrix, activation_function, bias_vector))
            curr_dim = hidden_layer_dim_list[i]
        
        self.neural_network = Standard_Neural_Network(hidden_layer_functions)
        
        # Initialize output layer
        output_layer_weight_matrix = np.random.randn(curr_dim, output_dim) * np.sqrt(1 / curr_dim)  # Xavier initialization
        self.output_layer_function = Softmax.Softmax(output_layer_weight_matrix)
    
    def set_sample_matrix(self, sample_matrix):
        self.neural_network.sample_matrix = sample_matrix
    
    def set_label_matrix(self, label_matrix):
        self.output_layer_function.label_matrix = label_matrix

    def function(self):
        # Forward pass

        output = self.neural_network.function()
        self.output_layer_function.sample_matrix = output
        output = self.output_layer_function.function()

        return output
    
    def loss_function(self):
        # Forward pass

        output = self.neural_network.function()
        self.output_layer_function.sample_matrix = output
        output = self.output_layer_function.loss_function()

        return output

    def gradient_of_loss_on_weight(self):
        # Backward pass
            grad_nn = self.neural_network.jackTMV(self.output_layer_function.gradient_of_loss_on_samples()).flatten()
            grad_out = self.output_layer_function.gradient_of_loss_on_weight().flatten()
            return np.concatenate([grad_nn, grad_out])
    
    def gradient_of_loss_on_samples(self, test_input, epsilon):
        pass
    
    def get_weights(self):
        params = []
        params.append(self.output_layer_function.get_weights().flatten())
        params.append(self.neural_network.get_weights().flatten())

        return np.concatenate(params)

    def set_weights(self, param_vector):
        nn_weights_size = self.neural_network.get_weights().size
        nn_params = param_vector[:nn_weights_size]
        out_params = param_vector[nn_weights_size:]
        
        self.neural_network.set_weights(nn_params)
        correct_shape = self.output_layer_function.weight_matrix.shape
        self.output_layer_function.set_weights(out_params.reshape(correct_shape))
    
    def gradient_test_on_weight(self, epsilon):

        # Step 1: Generate normalized random vector d
        d = self.get_normalized_random_vector()

        # Step 2: Compute numerical approximation using finite differences
        original_weight_matrix = self.get_weights()
        self.set_weights(original_weight_matrix + epsilon * d)
        f_w_plus_eps_d = self.loss_function()
        self.set_weights(original_weight_matrix)
        f_w = self.loss_function()
        zero_order_approximation = f_w_plus_eps_d - f_w

        # Step 3: Compute analytical approximation using the gradient
        calculated_gradient = self.gradient_of_loss_on_weight()
        first_order_approximation = np.dot(calculated_gradient, d * epsilon)

        # Step 4: Compute the differences
        first_degree_approximation = np.abs(zero_order_approximation - first_order_approximation)
        zero_order_approximation = np.abs(zero_order_approximation)

        return zero_order_approximation, first_degree_approximation
    
    def get_normalized_random_vector(self):
        params = self.get_weights()
        d = np.random.rand(params.size)
        return d / np.linalg.norm(d)