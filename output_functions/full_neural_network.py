import numpy as np # type: ignore
import logging
from Hidden_Layer_Functions.Hidden_Layer_Function import Hidden_Layer_Function
from neural_networks.neural_network import Neural_Network
from output_functions.ObjectiveFunction import ObjectiveFunction
import Softmax


class Neural_Network(ObjectiveFunction):
    def __init__(self, sample_matrix_dim, label_matrix, hidden_layer_dim_list, activation_function):

        self.sample_matrix_dim = sample_matrix_dim

        # Initialize hidden layers
        for i in range(hidden_layer_dim_list):
            weight_matrix = np.random.randn(hidden_layer_dim_list[i], curr_dim) * np.sqrt(2 / curr_dim)  # He initialization
            bias_vector = np.random.randn(hidden_layer_dim_list[i], 1) * np.sqrt(2 / curr_dim)  # He initialization
            self.hidden_layer_functions.append(Hidden_Layer_Function(weight_matrix, activation_function, bias_vector))
            curr_dim = hidden_layer_dim_list[i]
        
        # Initialize output layer
        output_dim = self.label_matrix.shape[1]
        self.output_layer_function = Softmax(label_matrix)
        self.output_layer_weight_matrix = np.random.randn(curr_dim, output_dim) * np.sqrt(1 / curr_dim)  # Xavier initialization

    # def function(self, sample_matrix):
    #     # Forward pass
    #     output = sample_matrix 
    #     logging.info(f"output : {output}")
    #     # Forward pass through hidden layers and set the output as the input for the next layer
    #     for i in range(self.hidden_layers_size):
    #         hidden_layer_function = self.hidden_layer_functions[i]
    #         hidden_layer_function.set_input(output)
    #         output = hidden_layer_function.function()

    #     # Forward pass through output layer   
    #     self.output_layer_function.set_sample_matrix(output)
    #     output = self.output_layer_function.function(self.output_layer_weight_matrix)
    #     return output
    
    # def loss_function(self, sample_matrix):
    #     # Forward pass
    #     output = sample_matrix

    #     # Forward pass through hidden layers and set the output as the input for the next layer
    #     for i in range(self.hidden_layers_size):
    #         hidden_layer_function = self.hidden_layer_functions[i]
    #         hidden_layer_function.set_input(output)
    #         output = hidden_layer_function.function()

    #     # Forward pass through output layer   
    #     self.output_layer_function.set_sample_matrix(output)
    #     output = self.output_layer_function.loss_function(self.output_layer_weight_matrix)

    #     return output

    # def gradient_of_loss_on_weight(self):
    #     output_grad_w = self.output_layer_function.gradient_of_loss_on_weight(self.output_layer_weight_matrix)
    #     # Output layer delta for the previous layer:
    #     delta = self.output_layer_function.gradient_of_loss_on_samples(self.output_layer_weight_matrix)

    #     gradient = [output_grad_w]

    #     for i in range(len(self.hidden_layer_functions)-1, -1, -1):

    #         hidden_layer = self.hidden_layer_functions[i]
    #         grad_w = np.dot(hidden_layer.derivative() * delta, hidden_layer.sample_matrix.T)
    #         gradient.append(grad_w)

    #         grad_b = np.sum(hidden_layer.derivative() * delta, axis=1).reshape(-1, 1)
    #         gradient.append(grad_b)

    #         delta = np.dot(hidden_layer.weight_matrix.T, hidden_layer.derivative() * delta)
        
    #     return np.hstack([j.ravel() for j in gradient])
    
    # def gradient_of_loss_on_samples(self, test_input, epsilon):
    #     return self.gradient_test_on_weight(test_input, epsilon)
    
    # def get_weights(self):
    #     params = []
    #     for layer in self.hidden_layer_functions:
    #         params.append(layer.weight_matrix.ravel())
    #         params.append(layer.bias_vector.ravel())
    #     params.append(self.output_layer_weight_matrix.ravel())
    #     return np.concatenate(params)

    # def set_weights(self, param_vector):
    #     start = 0
    #     for layer in self.hidden_layer_functions:
    #         weight_size = layer.weight_matrix.size
    #         bias_size = layer.bias_vector.size

    #         layer.weight_matrix = param_vector[start:start+weight_size].reshape(layer.weight_matrix.shape)
    #         start += weight_size
    #         layer.bias_vector = param_vector[start:start+bias_size].reshape(layer.bias_vector.shape)
    #         start += bias_size
    #     size = self.output_layer_weight_matrix.size
    #     self.output_layer_weight_matrix = param_vector[start:start+size].reshape(self.output_layer_weight_matrix.shape)
    
    # def gradient_test_on_weight(self, test_input, epsilon):
    #     # Get current parameters
    #     params = self.get_parameters()

    #     # Get random direction in parameter space
    #     d = self.get_normalized_random_vector()

    #     # Compute function outputs with and without parameter perturbation
    #     self.set_parameters(params + epsilon * d)
    #     f_perturbed = self.loss_function(test_input)

    #     self.set_parameters(params)  # restore original parameters
    #     f_original = self.loss_function(test_input)

    #     # zero_degree_approximation (numerical gradient estimate)
    #     zero_degree_approximation = f_perturbed - f_original

    #     grad = self.gradient_of_loss_on_weight()
    #     logging.info(f"grad : {grad}")
    #     analytical_approx = epsilon * np.dot(d.T, grad)

    #     first_degree_approximation = zero_degree_approximation - analytical_approx

    #     return np.abs(zero_degree_approximation), np.abs(first_degree_approximation)
    
    # def get_normalized_random_vector(self):
    #     params = self.get_parameters()
    #     d = np.random.rand(params.size)
    #     return d / np.linalg.norm(d)