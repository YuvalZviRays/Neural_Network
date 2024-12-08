import numpy as np # type: ignore
import logging
from output_functions.ObjectiveFunction import ObjectiveFunction
from output_functions.Softmax import Softmax
from Hidden_Layer_Functions.Hidden_Layer_Function import Hidden_Layer_Function

class Neural_Network():
    def __init__(self, sample_matrix_dim, label_matrix, hidden_layer_size, hidden_layer_dim, activation_function):

        self.label_matrix = label_matrix
        self.hidden_layer_functions = []
        self.hidden_layers_size = hidden_layer_size

        curr_dim = sample_matrix_dim + 1 # +1 for bias
        
        # Initialize hidden layers
        for i in range(hidden_layer_size):
            weight_matrix = np.random.randn(curr_dim, hidden_layer_dim[i]) * np.sqrt(2 / curr_dim)  # He initialization
            self.hidden_layer_functions.append(Hidden_Layer_Function(weight_matrix, activation_function))
            curr_dim = hidden_layer_dim[i] + 1 # +1 for bias
        
        # Initialize output layer
        output_dim = self.label_matrix.shape[1]
        self.output_layer_function = Softmax(label_matrix)
        self.output_layer_weight_matrix = np.random.randn(curr_dim - 1, output_dim) * np.sqrt(1 / curr_dim)  # Xavier initialization
    
    def function(self, sample_matrix):
        # Forward pass
        output = sample_matrix 
        # Forward pass through hidden layers and set the output as the input for the next layer
        for i in range(self.hidden_layers_size):
            hidden_layer_function = self.hidden_layer_functions[i]
            hidden_layer_function.set_input(output)
            output = hidden_layer_function.function()

        # Forward pass through output layer   
        self.output_layer_function.set_sample_matrix(output)
        output = self.output_layer_function.function(self.output_layer_weight_matrix)
        return output
    
    def loss_function (self, sample_matrix):
        # Forward pass
        output = sample_matrix 
        # Forward pass through hidden layers and set the output as the input for the next layer
        for i in range(self.hidden_layers_size):
            hidden_layer_function = self.hidden_layer_functions[i]
            hidden_layer_function.set_input(output)
            output = hidden_layer_function.function()


        # Forward pass through output layer   
        self.output_layer_function.set_sample_matrix(output)
        output = self.output_layer_function.loss_function(self.output_layer_weight_matrix)
        return output
    
    def jacobian(self):
        # Output layer gradient of weights:
        output_grad_w = self.output_layer_function.gradient_of_loss_on_weight(self.output_layer_weight_matrix)
        # Output layer delta for the previous layer:
        delta = np.sum(self.output_layer_function.gradient_of_loss_on_samples(self.output_layer_weight_matrix), axis = 1, keepdims = True)

        jacobian = [output_grad_w]

        for i in range(len(self.hidden_layer_functions)-1, -1, -1):
            # Compute the gradient of the weights

            grad_w = self.hidden_layer_functions[i].gradient_of_weight().T * delta
            jacobian.append(grad_w)


            curr = np.sum(self.hidden_layer_functions[i].gradient_of_samples(), axis = 1, keepdims = True)
            logging.info(f"curr shape: {curr.shape}")
            curr = curr[:-1, :]  # remove the bias 

            # Now compute new delta = gradient_of_samples * old delta
            delta = np.sum(np.dot(curr, delta) , axis = 1, keepdims = True)

        return np.hstack([j.ravel() for j in jacobian])
    
    def get_parameters(self):
        params = []
        for layer in self.hidden_layer_functions:
            params.append(layer.weight_matrix.ravel())
        params.append(self.output_layer_weight_matrix.ravel())
        return np.concatenate(params)

    def set_parameters(self, param_vector):
        start = 0
        for layer in self.hidden_layer_functions:
            size = layer.weight_matrix.size
            layer.weight_matrix = param_vector[start:start+size].reshape(layer.weight_matrix.shape)
            start += size
        size = self.output_layer_weight_matrix.size
        self.output_layer_weight_matrix = param_vector[start:start+size].reshape(self.output_layer_weight_matrix.shape)
    
    def jacobian_test(self, test_input, epsilon):
        # Get current parameters
        params = self.get_parameters()

        # Get random direction in parameter space
        d = self.get_normalized_random_vector()

        # Compute function outputs with and without parameter perturbation
        self.set_parameters(params + epsilon * d)
        f_perturbed = self.loss_function(test_input)

        self.set_parameters(params)  # restore original parameters
        f_original = self.loss_function(test_input)

        # zero_degree_approximation (numerical gradient estimate)
        zero_degree_approximation = f_perturbed - f_original

        jac = self.jacobian()
        analytical_approx = np.dot(jac, epsilon * d)

        first_degree_approximation = zero_degree_approximation - analytical_approx

        return np.abs(zero_degree_approximation), np.abs(first_degree_approximation)
    
    def get_normalized_random_vector(self):
        params = self.get_parameters()
        d = np.random.rand(params.size)
        return d / np.linalg.norm(d)