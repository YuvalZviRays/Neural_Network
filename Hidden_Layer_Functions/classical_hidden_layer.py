import numpy as np # type: ignore
import logging

class Classical_Hidden_Layer():
    def __init__(self, weight_matrix, activation_function, bias_vector):
        self.weight_matrix = weight_matrix
        self.activation_function = activation_function
        self.sample_matrix = None 
        self.bias_vector = bias_vector
    
    def set_input(self, input):
        self.sample_matrix = input
    
    def set_input_for_test(self, input):
        self.sample_matrix = input
    
    def function(self):
        return self.activation_function.function(self.weight_matrix, self.sample_matrix, self.bias_vector)
    
    def jacobian_of_weight(self):
        return self.activation_function.jacobian_of_weight(self.weight_matrix, self.sample_matrix, self.bias_vector)
    
    def jacobian_of_samples(self):
        return self.activation_function.jacobian_of_samples(self.weight_matrix, self.sample_matrix, self.bias_vector)
    
    def derivative(self):
        return self.activation_function.derivative(self.weight_matrix, self.sample_matrix, self.bias_vector)
    
    def jacobian_test_for_weight(self, epsilon):
        d = self.get_normalized_random_vector(self.sample_matrix)
        weight_matrix_original = self.weight_matrix
        self.weight_matrix = self.weight_matrix + epsilon * d

        # Step 2: Compute numerical approximation using finite differences
        f_w_plus_eps_d = self.function()
        self.weight_matrix = weight_matrix_original
        f_w = self.function()
        zero_order_approximation = f_w_plus_eps_d - f_w

        # Step 3: Compute analytical approximation using the jacobian
        calculated_jacobian = self.jacobian_of_weight()
        first_order_approximation = np.dot(calculated_jacobian, epsilon * d)

        # Step 4: Compute the differences
        first_degree_approximation = np.abs(np.sum(zero_order_approximation - first_order_approximation))
        zero_order_approximation = np.abs(np.sum(zero_order_approximation))

        return zero_order_approximation, first_degree_approximation
    
    def jacobian_test_for_samples(self, epsilon):
        d = self.get_normalized_random_vector(self.sample_matrix)
        sample_matrix_original = self.sample_matrix
        self.sample_matrix = self.sample_matrix + epsilon * d

        # Step 2: Compute numerical approximation using finite differences
        f_w_plus_eps_d = self.function()
        self.sample_matrix = sample_matrix_original
        f_w = self.function()
        zero_order_approximation = f_w_plus_eps_d - f_w

        # Step 3: Compute analytical approximation using the jacobian
        calculated_jacobian = self.jacobian_of_samples()
        logging.info(f"calculated_jacobian: {calculated_jacobian}")
        first_order_approximation = np.dot(calculated_jacobian, epsilon * d)

        # Step 4: Compute the differences
        first_degree_approximation = np.abs(np.sum(zero_order_approximation - first_order_approximation))
        zero_order_approximation = np.abs(np.sum(zero_order_approximation))

        return zero_order_approximation, first_degree_approximation
    

    def get_normalized_random_vector(self, matrix):
        """
        Generate a normalized random vector for the jacobian test.
        :param weightMatrix: The weight matrix.
        :return: Normalized random vector.
        """
        d = np.random.rand(matrix.shape[0],1)
        norm = np.linalg.norm(d)
        return d / norm

        