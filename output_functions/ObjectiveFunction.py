from abc import ABC, abstractmethod
import numpy as np # type: ignore
import logging

class ObjectiveFunction(ABC):
    """
    Abstract base class for any objective function.
    This serves as an interface for gradient-based optimization tasks.
    """

    def __init__(self, sample_matrix, label_matrix, weight_matrix):
        self.sample_matrix = sample_matrix
        self.label_matrix = label_matrix   
        self.weight_matrix = weight_matrix

    @abstractmethod
    def function(self):
        """
        Compute the function value.
        :param weightMatrix: The weight matrix.
        :return: The function value.
        """

    @abstractmethod
    def loss_function(self):
        """
        Abstract method to compute the loss.
        Subclasses must implement this method.
        """
        pass

    @abstractmethod
    def gradient_of_loss_on_weight(self):
        """
        Abstract method to compute the gradient of the loss.
        Subclasses must implement this method.
        """
        pass

    @abstractmethod
    def gradient_of_loss_on_samples(self):
        """
        Abstract method to compute the gradient of the loss.
        Subclasses must implement this method.
        """
        pass

    @abstractmethod
    def get_weights(self):
        """
        Get the parameters of the objective function.
        :return: The parameters.
        """
        pass

    @abstractmethod
    def set_weights(self, weight_matrix):
        """
        Get the parameters of the objective function.
        :return: The parameters.
        """
        pass

    def gradient_test_for_loss_function_on_weight(self, epsilon):
        """
        Perform a gradient test to verify the implementation of the gradient.
        :param weightMatrix: The weight matrix.
        :param epsilon: Perturbation value for the gradient test.
        """
        # Step 1: Generate normalized random vector d
        d = self.get_normalized_random_vector(self.weight_matrix)

        # Step 2: Compute numerical approximation using finite differences
        original_weight_matrix = self.weight_matrix
        self.weight_matrix = self.weight_matrix + epsilon * d
        f_w_plus_eps_d = self.loss_function()
        self.weight_matrix = original_weight_matrix
        f_w = self.loss_function()
        zero_order_approximation = f_w_plus_eps_d - f_w

        # Step 3: Compute analytical approximation using the gradient
        calculated_gradient = self.gradient_of_loss_on_weight()
        first_order_approximation = epsilon * np.sum(calculated_gradient * d)

        # Step 4: Compute the differences
        first_degree_approximation = np.abs(zero_order_approximation - first_order_approximation)
        zero_order_approximation = np.abs(zero_order_approximation)

        return zero_order_approximation, first_degree_approximation
    
    def gradient_test_for_loss_function_on_samples(self, epsilon):
        """
        Perform a gradient test to verify the implementation of the gradient.
        :param weightMatrix: The weight matrix.
        :param epsilon: Perturbation value for the gradient test.
        """
        # Step 1: Generate normalized random vector d
        d = self.get_normalized_random_vector(self.sample_matrix)
        sample_matrix_original = self.sample_matrix
        self.sample_matrix = self.sample_matrix + epsilon * d

        # Step 2: Compute numerical approximation using finite differences
        f_w_plus_eps_d = self.loss_function()
        self.sample_matrix = sample_matrix_original
        f_w = self.loss_function()
        zero_order_approximation = f_w_plus_eps_d - f_w

        # Step 3: Compute analytical approximation using the gradient
        calculated_gradient = self.gradient_of_loss_on_samples()
        first_order_approximation = epsilon * np.sum(calculated_gradient * d)

        # Step 4: Compute the differences
        first_degree_approximation = np.abs(zero_order_approximation - first_order_approximation)
        zero_order_approximation = np.abs(zero_order_approximation)

        return zero_order_approximation, first_degree_approximation

    def get_normalized_random_vector(self, matrix):
        """
        Generate a normalized random vector for the gradient test.
        :param weightMatrix: The weight matrix.
        :return: Normalized random vector.
        """
        d = np.random.rand(matrix.shape[0], matrix.shape[1])
        norm = np.linalg.norm(d)
        return d / norm