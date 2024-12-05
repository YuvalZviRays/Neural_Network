from abc import ABC, abstractmethod
import numpy as np # type: ignore
import logging

class ObjectiveFunction(ABC):
    """
    Abstract base class for any objective function.
    This serves as an interface for gradient-based optimization tasks.
    """

    def __init__(self, sampleMatrix, labelMatrix):
        self.sampleMatrix = sampleMatrix
        self.labelMatrix = labelMatrix   

    def set_sample_matrix(self, sampleMatrix):
        self.sampleMatrix = sampleMatrix
    
    def set_label_matrix(self, labelMatrix):
        self.labelMatrix = labelMatrix
    
    def get_sample_matrix(self):
        return self.sampleMatrix
    
    def get_label_matrix(self):
        return self.labelMatrix

    @abstractmethod
    def function(self, weightMatrix):
        """
        Compute the function value.
        :param weightMatrix: The weight matrix.
        :return: The function value.
        """
        return self.loss_function(weightMatrix)

    @abstractmethod
    def loss_function(self, weightMatrix):
        """
        Abstract method to compute the loss.
        Subclasses must implement this method.
        """
        pass

    @abstractmethod
    def gradient_of_loss_on_weight(self, weightMatrix):
        """
        Abstract method to compute the gradient of the loss.
        Subclasses must implement this method.
        """
        pass

    def gradient_test_for_loss_function_on_weight(self, weightMatrix, epsilon):
        """
        Perform a gradient test to verify the implementation of the gradient.
        :param weightMatrix: The weight matrix.
        :param epsilon: Perturbation value for the gradient test.
        """
        # Step 1: Generate normalized random vector d
        d = self.get_normalized_random_vector(weightMatrix)

        # Step 2: Compute numerical approximation using finite differences
        f_w_plus_eps_d = self.loss_function(weightMatrix + epsilon * d)
        f_w = self.loss_function(weightMatrix)
        zero_order_approximation = f_w_plus_eps_d - f_w

        # Step 3: Compute analytical approximation using the gradient
        calculated_gradient = self.gradient_of_loss_on_weight(weightMatrix)
        first_order_approximation = epsilon * np.sum(calculated_gradient * d)

        # Step 4: Compute the differences
        first_degree_approximation = np.abs(zero_order_approximation - first_order_approximation)
        zero_order_approximation = np.abs(zero_order_approximation)

        return zero_order_approximation, first_degree_approximation

    def get_normalized_random_vector(self, weightMatrix):
        """
        Generate a normalized random vector for the gradient test.
        :param weightMatrix: The weight matrix.
        :return: Normalized random vector.
        """
        d = np.random.rand(weightMatrix.shape[0], weightMatrix.shape[1])
        norm = np.linalg.norm(d)
        return d / norm