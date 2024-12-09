from abc import ABC, abstractmethod
import numpy as np # type: ignore
import logging

class Neural_Network(ABC):
    """
    Abstract base class for any objective function.
    This serves as an interface for gradient-based optimization tasks.
    """

    def __init__(self, hidden_layers, sample_matrix):
        self.hidden_layers = hidden_layers
        self.sample_matrix = sample_matrix

    @abstractmethod
    def function(self):
        """
        Compute the function value.
        propogated through the network to result in an output
        """
        pass

    @abstractmethod
    def jack_on_weights(self):
        """
        compute the Jacobian-vector product
        Does this by propogating the input through the network and then multiplying the result by the input vector
        :param v: The vector to multiply the Jacobian with.
        :return: The product.

        """

    @abstractmethod
    def jackTMV(self):
        """
        compute the Jacobian-vector product
        Does this by propogating the input through the network and then multiplying the result by the input vector
        :param v: The vector to multiply the Jacobian with.
        :return: The product.

        """

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

    def jacobian_test_for_weight(self, epsilon):
        """
        Test the Jacobian-vector product for the weight matrix.
        """
        # Initialize the perturbation vector
        perturbation = np.random.randn(self.get_weights().size)
        perturbation = perturbation / np.linalg.norm(perturbation)
        original_weights = self.get_weights()   

        # Compute the Jacobian-vector product
        self.set_weights(original_weights + epsilon * perturbation)
        f_w_plus_epsilon = self.function()
        self.set_weights(original_weights)
        f_w = self.function()

        zero_order_approximations = f_w_plus_epsilon - f_w

        jack_product = self.jack_on_weights().dot(perturbation * epsilon)

        first_order_approximations = zero_order_approximations - jack_product

        return np.abs(np.sum(zero_order_approximations)), np.abs(np.sum(first_order_approximations))