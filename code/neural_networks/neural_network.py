from abc import ABC, abstractmethod
import numpy as np # type: ignore
import logging

class Neural_Network(ABC):
    """

    A class representing a neural network without the final softmax layer.
    """

    def __init__(self, hidden_layers):
        self.hidden_layers = hidden_layers

    @abstractmethod
    def function(self):
        """
        Compute the function value.
        propogated through the network to result in an output
        """
        pass
    
    @abstractmethod
    def jackMV(self, p):
        """
        compute the Jacobian-vector product
        Does this by propogating the input through the network and then multiplying the result by the input vector
        :param v: The vector to multiply the Jacobian with.
        :return: The product.

        """

    @abstractmethod
    def jackTMV(self):
        """
        compute the Jacobian-vector transpose product
        Does this by propogating the input through the network and then multiplying the result by the input vector
        :param v: The vector to multiply the Jacobian with.
        :return: The product.

        """

    @abstractmethod
    def get_weights(self):
        """
        Get the parameters of the neural network ordered in a vector.
        :return: The parameters.
        """
        pass

    @abstractmethod
    def set_weights(self, weight_matrix):
        """

        Set the parameters of the neural network from a vector.
        :param weight_matrix: The parameters.
        """
        pass

    def jacobian_test_for_weight(self, epsilon):
        """
        Test the Jacobian-vector product for the weight matrix.
        """
        # Initialize the perturbation vector size is the same as the number of weights
        v = np.random.randn(self.get_weights().size)
        v = v / np.linalg.norm(v)
        original_weights = self.get_weights()   

        # Compute the Jacobian-vector product
        self.set_weights(original_weights + epsilon * v)
        f_w_plus_epsilon = self.function()
        self.set_weights(original_weights)
        f_w = self.function()

        zero_order_approximations = f_w_plus_epsilon - f_w

        jack_product = self.jackMV(v * epsilon)
        first_order_approximations = zero_order_approximations - jack_product

        # A random variable to test the transpose test, size is the same as the number of output neurons
        f_w = self.function()
        u = np.random.randn(*f_w.shape)
        u = u / np.linalg.norm(u)

        transpose_test = np.abs(np.dot(u.T, jack_product) - np.dot(v.T, self.jackTMV(u)))
        logging.info(f"Transpose test: {transpose_test}")

        return np.abs(np.sum(zero_order_approximations)), np.abs(np.sum(first_order_approximations))