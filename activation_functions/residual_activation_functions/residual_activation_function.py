from abc import ABC, abstractmethod
import numpy as np # type: ignore
import logging

class Residual_Activation_Function(ABC):
    """
    Abstract base class for any activation function.
    This serves as an interface for gradient-based optimization tasks.
    """

    @abstractmethod
    def function(self, weight_matrix_1,weight_matrix_2, input_matrix, bias_vector):
        """
        Compute the function value.
        :param weightMatrix: The weight matrix.
        :return: The function value.
        """
        pass
    
    @abstractmethod
    def jacobian_of_weight_1(self, weight_matrix_1, weight_matrix_2, input_matrix, bias_vector):
        """
        Compute the gradient of the function with respect to the inner weight matrix.
        :param weightMatrix: The weight matrix.
        :return: The gradient of the function with respect to the weight matrix.
        """
        pass

    @abstractmethod
    def jacobian_of_weight_2(self, weight_matrix_1, weight_matrix_2, input_matrix, bias_vector):
        """
        Compute the gradient of the function with respect to the outer weight matrix.
        :param weightMatrix: The weight matrix.
        :return: The gradient of the function with respect to the weight matrix.
        """
        pass
    
    @abstractmethod
    def jacobian_of_samples(self, weight_matrix_1, weight_matrix_2, input_matrix, bias_vector):
        """
        Compute the gradient of the function with respect to the samples.
        :param weightMatrix: The weight matrix.
        :return: The gradient of the function with respect to the samples.
        """
        pass
    
    # @abstractmethod
    # def derivative(self, weight_matrix_1, weight_matrix_2, input_matrix, bias_vector):
    #     """
    #     Compute the derivative of the function.
    #     :param weightMatrix: The weight matrix.
    #     :return: The derivative of the function.
    #     """
    #     pass