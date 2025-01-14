from abc import ABC, abstractmethod
import numpy as np # type: ignore
import logging

class Activation_Function(ABC):
    """
    Abstract base class for any activation function.
    This serves as an interface for gradient-based optimization tasks.
    """

    @abstractmethod
    def function(self, weightMatrix):
        """
        Compute the function value.
        :param weightMatrix: The weight matrix.
        :return: The function value.
        """
        return self.loss_function(weightMatrix)
    
    @abstractmethod
    def jacobian_of_weight(self, weightMatrix):
        """
        Compute the gradient of the function with respect to the weight matrix.
        :param weightMatrix: The weight matrix.
        :return: The gradient of the function with respect to the weight matrix.
        """
        return self.loss_function_gradient(weightMatrix)
    
    @abstractmethod
    def jacobian_of_samples(self, weightMatrix):
        """
        Compute the gradient of the function with respect to the samples.
        :param weightMatrix: The weight matrix.
        :return: The gradient of the function with respect to the samples.
        """
        return self.loss_function_gradient(weightMatrix)
    
    @abstractmethod
    def derivative(self, weightMatrix):
        """
        Compute the derivative of the function.
        :param weightMatrix: The weight matrix.
        :return: The derivative of the function.
        """
        return self.loss_function_gradient(weightMatrix)