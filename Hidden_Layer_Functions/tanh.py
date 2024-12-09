from abc import ABC, abstractmethod
import numpy as np # type: ignore
import logging
from Hidden_Layer_Functions.activation_function import Activation_function

class Tanh(Activation_function):

    def function(self, weight_matrix, input_matrix, bias_vector):
        return np.tanh(np.dot(weight_matrix, input_matrix) + bias_vector)

    def jacobian_of_weight(self, weight_matrix, input_matrix, bias_vector):
        return np.dot(input_matrix, (1 - np.tanh(np.dot(weight_matrix, input_matrix) + bias_vector )**2).T)

    def jacobian_of_samples(self, weight_matrix, input_matrix, bias_vector):
        return weight_matrix * (1 - np.tanh(np.dot(weight_matrix, input_matrix) + bias_vector)**2)
    
    def derivative(self, weight_matrix, input_matrix, bias_vector):
        return 1 - np.tanh(np.dot(weight_matrix, input_matrix) + bias_vector)**2