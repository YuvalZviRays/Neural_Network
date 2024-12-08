from abc import ABC, abstractmethod
import numpy as np # type: ignore
import logging
from Hidden_Layer_Functions.activation_function import Activation_function

class Tanh(Activation_function):

    def function(self, weight_matrix, input_matrix):
        return np.tanh(np.dot(weight_matrix, input_matrix))

    def jacobian_of_weight(self, weight_matrix, input_matrix):
        return np.dot(input_matrix, (1 - np.tanh(np.dot(weight_matrix, input_matrix))**2).T)

    def jacobian_of_samples(self, weight_matrix, input_matrix):
        return weight_matrix * (1 - np.tanh(np.dot(weight_matrix, input_matrix))**2)