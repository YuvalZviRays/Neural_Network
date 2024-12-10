from abc import ABC, abstractmethod
import numpy as np  # type: ignore
from Hidden_Layer_Functions.activation_function import Activation_function

class ReLU(Activation_function):

    def function(self, weight_matrix, input_matrix, bias_vector):
        return np.maximum(0, np.dot(weight_matrix, input_matrix) + bias_vector)
    
    def jacobian_of_weight(self, weight_matrix, input_matrix, bias_vector):
        mask = (np.dot(weight_matrix, input_matrix) + bias_vector > 0).astype(float)
        return np.outer(mask, input_matrix.T)  
    
    def jacobian_of_samples(self, weight_matrix, input_matrix, bias_vector):
        mask = (np.dot(weight_matrix, input_matrix) + bias_vector > 0).astype(float)
        return weight_matrix * mask
    
    def derivative(self, weight_matrix, input_matrix, bias_vector):
        return ((np.dot(weight_matrix, input_matrix) + bias_vector) > 0).astype(float)