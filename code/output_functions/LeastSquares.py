import numpy as np
import logging
from output_functions.ObjectiveFunction import ObjectiveFunction

class LeastSquaresObjectiveFunction(ObjectiveFunction):
    
    def function(self):
        return self.loss_function()

    def loss_function(self):
        predictions = np.dot(self.sample_matrix.T, self.weight_matrix)
        residuals = predictions - self.label_matrix
        loss = np.mean(residuals ** 2)  # Mean squared error
        return loss

    def gradient_of_loss_on_weight(self):
        predictions = np.dot(self.sample_matrix.T, self.weight_matrix)
        residuals = predictions - self.label_matrix
        gradient = 2 * np.dot(self.sample_matrix, residuals) / self.sample_matrix.shape[0]
        return gradient
    
    def get_weights(self):
        return self.weight_matrix
    
    def set_weights(self, weight_matrix):
        self.weight_matrix = weight_matrix
    
    def set_sample_matrix(self, sample_matrix):
        self.sample_matrix = sample_matrix
    
    def set_label_matrix(self, label_matrix):
        self.label_matrix = label_matrix

    def gradient_of_loss_on_samples(self):
        pass