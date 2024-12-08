import numpy as np
import logging
from output_functions.ObjectiveFunction import ObjectiveFunction

class LeastSquaresObjectiveFunction(ObjectiveFunction):
    def __init__(self, sampleMatrix, labelMatrix):
        self.sampleMatrix = sampleMatrix
        self.labelMatrix = labelMatrix
    
    def function(self, weightMatrix):
        return self.loss_function(weightMatrix)

    def loss_function(self, weightMatrix):
        predictions = np.dot(self.sampleMatrix, weightMatrix)
        residuals = predictions - self.labelMatrix
        loss = np.mean(residuals ** 2)  # Mean squared error
        return loss

    def gradient_of_loss_on_weight(self, weightMatrix):
        predictions = np.dot(self.sampleMatrix, weightMatrix)
        residuals = predictions - self.labelMatrix
        gradient = 2 * np.dot(self.sampleMatrix.T, residuals) / self.sampleMatrix.shape[0]
        return gradient