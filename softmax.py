import numpy as np # type: ignore
import logging
from ObjectiveFunction import ObjectiveFunction

class Softmax(ObjectiveFunction):
	
    def __init__(self, sampleMatrix, labelMatrix):
        self.sampleMatrix = sampleMatrix
        self.labelMatrix = labelMatrix

    ## Function to calculate the softmax
    ## @param weightMatrix: The weight matrix
    ## @return: The softmax
    def function(self, weightMatrix):
        z = np.dot(self.sampleMatrix.T, weightMatrix)
        exp_z = np.exp(z)

        denominator = np.sum(exp_z, axis=1, keepdims=True)

        softmax = exp_z / denominator

        return softmax
    
    ## Function to calculate the softmax loss
    ## @param weightMatrix: The weight matrix
    ## @return: The softmax loss
    def loss_function(self, weightMatrix):

        softmax = self.function(weightMatrix)

        log_softmax = np.log(softmax)

        loss = -np.sum(self.labelMatrix * log_softmax) / self.sampleMatrix.shape[0]

        return loss

    ## Function to calculate the gradient of the softmax
    ## @param weightMatrix: The weight matrix
    ## @return: The gradient of the loss softmax
    def gradient_of_loss_on_weight(self, weightMatrix):
        softmax = self.function(weightMatrix)
        number_of_samples = self.sampleMatrix.shape[0]

        gradient = np.dot(self.sampleMatrix, softmax - self.labelMatrix) / number_of_samples

        return gradient