import numpy as np # type: ignore
import logging
from output_functions.ObjectiveFunction import ObjectiveFunction

class Softmax(ObjectiveFunction):

    ## Function to calculate the softmax
    ## @param weightMatrix: The weight matrix
    ## @return: The softmax
    def function(self):
        z = np.dot(self.sample_matrix.T, self.weight_matrix)
        exp_z = np.exp(z)

        denominator = np.sum(exp_z, axis=1, keepdims=True)

        softmax = exp_z / denominator

        return softmax
    
    ## Function to calculate the softmax loss
    ## @param weightMatrix: The weight matrix
    ## @return: The softmax loss
    def loss_function(self):

        softmax = self.function()

        log_softmax = np.log(softmax)

        loss = -np.sum(self.label_matrix * log_softmax) / self.sample_matrix.shape[0]

        return loss

    ## Function to calculate the gradient of the loss of softmax function on the weight
    ## @param weightMatrix: The weight matrix
    ## @return: The gradient of the loss softmax on the weight
    def gradient_of_loss_on_weight(self):
        softmax = self.function()
        number_of_samples = self.sample_matrix.shape[0]

        gradient = np.dot(self.sample_matrix, softmax - self.label_matrix) / number_of_samples

        return gradient
    
    ## Function to calculate the gradient test for the softmax loss function on the samples
    ## @param weightMatrix: The weight matrix
    ## @return: The gradient of the loss softmax on the samples
    def gradient_of_loss_on_samples(self):
        softmax = self.function()
        number_of_samples = self.sample_matrix.shape[0]

        gradient = np.dot(self.weight_matrix, (softmax - self.label_matrix).T) / number_of_samples

        return gradient
    
    ## Function to get the parameters of the softmax function
    ## @return: The parameters of the softmax function
    def get_weights(self):
        return self.weight_matrix

    ## Function to set the parameters of the softmax function   
    def set_weights(self, weight_matrix):
        self.weight_matrix = weight_matrix