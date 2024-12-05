
import numpy as np # type: ignore

class softmax:
	
    def __init__(self, inputMatrix, labelMatrix):
        self.inputMatrix = inputMatrix
        self.labelMatrix = labelMatrix
    
    def set_input_matrix(self, inputMatrix):
        self.inputMatrix = inputMatrix
    
    def set_label_matrix(self, labelMatrix):
        self.labelMatrix = labelMatrix

    ## Function to calculate the softmax
    ## @param weightMatrix: The weight matrix
    ## @return: The softmax
    def softmax_function(self, weightMatrix):
        z = np.dot(self.inputMatrix, weightMatrix)
        exp_z = np.exp(z)

        denominator = np.sum(exp_z, axis=1)

        softmax = exp_z / denominator[:, None]

        return softmax
    
    ## Function to calculate the softmax loss
    ## @param weightMatrix: The weight matrix
    ## @return: The softmax loss
    def loss_function_of_softmax(self, weightMatrix):

        softmax = self.get_softmax(weightMatrix)

        log_softmax = np.log(softmax)

        loss = -np.sum(np.multiply(self.labelMatrix, log_softmax))

        return loss

    ## Function to calculate the gradient of the softmax
    ## @param weightMatrix: The weight matrix
    ## @return: The gradient of the loss softmax
    def gradient_of_loss(self, weightMatrix):
        softmax = self.get_softmax(weightMatrix)

        gradient = np.dot(self.inputMatrix.T, softmax - self.labelMatrix) / self.inputMatrix.shape[0]

        return gradient

    ## Gradient test of the result
    ## @param weightMatrix: The weight matrix
    ## @return: The gradient test result
    def gradient_test_for_loss_function(self, weightMatrix):

        epsilon = 1e-5
        d = self.get_normlized_random_vector(weightMatrix)
        calculated_gradient = self.gradient_of_loss(weightMatrix)
        estimated_gradient = self.loss_function_of_softmax(weightMatrix + epsilon * d) - self.loss_function_of_softmax(weightMatrix)
    
    def get_normlized_random_vector(self, weightMatrix):
        d = np.random.rand(weightMatrix.shape[0], weightMatrix.shape[1])
        norm = np.linalg.norm(d)
        d_normalized = d / norm
        return d_normalized






        


