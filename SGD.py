
import numpy as np # type: ignore
import logging

class SGD:
    def __init__(self, objectiveFunction, learningRate, maxIterations):
        self.objectiveFunction = objectiveFunction
        self.learningRate = learningRate
        self.maxIterations = maxIterations
        self.sampleMatrix = objectiveFunction.get_sample_matrix()
        self.labelMatrix = objectiveFunction.get_label_matrix()
        self.momentum = 0.9
        self.batchSize = 10

    def optimize(self, initialWeightMatrix):

        losses = []

        weightMatrix = initialWeightMatrix

        num_samples = self.objectiveFunction.sampleMatrix.shape[0]

        self.velocity = np.zeros_like(initialWeightMatrix)

        for i in range(self.maxIterations):
            # Step 1: Randomly sample a mini-batch
            indices = np.random.choice(num_samples, self.batchSize, replace=False)
            batch_sample = self.sampleMatrix[indices]
            batch_labels = self.labelMatrix[indices]

            # Step 2: Temporarily set the batch data in the objective function
            self.objectiveFunction.set_sample_matrix(batch_sample)
            self.objectiveFunction.set_label_matrix(batch_labels)

            # Step 3: Compute the velocity on the mini-batch
            gradient = self.objectiveFunction.gradient_of_loss_on_weight(weightMatrix)
            self.velocity = self.momentum * self.velocity + (1 - self.momentum) * gradient         

            # Step 4: Update the weights
            weightMatrix  = weightMatrix - self.learningRate * self.velocity

            # Step 5: Log the loss on the entire dataset for monitoring
            full_loss = self.objectiveFunction.loss_function(weightMatrix)
            losses.append(full_loss)
            logging.info(f"Iteration: {i}, Loss: {full_loss}")

        return weightMatrix , losses
    
