import numpy as np # type: ignore
import logging

class SGD:
    def __init__(self, objectiveFunction, learningRate, maxEpochs, batchSize):
        self.objectiveFunction = objectiveFunction
        self.sampleMatrix = objectiveFunction.get_sample_matrix()
        self.labelMatrix = objectiveFunction.get_label_matrix()
        self.momentum = 0.9
        self.batchSize = batchSize
        self.learningRate = learningRate
        self.maxEpochs = maxEpochs
        self.num_samples = self.sampleMatrix.shape[1]

    def single_epoch_optimize(self, weightMatrix, epoch_number):
        losses = []
        self.velocity = np.zeros_like(weightMatrix)
        
        # Shuffle the data
        shuffled_indices = np.random.permutation(self.num_samples)

        for start in range(0, self.num_samples, self.batchSize):
            end = start + self.batchSize
            indices = shuffled_indices[start:end]
            batch_sample = self.sampleMatrix[:, indices]
            batch_labels = self.labelMatrix[indices, :]

            # Temporarily set batch data in the objective function
            self.objectiveFunction.set_sample_matrix(batch_sample)
            self.objectiveFunction.set_label_matrix(batch_labels)

            # Compute gradient and velocity
            gradient = self.objectiveFunction.gradient_of_loss_on_weight(weightMatrix)
            self.velocity = self.momentum * self.velocity + (1 - self.momentum) * gradient

            # Update weights
            weightMatrix = weightMatrix - self.learningRate * self.velocity

            # Compute and log loss on the mini-batch
            batch_loss = self.objectiveFunction.loss_function(weightMatrix)
            losses.append(batch_loss)
            
        return weightMatrix, losses

    def optimize(self, initialWeightMatrix, testSampleMatrix, testLabelMatrix):
        weightMatrix = initialWeightMatrix
        all_losses = []
        sucess_precentage_train = []
        sucess_precentage_test = []

        for epoch in range(self.maxEpochs):
            weightMatrix, epoch_losses = self.single_epoch_optimize(weightMatrix, epoch)
            sucess_precentage_train_epoch, sucess_precentage_test_epoch = self.calculate_success_precentages(weightMatrix, testSampleMatrix, testLabelMatrix)
            sucess_precentage_train.append(sucess_precentage_train_epoch)
            sucess_precentage_test.append(sucess_precentage_test_epoch)
            all_losses += epoch_losses

        return weightMatrix, all_losses, sucess_precentage_train, sucess_precentage_test
    
    def calculate_success_precentages(self, weightMatrix, testSampleMatrix, testLabelMatrix):

        # Temporarily set train data in the objective function 
        indices = np.random.choice(self.num_samples, min(self.batchSize, self.num_samples), replace=False)
        batch_inputs = self.sampleMatrix[:, indices]
        batch_labels = self.labelMatrix[indices ,:]
        self.objectiveFunction.set_sample_matrix(batch_inputs)
        self.objectiveFunction.set_label_matrix(batch_labels)

        #compute the sucess precentage for train data
        function_output = self.objectiveFunction.function(weightMatrix)
        sucess_precentage_train = (np.sum(np.argmax(function_output, axis=1) == np.argmax(batch_labels, axis=1)) / self.batchSize) * 100

        # Temporarily set test data in the objective function 
        test_size = testSampleMatrix.shape[1]
        indices = np.random.choice(test_size, min(self.batchSize, test_size), replace=False)
        batch_inputs = testSampleMatrix[:, indices]
        batch_labels = testLabelMatrix[indices ,:]
        self.objectiveFunction.set_sample_matrix(batch_inputs)
        self.objectiveFunction.set_label_matrix(batch_labels)

        #compute the sucess precentage
        function_output = self.objectiveFunction.function(weightMatrix)
        sucess_precentage_test = (np.sum(np.argmax(function_output, axis=1) == np.argmax(batch_labels, axis=1)) / self.batchSize) * 100

        return sucess_precentage_train, sucess_precentage_test

