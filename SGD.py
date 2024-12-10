import numpy as np # type: ignore
import logging

class SGD:
    def __init__(self, objective_function, learning_rate, max_epochs, batch_size, sample_matrix, label_matrix):
        self.objective_function = objective_function
        self.sample_matrix = sample_matrix
        self.label_matrix = label_matrix
        self.momentum = 0.9
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.num_samples = self.sample_matrix.shape[1]
        self.weights = objective_function.get_weights()

    def single_epoch_optimize(self, epoch):
        losses = []
        self.velocity = np.zeros_like(self.weights)
        
        # Shuffle the data
        shuffled_indices = np.random.permutation(self.num_samples)

        for start in range(0, self.num_samples, self.batch_size):
            end = start + self.batch_size
            indices = shuffled_indices[start:end]
            batch_sample = self.sample_matrix[:, indices]
            batch_labels = self.label_matrix[indices, :]

            # Temporarily set batch data in the objective function
            self.objective_function.set_sample_matrix(batch_sample)
            self.objective_function.set_label_matrix(batch_labels)

            self.objective_function.function()

            # Compute gradient and velocity
            gradient = self.objective_function.gradient_of_loss_on_weight()
            self.velocity = self.momentum * self.velocity + (1 - self.momentum) * gradient

            # Update weights
            self.weights = self.weights - self.learning_rate * self.velocity

            self.objective_function.set_weights(self.weights)

            # Compute and log loss on the mini-batch
            batch_loss = self.objective_function.loss_function()
            losses.append(batch_loss)
            
        return self.weights, losses

    def optimize_with_precentages(self, test_sample_matrix, test_label_matrix):
        all_losses = []
        sucess_precentage_train = []
        sucess_precentage_test = []

        for epoch in range(self.max_epochs):
            self.weights, epoch_losses = self.single_epoch_optimize(epoch)
            sucess_precentage_train_epoch, sucess_precentage_test_epoch = self.calculate_success_precentages(test_sample_matrix, test_label_matrix)
            sucess_precentage_train.append(sucess_precentage_train_epoch)
            sucess_precentage_test.append(sucess_precentage_test_epoch)
            all_losses += epoch_losses

        return self.weights, all_losses, sucess_precentage_train, sucess_precentage_test
    
    def optimize(self):
        all_losses = []

        for epoch in range(self.max_epochs):
            self.weights, epoch_losses = self.single_epoch_optimize(epoch)
            all_losses += epoch_losses

        return self.weights, all_losses
    
    def calculate_success_precentages(self, test_sample_matrix, test_label_matrix):

        # Temporarily set train data in the objective function 
        indices = np.random.choice(self.num_samples, min(self.batch_size, self.num_samples), replace=False)
        batch_inputs = self.sample_matrix[:, indices]
        batch_labels = self.label_matrix[indices ,:]
        self.objective_function.set_sample_matrix(batch_inputs)
        self.objective_function.set_label_matrix(batch_labels)

        #compute the sucess precentage for train data
        function_output = self.objective_function.function()
        sucess_precentage_train = (np.sum(np.argmax(function_output, axis=1) == np.argmax(batch_labels, axis=1)) / self.batch_size) * 100

        # Temporarily set test data in the objective function 
        test_size = test_sample_matrix.shape[1]
        indices = np.random.choice(test_size, min(self.batch_size, test_size), replace=False)
        batch_inputs = test_sample_matrix[:, indices]
        batch_labels = test_label_matrix[indices ,:]
        self.objective_function.set_sample_matrix(batch_inputs)
        self.objective_function.set_label_matrix(batch_labels)

        #compute the sucess precentage
        function_output = self.objective_function.function()
        sucess_precentage_test = (np.sum(np.argmax(function_output, axis=1) == np.argmax(batch_labels, axis=1)) / self.batch_size) * 100

        return sucess_precentage_train, sucess_precentage_test

