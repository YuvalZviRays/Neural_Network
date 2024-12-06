# Import necessary libraries
import os
import sys
import logging
import numpy as np # type: ignore
import Softmax 
import matplotlib.pyplot as plt # type: ignore
from SGD import SGD
from LeastSquares import LeastSquaresObjectiveFunction
from scipy.io import loadmat # type: ignore

class Tester:


    def test_sgd_on_softmax(self):
        # Step 1: Load data from the .mat file
        mat_data = loadmat('data_sets/PeaksData.mat')
        
        # Extract matrices
        training_features = mat_data['Yt']  # Transpose: features × samples
        training_labels = mat_data['Ct'].T  # Training labels (class)
        test_features = mat_data['Yv']  # Transpose: features × samples
        test_labels = mat_data['Cv'].T  # Validation labels (class)

        # Step 3: Initialize the objective function
        objective_function = Softmax.Softmax(training_features, training_labels)

        # Step 4: Initialize SGD optimizer with hyperparameters
        learning_rate = 0.00001
        max_iterations = 100
        batch_size = 100
        sgd_optimizer = SGD(objective_function, learning_rate, max_iterations, batch_size)

        # Step 5: Initialize weights randomly
        num_features = training_features.shape[0]  # Features are rows now
        initial_weights = np.random.randn(num_features, training_labels.shape[1])
        logging.info(f"Initial weights: {initial_weights}")

        # Step 6: Optimize the weights
        optimized_weights, losses, success_percentage_train, success_percentage_test = sgd_optimizer.optimize(
            initial_weights, test_features, test_labels
        )

        # Plot both success percentages on the same graph
        plt.figure(figsize=(10, 6))
        plt.plot(range(len(success_percentage_train)), success_percentage_train, label="Training Accuracy", color="blue", linewidth=2)
        plt.plot(range(len(success_percentage_test)), success_percentage_test, label="Validation Accuracy", color="red", linewidth=2)

        # Add gridlines
        plt.grid(alpha=0.5, linestyle="--")

        # Add title and labels
        plt.title("Training and Validation Accuracy per Epoch", fontsize=16, fontweight="bold")
        plt.xlabel("Epochs", fontsize=12)
        plt.ylabel("Accuracy (%)", fontsize=12)

        # Add legend
        plt.legend(loc="lower right", fontsize=12)

        # Customize x and y ticks
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)

        # Adjust margins for better spacing
        plt.tight_layout()

        # Show the plot
        plt.show()

        # Log final results
        logging.info(f"Final Train Success %: {success_percentage_train[-1]}")
        logging.info(f"Final Test Success %: {success_percentage_test[-1]}")
        logging.info(f"Optimized Weights: \n{optimized_weights}")


    def test_sgd_on_least_squares(self):
        # Step 1: Create synthetic data
        np.random.seed(88)
        num_samples = 100
        num_features = 2

        # Generate random data
        X = np.random.rand(num_samples, num_features)
        true_weights = np.array([250.0, -50.0])  # True weights for testing
        y = np.dot(X, true_weights) + np.random.normal(scale=0.1, size=num_samples)

        # Reshape y for consistency
        y = y.reshape(-1, 1)

        # Step 2: Initialize the objective function
        objective_function = LeastSquaresObjectiveFunction(X, y)

        # Step 3: Initialize SGD optimizer
        learning_rate = 0.2
        max_iterations = 30
        batch_size = 10
        sgd_optimizer = SGD(objective_function, learning_rate, max_iterations, batch_size)

        # Step 4: Initialize weights randomly
        initial_weights = np.random.randn(num_features, 1)

        logging.info(f"Initial weights: {initial_weights}")
        
        optimized_weights , losses = sgd_optimizer.optimize(initial_weights)

        # Plot convergence of loss
        plt.figure()
        plt.plot(range(len(losses)), losses, marker="o")
        plt.xlabel("Iteration")
        plt.ylabel("Loss (Mean Squared Error)")
        plt.title("Convergence of SGD on Least Squares")
        plt.grid()
        plt.show()

        logging.info(f"True weights: {true_weights}")
        logging.info(f"Optimized weights: {optimized_weights}")
        

    # gradient test for softmax loss function
    def softmax_loss_function_gradient_test(self):
        inputMatrix = np.array([[1, 4, 7, 1, 0],  # Feature 1
                                [2, 5, 8, 0, 1],  # Feature 2
                                [3, 6, 9, 0, 0]])  # Feature 3
        
        labelMatrix = np.array([[1, 0, 0], 
                                [0, 1, 0], 
                                [0, 0, 1], 
                                [1, 0, 0], 
                                [0, 1, 0]])

        softmax_model = Softmax.Softmax(inputMatrix, labelMatrix)

        weightMatrix = np.array([[0.1, 0.2, 0.3], 
                                [0.4, 0.5, 0.6], 
                                [0.7, 0.8, 0.9]])

        epsilon_values = np.logspace(1, 8, 8)  # Epsilon values
        first_degree_approximations = []
        second_degree_approximations = []

        # Assume softmax_model.gradient_test_for_loss_function calculates and returns approximations
        for epsilon in epsilon_values:
            softmax_model.epsilon = epsilon
            first, second = softmax_model.gradient_test_for_loss_function_on_weight(weightMatrix,1/epsilon)
            first_degree_approximations.append(first)
            second_degree_approximations.append(second)

        # Plotting the approximations on a semilogarithmic scale
        plt.figure()
        plt.loglog(epsilon_values, first_degree_approximations, label='Zero Order Approximation', marker='o')
        plt.loglog(epsilon_values, second_degree_approximations, label='First Order Approximation', marker='s')
        plt.xlabel('Epsilon (log scale)')
        plt.ylabel('Approximation/Error (log scale)')
        plt.title('Gradient Test For Softmax Loss Function')
        plt.legend()
        plt.show()