# Import necessary libraries
import os
import sys
import logging
import numpy as np # type: ignore
import output_functions.Softmax as Softmax 
import matplotlib.pyplot as plt # type: ignore
from SGD import SGD
from output_functions.LeastSquares import LeastSquaresObjectiveFunction

from activation_functions.classical_activation_functions.classical_tanh import Classical_Tanh
from activation_functions.classical_activation_functions.classical_ReLU import Classical_ReLU
from Hidden_Layer_Functions.classical_hidden_layer import Classical_Hidden_Layer

from activation_functions.residual_activation_functions.residual_tanh import Residual_Tanh
from Hidden_Layer_Functions.residual_block import Residual_Block



from neural_networks.standard_neural_network import Standard_Neural_Network
from full_neural_network import Neural_Network
from scipy.io import loadmat # type: ignore

class Tester:

         #=============================================Task 2.2.5 test===================================================
    def test_neural_network_on_200_samples(self, hidden_layer_dim, activation_function):

        mat_data = loadmat('data_sets/GMMData.mat')
        
        # Extract full training data
        full_training_features = mat_data['Yt']  
        full_training_labels = mat_data['Ct'].T  

        # Randomly sample 200 data points from the training set
        sample_indices = np.random.choice(full_training_features.shape[1], 200, replace=False)
        training_features = full_training_features[:, sample_indices] 
        training_labels = full_training_labels[sample_indices, :]  

        test_features = mat_data['Yv']  
        test_labels = mat_data['Cv'].T 

        # Determine input and output dimensions
        input_dim = training_features.shape[0]  # Number of input features
        output_dim = training_labels.shape[1]  # Number of output classes

        if activation_function == 'tanh':
            activation_function = Classical_Tanh()
        elif activation_function == 'ReLU':
            activation_function = Classical_ReLU() 

        # Step 3: Initialize the objective function
        objective_function = Neural_Network(input_dim, hidden_layer_dim, activation_function, output_dim)

        # Step 4: Initialize SGD optimizer with hyperparameters
        learning_rate = 0.1
        max_iterations = 200
        batch_size = 5
        sgd_optimizer = SGD(objective_function, learning_rate, max_iterations, batch_size, training_features, training_labels)

        # Step 6: Optimize the weights
        optimized_weights, losses, success_percentage_train, success_percentage_test = sgd_optimizer.optimize_with_precentages(
            test_features, test_labels
        )

        # Plot both success percentages on the same graph
        plt.figure(figsize=(10, 6))
        plt.plot(range(len(success_percentage_train)), success_percentage_train, label="Training Accuracy", color="blue", linewidth=2)
        plt.plot(range(len(success_percentage_test)), success_percentage_test, label="Validation Accuracy", color="red", linewidth=2)

        # Add gridlines
        plt.grid(alpha=0.5, linestyle="--")

        # Add title and labels
        plt.title("Training and Validation Accuracy per Epoch on 200 samples", fontsize=16, fontweight="bold")
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

     #=============================================Task 2.2.4 test===================================================
    def test_neural_network(self, hidden_layer_dim, activation_function):
            # Step 1: Load data from the .mat file
            mat_data = loadmat('data_sets/GMMData.mat')
            
            # Extract matrices
            training_features = mat_data['Yt']  # Transpose: features × samples
            training_labels = mat_data['Ct'].T  # Training labels (class)
            test_features = mat_data['Yv']  # Transpose: features × samples
            test_labels = mat_data['Cv'].T  # Validation labels (class)
            num_features = training_features.shape[0]  # Features are rows now

            if activation_function == 'tanh':
                activation_function = Classical_Tanh()
            if activation_function == 'ReLU':
                activation_function = Classical_ReLU()
            
            input_dim = training_features.shape[0]  # Number of input features
            output_dim = training_labels.shape[1]  # Number of output classes

            # Step 3: Initialize the objective function
            objective_function = Neural_Network(input_dim, hidden_layer_dim, activation_function, output_dim)

            # Step 4: Initialize SGD optimizer with hyperparameters
            learning_rate = 0.003
            max_iterations =  100
            batch_size = 100
            sgd_optimizer = SGD(objective_function, learning_rate, max_iterations, batch_size, training_features, training_labels)

            # Step 6: Optimize the weights
            optimized_weights, losses, success_percentage_train, success_percentage_test = sgd_optimizer.optimize_with_precentages(
                test_features, test_labels
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

    #=============================================Task 2.2.3 test===================================================
    def gradient_test_FF_neural_network(self):
        """
        Perform a gradient test for the feedforward neural network.
        """
        # Step 1: Initialize a small test feedforward neural network
        input_dim = 2  # Number of input features
        hidden_layer_dim = [3, 2]  # Dimensions of each hidden layer
        output_dim = 2  # Number of output classes

        # Create random input and labels for testing
        test_input = np.random.rand(input_dim, 1)  # 1 samples with input_dim features
        test_labels = np.eye(output_dim)[np.random.choice(output_dim, 1)]  # Random one-hot encoded label

        logging.info(f"Test input: \n{test_input}")
        logging.info(f"Test labels: \n{test_labels}")

        # Initialize the neural network
        activation_function = Classical_Tanh()  # Use the tanh activation function
        nn_model = Neural_Network(input_dim, hidden_layer_dim, activation_function, output_dim)

        nn_model.set_sample_matrix(test_input)
        nn_model.set_label_matrix(test_labels)

        # Step 2: Generate epsilon values for gradient testing
        epsilon_values = np.logspace(1, 8, 8)  # Epsilon values from 10^-8 to 10^-1
        zero_order_approximations = []
        first_order_approximations = []

        # Step 3: Iterate through epsilon values and compute approximations
        for epsilon in epsilon_values:
            zero_order, first_order = nn_model.gradient_test_on_weight(1/epsilon)
            # Aggregate errors (e.g., take the mean across all samples and outputs)
            zero_order_approximations.append(zero_order)
            first_order_approximations.append(first_order)

        # Step 4: Plot the results
        plt.figure(figsize=(10, 6))
        plt.loglog(epsilon_values, zero_order_approximations, label="Zero-Order Approximation", marker="o", linewidth=2)
        plt.loglog(epsilon_values, first_order_approximations, label="First-Order Approximation", marker="s", linewidth=2)

        # Add labels, title, and legend
        plt.xlabel("Epsilon (log scale)", fontsize=12)
        plt.ylabel("Error (log scale)", fontsize=12)
        plt.title("Gradient Test for full Feedforward Neural Network", fontsize=16, fontweight="bold")
        plt.legend(fontsize=12)

        # Add gridlines and show plot
        plt.grid(alpha=0.5, linestyle="--")
        plt.show()
    
       #=============================================Task 2.2.2 test===================================================
        
    def jacobian_test_for_residual_neural_network(self):
                # a single hidden layer with 3 neurons and 3 features
        sample_matrix = np.array([[1],  # Feature 1
                                [2],  # Feature 2
                                [3]])  # Feature 3
        
        weight_matrix_1 = np.array([[0.1, 0.2, 0.3], 
                                [0.4, 0.5, 0.6], 
                                [0.7, 0.8, 0.9]])
        
        weight_matrix_2 = np.array([[0.1, 0.2, 0.3], 
                                [0.4, 0.5, 0.6], 
                                [0.7, 0.8, 0.9]])
        
        bias_vector = np.array([[0.1],
                                [0.2],
                                [0.3]])
        tanh_model = Residual_Tanh()
        
        hidden_layer_1 = Classical_Hidden_Layer(weight_matrix_1, tanh_model, bias_vector)
        hidden_layer_2 = Classical_Hidden_Layer(weight_matrix_2, tanh_model, bias_vector)
        hidden_layers = [hidden_layer_1, hidden_layer_2]

        neural_network = Standard_Neural_Network(hidden_layers)
        neural_network.sample_matrix = sample_matrix

        epsilon_values = np.logspace(1, 8, 8)  # Epsilon values
        first_degree_approximations_weight = []
        second_degree_approximations_weight = []

        # Assume softmax_model.gradient_test_for_loss_function calculates and returns approximations
        for epsilon in epsilon_values:

            first, second = neural_network.jacobian_test_for_weight(1/epsilon)
            first_degree_approximations_weight.append(first)
            second_degree_approximations_weight.append(second)
        
        # Plotting the approximations on a semilogarithmic scale
        plt.figure()
        plt.loglog(epsilon_values, first_degree_approximations_weight, label='Zero Order Approximation', marker='o')
        plt.loglog(epsilon_values, second_degree_approximations_weight, label='First Order Approximation', marker='s')
        plt.xlabel('Epsilon (log scale)')
        plt.ylabel('Approximation/Error (log scale)')
        plt.title('Jacobian Test For standard Neural Network on Weights')
        plt.legend()
        plt.show()
    
    def jacobian_test_residuel_block(self):
        # a single hidden layer with 3 neurons and 3 features
        sample_matrix = np.array([[1],  # Feature 1
                                [2],  # Feature 2
                                [3]])  # Feature 3
        
        weight_matrix_1 = np.array([[0.1, 0.2, 0.3], 
                                [0.4, 0.5, 0.6], 
                                [0.7, 0.8, 0.9]])
        
        weight_matrix_2 = np.array([[0.1, 0.2, 0.3], 
                                [0.4, 0.5, 0.6], 
                                [0.7, 0.8, 0.9]])
        
        bias_vector = np.array([[0.1],
                                [0.2],
                                [0.3]])
        
        tanh_model = Residual_Tanh()
        
        hidden_layer = Residual_Block(weight_matrix_1, weight_matrix_2, tanh_model, bias_vector)

        hidden_layer.set_input(sample_matrix)

        test = hidden_layer.function()
        logging

        epsilon_values = np.logspace(1, 8, 8)  # Epsilon values
        first_degree_approximations_weight_1 = []
        second_degree_approximations_weight_1 = []

        first_degree_approximations_weight_2 = []
        second_degree_approximations_weight_2 = []

        first_degree_approximations_samples = []
        second_degree_approximations_samples = []

        d_weight_1 = hidden_layer.get_normalized_random_vector(weight_matrix_1)
        d_weight_2 = hidden_layer.get_normalized_random_vector(weight_matrix_2)
        d_sample = hidden_layer.get_normalized_random_vector(sample_matrix)

        # Assume softmax_model.gradient_test_for_loss_function calculates and returns approximations
        for epsilon in epsilon_values:

            # first, second = hidden_layer.jacobian_test_for_weight_1(1/epsilon, d_weight_1)
            # first_degree_approximations_weight_1.append(first)
            # second_degree_approximations_weight_1.append(second)

            first, second = hidden_layer.jacobian_test_for_weight_2(1/epsilon, d_weight_2)
            first_degree_approximations_weight_2.append(first)
            second_degree_approximations_weight_2.append(second)

            # first, second = hidden_layer.jacobian_test_for_samples(1/epsilon, d_sample)
            # first_degree_approximations_samples.append(first)
            # second_degree_approximations_samples.append(second)
        
        # Plotting the approximations on a semilogarithmic scale
        # plt.figure()
        # plt.loglog(epsilon_values, first_degree_approximations_weight_1, label='Zero Order Approximation', marker='o')
        # plt.loglog(epsilon_values, second_degree_approximations_weight_1, label='First Order Approximation', marker='s')
        # plt.xlabel('Epsilon (log scale)')
        # plt.ylabel('Approximation/Error (log scale)')
        # plt.title('Jacobian Test for residual block on Weight 1')   
        # plt.legend()
        # plt.show()

        # Plotting the approximations on a semilogarithmic scale
        plt.figure()
        plt.loglog(epsilon_values, first_degree_approximations_weight_2, label='Zero Order Approximation', marker='o')
        plt.loglog(epsilon_values, second_degree_approximations_weight_2, label='First Order Approximation', marker='s')
        plt.xlabel('Epsilon (log scale)')
        plt.ylabel('Approximation/Error (log scale)')
        plt.title('Jacobian Test for residual block on Weight 2')
        plt.legend()
        plt.show()

        # # Plotting the approximations on a semilogarithmic scale
        # plt.figure()
        # plt.loglog(epsilon_values, first_degree_approximations_samples, label='Zero Order Approximation', marker='o')
        # plt.loglog(epsilon_values, second_degree_approximations_samples, label='First Order Approximation', marker='s')
        # plt.xlabel('Epsilon (log scale)')
        # plt.ylabel('Approximation/Error (log scale)')
        # plt.title('Jacobian Test for residual block on sample')
        # plt.legend()
        # plt.show()
    
    #=============================================Task 2.2.1 test===================================================
        
    def jacobian_test_for_standard_neural_network(self):
                # a single hidden layer with 3 neurons and 3 features
        sample_matrix = np.array([[1],  # Feature 1
                                [2],  # Feature 2
                                [3]])  # Feature 3
        
        weight_matrix_1 = np.array([[0.1, 0.2, 0.3], 
                                [0.4, 0.5, 0.6], 
                                [0.7, 0.8, 0.9]])
        
        weight_matrix_2 = np.array([[0.1, 0.2, 0.3], 
                                [0.4, 0.5, 0.6], 
                                [0.7, 0.8, 0.9]])
        
        bias_vector = np.array([[0.1],
                                [0.2],
                                [0.3]])
        tanh_model = Classical_Tanh()
        
        hidden_layer_1 = Classical_Hidden_Layer(weight_matrix_1, tanh_model, bias_vector)
        hidden_layer_2 = Classical_Hidden_Layer(weight_matrix_2, tanh_model, bias_vector)
        hidden_layers = [hidden_layer_1, hidden_layer_2]

        neural_network = Standard_Neural_Network(hidden_layers)
        neural_network.sample_matrix = sample_matrix

        epsilon_values = np.logspace(1, 8, 8)  # Epsilon values
        first_degree_approximations_weight = []
        second_degree_approximations_weight = []

        # Assume softmax_model.gradient_test_for_loss_function calculates and returns approximations
        for epsilon in epsilon_values:

            first, second = neural_network.jacobian_test_for_weight(1/epsilon)
            first_degree_approximations_weight.append(first)
            second_degree_approximations_weight.append(second)
        
        # Plotting the approximations on a semilogarithmic scale
        plt.figure()
        plt.loglog(epsilon_values, first_degree_approximations_weight, label='Zero Order Approximation', marker='o')
        plt.loglog(epsilon_values, second_degree_approximations_weight, label='First Order Approximation', marker='s')
        plt.xlabel('Epsilon (log scale)')
        plt.ylabel('Approximation/Error (log scale)')
        plt.title('Jacobian Test For standard Neural Network on Weights')
        plt.legend()
        plt.show()

    
    def jacobian_test_hidden_layer(self):
        # a single hidden layer with 3 neurons and 3 features
        sample_matrix = np.array([[1],  # Feature 1
                                [2],  # Feature 2
                                [3]])  # Feature 3
        
        weight_matrix = np.array([[0.1, 0.2, 0.3], 
                                [0.4, 0.5, 0.6], 
                                [0.7, 0.8, 0.9]])
        
        bias_vector = np.array([[0.1],
                                [0.2],
                                [0.3]])
        tanh_model = Classical_Tanh()
        
        hidden_layer = Classical_Hidden_Layer(weight_matrix, tanh_model, bias_vector)

        hidden_layer.set_input_for_test(sample_matrix)

        epsilon_values = np.logspace(1, 8, 8)  # Epsilon values
        first_degree_approximations_weight = []
        second_degree_approximations_weight = []

        first_degree_approximations_samples = []
        second_degree_approximations_samples = []

        # Assume softmax_model.gradient_test_for_loss_function calculates and returns approximations
        for epsilon in epsilon_values:

            first, second = hidden_layer.jacobian_test_for_weight(1/epsilon)
            first_degree_approximations_weight.append(first)
            second_degree_approximations_weight.append(second)

            first, second = hidden_layer.jacobian_test_for_samples(1/epsilon)
            first_degree_approximations_samples.append(first)
            second_degree_approximations_samples.append(second)
        
        # Plotting the approximations on a semilogarithmic scale
        plt.figure()
        plt.loglog(epsilon_values, first_degree_approximations_weight, label='Zero Order Approximation', marker='o')
        plt.loglog(epsilon_values, second_degree_approximations_weight, label='First Order Approximation', marker='s')
        plt.xlabel('Epsilon (log scale)')
        plt.ylabel('Approximation/Error (log scale)')
        plt.title('Jacobian Test For Tanh Activation Function on Weights')
        plt.legend()
        plt.show()

        # Plotting the approximations on a semilogarithmic scale
        plt.figure()
        plt.loglog(epsilon_values, first_degree_approximations_samples, label='Zero Order Approximation', marker='o')
        plt.loglog(epsilon_values, second_degree_approximations_samples, label='First Order Approximation', marker='s')
        plt.xlabel('Epsilon (log scale)')
        plt.ylabel('Approximation/Error (log scale)')
        plt.title('Jacobian Test For Tanh Activation Function on Samples')
        plt.legend()
        plt.show()

    #=============================================Task 2.1.3 test===================================================
    def test_sgd_on_softmax(self):
            # Step 1: Load data from the .mat file
            mat_data = loadmat('data_sets/PeaksData.mat')
            
            # Extract matrices
            training_features = mat_data['Yt']  # Transpose: features × samples
            training_labels = mat_data['Ct'].T  # Training labels (class)
            test_features = mat_data['Yv']  # Transpose: features × samples
            test_labels = mat_data['Cv'].T  # Validation labels (class)
            num_features = training_features.shape[0]  # Features are rows now
            initial_weights = np.random.randn(num_features, training_labels.shape[1])

            # Step 3: Initialize the objective function
            objective_function = Softmax.Softmax(initial_weights)

            # Step 4: Initialize SGD optimizer with hyperparameters
            learning_rate = 0.00001
            max_iterations = 80
            batch_size = 100
            sgd_optimizer = SGD(objective_function, learning_rate, max_iterations, batch_size, training_features, training_labels)


            # Step 6: Optimize the weights
            optimized_weights, losses, success_percentage_train, success_percentage_test = sgd_optimizer.optimize_with_precentages(
                test_features, test_labels
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

    #=============================================Task 2.1.2 test===================================================
    def test_sgd_on_least_squares(self):
        # Step 1: Create synthetic data
        np.random.seed(42)
        num_samples = 100
        num_features = 2

        # Generate random data
        X = np.random.rand(num_samples, num_features)
        true_weights = np.array([250.0, -50.0])  # True weights for testing
        y = np.dot(X, true_weights) + np.random.normal(scale=0.1, size=num_samples)
        initial_weights = np.random.randn(num_features, 1)

        # Reshape y for consistency
        y = y.reshape(-1, 1)

        # Step 2: Initialize the objective function
        objective_function = LeastSquaresObjectiveFunction(initial_weights)

        # Step 3: Initialize SGD optimizer
        learning_rate = 0.6
        max_iterations = 10
        batch_size = 10
        sgd_optimizer = SGD(objective_function, learning_rate, max_iterations, batch_size, X.T, y)
        
        optimized_weights , losses = sgd_optimizer.optimize()

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

    #=============================================Task 2.1.1 test===================================================
        
    # gradient test for softmax loss function
    def softmax_loss_function_gradient_test(self):
        input_matrix = np.array([[1],  # Feature 1
                                [2],  # Feature 2
                                [3]])  # Feature 3
        
        label_matrix = np.array([[1, 0, 0]])
        
        weight_matrix = np.array([[0.1, 0.2, 0.3], 
                                [0.4, 0.5, 0.6], 
                                [0.7, 0.8, 0.9]])

        softmax_model = Softmax.Softmax(weight_matrix)
        softmax_model.sample_matrix = input_matrix
        softmax_model.label_matrix = label_matrix

        epsilon_values = np.logspace(1, 8, 8)  # Epsilon values
        first_degree_approximations_weight = []
        second_degree_approximations_weight = []

        first_degree_approximations_samples = []
        second_degree_approximations_samples = []

        d_weight = softmax_model.get_normalized_random_vector(weight_matrix)
        d_sample = softmax_model.get_normalized_random_vector(input_matrix)

        # Assume softmax_model.gradient_test_for_loss_function calculates and returns approximations
        for epsilon in epsilon_values:
            first, second = softmax_model.gradient_test_for_loss_function_on_weight(1/epsilon,d_weight)
            first_degree_approximations_weight.append(first)
            second_degree_approximations_weight.append(second)

            first, second = softmax_model.gradient_test_for_loss_function_on_samples(1/epsilon,d_sample)
            first_degree_approximations_samples.append(first)
            second_degree_approximations_samples.append(second)

        # Plotting the approximations on a semilogarithmic scale
        plt.figure()
        plt.loglog(epsilon_values, first_degree_approximations_weight, label='Zero Order Approximation', marker='o')
        plt.loglog(epsilon_values, second_degree_approximations_weight, label='First Order Approximation', marker='s')
        plt.xlabel('Epsilon (log scale)')
        plt.ylabel('Approximation/Error (log scale)')
        plt.title('Gradient Test For Softmax Loss Function on Weights')
        plt.legend()
        plt.show()

        # Plotting the approximations on a semilogarithmic scale
        plt.figure()
        plt.loglog(epsilon_values, first_degree_approximations_samples, label='Zero Order Approximation', marker='o')
        plt.loglog(epsilon_values, second_degree_approximations_samples, label='First Order Approximation', marker='s')
        plt.xlabel('Epsilon (log scale)')
        plt.ylabel('Approximation/Error (log scale)')
        plt.title('Gradient Test For Softmax Loss Function on Samples')
        plt.legend()
        plt.show()