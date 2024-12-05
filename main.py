# Import necessary libraries
import os
import sys
import logging
import numpy as np # type: ignore
import Softmax 
import matplotlib.pyplot as plt # type: ignore
from SGD import SGD
from LeastSquares import LeastSquaresObjectiveFunction

# to install requirements
# pip install -r requirements.txt
# to turn on virtual environment
# source myenv/bin/activate
# to turn off virtual environment
# deactivate

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Constants
CONSTANT_VAR = "Example Constant"

# Helper functions
def helper_function(param):
    """
    Example helper function.
    :param param: Description of the parameter
    :return: Description of the return value
    """
    logging.info(f"Processing parameter: {param}")
    return f"Processed {param}"

# Main function
def main():
    logging.info("Starting the script.")

    test_sgd_on_least_squares()

def test_sgd_on_least_squares():
    # Step 1: Create synthetic data
    np.random.seed(42)
    num_samples = 100
    num_features = 2

    # Generate random data
    X = np.random.rand(num_samples, num_features)
    true_weights = np.array([3.0, -2.0])  # True weights for testing
    y = np.dot(X, true_weights) + np.random.normal(scale=0.1, size=num_samples)

    # Reshape y for consistency
    y = y.reshape(-1, 1)

    # Step 2: Initialize the objective function
    objective_function = LeastSquaresObjectiveFunction(X, y)

    # Step 3: Initialize SGD optimizer
    learning_rate = 0.1
    max_iterations = 300
    sgd_optimizer = SGD(objective_function, learning_rate, max_iterations)

    # Step 4: Initialize weights randomly
    initial_weights = np.random.randn(num_features, 1)

    
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
def softmax_loss_function_gradient_test():
    inputMatrix = np.array([[1, 2, 3], 
                            [4, 5, 6], 
                            [7, 8, 9], 
                            [1, 0, 0], 
                            [0, 1, 0]])
    
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

# Entry point
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.error(f"An error occurred: {e}", exc_info=True)
        sys.exit(1)