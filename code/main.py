# Import necessary libraries
import os
import logging
from tests import Tester
import sys

# to install requirements
# pip install -r requirements.txt

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Main function
def main():
    # Run the tests
    tester = Tester()
    tester.softmax_loss_function_gradient_test() #2.1.1
    tester.test_sgd_on_least_squares() #2.1.2
    tester.test_sgd_on_softmax() #2.1.3
    tester.jacobian_test_hidden_layer() #2.2.1
    tester.jacobian_test_for_standard_neural_network() #2.2.1
    tester.jacobian_test_residuel_block() #2.2.2
    tester.gradient_test_FF_neural_network() #2.2.3
    tester.test_neural_network([8, 8, 8], "ReLU") #2.2.4
    tester.test_neural_network_on_200_samples([8, 8, 8], "ReLU") #2.2.5


# Entry point
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.error(f"An error occurred: {e}", exc_info=True)
        sys.exit(1)