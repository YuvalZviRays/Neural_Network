# Import necessary libraries
import os
import logging
from tests import Tester
import sys

# to install requirements
# pip install -r requirements.txt
# to turn on virtual environment
# source myenv/bin/activate
# to turn off virtual environment
# deactivate

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Main function
def main():
    # Run the tests
    tester = Tester()
    hidden_layers = [16, 16]
    tester.test_neural_network(hidden_layers, "ReLU")

# Entry point
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.error(f"An error occurred: {e}", exc_info=True)
        sys.exit(1)