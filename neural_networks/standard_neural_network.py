import os
import sys
import logging
import numpy as np # type: ignore
from Hidden_Layer_Functions.Hidden_Layer_Function import Hidden_Layer_Function
from neural_networks.neural_network import Neural_Network

class Standard_Neural_Network(Neural_Network):

    def function(self):
        # Forward pass
        output = self.sample_matrix 
        logging.info(f"output : {output}")
        # Forward pass through hidden layers and set the output as the input for the next layer
        for i in range(len(self.hidden_layers)):
            hidden_layer_function = self.hidden_layers[i]
            hidden_layer_function.set_input(output)
            output = hidden_layer_function.function()
        
        return output
    
    def jack_on_weights(self):
        
        output_size = self.hidden_layers[-1].weight_matrix.shape[0]  # Number of neurons in the last layer
        delta = np.ones((output_size, 1))

        jack = []
        for i in range(len(self.hidden_layers)-1, -1, -1):
                
                hidden_layer = self.hidden_layers[i]
                grad_w = hidden_layer.jacobian_of_weight() * delta
                jack.append(grad_w.ravel())
                grad_b = np.sum(hidden_layer.jacobian_of_samples() * delta, axis=1, keepdims=True)
                jack.append(grad_b.ravel())
    
                delta = hidden_layer.jacobian_of_samples() * delta
        
        return np.concatenate(jack)

    def jackTMV(self, delta):
        jack = []
        for i in range(len(self.hidden_layers)-1, -1, -1):

            hidden_layer = self.hidden_layers[i]
            grad_w = np.dot(hidden_layer.derivative() * delta, hidden_layer.sample_matrix.T)
            jack.append(grad_w.ravel())

            grad_b = np.sum(hidden_layer.derivative() * delta, axis=1, keepdims=True)
            jack.append(grad_b.ravel())

            delta = np.dot(hidden_layer.weight_matrix.T, hidden_layer.derivative() * delta)
        
        return np.concatenate(jack)
    
    def get_weights(self):
        params = []
        for layer in self.hidden_layers:
            params.append(layer.weight_matrix.ravel())
            params.append(layer.bias_vector.ravel())
        return np.hstack(params)
    
    def set_weights(self, param_vector):
        start = 0
        for layer in self.hidden_layers:
            weight_size = layer.weight_matrix.size
            bias_size = layer.bias_vector.size

            layer.weight_matrix = param_vector[start:start+weight_size].reshape(layer.weight_matrix.shape)
            start += weight_size
            layer.bias_vector = param_vector[start:start+bias_size].reshape(layer.bias_vector.shape)
            start += bias_size