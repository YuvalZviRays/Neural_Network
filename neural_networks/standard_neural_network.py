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
        # Forward pass through hidden layers and set the output as the input for the next layer
        for i in range(len(self.hidden_layers)):
            hidden_layer_function = self.hidden_layers[i]
            hidden_layer_function.set_input(output)
            output = hidden_layer_function.function()
        
        return output
    

    def jackMV(self, p):
        # p is a 1D array representing perturbations to all weights and biases.
        # We must split p into p_W and p_b for each layer.

        # Step 1: Slice p into (p_W, p_b) for each layer
        p_weights = []
        p_biases = []
        idx = 0
        for layer in self.hidden_layers:
            w_shape = layer.weight_matrix.shape  # (num_neurons, num_inputs)
            w_size = w_shape[0] * w_shape[1]

            p_w = p[idx : idx + w_size].reshape(w_shape)
            idx += w_size

            # bias shape: (num_neurons, 1)
            b_shape = (w_shape[0], 1)
            b_size = b_shape[0]
            p_b = p[idx : idx + b_size].reshape(b_shape)
            idx += b_size

            p_weights.append(p_w)
            p_biases.append(p_b)

        delta_a = np.zeros_like(self.hidden_layers[0].sample_matrix)

        # Step 2: Compute the effect of p on the output
        for i, layer in enumerate(self.hidden_layers):
            W = layer.weight_matrix
            a_prev = layer.sample_matrix       # input to this layer
            d = layer.derivative()             # derivative w.r.t. preactivation
            pW = p_weights[i]
            pB = p_biases[i]

            delta_z = W.dot(delta_a) + pW.dot(a_prev) + pB

            delta_a = d * delta_z

        # After the last layer, delta_a is how the final output changes due to p
        return delta_a

    def jackTMV(self, delta):
        jack = []
        for i in range(len(self.hidden_layers)-1, -1, -1):
            hidden_layer = self.hidden_layers[i]
            grad_w = np.dot(hidden_layer.derivative() * delta, hidden_layer.sample_matrix.T)
            grad_b = np.sum(hidden_layer.derivative() * delta, axis=1, keepdims=True)
            jack.append(grad_b.ravel())
            jack.append(grad_w.ravel())
            delta = np.dot(hidden_layer.weight_matrix.T, hidden_layer.derivative() * delta)

        jack.reverse()  
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