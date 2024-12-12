from abc import ABC, abstractmethod
import numpy as np  # type: ignore
from activation_functions.residual_activation_functions.residual_activation_function import Residual_Activation_Function

class Residual_Tanh(Residual_Activation_Function):
    def function(self, weight_matrix_1, weight_matrix_2, input_matrix, bias_vector):
        """
        Compute the forward pass of the residual block with Tanh activation.
        y = x + W2 * tanh(W1 * x + b)
        """
        return input_matrix + np.dot(weight_matrix_2, np.tanh(np.dot(weight_matrix_1, input_matrix) + bias_vector))

    def jacobian_of_weight_1(self, weight_matrix_1, weight_matrix_2, input_matrix, bias_vector):
    # weight_matrix_1: shape (n, n)
        # weight_matrix_2: shape (n, n)
        # input_matrix: shape (n,) 
        # bias_vector: shape (n,)
        # sigma_prime: a function that applies the derivative of the activation element-wise.

        # 1. Compute a = W1 x + b
        a = weight_matrix_1.dot(input_matrix) + bias_vector  # shape (n,)

        # 2. Compute sigma'(a)
        da = 1 - np.tanh(a) ** 2  # shape (n,)
        da = da.ravel()

        # 3. Form diag(sigma'(a))
        diag_da = np.diag(da)  # shape (n,n)

        # 4. Multiply W2 * diag_da
        part = weight_matrix_2.dot(diag_da)  # shape (n,n)

        # 5. Construct x^T ⊗ I
        n = input_matrix.shape[0]
        kron_x_I = np.kron(input_matrix.reshape(1, -1), np.eye(n))  # shape (n, n²)

        # 6. Final multiplication
        jac = part.dot(kron_x_I)  # shape (n, n²)

        return jac

    def jacobian_of_weight_2(self, weight_matrix_1, weight_matrix_2, input_matrix, bias_vector):
        return np.kron(np.tanh(np.dot(weight_matrix_1, input_matrix) + bias_vector).T, np.eye(input_matrix.shape[0]))

    def jacobian_of_samples(self, weight_matrix_1, weight_matrix_2, input_matrix, bias_vector):
        """
        Compute the gradient of the residual output with respect to the input samples x.
        ∂y/∂x = I + W2 * diag(tanh'(W1 * x + b)) * W1
        """
        tanh_derivative = 1 - np.tanh(np.dot(weight_matrix_1, input_matrix) + bias_vector) ** 2
        identity_matrix = np.eye(input_matrix.shape[0])
        return identity_matrix + np.dot(weight_matrix_2, tanh_derivative * weight_matrix_1)
    
    def derivative(self, weight_matrix, input_matrix, bias_vector):
        """
        Compute the derivative of the residual block with Tanh activation.
        dy/dx = 1 - tanh(W * x + b)^2
        """
        return 1 - np.tanh(np.dot(weight_matrix, input_matrix) + bias_vector) ** 2