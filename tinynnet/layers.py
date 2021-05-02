"""Layers"""
import numpy as np
from tinynnet import Tensor

class Dense:
    """Dense Layer"""

    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
        self.output = []
        # For backpropagation
        self.grad_b = []
        self.grad_w = []

    def forward(self, inputs) -> None:
        """Forward pass of the dense layer"""
        self.output = np.dot(inputs, self.weights) + self.biases

    def backward(self, grad: Tensor) -> Tensor:
        """Backpropagation of the dense layer"""
        self.grad_b = np.sum(grad, axis=0)
        self.grad_w = np.dot(self.output.T, grad)
        return np.dot(grad, self.weights.T)
