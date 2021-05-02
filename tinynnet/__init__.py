import numpy as np

from numpy import ndarray as Tensor

class SequentialNet:
    """Sequential neural network architecture"""
    def __init__(self, layers):
        self.layers = layers
        self.output = []

    def forward(self, inputs):
        """Forward pass of sequential net"""
        data = inputs
        for layer in self.layers:
            layer.forward(data)
            data = layer.output

        self.output = data
        return data

    def backward(self, grad: Tensor) -> Tensor:
        """Backpropagation of sequential net"""
        raise NotImplementedError
