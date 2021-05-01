"""Activation functions"""
import numpy as np

class ReLU:
    """Rectified linear activation function"""

    def __init__(self):
        self.output = []

    def forward(self, inputs):
        """Forward of ReLU"""
        self.output = np.maximum(0, inputs)


class Softmax:
    """Softmax activation function"""

    def __init__(self):
        self.output = []

    def forward(self, inputs):
        """Forward of Softmax"""
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        self.output = exp_values / np.sum(exp_values, axis=1, keepdims=True)
