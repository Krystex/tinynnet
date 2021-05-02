"""Activation functions"""
import numpy as np
from tinynnet import Tensor

class ReLU:
    """Rectified linear activation function"""

    def __init__(self):
        self.output = []

    def forward(self, inputs):
        """Forward of ReLU"""
        self.output = np.maximum(0, inputs)

    def backward(self) -> Tensor:
        raise NotImplementedError


class Softmax:
    """Softmax activation function"""

    def __init__(self):
        self.output = []

    def forward(self, inputs):
        """Forward of Softmax"""
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        self.output = exp_values / np.sum(exp_values, axis=1, keepdims=True)

class Tanh:
    """Tanh activation function"""

    def __init__(self):
        self.output = []

    def forward(self, inputs: Tensor) -> None:
        """Forward pass of tanh"""
        self.output = np.tanh(inputs)

    def backward(self, _: Tensor) -> Tensor:
        """Derivative of tanh function"""
        return 1 - self.output ** 2
