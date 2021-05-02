"""Loss functions, measures how good predictions are"""
import numpy as np
from tinynnet import Tensor

class Loss:
    """Abstract loss function"""

    def forward(self, predicted: Tensor, actual: Tensor) -> float:
        """Forward pass of Loss function"""
        raise NotImplementedError

    def backward(self, predicted: Tensor, actual: Tensor) -> Tensor:
        """Backward pass of Loss function"""
        raise NotImplementedError


class CategoricalCrossentropy:
    """Categorical Cross-entropy"""

    def __init__(self):
        self.output = []

    def forward(self, inputs, target_output) -> None:
        """Forward categorical crossentropy"""
        self.output = -np.sum(np.log(inputs) * target_output, axis=1, keepdims=True)


class MeanSquaredError(Loss):
    """Mean squared error (but just total squared error)"""

    def forward(self, predicted: Tensor, actual: Tensor) -> float:
        return np.sum((predicted - actual) ** 2, axis=1, keepdims=True)

    def backward(self, predicted: Tensor, actual: Tensor) -> Tensor:
        return 2 * (predicted - actual)
