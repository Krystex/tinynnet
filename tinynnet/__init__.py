import numpy as np

class SequentialNet:
    def __init__(self, layers):
        self.layers = layers
        self.output = []

    def forward(self, inputs):
        data = inputs
        for layer in self.layers:
            layer.forward(data)
            data = layer.output
        
        self.output = data
        return data