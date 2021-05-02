"""Some tools"""
import numpy as np

def to_categorical(arr):
    """Convert array to one-hot encoding"""
    matrix = np.zeros((arr.size, np.max(arr)+1))
    matrix[np.arange(arr.size),arr] = 1
    return matrix
