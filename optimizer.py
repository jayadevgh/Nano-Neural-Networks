import numpy as np
from numpy import ndarray

class Optimizer(object):
    def __init__(self, learning_rate:float=0.01):
       self.learning_rate = learning_rate

    def gradient_descent(self):
        raise NotImplementedError


class SGD(Optimizer):
    def __init__(self, learning_rate:float = 0.01):
       self.learning_rate = learning_rate

    def gradient_descent(self, input:ndarray, input_gradient:ndarray):
        input = input - self.learning_rate * input_gradient










