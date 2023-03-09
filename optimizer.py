import numpy as np
from numpy import ndarray

from data import LayerData


class Optimizer(object):
    def __init__(self, learning_rate:float=0.01):
       self.learning_rate = learning_rate

    def optimize(self, ):
        raise NotImplementedError


class SGD(Optimizer):
    def __init__(self, learning_rate:float = 0.01):
       self.learning_rate = learning_rate

    def optimize(self):
        for (parameter, parameter_gradient) in zip(self.data.parameters, self.data.parameter_gradients):
            parameter -= self.learning_rate * parameter_gradient
        # Once updating the parameters, parameter_gradients are not required.
        self.data.parameter_gradients.clear()










