from typing import Callable, List
import numpy as np
from numpy import ndarray
from optimizer import Optimizer


class Function(object):
    def __init__(self, input):
        self.input = input
        self.output = None
        self.input_gradient = None

    def forward(self, input: ndarray) -> ndarray:
        raise NotImplementedError

    def backward(self, output_gradient: ndarray)->ndarray:
        raise NotImplementedError


class WeightAndBiasFunction(Function):
    def __init__(self, input:ndarray, weight:ndarray, bias:ndarray, optimizer:Optimizer):
        super.__init__(input)
        self.weight = weight
        self.bias = bias
        self.weight_gradient = None
        self.bias_gradient = None
        self.optimizer = optimizer

    def forward(self) -> ndarray:
        '''
          Apply weight and bias forward propagation functions
        '''
        output = np.dot(self.input + self.bias, self.weight)

        return output

    def backward(self, output_gradient: ndarray) -> ndarray:
        '''
            Calculate backward propagation for input
        '''
        self.input_gradient =  np.dot(np.ones_like(self.input) * output_gradient, np.transpose(self.weight, (1, 0)))

        '''
            Calculate backward propagation for weight
        '''

        bias_gradient =  np.ones_like(self.bias) * output_gradient
        np.sum(bias_gradient, axis=0).reshape(1, bias_gradient.shape[1])

        self.weight_gradient = np.dot(np.transpose(self.input_, (1, 0)), np.ones_like(self.input) * output_gradient)

        '''
            Update Bias and Weights based on gradients
        '''
        self.optimizer.gradient_descent(self.weight, self.weight_gradient)
        self.optimizer.gradient_descent(self.bias, self.bias_gradient)


        return self.input_gradient

class SigmoidFunction(Function):
    def __init__(self, input):
        super.__init__(input)

    def forward(self) -> ndarray:
        self.output =  1.0/(1.0+np.exp(-1.0 * self.input))

        return self.output

    def backward(self, output_gradient:ndarray) -> ndarray:
        self.input_gradient = self.output * (1.0 - self.output) * output_gradient

        return self.input_gradient












