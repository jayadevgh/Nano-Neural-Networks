import math
from typing import Callable, List
import numpy as np
from numpy import ndarray

from data import LayerData
from optimizer import Optimizer


class Function(object):
    def __init__(self):
       pass

    def feed_forward(self, input: ndarray, data: LayerData) -> ndarray:
        raise NotImplementedError

    def back_propagate(self, output_gradient: ndarray)->ndarray:
        raise NotImplementedError


class WeightMultiplyFunction(Function):
    def __init__(self, weight:ndarray):
        super().__init__()
        self.weight = weight



    def feed_forward(self, input: ndarray, data: LayerData) -> ndarray:
        self.input = input
        self.data = data
       # print("Before Multiply")
       #  if math.isnan(input[0][0]) :
       #      print("WeightMultiply Feed Forward", input, self.weight)
       #      exit(1)

        output = np.dot(self.input, self.weight)
        self.data.input.append(input)

        return output

    def back_propagate(self, output_gradient: ndarray) -> ndarray:
        input_gradient = np.dot(output_gradient, np.transpose(self.weight, (1, 0)))
        self.data.input_gradients.append(input_gradient)

        weight_gradient =  np.dot(np.transpose(self.input, (1, 0)), output_gradient)
        self.data.parameter_gradients.insert(0, weight_gradient)

        return input_gradient

class BiasAddFunction(Function):
    def __init__(self, bias:ndarray):
        super().__init__()
        self.bias = bias



    def feed_forward(self, input: ndarray, data: LayerData) -> ndarray:
        self.input = input
        self.data = data
        output = self.input + self.bias
        self.data.input.append(input)
        return output

    def back_propagate(self, output_gradient: ndarray) -> ndarray:

        input_gradient =  np.ones_like(self.input) * output_gradient
        self.data.input_gradients.append(input_gradient)

        gradient = np.ones_like(self.bias) * output_gradient
        bias_gradient = np.sum(gradient, axis=0).reshape(1, gradient.shape[1])
        self.data.parameter_gradients.insert(0, bias_gradient)

        return input_gradient

class SigmoidFunction(Function):
    def __init__(self):
        super().__init__()


    def feed_forward(self, input: ndarray, data: LayerData) -> ndarray:
        self.output =  1.0/(1.0+np.exp(-1.0 * input))

        return self.output

    def back_propagate(self, output_gradient:ndarray) -> ndarray:
        self.input_gradient = self.output * (1.0 - self.output) * output_gradient

        return self.input_gradient
class LinearFunction():

    def __init__(self) -> None:
        super().__init__()

    def feed_forward(self, input: ndarray, data: LayerData) -> ndarray:
        return input

    def back_propagate(self, output_gradient: ndarray) -> ndarray:
        '''Pass through'''
        return output_gradient

class LayerFunctions(object):

    def __init__(self):

        self.functions: List[Function] = []
        pass

    def setup(self, functions : List[Function] ):
        self.functions = functions

    def feed_forward(self, input:ndarray, data: LayerData) -> ndarray:
        for function in self.functions:
            input = function.feed_forward(input, data)

        return input

    def back_propagate(self, grads: ndarray) -> ndarray:
        for function in reversed(self.functions):
           grads = function.back_propagate(grads)

        return grads












