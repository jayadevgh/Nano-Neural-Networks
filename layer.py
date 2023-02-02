from typing import Callable, List
import numpy as np
from numpy import ndarray

from function import Function, WeightAndBiasFunction
from optimizer import SGD
from errors import is_same_shape

class Layer(object):
    '''
    A "layer" takes the input data and transforms the data through series of functions
    '''

    def __init__(self,
                 num_neurons: int):

        self.num_neurons = num_neurons
        self.functions: List[Function] = []

    def _setup(self) -> None:
        '''
        The setup must be implemented at subclasses. Mostly setup the input data, and parameter data
        '''
        raise NotImplementedError()

    def forward(self, input: ndarray) -> ndarray:
        '''
        Transforms the input through a series of functions
        '''

        for function in self.functions:
            input = function(input)

        return input

    def backward(self, grad: ndarray) -> ndarray:
        '''
        Takes the gradients from the next layer and passes through functions in reverse order.
        Calculate both input gradient, and weight and bias gradients at every layer.
        '''

        for function in reversed(self.functions):
            output_grad = function.backward(output_grad)

        return output_grad

    # def _param_grads(self) -> ndarray:
    #     '''
    #     Extracts the _param_grads from a layer's operations
    #     '''
    #
    #     self.param_grads = []
    #     for operation in self.operations:
    #         if issubclass(operation.__class__, ParamOperation):
    #             self.param_grads.append(operation.param_grad)
    #
    # def _params(self) -> ndarray:
    #     '''
    #     Extracts the _params from a layer's operations
    #     '''
    #
    #     self.params = []
    #     for operation in self.operations:
    #         if issubclass(operation.__class__, ParamOperation):
    #             self.params.append(operation.param)

class NetworkLayer(Layer):
    def __init__(self, num_neurons: int, activation:Function):
        super.__init__(num_neurons)
        self.activation = activation


    def setup(self,  input):
        weights = np.random.randn(input.shape[1], self.num_neurons)
        bias = np.random.randn(1, self.num_neurons)

        self.functions = [WeightAndBiasFunction(input, weights, bias, SGD()),self.activation]


class DeepNetworkLayer(object):
    def __init__(self, layers: List[Layer], seed:float):
        self.layers = layers
        self.seed = seed

    def forward(self, input:ndarray) -> input:
        for layer in self.layers:
            input = layer.forward(input)

        return input

    def backward(self, input: ndarray) -> input:
        temp = input
        for layer in reversed(self.layers):
            temp = layer.backward(temp)

        return temp









