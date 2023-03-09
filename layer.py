from typing import Callable, List
import numpy as np
from numpy import ndarray

from data import LayerData
from error import Error
from function import Function, LayerFunctions, WeightMultiplyFunction, BiasAddFunction, LinearFunction
from optimizer import SGD, Optimizer
from errors import is_same_shape

class Layer(object):

    def __init__(self,
                 data: LayerData, activation: Function, functions: LayerFunctions, seed : int):

        self.data = data
        self.functions = functions
        self.activation = activation
        self.seed = seed


    def setup(self, input: ndarray) -> None:

        np.random.seed(self.seed)



        # weights
        self.data.parameters.append(np.random.randn(input.shape[1], self.data.neurons))
        # print("In setup input size:", input.shape, "params size", (self.data.parameters[0]).shape,
        #       "neurons", self.data.neurons)

        # bias
        self.data.parameters.append(np.random.randn(1, self.data.neurons))
        # print("Bias Size", (self.data.parameters[1]).shape)

        self.functions.functions = [WeightMultiplyFunction(self.data.parameters[0]),
                                    BiasAddFunction(self.data.parameters[1]), self.activation]

        return None

    def feed_forward(self, input: ndarray) -> ndarray:
        if len(self.data.parameters) == 0:
            #print("Setting up weights and biases")
            self.setup(input)
        setattr(self.functions, "data", self.data)
        output = self.functions.feed_forward(input, self.data)

        return output

    def back_propagate(self, grads: ndarray) -> ndarray:
        setattr(self.functions, "data", self.data)
        input_grads = self.functions.back_propagate(grads)
        self.data.input_grads = input_grads

        return input_grads

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

# class NetworkLayer(Layer):
#     def __init__(self, num_neurons: int, activation:Function):
#         super.__init__(num_neurons)
#         self.activation = activation
#
#
#     def setup(self,  input):
#         weights = np.random.randn(input.shape[1], self.num_neurons)
#         bias = np.random.randn(1, self.num_neurons)
#
#         self.functions = [WeightAndBiasFunction(input, weights, bias, SGD()),self.activation]
#

