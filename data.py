from typing import Callable, List
from numpy import ndarray

'''Repository of all the data at layer level.'''
class LayerData(object):

    def __init__(self,
                 num_neurons: int):
        self.neurons = num_neurons
        self.input: List[ndarray] = []
        self.output: List[ndarray] = []
        self.parameters: List[ndarray] = []
        self.input_gradients: List[ndarray] = []
        self.parameter_gradients: List[ndarray] = []


    def add_input(self, input : ndarray):

        self.input.append(input)


