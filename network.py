from typing import  List
from numpy import ndarray
from error import Error, MSE
from layer import Layer
from optimizer import  Optimizer

'''Trains the data by going through each layer and running feed_forward and backpropagation methods
    and update the paramenters with optimize method'''
class DeepNeuralNetwork(object):
    def __init__(self, layers: List[Layer], error: Error, seed:float):
        self.layers = layers
        self.seed = seed
        self.error = error

    def feed_forward(self, input:ndarray) -> ndarray:
        for layer in self.layers:
           input = layer.feed_forward(input)

        return input

    def back_propagate(self, input: ndarray) -> ndarray:
        temp = input
        for layer in reversed(self.layers):
            temp = layer.back_propagate(temp)

        return temp

    def train(self, input_batch: ndarray, target_batch: ndarray) -> None:
        predictions = self.feed_forward(input_batch)

        self.error.feed_forward(predictions, target_batch)
        err_back = self.error.back_propagate(predictions, target_batch)

        self.back_propagate(err_back)


    def optimize(self, optim: Optimizer) -> None :

        for layer in self.layers:
            setattr(optim, "data", layer.data)
            optim.optimize()
