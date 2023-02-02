from __future__ import annotations
import numpy as np
from numpy import ndarray
# from multimethod import multimethod


import errors as err


class Tensor(object):

    # Take the ndarray and convert into Tensor object.
    def __init__(self, data:ndarray, type:str="float", device:str="cpu"):
        if not isinstance(data, ndarray):
            raise(TypeError("input type must be Numpy Array"))

        self.data = data
        self.type = type
        self.shape = data.shape
        self.device = device

    # @multimethod
    def __add__(self, other: Tensor = None):
        if other is not None:
            err.is_same_shape(self, other)
            self.data = self.data + other.data

        return self

    # def _setup(self, layer: Layer):
    #     self.layer = layer


    # @multimethod
    def add(self, other: float):

        self.data = self.data + other
        return self

    # @multimethod
    def __sub__(self, other: Tensor):
        if other is not None:
            err.is_same_shape(self, other)
            self.data = self.data - other.data


        return self

    # @multimethod
    def sub(self, other: float):

        self.data = self.data - other
        print(self.data)
        return self

    def __mul__(self, other: Tensor):
        err.is_multi_possible(self, other)
        result = np.dot(self.data, other.data)

        return result

    def __truediv__(self, divisor:float):
        self.data = self.data /  divisor
        print(self.data)
        return self

    def __repr__(self):
        return f'Tensor({self.data!r})'

    def __getitem__(self, ):
        pass




