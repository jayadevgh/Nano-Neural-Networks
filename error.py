from typing import Callable, List
import numpy as np
from numpy import ndarray

class Error(object):
    def __init__(self, prediction:ndarray, target:ndarray):
        self.prediction = prediction
        self.target = target

    def forward(self) -> float:
        raise NotImplementedError

    def backward(self):
        raise NotImplementedError


class MSE(Error):
    def __init__(self, prediction: ndarray, target: ndarray):
        super.__init__(prediction, target)

    def forward(self) -> float:
       return np.sum(np.power(self.target - self.prediction, 2)) / self.target.shape[0]

    def backward(self)->ndarray :
        return 2.0 * (self.target - self.prediction) / self.target.shape[0]






