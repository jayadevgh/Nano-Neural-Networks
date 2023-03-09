from typing import Callable, List
import numpy as np
from numpy import ndarray

'''Currently error calculations are done using MSE'''
class Error(object):
    def __init__(self):
      pass

    def feed_forward(self, prediction:ndarray, target:ndarray) -> float:
        raise NotImplementedError

    def back_propagate(self, prediction:ndarray, target:ndarray):
        raise NotImplementedError


class MSE(Error):
    def __init__(self):
        super().__init__()

    def feed_forward(self, prediction:ndarray, target:ndarray) -> float:
        err = np.sum(np.power(target - prediction, 2)) / target.shape[0]
        return err

    def back_propagate(self, prediction:ndarray, target:ndarray)->ndarray :
        return 2.0 * (prediction - target) / prediction.shape[0]






