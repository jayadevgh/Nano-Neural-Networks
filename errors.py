import numpy as np
from tensor import Tensor

def is_same_type(t1: Tensor, t2: Tensor):
    assert(isinstance(t1, t2), '''Input type must be Tensor type''')
    return None

def is_same_shape(t1 : Tensor, t2 : Tensor):
    assert((t1.shape == t2.shape), "Both Tensors should have the same shape: First parameter's shape:{0},"
        "Second parameter's shape:{1}".format(tuple(t1.shape), tuple(t2.shape)))
    return None

def is_multi_possible(t1 : Tensor, t2 : Tensor):
    assert(t1.shape[1] == t2.shape[0], "Number of cols in the first tensor hould match number of rows in the second"
           "Shape of first tesnor:{0}, Shape of second Tensor:{1}".format(tuple(t1.shape), tuple(t2.shape)))





