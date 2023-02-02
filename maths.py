from DeepMath import tensor
from tensor import Tensor
import numpy as np
from numpy import ndarray
from typing import Callable, List


def sigmoid(input: Tensor) -> Tensor:
    data:ndarray = np.copy(input.data)
    np_data = 1 / (1 + np.exp(-data))

    return Tensor(np_data)


def square(input: Tensor) -> Tensor:
    return Tensor(np.power(input.data, 2))


def leaky_relu(input: Tensor) -> Tensor:
    return Tensor(np.maximum(input.data * 0.1, input.data))


def derive(input: Tensor, func) -> Tensor:
    delta = 0.001


    derived = (func(input.data + delta) - func(input.data - delta)) / (2 * delta)

    return derived


def chain_rule_derive(input: Tensor, funcs: List):
    '''
      Generalizing the Chain rule
      f2(f1)))' = f2'(f1(x)) * f1'(x)
      '''

    chain = []
    for idx, f in enumerate(funcs):
        if idx == len(funcs):
            break
        if idx == 0:
            chain.append(f(input))
        else:
            chain.append(f(chain[idx - 1]))

    results = []
    for idx, f in enumerate(funcs):
        if idx == 0:
            results.append(derive(input, f))
        else:
            results.append(derive(results[idx - 1], f))

   # product:ndarray = []
    product = results[0]
    for idx, result in enumerate(reversed(results)):
        if idx != 0 :
            product = product * result


    # Multiplying these quantities together at each point
    return product
def main():
    a = np.array([[1, 2, 3],
                  [4, 5, 6]])
    b = np.array([[1, 3],
                  [4, 6],
                  [2, 5]])
    c= np.array([1,2,3,4])
    t1 = Tensor(a)
    t2 = Tensor(b)
    t3 = Tensor(c)
 #   print(t1 * t2)
  #  print(derive(t1, square))
    print(chain_rule_derive(t3, [square, sigmoid]))
    #print(sigmoid(t1))

if __name__ == "__main__":
    main()
