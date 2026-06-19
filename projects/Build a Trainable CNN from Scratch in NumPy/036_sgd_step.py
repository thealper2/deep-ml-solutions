import numpy as np

def sgd_step(param, grad, lr):
    return param - lr * grad
