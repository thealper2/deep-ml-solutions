import numpy as np

class Neg(Function):
    def forward(self, x):
        return LazyBuffer(-x._np)
    
    def backward(self, grad_output):
        return LazyBuffer(-grad_output._np)
