import numpy as np

def adam_update_v(v, grad, beta_two):
    return beta_two * v + (1 - beta_two) * (grad ** 2)
