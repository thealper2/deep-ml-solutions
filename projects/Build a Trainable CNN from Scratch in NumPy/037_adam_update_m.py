import numpy as np

def adam_update_m(m, grad, beta_one):
    return beta_one * m + (1 - beta_one) * grad
