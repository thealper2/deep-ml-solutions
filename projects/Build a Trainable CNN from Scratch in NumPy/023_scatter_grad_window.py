import numpy as np

def scatter_grad_window(grad_value, argmax_index, kernel):
    window = np.zeros((kernel, kernel))
    row = argmax_index // kernel
    col = argmax_index % kernel
    window[row, col] = grad_value
    return window
