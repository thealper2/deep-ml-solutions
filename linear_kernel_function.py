import numpy as np


def kernel_function(x1, x2):
    return np.matmul(x1, x2)


x1 = np.array([1, 2, 3])
x2 = np.array([4, 5, 6])

result = kernel_function(x1, x2)
print(result)
