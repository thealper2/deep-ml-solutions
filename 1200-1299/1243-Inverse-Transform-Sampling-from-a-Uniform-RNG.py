import math

def inverse_transform_exponential(u_samples, rate):
    return [(-math.log(1 - u) / rate) for u in u_samples]
