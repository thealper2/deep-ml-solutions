from tinygrad import Tensor, dtypes

def to_float_tensor(values):
    t = Tensor(values, dtype=dtypes.float32)
    return t