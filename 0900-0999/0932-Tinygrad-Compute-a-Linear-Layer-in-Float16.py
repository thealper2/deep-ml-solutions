from tinygrad import Tensor, dtypes

def linear_fp16(x, W, b):
    return (x @ W.T + b).cast(dtype=dtypes.float32)