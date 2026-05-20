from tinygrad import Tensor

def dropout(x, p, training):
    if not training:
        return x

    keep_prob = 1 - p
    mask = (Tensor.rand(*x.shape) < keep_prob)
    mask = mask / keep_prob
    out = x * mask
    return out