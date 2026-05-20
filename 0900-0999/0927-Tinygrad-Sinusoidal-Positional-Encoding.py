import math
from tinygrad import Tensor, dtypes

def sinusoidal_positional_encoding(seq_len, d_model):
    pe = Tensor.zeros(seq_len, d_model).contiguous()
    position = Tensor.arange(0, seq_len).unsqueeze(1)
    div_term = Tensor.exp(Tensor.arange(0, d_model, 2, dtype=dtypes.float) * -(math.log(10000.0) / d_model))
    pe[:, 0::2] = Tensor.sin(position.float() * div_term)
    pe[:, 1::2] = Tensor.cos(position.float() * div_term)
    return pe