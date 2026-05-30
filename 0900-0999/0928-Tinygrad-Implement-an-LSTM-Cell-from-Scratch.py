import math
from tinygrad import Tensor

class LSTMCell:
    def __init__(self, input_size, hidden_size):
        self.input_size = input_size
        self.hidden_size = hidden_size

        std = 1.0 / math.sqrt(hidden_size)

        self.W_ih = Tensor.uniform(
            4 * hidden_size, input_size,
            low=-std, high=std, requires_grad=True
        )
        self.W_hh = Tensor.uniform(
            4 * hidden_size, hidden_size,
            low=-std, high=std, requires_grad=True
        )
        self.b_ih = Tensor.uniform(
            4 * hidden_size,
            low=-std, high=std, requires_grad=True
        )
        self.b_hh = Tensor.uniform(
            4 * hidden_size,
            low=-std, high=std, requires_grad=True
        )

    def __call__(self, x, state):
        h_prev, c_prev = state

        gates = (
            x @ self.W_ih.T + self.b_ih +
            h_prev @ self.W_hh.T + self.b_hh
        )

        hs = self.hidden_size

        i = gates[:, :hs].sigmoid()
        f = gates[:, hs:2*hs].sigmoid()
        g = gates[:, 2*hs:3*hs].tanh()
        o = gates[:, 3*hs:].sigmoid()

        c_new = f * c_prev + i * g
        h_new = o * c_new.tanh()

        return h_new, c_new