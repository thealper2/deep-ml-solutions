from tinygrad import Tensor, nn

class MLP:
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        self.l1 = nn.Linear(in_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, out_dim)

    def __call__(self, x: Tensor) -> Tensor:
        x = self.l1(x)
        x = x.relu()
        x = self.l2(x)
        return x