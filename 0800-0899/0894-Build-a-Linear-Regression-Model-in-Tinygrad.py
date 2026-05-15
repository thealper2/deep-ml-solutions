from tinygrad import Tensor, nn

class LinearRegression:
    def __init__(self, in_features: int, out_features: int):
        self.linear = nn.Linear(in_features, out_features)

    def __call__(self, x: Tensor) -> Tensor:
        return self.linear(x)