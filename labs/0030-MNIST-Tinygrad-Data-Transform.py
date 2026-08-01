from tinygrad import Tensor

class MyTransform:
    def __call__(self, x: Tensor) -> Tensor:
        """
        x: tinygrad Tensor of shape (1, 28, 28), float, in [0, 1].
        Return: transformed Tensor, same shape and dtype.
        Must be non-identity and deterministic under Tensor.manual_seed.
        """
        return (x * 2.0 - 1.0).clip(0.0, 1.0)
