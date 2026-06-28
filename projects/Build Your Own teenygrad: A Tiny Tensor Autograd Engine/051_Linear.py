class Linear:
    def __init__(self, in_features, out_features, seed=None):
        self.in_features = in_features
        self.out_features = out_features
        self.weight = tensor_randn((in_features, out_features), seed=seed, requires_grad=True)
        self.bias = tensor_randn((out_features,), seed=seed, requires_grad=True)

    def __call__(self, x):
        out = tensor_matmul_2d(x, self.weight)
        n = out.shape[0]
        bias_b = self.bias.reshape((1, self.out_features)).expand((n, self.out_features))
        return Add.apply(out, bias_b)

    def parameters(self):
        return [self.weight, self.bias]
