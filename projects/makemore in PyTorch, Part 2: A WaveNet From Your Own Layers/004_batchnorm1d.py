import torch

class BatchNorm1d:
    def __init__(self, dim, eps=1e-5, momentum=0.1):
        """gamma, beta (parameters); running_mean, running_var (buffers); training flag."""
        self.gamma = torch.ones(dim)
        self.beta = torch.zeros(dim)
        self.running_mean = torch.zeros(dim)
        self.running_var = torch.ones(dim)
        self.eps = eps
        self.momentum = momentum
        self.training = True

    def __call__(self, x):
        """Normalize over every dimension but the last; batch stats in training, running stats in eval."""
        if self.training:
            dims = tuple(range(x.ndim - 1))
            mean = x.mean(dim=dims)
            var = x.var(dim=dims, unbiased=True)
            with torch.no_grad():
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
                self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var
        else:
            mean = self.running_mean
            var = self.running_var
        xhat = (x - mean) / torch.sqrt(var + self.eps)
        self.out = self.gamma * xhat + self.beta
        return self.out

    def parameters(self):
        return [self.gamma, self.beta]
