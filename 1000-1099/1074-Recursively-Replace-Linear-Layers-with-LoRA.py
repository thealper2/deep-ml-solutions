import numpy as np

class Module:
    def __init__(self):
        self._modules = {}

class Linear:
    def __init__(self, in_features, out_features):
        self.in_features = in_features
        self.out_features = out_features
        self.weight = np.zeros((out_features, in_features))
        self.bias = np.zeros(out_features)
        self.weight_requires_grad = True
        self.bias_requires_grad = True

class LoRALayer:
    def __init__(self, in_features, out_features, rank, alpha):
        self.rank = rank
        self.alpha = alpha
        self.A = np.zeros((rank, in_features))
        self.B = np.zeros((out_features, rank))
        self.A_requires_grad = True
        self.B_requires_grad = True

class LinearWithLoRA:
    def __init__(self, linear, rank, alpha):
        self.linear = linear
        self.linear.weight_requires_grad = False
        self.linear.bias_requires_grad = False
        self.lora = LoRALayer(linear.in_features, linear.out_features, rank, alpha)

def replace_linear_with_lora(module, rank, alpha):
    for name, child in list(module._modules.items()):
        if isinstance(child, Linear):
            module._modules[name] = LinearWithLoRA(child, rank, alpha)
        elif isinstance(child, Module):
            replace_linear_with_lora(child, rank, alpha)

    return module