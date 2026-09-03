import torch
import torch.nn as nn


class EMA:
    def __init__(self, model, decay):
        self.decay = decay
        self.shadow = {}

        with torch.no_grad():
            for name, param in model.named_parameters():
                self.shadow[name] = param.data.clone()

    def update(self, model):
        with torch.no_grad():
            for name, param in model.named_parameters():
                self.shadow[name] = self.decay * self.shadow[name] + (1 - self.decay) * param.data

    def copy_to(self, model):
        with torch.no_grad():
            for name, param in model.named_parameters():
                param.data.copy_(self.shadow[name])