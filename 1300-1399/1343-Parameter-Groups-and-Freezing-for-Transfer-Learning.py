import torch
import torch.nn as nn


class TinyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 8))
        self.head = nn.Linear(8, 2)

    def forward(self, x):
        return self.head(self.backbone(x))


def build_optimizer(model, base_lr, head_lr, freeze_backbone):
    if freeze_backbone:
        for param in model.backbone.parameters():
            param.requires_grad = False

        return torch.optim.SGD(model.head.parameters(), lr=head_lr)
    else:
        return torch.optim.SGD([
            {'params': model.backbone.parameters(), 'lr': base_lr},
            {'params': model.head.parameters(), 'lr': head_lr},
        ])