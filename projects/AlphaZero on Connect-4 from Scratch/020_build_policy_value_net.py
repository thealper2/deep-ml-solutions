import torch
import torch.nn as nn

def build_policy_value_net(in_channels=2, hidden_channels=16, num_columns=7):
    """Compose backbone + policy head + value head into one nn.Module."""
    class PolicyValueNet(nn.Module):
        def __init__(self, in_channels=2, hidden_channels=16, num_columns=7):
            super().__init__()
            self.backbone = init_conv_backbone(in_channels, hidden_channels)
            self.policy_head = init_policy_head(hidden_channels, num_columns)
            self.value_head = init_value_head(hidden_channels)

        def forward(self, x):
            features = self.backbone(x)
            logits = self.policy_head(features)
            value = self.value_head(features)
            return logits, value

    return PolicyValueNet(in_channels, hidden_channels, num_columns)
