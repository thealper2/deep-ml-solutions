def apply_ffn_first_linear_and_relu(x, w1, b1):
    hidden = x @ w1 + b1
    return torch.relu(hidden)
