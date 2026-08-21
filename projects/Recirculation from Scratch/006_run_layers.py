import torch

def run_layers(x, blocks):
    """Return residual streams after every layer including the embedding as index 0."""
    residual_streams = [x]

    for block in blocks:
        x = pre_norm_block(x, block)
        residual_streams.append(x)

    return residual_streams