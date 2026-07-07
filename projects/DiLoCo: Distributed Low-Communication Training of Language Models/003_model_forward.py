import numpy as np

def model_forward(params, x):
    """Run the 2-layer MLP forward pass and stash intermediates for backprop."""
    W1, b1 = params['W1'], params['b1']
    W2, b2 = params['W2'], params['b2']
    z1 = x @ W1 + b1
    h1 = np.maximum(0, z1)
    logits = h1 @ W2 + b2
    cache = {'x': x, 'z1': z1, 'h1': h1, 'logits': logits}
    return logits, cache
