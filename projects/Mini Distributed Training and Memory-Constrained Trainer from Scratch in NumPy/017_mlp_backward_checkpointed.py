def mlp_backward_checkpointed(dy_pred, light_cache, params):
    x = light_cache['x']
    cache = recompute_block_activations(x, params)
    grads = mlp_backward(dy_pred, cache, params)
    return grads
