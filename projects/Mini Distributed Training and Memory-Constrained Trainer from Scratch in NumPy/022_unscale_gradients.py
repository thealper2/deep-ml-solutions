def unscale_gradients(grads, scale):
    return {k: (v.copy() / scale).astype(np.float32) for k, v in grads.items()}
