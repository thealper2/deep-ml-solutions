def has_non_finite_gradients(grads):
    for k, v in grads.items():
        if not np.all(np.isfinite(v)):
            return True

    return False
