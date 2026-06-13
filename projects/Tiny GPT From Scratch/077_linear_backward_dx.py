def linear_backward_dx(dy, cache):
    W = cache['w']
    dx = np.dot(dy, W.T)
    return dx
