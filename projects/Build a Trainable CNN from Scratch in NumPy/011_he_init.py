def he_init(shape, fan_in, seed):
    np.random.seed(seed)
    std_dev = np.sqrt(2.0 / fan_in)
    weights_normal = np.random.randn(*shape) * std_dev
    return weights_normal
