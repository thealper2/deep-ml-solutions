def design_network(input_dim, num_classes, seed=0):
    """Design and train a net that solves a nonlinear classification task.

    Inputs:
      input_dim: int, feature dimension
      num_classes: int, number of classes
      seed: int, RNG seed for reproducibility

    Returns:
      model: trained sequential model (forward/backward/params)
      metrics: dict with
        'accuracy': float >= 0.90 on an evaluation set,
        'x': np.ndarray (N, input_dim) eval features (N >= 50),
        'y': np.ndarray (N,) integer eval labels.
      The eval set (x, y) must not be linearly separable to high accuracy
      (< 0.82 for a linear classifier), and the model's true accuracy on
      it must match metrics['accuracy'] and be >= 0.90.
    """
    np.random.seed(seed)
    rng = np.random.RandomState(seed)

    N_per = max(120, 60 * num_classes)
    xs, ys = [], []
    for c in range(num_classes):
        r0 = 1.0 + 2.0 * c
        theta = rng.uniform(0, 2 * np.pi, N_per)
        r = r0 + rng.normal(0, 0.18, N_per)
        pts = np.zeros((N_per, input_dim))
        pts[:, 0] = r * np.cos(theta)
        if input_dim >= 2:
            pts[:, 1] = r * np.sin(theta)
        if input_dim > 2:
            pts[:, 2:] = rng.normal(0, 0.05, (N_per, input_dim - 2))
        xs.append(pts)
        ys.append(np.full(N_per, c))
    x = np.vstack(xs)
    y = np.concatenate(ys)
    perm = rng.permutation(len(y))
    x, y = x[perm], y[perm]

    H = 64
    def init(a, b):
        return initialize_weights(a, b, 'he')
    layers = [
        make_dense(input_dim, H, init),
        make_activation('relu'),
        make_dense(H, H, init),
        make_activation('relu'),
        make_dense(H, num_classes, init),
    ]
    model = make_sequential(layers)

    loss_fn = make_loss('cross_entropy')
    opt = make_optimizer(model['params'], lr=0.05, kind='momentum')
    train(model, loss_fn, opt, x, y, epochs=200, batch_size=32, seed=seed)

    logits, _ = model['forward'](x)
    acc = float(np.mean(np.argmax(logits, axis=1) == y))

    metrics = {'accuracy': acc, 'x': x, 'y': y}
    return model, metrics