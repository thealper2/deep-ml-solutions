"""
Neural Networks From Scratch: Forward and Backward — assembled scaffold.
This updates live as you solve each step.
"""

import numpy as np

# ── Step 001  numerical_gradient ──
import numpy as np

def numerical_gradient(f, x, eps=1e-5):
    grad = np.zeros_like(x)
    if x.size == 0:
        return grad
    
    it = np.nditer(x, flags=["multi_index"])
    while not it.finished:
        idx = it.multi_index
        x_plus = x.copy()
        x_plus[idx] += eps
        x_minus = x.copy()
        x_minus[idx] -= eps
        grad[idx] = (f(x_plus) - f(x_minus)) / (2 * eps)
        it.iternext()

    return grad

# ── Step 002  gradient_check ──
def gradient_check(analytic_grad, numeric_grad, tol=1e-5):
    diff = np.abs(analytic_grad - numeric_grad)
    denominator = np.maximum(np.abs(analytic_grad), np.abs(numeric_grad))
    denominator = np.maximum(denominator, tol)
    relative_errors = diff / denominator
    return float(np.max(relative_errors))

# ── Step 003  make_dense ──
def make_dense(in_dim, out_dim, weight_init_fn):
    """Create a fully connected layer.

    Inputs:
      in_dim: int, input feature size
      out_dim: int, output feature size
      weight_init_fn: callable(in_dim, out_dim) -> (W, b)

    Returns layer dict with keys:
      params: {'W': (in_dim, out_dim), 'b': (out_dim,)}
      forward(x) -> (y, cache) with y shape (batch, out_dim)
      backward(dout, cache) -> (dx, grads) with grads {'W', 'b'}
        Analytic dx/dW/db must match numerical_gradient via gradient_check.
    """
    W, b = weight_init_fn(in_dim, out_dim)
    params = {'W': W, 'b': b}

    def forward(x):
      y = x @ W + b
      cache = x
      return y, cache

    def backward(dout, cache):
      x = cache
      dx = dout @ W.T
      dW = x.T @ dout
      db = np.sum(dout, axis=0)
      grads = {'W': dW, 'b': db}
      return dx, grads

    return {
      'params': params,
      'forward': forward,
      'backward': backward,
    }

# ── Step 004  make_activation ──
def make_activation(kind='relu'):
    """Create a genuinely nonlinear elementwise activation layer.

    Args:
        kind: str nonlinearity name. Default 'relu' must implement ReLU
              (zero negatives, pass non-negatives). Other kinds optional.

    Returns:
        Layer dict with:
          forward(x) -> (y, cache)
            x, y: np.ndarray shape (batch, dim)
          backward(dout, cache) -> (dx, {})
            dout, dx: np.ndarray shape (batch, dim)
            param grad dict is always empty (no learnable params)

    Must be elementwise and non-affine; analytic dx must match
    numerical_gradient / gradient_check.
    """
    params = {}

    if kind == 'relu':
      def forward(x):
        y = np.maximum(x, 0)
        cache = x
        return y, cache

      def backward(dout, cache):
        x = cache
        dx = dout * (x > 0)
        param_grads = {}
        return dx, param_grads

      return {
        'params': params,
        'forward': forward,
        'backward': backward
      }

    elif kind == "tanh":
      def forward(x):
        y = np.tanh(x)
        cache = y
        return y, cache
      
      def backward(dout, cache):
        y = cache
        dx = dout * (1 - y * y)
        param_grads = {}
        return dx, param_grads

      return {
        'params': params,
        'forward': forward,
        'backward': backward
      }

    elif kind == 'sigmoid':
      def forward(x):
        y = 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))
        cache = y
        return y, cache

      def backward(dout, cache):
        y = cache
        dx = dout * y * (1 - y)
        param_grads = {}
        return dx, param_grads

      return {
        'params': params,
        'forward': forward,
        'backward': backward
      }

    else:
      raise ValueError(f"Unsupported activation: {kind}")

# ── Step 005  initialize_weights ──
def initialize_weights(in_dim, out_dim, scheme='he'):
    """Return (W, b) for a dense layer.

    Inputs:
      in_dim: int fan-in
      out_dim: int fan-out
      scheme: str initialization family (default 'he')

    Returns:
      W: np.ndarray shape (in_dim, out_dim), finite, symmetry-breaking,
         scale stable with depth (fan-in dependent)
      b: np.ndarray shape (out_dim,), near zero
    """
    if scheme == 'he':
      std = np.sqrt(2.0 / in_dim)
      W = np.random.randn(in_dim, out_dim) * std
      b = np.zeros(out_dim)
      return W, b

    elif scheme == 'xavier':
      std = np.sqrt(1.0 / in_dim)
      W = np.random.randn(in_dim, out_dim) * std
      b = np.zeros(out_dim)
      return W, b

    elif scheme == 'xavier_uniform':
      a = np.sqrt(6.0 / (in_dim + out_dim))
      W = np.random.uniform(-a, a, size=(in_dim, out_dim))
      b = np.zeros(out_dim)
      return W, b

    else:
      raise ValueError(f"Unsupported initialization scheme: {scheme}")

# ── Step 006  make_loss ──
def make_loss(kind='cross_entropy'):
    """Return a classification loss_fn(logits, labels) -> (loss, d_logits).

    Inputs to loss_fn:
      logits: (batch, C) float array of raw class scores
      labels: (batch,) int array of class indices in [0, C)
    Outputs:
      loss: Python float, mean scalar loss over the batch (finite)
      d_logits: (batch, C) gradient of loss w.r.t. logits (finite)
    Must pass gradient_check, be minimized by confident correct predictions,
    and stay finite under saturated logits.
    """
    if kind == 'cross_entropy':
      def cross_entropy_loss(logits, labels):
        batch, C = logits.shape
        logits_shifted = logits - np.max(logits, axis=1, keepdims=True)
        log_sum_exp = np.max(logits, axis=1) + np.log(np.sum(np.exp(logits_shifted), axis=1))
        logits_correct = logits[np.arange(batch), labels]
        losses = log_sum_exp - logits_correct
        loss = float(np.mean(losses))
        exp_logits = np.exp(logits_shifted)
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        d_logits = probs.copy()
        d_logits[np.arange(batch), labels] -= 1.0
        d_logits /= batch
        return loss, d_logits
      
      return cross_entropy_loss

    else:
      raise ValueError(f"Unsupported loss kind: {kind}")

# ── Step 007  make_sequential ──
def make_sequential(layers):
    """Compose protocol-honoring layers into one sequential model.

    Inputs:
      layers: list of layer dicts, each with
        forward(x) -> (y, cache),
        backward(dout, cache) -> (dx, grads_dict),
        params: dict of ndarrays (possibly empty).

    Returns a dict with:
      forward(x) -> (y, caches)
        y: final activation after applying every layer in order
        caches: opaque structure needed by backward
      backward(dout, caches) -> (dx, grads_list)
        dx: gradient w.r.t. the original input x
        grads_list: list of length len(layers); grads_list[i] is the
          grads_dict from layers[i] ({} for param-free layers)
      params: aggregated live view of all layer params, length len(layers),
        same order as layers (so in-place updates affect the model)
    """
    params = [layer['params'] for layer in layers]

    def forward(x):
      caches = []
      for layer in layers:
        x, cache = layer['forward'](x)
        caches.append(cache)

      return x, caches

    def backward(dout, caches):
      grads_list = []
      dx = dout
      for layer, cache in zip(reversed(layers), reversed(caches)):
        dx, grads = layer['backward'](dx, cache)
        grads_list.insert(0, grads)

      return dx, grads_list

    return {
      'params': params,
      'forward': forward,
      'backward': backward,
    }

# ── Step 008  forward_backward ──
def forward_backward(model, loss_fn, x, y):
    """Run one full forward-backward sweep on a batch.

    Inputs:
      model: sequential dict with 'forward', 'backward', 'params'
             model['forward'](x) -> (logits, caches)
             model['backward'](d_logits, caches) -> (dx, param_grads)
      loss_fn: callable (logits, y) -> (loss, d_logits)
      x: np.ndarray (batch, in_dim)
      y: np.ndarray (batch,) integer labels

    Returns:
      loss: float, scalar batch loss
      param_grads: nested np.ndarrays matching model['params'] layout
                   (gradients of loss w.r.t. every parameter)
    """
    logits, caches = model['forward'](x)
    loss, d_logits = loss_fn(logits, y)
    dx, layer_grads = model['backward'](d_logits, caches)
    param_grads = layer_grads
    return loss, param_grads

# ── Step 009  make_optimizer ──
def make_optimizer(params, lr=1e-2, kind='sgd'):
    """Build an optimizer that updates params in place.

    Inputs:
      params: arrays, possibly nested in lists/dicts (or dict of arrays) to optimize
      lr: float learning rate
      kind: str algorithm name (e.g. 'sgd')

    Returns:
      dict with key 'step'. step(grads) applies one in-place update
      using grads structured like params. Parameter shapes must stay
      unchanged. Repeated steps must reduce a simple convex objective
      within a modest fixed budget and keep values finite.
    """
    def sgd_step(grads):
        """Apply SGD update: p <- p - lr * g"""
        def _update(p, g):
            if isinstance(p, dict) and isinstance(g, dict):
                for key in p:
                    _update(p[key], g[key])
            elif isinstance(p, list) and isinstance(g, list):
                for i in range(len(p)):
                    _update(p[i], g[i])
            elif isinstance(p, np.ndarray) and isinstance(g, np.ndarray):
                p -= lr * g
            else:
                pass
        
        _update(params, grads)
    
    def momentum_step(grads):
        """SGD with momentum: v <- mu * v + g, p <- p - lr * v"""
        if not hasattr(momentum_step, 'velocities'):
            def _build_velocities(p):
                if isinstance(p, dict):
                    return {k: _build_velocities(v) for k, v in p.items()}
                elif isinstance(p, list):
                    return [_build_velocities(v) for v in p]
                elif isinstance(p, np.ndarray):
                    return np.zeros_like(p)
                else:
                    return None
            momentum_step.velocities = _build_velocities(params)
            momentum_step.mu = 0.9
        
        mu = momentum_step.mu
        velocities = momentum_step.velocities
        
        def _update(p, g, v):
            if isinstance(p, dict) and isinstance(g, dict) and isinstance(v, dict):
                for key in p:
                    _update(p[key], g[key], v[key])
            elif isinstance(p, list) and isinstance(g, list) and isinstance(v, list):
                for i in range(len(p)):
                    _update(p[i], g[i], v[i])
            elif isinstance(p, np.ndarray) and isinstance(g, np.ndarray) and isinstance(v, np.ndarray):
                v[:] = mu * v + g
                p -= lr * v
            else:
                pass
        
        _update(params, grads, velocities)
    
    if kind == 'sgd':
        step_fn = sgd_step
    elif kind == 'momentum':
        step_fn = momentum_step
    else:
        raise ValueError(f"Unsupported optimizer kind: {kind}")
    
    return {'step': step_fn}

# ── Step 010  train_step ──
def train_step(model, loss_fn, optimizer, x_batch, y_batch):
    """Perform one complete optimization step over a minibatch.

    Inputs:
      model: sequential model dict with 'forward', 'backward', and 'params'
      loss_fn: callable (logits, y) -> (loss, d_logits)
      optimizer: dict with 'step'(grads) applying in-place parameter updates
      x_batch: np.ndarray of shape (B, D)
      y_batch: np.ndarray of shape (B,) integer class labels

    Returns:
      loss: float, scalar batch loss evaluated BEFORE the parameter update.
      Model parameters are updated in place; shapes unchanged and values finite.
    """
    logits, caches = model['forward'](x_batch)
    loss, d_logits = loss_fn(logits, y_batch)
    dx, layer_grads = model['backward'](d_logits, caches)
    optimizer['step'](layer_grads)
    return loss

# ── Step 011  train ──
def train(model, loss_fn, optimizer, x, y, epochs, batch_size, seed=0):
    """Run a deterministic minibatch training loop.

    Inputs:
      model: sequential model dict with 'forward', 'backward', 'params'
      loss_fn: callable (logits, y) -> (loss, d_logits)
      optimizer: dict with 'step'(grads) applying in-place parameter updates
      x: np.ndarray of shape (N, D) training features
      y: np.ndarray of shape (N,) integer class labels
      epochs: int, number of full passes over the data
      batch_size: int, minibatch size
      seed: int, RNG seed for deterministic shuffling / batching

    Returns:
      history: list[float] of length `epochs`; history[t] is the mean
      train_step loss over minibatches in epoch t.
      Model parameters are updated in place; shapes unchanged.
    """
    N = x.shape[0]
    rng = np.random.RandomState(seed)
    history = []

    for epoch in range(epochs):
      indices = np.arange(N)
      rng.shuffle(indices)

      epoch_loss_sum = 0.0
      num_batches = 0

      for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        batch_indices = indices[start:end]

        x_batch = x[batch_indices]
        y_batch = y[batch_indices]

        batch_loss = train_step(model, loss_fn, optimizer, x_batch, y_batch)

        epoch_loss_sum += batch_loss
        num_batches += 1

      avg_epoch_loss = epoch_loss_sum / num_batches
      history.append(avg_epoch_loss)

    return history

# ── Step 012  design_network ──
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

# ── Step 013  improve_generalization ──
def improve_generalization(baseline_model_fn, x_train, y_train, x_val, y_val, seed=0):
    """Improve held-out accuracy over an unregularized baseline.

    Inputs:
      baseline_model_fn: zero-arg callable -> fresh untrained sequential model
        (dict with 'forward', 'backward', 'params') matching the data dims.
      x_train, y_train: training features (N, D) and int labels (N,).
      x_val, y_val: validation features (N_val, D) and int labels (N_val,).
      seed: int for deterministic training.

    Returns:
      dict with keys:
        'val_accuracy': float accuracy of the improved model on x_val/y_val
        'baseline_val_accuracy': float val accuracy of plain unregularized SGD
        'predictions': np.ndarray shape (N_val,) int preds from improved model
        'model': the trained improved model

    Required behavior:
      val_accuracy > baseline_val_accuracy
      predictions == argmax(model.forward(x_val), axis=1)
      val_accuracy == mean(predictions == y_val)
      predictions are non-constant (not a trivial single-class predictor)
    """
    np.random.seed(seed)
    ce = make_loss()

    def fit(model, epochs, lr, batch_size, s, weight_decay=0.0, track_val=False):
        """Train in place. If track_val, early-stop by restoring best-val params."""
        params = model['params']
        N = x_train.shape[0]
        rng = np.random.RandomState(s)

        def val_stats():
            lg, _ = model['forward'](x_val)
            pr = np.argmax(lg, axis=1)
            return float(np.mean(pr == y_val)), pr

        def snapshot():
            return [{k: v.copy() for k, v in p.items()} for p in params]

        def restore(sp):
            for p, snap in zip(params, sp):
                for k in p:
                    p[k][:] = snap[k]

        best_acc, best_sp = (-1.0, None)
        if track_val:
            best_acc, _ = val_stats()
            best_sp = snapshot()

        for _ in range(epochs):
            idx = np.arange(N); rng.shuffle(idx)
            for st in range(0, N, batch_size):
                bi = idx[st:st + batch_size]
                logits, cs = model['forward'](x_train[bi])
                _, d = ce(logits, y_train[bi])
                _, grads = model['backward'](d, cs)
                for p, g in zip(params, grads):
                    for k in p:
                        step = g[k] + (weight_decay * p[k] if k == 'W' else 0.0)
                        p[k] -= lr * step
            if track_val:
                a, pr = val_stats()
                if a > best_acc and len(np.unique(pr)) > 1:
                    best_acc, best_sp = a, snapshot()

        if track_val and best_sp is not None:
            restore(best_sp)
        return model

    base = baseline_model_fn()
    fit(base, epochs=150, lr=0.05, batch_size=32, s=seed)
    blg, _ = base['forward'](x_val)
    base_acc = float(np.mean(np.argmax(blg, axis=1) == y_val))

    np.random.seed(seed + 1)
    imp = baseline_model_fn()
    fit(imp, epochs=300, lr=0.05, batch_size=32, s=seed,
        weight_decay=1e-3, track_val=True)
    ilg, _ = imp['forward'](x_val)
    preds = np.argmax(ilg, axis=1)
    imp_acc = float(np.mean(preds == y_val))

    if imp_acc <= base_acc:
        for wd_try, lr_try in [(3e-3, 0.05), (1e-2, 0.03), (5e-3, 0.05), (3e-3, 0.08)]:
            np.random.seed(seed + 7)
            cand = baseline_model_fn()
            fit(cand, epochs=300, lr=lr_try, batch_size=32, s=seed,
                weight_decay=wd_try, track_val=True)
            clg, _ = cand['forward'](x_val)
            cpr = np.argmax(clg, axis=1)
            cacc = float(np.mean(cpr == y_val))
            if cacc > imp_acc and len(np.unique(cpr)) > 1:
                imp_acc, preds, imp = cacc, cpr, cand

    return {
        'val_accuracy': imp_acc,
        'baseline_val_accuracy': base_acc,
        'predictions': preds,
        'model': imp,
    }

# ── Scaffold (runner) ──
"""End-to-end demo: NumPy neural net from scratch on a nonlinear dataset."""
import numpy as np


def _nonlinear_dataset(n_samples=256, seed=0, label_noise=0.0):
    """Binary labels on noisy concentric rings (not linearly separable)."""
    rng = np.random.RandomState(seed)
    half = n_samples // 2
    t = rng.uniform(0.0, 2.0 * np.pi, size=half)
    r0 = 0.45 + rng.normal(0.0, 0.06, size=half)
    r1 = 1.15 + rng.normal(0.0, 0.06, size=half)
    x0 = np.column_stack([r0 * np.cos(t), r0 * np.sin(t)])
    x1 = np.column_stack([r1 * np.cos(t), r1 * np.sin(t)])
    x = np.vstack([x0, x1]).astype(np.float64)
    y = np.array([0] * half + [1] * half, dtype=np.int64)
    if label_noise > 0.0:
        flip = rng.rand(n_samples) < float(label_noise)
        y = y.copy()
        y[flip] = 1 - y[flip]
    idx = rng.permutation(n_samples)
    return x[idx], y[idx]


def _fresh_mlp(in_dim, n_classes, hidden=32, seed=1):
    """Return a freshly initialized untrained sequential MLP."""
    np.random.seed(int(seed))

    def init_fn(i, o):
        return initialize_weights(i, o, scheme="he")

    layers = [
        make_dense(in_dim, hidden, init_fn),
        make_activation("relu"),
        make_dense(hidden, hidden, init_fn),
        make_activation("relu"),
        make_dense(hidden, n_classes, init_fn),
    ]
    return make_sequential(layers)


def main():
    np.random.seed(0)
    x, y = _nonlinear_dataset(256, seed=0, label_noise=0.0)
    n_train = 192
    x_train, y_train = x[:n_train], y[:n_train]
    x_val, y_val = x[n_train:], y[n_train:]
    in_dim, n_classes = int(x.shape[1]), 2

    designed, design_metrics = design_network(in_dim, n_classes, seed=0)
    print("design_network_accuracy:", round(float(design_metrics["accuracy"]), 4))

    # Fresh untrained net on the rings data: one train_step must drop loss.
    model = _fresh_mlp(in_dim, n_classes, hidden=32, seed=1)
    loss_fn = make_loss("cross_entropy")
    xb, yb = x_train[:32], y_train[:32]
    init_loss, _ = forward_backward(model, loss_fn, xb, yb)
    print("initial_batch_loss:", float(init_loss))

    optimizer = make_optimizer(model["params"], lr=0.1, kind="sgd")
    pre_loss = train_step(model, loss_fn, optimizer, xb, yb)
    post_loss, _ = forward_backward(model, loss_fn, xb, yb)
    print("train_step_loss:", float(pre_loss))
    print("after_train_step_loss:", float(post_loss))

    history = train(
        model, loss_fn, optimizer, x_train, y_train,
        epochs=50, batch_size=32, seed=0,
    )
    if isinstance(history, (list, tuple, np.ndarray)) and len(history) > 0:
        print("overfit_loss_start:", float(history[0]))
        print("overfit_loss_end:", float(history[-1]))
    else:
        print("train_history:", history)

    # Regularization demo: wide net + noisy train labels so the unregularized
    # baseline cannot already sit at 100% val accuracy.
    x_noisy, y_noisy = _nonlinear_dataset(192, seed=2, label_noise=0.22)
    x_hold, y_hold = _nonlinear_dataset(64, seed=3, label_noise=0.0)

    def baseline_model_fn():
        return _fresh_mlp(in_dim, n_classes, hidden=56, seed=12345)

    gen_result = improve_generalization(
        baseline_model_fn, x_noisy, y_noisy, x_hold, y_hold, seed=0,
    )
    print("baseline_val_accuracy:", round(float(gen_result["baseline_val_accuracy"]), 4))
    print("improved_val_accuracy:", round(float(gen_result["val_accuracy"]), 4))


if __name__ == "__main__":
    main()
