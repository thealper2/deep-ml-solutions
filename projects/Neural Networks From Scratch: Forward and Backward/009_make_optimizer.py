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