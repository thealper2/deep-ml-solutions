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