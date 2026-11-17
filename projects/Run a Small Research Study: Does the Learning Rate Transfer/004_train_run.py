def train_run(width, param, lr, seed, data, steps=300, batch_size=128):
    """Train one model; return val_loss, per_example losses, diverged and steps_done."""
    X_train, Y_train, X_val, Y_val = data
    n_train = X_train.shape[0]

    model = build_mlp(width, param, seed)
    opt = make_optimizer(model, lr, param)
    g = torch.Generator().manual_seed(1000 + seed)

    steps_done = 0
    for _ in range(steps):
        idx = torch.randint(0, n_train, (batch_size,), generator=g)
        opt.zero_grad()
        out = model(X_train[idx])
        loss = ((out - Y_train[idx]) ** 2).mean()

        lv = float(loss.item())
        if not np.isfinite(lv) or lv > 100:
            return {
                "val_loss": None,
                "per_example": None,
                "diverged": True,
                "steps_done": steps_done,
            }

        loss.backward()
        opt.step()
        steps_done += 1

    with torch.no_grad():
        val_out = model(X_val)
        per_ex = ((val_out - Y_val) ** 2).mean(dim=1).numpy()

    if not np.all(np.isfinite(per_ex)):
        return {
            "val_loss": None,
            "per_example": None,
            "diverged": True,
            "steps_done": steps,
        }

    return {
        "val_loss": float(per_ex.mean()),
        "per_example": per_ex,
        "diverged": False,
        "steps_done": steps,
    }