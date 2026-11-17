def coord_check(param, widths, data, lr=2 ** -10, steps=1, seed=0):
    """{width: RMS change of the output on X_val[:256] after `steps` Adam steps on X_train[:128]}."""
    X_train, Y_train, X_val, Y_val = data
    xb = X_train[:128]
    yb = Y_train[:128]
    probe = X_val[:256]

    result = {}
    for width in widths:
        model = build_mlp(width, param, seed)
        opt = make_optimizer(model, lr, param)

        with torch.no_grad():
            before = model(probe).clone()

        for _ in range(steps):
            opt.zero_grad()
            out = model(xb)
            loss = ((out - yb) ** 2).mean()
            loss.backward()
            opt.step()

        with torch.no_grad():
            after = model(probe)
        
        diff = (after - before).flatten()
        rms = float(torch.sqrt((diff ** 2).mean()))
        result[width] = rms

    return result