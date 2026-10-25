def conv2d(x, w, pad=0):
    N, Cin, H, W = x.shape
    Cout, _, kh, kw = w.shape
    Ho = H + 2 * pad - kh + 1
    Wo = W + 2 * pad - kw + 1
    
    if pad > 0:
        xp = x.pad(((0, 0), (0, 0), (pad, pad), (pad, pad)))
    else:
        xp = x
    
    total = None
    for dh in range(kh):
        for dw in range(kw):
            win = xp.shrink(((0, N), (0, Cin), (dh, dh + Ho), (dw, dw + Wo)))
            win = win.reshape(N, 1, Cin, Ho, Wo).expand(N, Cout, Cin, Ho, Wo)
            wt = w.shrink(((0, Cout), (0, Cin), (dh, dh + 1), (dw, dw + 1)))
            wt = wt.reshape(1, Cout, Cin, 1, 1).expand(N, Cout, Cin, Ho, Wo)
            prod = win * wt
            total = prod if total is None else total + prod
    
    return total.sum(2)

Tensor.conv2d = conv2d
